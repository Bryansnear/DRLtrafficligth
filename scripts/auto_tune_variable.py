#!/usr/bin/env python3

import os
import argparse
import yaml
import optuna
import random
from typing import Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Asegurar PYTHONPATH
if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = os.getcwd()

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_variable_demand_env(cfg: dict, overrides: Dict, demand_profiles: List[str]) -> TrafficLightGymEnv:
    """Crear entorno con demanda variable para autotuning."""
    
    class VariableDemandTuningEnv(TrafficLightGymEnv):
        def __init__(self, base_config, profiles):
            self.demand_profiles = profiles
            self.current_profile = None
            super().__init__(base_config)
            
        def reset(self, *, seed=None, options=None):
            # Cambiar a perfil aleatorio en cada reset
            self.current_profile = random.choice(self.demand_profiles)
            self.cfg.demand_profile_path = self.current_profile
            self.cfg.dynamic_demand = True
            return super().reset(seed=seed, options=options)
    
    # Aplicar overrides de autotuning
    rc = cfg["reward"]
    rc = {**rc, **overrides}
    
    ec = EnvConfig(
        sumo_cfg_path=cfg["sumo"]["cfg_path"],
        step_length=float(cfg["sumo"]["step_length"]),
        control_interval=int(cfg["control"]["control_interval"]),
        min_green=int(cfg["control"]["min_green"]),
        yellow=int(cfg["control"]["yellow"]),
        all_red=int(cfg["control"]["all_red"]),
        e2_ids=tuple(cfg["detectors"]["e2_ids"]),
        v_free=float(cfg["detectors"]["v_free"]),
        jam_length_thr_m=float(cfg["detectors"]["jam_length_thr_m"]),
        e2_capacity_per_lane=int(cfg["detectors"]["e2_capacity_per_lane"]),
        w_served=float(rc["w_served"]),
        w_queue=float(rc["w_queue"]),
        w_backlog=float(rc["w_backlog"]),
        w_switch=float(rc["w_switch"]),
        w_spill=float(rc["w_spill"]),
        kappa_backlog=int(rc["kappa_backlog"]),
        sat_headway_s=float(rc["sat_headway_s"]),
        use_gui=False,
        dynamic_demand=True
    )
    
    env = VariableDemandTuningEnv(ec, demand_profiles)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def objective_variable_demand(trial: optuna.Trial, cfg: dict, demand_profiles: List[str]) -> float:
    """Función objetivo optimizada para demanda variable."""
    
    # Espacio de búsqueda adaptado para demanda variable
    overrides = {
        # Rebalancear para demanda variable (menos reactivo, más estable)
        "w_served": trial.suggest_float("w_served", 0.2, 0.8),        
        "w_queue": trial.suggest_float("w_queue", 0.1, 0.6),          # MENOR penalización inmediata
        "w_backlog": trial.suggest_float("w_backlog", 0.3, 0.9),      # MAYOR penalización acumulativa  
        "w_switch": trial.suggest_float("w_switch", 0.05, 0.25),      # MAYOR penalización por cambios
        "w_spill": trial.suggest_float("w_spill", 0.1, 0.5),         
        "kappa_backlog": trial.suggest_int("kappa_backlog", 8, 20),   # MAYOR memoria histórica
    }
    
    # Parámetros de control adaptados
    cfg2 = yaml.safe_load(yaml.dump(cfg))
    cfg2["control"]["min_green"] = int(trial.suggest_int("min_green", 8, 18))      # Verde MÁS LARGO
    cfg2["control"]["control_interval"] = int(trial.suggest_int("control_interval", 5, 12))  # Intervalos variables
    
    env = make_variable_demand_env(cfg2, overrides, demand_profiles)
    model = PPO(
        policy=cfg["ppo"]["policy"], 
        env=env, 
        learning_rate=cfg["ppo"]["lr"], 
        gamma=cfg["ppo"]["gamma"],
        n_steps=cfg["ppo"]["n_steps"], 
        batch_size=cfg["ppo"]["batch_size"], 
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"], 
        device="cpu", 
        verbose=0
    )
    
    # Entrenamiento más largo para demanda variable
    model.learn(total_timesteps=8192)
    
    # Evaluación robusta con múltiples resets para probar adaptabilidad
    total_performance = 0.0
    num_evaluations = 4  # Múltiples episodios con diferentes perfiles
    
    for eval_idx in range(num_evaluations):
        obs, info = env.reset()
        episode_served = 0.0
        episode_switches = 0
        episode_queues = 0.0
        episode_spills = 0
        episode_rewards = 0.0
        
        steps_per_eval = 15  # 15 pasos por evaluación
        
        for step in range(steps_per_eval):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            
            episode_served += info.get("served_cnt", 0.0)
            episode_switches += int(info.get("switch", 0))
            episode_queues += sum(info.get("queues", {}).values())
            episode_rewards += reward
            
            # Contar spillbacks críticos
            if any(q > 6 for q in info.get("queues", {}).values()):
                episode_spills += 1
        
        # Score por episodio (adaptado para demanda variable)
        throughput_score = episode_served / steps_per_eval  # veh/paso
        stability_penalty = (episode_switches / steps_per_eval) * 0.4  # Penalizar inestabilidad MÁS
        queue_penalty = (episode_queues / (steps_per_eval * 24)) * 0.3  # Normalizado por capacidad total
        spill_penalty = (episode_spills / steps_per_eval) * 0.7  # Penalizar spillbacks MÁS
        adaptability_bonus = episode_rewards / steps_per_eval * 0.2  # Bonus por adaptarse bien
        
        episode_score = throughput_score - stability_penalty - queue_penalty - spill_penalty + adaptability_bonus
        total_performance += episode_score
    
    # Promedio de todas las evaluaciones
    avg_performance = total_performance / num_evaluations
    
    try:
        env.core.close()
    except Exception:
        pass
        
    return float(avg_performance)


def main():
    parser = argparse.ArgumentParser(description="Autotuning específico para demanda variable")
    parser.add_argument("--trials", type=int, default=20, help="Número de trials")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    args = parser.parse_args()
    
    # Perfiles de demanda para autotuning
    demand_profiles = [
        os.path.join("experiments", "demands", "rush_morning.yaml"),
        os.path.join("experiments", "demands", "rush_evening.yaml"), 
        os.path.join("experiments", "demands", "midday_moderate.yaml"),
        os.path.join("experiments", "demands", "off_peak_low.yaml")
    ]
    
    # Verificar que existen los perfiles
    existing_profiles = [p for p in demand_profiles if os.path.exists(p)]
    if len(existing_profiles) < 2:
        print("Error: Se necesitan al menos 2 perfiles de demanda")
        return 1
    
    cfg = load_cfg(os.path.join("experiments", "configs", "base.yaml"))
    
    print("AUTOTUNING ESPECÍFICO PARA DEMANDA VARIABLE")
    print("=" * 60)
    print(f"   Trials: {args.trials}")
    print(f"   Sampler: {args.sampler.upper()}")
    print(f"   Perfiles de demanda: {len(existing_profiles)}")
    print("   Objetivo: Optimizar estabilidad + adaptabilidad")
    print()
    
    # Configurar sampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    else:
        sampler = optuna.samplers.RandomSampler(seed=42)
    
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: objective_variable_demand(t, cfg, existing_profiles), 
        n_trials=args.trials
    )
    
    print("RESULTADOS DEL AUTOTUNING VARIABLE:")
    print("=" * 60)
    print(f"Mejor score: {study.best_value:.4f}")
    print("\nMejores parámetros:")
    for key, value in study.best_params.items():
        print(f"  {key:20}: {value}")
    
    # Guardar configuración optimizada para demanda variable
    optimized_cfg = yaml.safe_load(yaml.dump(cfg))
    
    # Aplicar mejores parámetros
    reward_params = ['w_served', 'w_queue', 'w_backlog', 'w_switch', 'w_spill', 'kappa_backlog']
    control_params = ['min_green', 'control_interval']
    
    for param in reward_params:
        if param in study.best_params:
            optimized_cfg['reward'][param] = study.best_params[param]
    
    for param in control_params:
        if param in study.best_params:
            optimized_cfg['control'][param] = study.best_params[param]
    
    # Guardar configuración optimizada
    output_path = os.path.join("experiments", "configs", "variable_demand_optimized.yaml")
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(optimized_cfg, f, default_flow_style=False, indent=2)
    
    print(f"\nConfiguracion optimizada guardada en: {output_path}")
    print(f"Entrena con: python scripts/train_variable_demand.py --config {output_path}")
    
    # Guardar study
    import pickle
    with open("autotuning_variable_study.pkl", "wb") as f:
        pickle.dump(study, f)
    
    return 0


if __name__ == "__main__":
    exit(main())


