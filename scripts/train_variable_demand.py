#!/usr/bin/env python3

import os
import yaml
import argparse
import random
from typing import List

# Asegurar PYTHONPATH
if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = os.getcwd()

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from src.rl.callbacks.tensorboard_callback import TensorboardKpiCallback


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_variable_demand(cfg: dict, demand_profiles: List[str]) -> TrafficLightGymEnv:
    """Crear entorno que cambia de perfil de demanda aleatoriamente en cada reset."""
    
    class VariableDemandEnv(TrafficLightGymEnv):
        def __init__(self, base_config, profiles):
            self.demand_profiles = profiles
            self.current_profile = None
            # Inicializar con perfil aleatorio
            self._switch_profile()
            super().__init__(base_config)
            
        def _switch_profile(self):
            """Cambiar a perfil de demanda aleatorio."""
            self.current_profile = random.choice(self.demand_profiles)
            print(f"🔄 Cambiando a perfil: {os.path.basename(self.current_profile)}")
            
        def reset(self, *, seed=None, options=None):
            """Reset con nuevo perfil de demanda aleatorio."""
            # Cambiar perfil antes del reset
            self._switch_profile()
            # Actualizar configuración del core
            self.cfg.demand_profile_path = self.current_profile
            self.cfg.dynamic_demand = True
            
            return super().reset(seed=seed, options=options)
    
    # Configuración base
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
        w_served=float(cfg["reward"]["w_served"]),
        w_queue=float(cfg["reward"]["w_queue"]),
        w_backlog=float(cfg["reward"]["w_backlog"]),
        w_switch=float(cfg["reward"]["w_switch"]),
        w_spill=float(cfg["reward"]["w_spill"]),
        kappa_backlog=int(cfg["reward"]["kappa_backlog"]),
        sat_headway_s=float(cfg["reward"]["sat_headway_s"]),
        use_gui=False,
        dynamic_demand=True
    )
    
    env = VariableDemandEnv(ec, demand_profiles)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PPO con demanda variable")
    parser.add_argument("--timesteps", type=int, default=15000, 
                       help="Total de timesteps de entrenamiento")
    parser.add_argument("--profiles", nargs="+", 
                       default=["rush_morning.yaml", "rush_evening.yaml", "midday_moderate.yaml", "off_peak_low.yaml"],
                       help="Perfiles de demanda a alternar durante entrenamiento")
    parser.add_argument("--model-name", default="ppo_variable_demand", 
                       help="Nombre del modelo a guardar")
    parser.add_argument("--config", default=os.path.join("experiments", "configs", "base.yaml"),
                       help="Archivo de configuración a usar")
    
    args = parser.parse_args()
    
    # Cargar configuración (puede ser optimizada o base)
    cfg = load_cfg(args.config)
    
    # Verificar que existan los perfiles de demanda
    demand_profiles = []
    for profile in args.profiles:
        full_path = os.path.join("experiments", "demands", profile)
        if os.path.exists(full_path):
            demand_profiles.append(full_path)
        else:
            print(f"⚠️  Perfil no encontrado: {full_path}")
    
    if not demand_profiles:
        print("❌ Error: No se encontraron perfiles de demanda válidos")
        return 1
    
    print("🚦 ENTRENAMIENTO PPO CON DEMANDA VARIABLE")
    print("=" * 50)
    print(f"📊 Perfiles de demanda ({len(demand_profiles)}):")
    for profile in demand_profiles:
        print(f"  - {os.path.basename(profile)}")
    print(f"🎯 Timesteps totales: {args.timesteps:,}")
    print(f"💾 Modelo final: models/{args.model_name}.zip")
    print()
    
    # Crear entorno con demanda variable
    env = make_env_variable_demand(cfg, demand_profiles)
    
    # Crear modelo PPO
    model = PPO(
        policy=cfg["ppo"]["policy"],
        env=env,
        learning_rate=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
        verbose=1,
        tensorboard_log=os.path.join("logs", "tensorboard"),
        device="cpu"
    )
    
    print("🤖 Iniciando entrenamiento con demanda variable...")
    print("   (El agente experimentará diferentes patrones de tráfico en cada episodio)")
    
    # Entrenar
    model.learn(
        total_timesteps=args.timesteps,
        callback=TensorboardKpiCallback(),
        progress_bar=True
    )
    
    # Guardar modelo
    model_path = os.path.join("models", f"{args.model_name}.zip")
    model.save(model_path)
    print(f"✅ Modelo guardado en: {model_path}")
    
    # Cerrar entorno
    try:
        if hasattr(env, "core"):
            env.core.close()
    except Exception:
        pass
    
    print("\n🎉 ¡Entrenamiento con demanda variable completado!")
    print(f"💡 Evalúa el modelo con: python scripts/eval_variable_demand.py --model {model_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

