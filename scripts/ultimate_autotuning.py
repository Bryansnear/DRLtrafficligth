#!/usr/bin/env python3

import os
import argparse
import yaml
import optuna
import random
import numpy as np
import time
from typing import Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gymnasium.wrappers import TimeLimit

# Asegurar PYTHONPATH
if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = os.getcwd()

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_ultimate_env(cfg: dict, overrides: Dict, demand_profiles: List[str]) -> TrafficLightGymEnv:
    """Crear entorno ultimate con demanda variable para autotuning extensivo."""
    
    class UltimateVariableDemandEnv(TrafficLightGymEnv):
        def __init__(self, base_config, profiles):
            self.demand_profiles = profiles
            self.current_profile = None
            self.reset_count = 0
            super().__init__(base_config)
            
        def reset(self, *, seed=None, options=None):
            # Rotar entre perfiles de manera balanceada
            profile_idx = self.reset_count % len(self.demand_profiles)
            self.current_profile = self.demand_profiles[profile_idx]
            self.reset_count += 1
            
            # Actualizar configuración
            self.cfg.demand_profile_path = self.current_profile
            self.cfg.dynamic_demand = True
            return super().reset(seed=seed, options=options)
    
    # Aplicar overrides del autotuning
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
    
    env = UltimateVariableDemandEnv(ec, demand_profiles)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def objective_ultimate(trial: optuna.Trial, cfg: dict, demand_profiles: List[str]) -> float:
    """Función objetivo ultimate con early stopping inteligente."""
    
    print(f"\n🔥 TRIAL {trial.number:3d}: Optimizando configuración avanzada...")
    
    # Espacio de búsqueda expandido y más granular
    overrides = {
        # Pesos de recompensa con más granularidad
        "w_served": trial.suggest_float("w_served", 0.1, 0.9, step=0.05),        
        "w_queue": trial.suggest_float("w_queue", 0.05, 0.8, step=0.05),        
        "w_backlog": trial.suggest_float("w_backlog", 0.1, 0.95, step=0.05),    
        "w_switch": trial.suggest_float("w_switch", 0.01, 0.3, step=0.01),      
        "w_spill": trial.suggest_float("w_spill", 0.05, 0.6, step=0.05),        
        "kappa_backlog": trial.suggest_int("kappa_backlog", 5, 25),             
        "sat_headway_s": trial.suggest_float("sat_headway_s", 1.5, 3.0, step=0.1),
    }
    
    # Parámetros de control más granulares
    cfg2 = yaml.safe_load(yaml.dump(cfg))
    cfg2["control"]["min_green"] = int(trial.suggest_int("min_green", 3, 20))      
    cfg2["control"]["control_interval"] = int(trial.suggest_int("control_interval", 3, 15))
    
    # Hiperparámetros de PPO también optimizables
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999, step=0.005)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4, step=0.05)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    
    env = make_ultimate_env(cfg2, overrides, demand_profiles)
    
    # Modelo con hiperparámetros optimizados
    model = PPO(
        policy="MlpPolicy", 
        env=env, 
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=cfg["ppo"]["n_steps"], 
        batch_size=cfg["ppo"]["batch_size"], 
        clip_range=clip_range,
        ent_coef=ent_coef,
        device="cpu", 
        verbose=0
    )
    
    # 🚀 EARLY STOPPING STAGE 1: Entrenamiento progresivo con checkpoints
    checkpoint_timesteps = [2048, 4096, 8192, 12288]  # Checkpoints progresivos
    early_scores = []
    
    for checkpoint_idx, target_timesteps in enumerate(checkpoint_timesteps):
        if checkpoint_idx == 0:
            model.learn(total_timesteps=target_timesteps)
        else:
            model.learn(total_timesteps=target_timesteps - checkpoint_timesteps[checkpoint_idx-1])
        
        # Evaluación rápida en cada checkpoint
        quick_score = quick_evaluation(model, env, trial, checkpoint_idx)
        early_scores.append(quick_score)
        
        # 🛑 EARLY STOPPING: Descartar trials pobres temprano (más tolerante)
        if checkpoint_idx >= 1:  # A partir del 2do checkpoint
            # Condición 1: Score extremadamente bajo (más tolerante)
            if quick_score < 0.3:
                print(f"   ⚠️  EARLY STOP: Score extremadamente bajo ({quick_score:.3f})")
                trial.report(quick_score, checkpoint_idx)
                raise optuna.TrialPruned()
            
            # Condición 2: No hay mejora entre checkpoints (más tolerante)
            if checkpoint_idx >= 2 and quick_score <= early_scores[checkpoint_idx-1] + 0.05:
                print(f"   ⚠️  EARLY STOP: Sin mejora significativa")
                trial.report(quick_score, checkpoint_idx)
                raise optuna.TrialPruned()
            
            # Condición 3: Tendencia descendente sostenida (requiere más evidencia)
            if checkpoint_idx >= 3 and all(early_scores[i] >= early_scores[i+1] + 0.02 for i in range(checkpoint_idx-2)):
                print(f"   ⚠️  EARLY STOP: Tendencia descendente sostenida")
                trial.report(quick_score, checkpoint_idx)
                raise optuna.TrialPruned()
        
        # Reportar score intermedio para pruning de Optuna
        trial.report(quick_score, checkpoint_idx)
        
        # Verificar si el trial fue podado por el pruner de Optuna
        if trial.should_prune():
            print(f"   ✂️  PRUNED por Optuna en checkpoint {checkpoint_idx}")
            raise optuna.TrialPruned()
    
    print(f"   ✅ Pasó todos los checkpoints: {early_scores}")
    
    # 🎯 EVALUACIÓN FINAL COMPLETA (solo para trials prometedores)
    performance_history = []
    total_performance = 0.0
    num_evaluations = 4  # Reducido de 6 a 4 para trials que llegaron aquí
    
    for eval_idx in range(num_evaluations):
        obs, info = env.reset()
        
        episode_served = 0.0
        episode_switches = 0
        episode_queues = 0.0
        episode_spills = 0
        episode_rewards = 0.0
        episode_violations = 0
        
        steps_per_eval = 20  # Reducido de 25 a 20
        
        for step in range(steps_per_eval):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            
            served = info.get("served_cnt", 0.0)
            switch = int(info.get("switch", 0))
            queues = info.get("queues", {})
            
            episode_served += served
            episode_switches += switch
            episode_queues += sum(queues.values())
            episode_rewards += reward
            
            # Métricas avanzadas
            if any(q > 8 for q in queues.values()):
                episode_spills += 1
            
            if any(q > 12 for q in queues.values()):
                episode_violations += 1
        
        # Score multi-objetivo
        throughput_score = episode_served / steps_per_eval
        efficiency_score = throughput_score / max(1, episode_switches / steps_per_eval)
        stability_penalty = (episode_switches / steps_per_eval) * 0.3
        queue_penalty = (episode_queues / (steps_per_eval * 24)) * 0.4
        spill_penalty = (episode_spills / steps_per_eval) * 0.8
        violation_penalty = (episode_violations / steps_per_eval) * 1.5
        reward_bonus = max(0, episode_rewards / steps_per_eval) * 0.3
        
        episode_score = (
            throughput_score + 
            efficiency_score * 0.2 + 
            reward_bonus - 
            stability_penalty - 
            queue_penalty - 
            spill_penalty - 
            violation_penalty
        )
        
        performance_history.append(episode_score)
        total_performance += episode_score
    
    # Métricas finales con estabilidad
    avg_performance = total_performance / num_evaluations
    performance_std = np.std(performance_history)
    stability_bonus = max(0, (2.0 - performance_std) * 0.1)
    
    final_score = avg_performance + stability_bonus
    
    try:
        env.core.close()
    except Exception:
        pass
    
    print(f"   📊 Score promedio: {avg_performance:.3f}")
    print(f"   📈 Estabilidad: {stability_bonus:.3f} (std: {performance_std:.3f})")  
    print(f"   🏆 Score final: {final_score:.3f}")
        
    return float(final_score)


def quick_evaluation(model, env, trial, checkpoint_idx: int) -> float:
    """Evaluación rápida para early stopping."""
    obs, info = env.reset()
    
    total_served = 0.0
    total_switches = 0
    total_queues = 0.0
    total_violations = 0
    steps = 10  # Evaluación muy rápida
    
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        
        served = info.get("served_cnt", 0.0)
        switch = int(info.get("switch", 0))
        queues = info.get("queues", {})
        
        total_served += served
        total_switches += switch  
        total_queues += sum(queues.values())
        
        # Violaciones críticas
        if any(q > 10 for q in queues.values()):
            total_violations += 1
    
    # Score rápido simple
    throughput_score = total_served / steps
    penalty = (total_queues / (steps * 24)) + (total_violations / steps) * 2.0
    quick_score = max(0, throughput_score - penalty)
    
    print(f"     Checkpoint {checkpoint_idx}: {quick_score:.3f} (served: {total_served:.1f}, violations: {total_violations})")
    
    return quick_score


def main():
    parser = argparse.ArgumentParser(description="Autotuning ULTIMATE para el modelo definitivo")
    parser.add_argument("--trials", type=int, default=50, help="Número de trials (default: 50)")
    parser.add_argument("--timeout", type=int, default=14400, help="Timeout en segundos (4 horas)")
    parser.add_argument("--sampler", choices=["tpe", "cmaes"], default="tpe")
    parser.add_argument("--pruning", action="store_true", help="Usar pruning automático")
    
    args = parser.parse_args()
    
    # Perfiles de demanda balanceados
    demand_profiles = [
        os.path.join("experiments", "demands", "rush_morning.yaml"),
        os.path.join("experiments", "demands", "rush_evening.yaml"), 
        os.path.join("experiments", "demands", "midday_moderate.yaml"),
        os.path.join("experiments", "demands", "off_peak_low.yaml")
    ]
    
    existing_profiles = [p for p in demand_profiles if os.path.exists(p)]
    if len(existing_profiles) < 3:
        print("❌ Error: Se necesitan al menos 3 perfiles de demanda")
        return 1
    
    cfg = load_cfg(os.path.join("experiments", "configs", "base.yaml"))
    
    print("🚀 AUTOTUNING ULTIMATE PARA MODELO DEFINITIVO")
    print("=" * 70)
    print(f"   🎯 Trials objetivo: {args.trials}")
    print(f"   ⏱️  Timeout: {args.timeout/3600:.1f} horas") 
    print(f"   🧠 Sampler: {args.sampler.upper()}")
    print(f"   ✂️  Pruning: {'✅' if args.pruning else '❌'}")
    print(f"   📊 Perfiles de demanda: {len(existing_profiles)}")
    print(f"   🎯 Optimizando: Pesos + Control + PPO hiperparámetros")
    print("   📈 Objetivo: Máxima estabilidad + throughput + eficiencia")
    print()
    
    # Configurar sampler avanzado
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=10,  # Más exploracion inicial
            n_ei_candidates=24,   # Más candidatos por iteración
            multivariate=True     # Optimización multivariada
        )
    else:  # CMA-ES
        sampler = optuna.samplers.CmaEsSampler(
            seed=42,
            n_startup_trials=10,
            restart_strategy="ipop"  # Restart strategy
        )
    
    # Configurar pruning si se solicita
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=5,
        interval_steps=5
    ) if args.pruning else optuna.pruners.NopPruner()
    
    # Crear study con configuración avanzada
    study = optuna.create_study(
        direction="maximize", 
        sampler=sampler,
        pruner=pruner,
        study_name="ultimate_traffic_optimization"
    )
    
    # Función objetivo con timeout
    def objective_with_timeout(trial):
        return objective_ultimate(trial, cfg, existing_profiles)
    
    print("🔥 Iniciando optimización ultimate...")
    start_time = time.time()
    
    try:
        study.optimize(
            objective_with_timeout, 
            n_trials=args.trials,
            timeout=args.timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimización interrumpida por usuario")
    except Exception as e:
        print(f"\n❌ Error durante optimización: {e}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🏁 OPTIMIZACIÓN COMPLETADA")
    print("=" * 70)
    print(f"⏱️  Tiempo transcurrido: {elapsed_time/3600:.2f} horas")
    print(f"🧪 Trials completados: {len(study.trials)}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"✅ Trials exitosos: {len(completed_trials)}")
    print(f"✂️  Trials podados: {len(pruned_trials)}")
    
    # NUEVO: Buscar mejor checkpoint de TODOS los trials (podados y completados)
    print(f"\n🔍 ANALIZANDO MEJOR CHECKPOINT DE TODOS LOS TRIALS...")
    
    best_trial_global = None
    best_score_global = float('-inf')
    best_checkpoint_idx = -1
    
    for trial in study.trials:
        print(f"\n  Trial {trial.number:2d} ({trial.state.name}):")
        
        # Analizar checkpoints intermedios
        if hasattr(trial, 'intermediate_values') and trial.intermediate_values:
            for checkpoint_idx, score in trial.intermediate_values.items():
                print(f"    Checkpoint {checkpoint_idx}: {score:.6f}")
                
                if score > best_score_global:
                    best_score_global = score
                    best_trial_global = trial
                    best_checkpoint_idx = checkpoint_idx
                    print(f"      🏆 NUEVO MEJOR CHECKPOINT!")
        
        # También considerar score final si existe
        if hasattr(trial, 'value') and trial.value is not None:
            print(f"    Final score: {trial.value:.6f}")
            if trial.value > best_score_global:
                best_score_global = trial.value
                best_trial_global = trial
                best_checkpoint_idx = -1  # -1 = score final
                print(f"      🏆 NUEVO MEJOR SCORE FINAL!")
    
    print(f"\n🎯 RESULTADO DEL ANÁLISIS GLOBAL:")
    print(f"   Mejor score encontrado: {best_score_global:.6f}")
    print(f"   Trial origen: {best_trial_global.number if best_trial_global else 'None'}")
    print(f"   Checkpoint: {best_checkpoint_idx if best_checkpoint_idx >= 0 else 'Final'}")
    
    # Usar los parámetros del mejor checkpoint global
    if best_trial_global is None:
        print("❌ No se encontraron trials válidos")
        return 1
        
    best_params = best_trial_global.params
    
    # Mostrar mejores parámetros del mejor checkpoint global
    print(f"\n🎯 MEJORES PARÁMETROS (del mejor checkpoint):")
    print("=" * 70)
    
    reward_params = ['w_served', 'w_queue', 'w_backlog', 'w_switch', 'w_spill', 'kappa_backlog', 'sat_headway_s']
    control_params = ['min_green', 'control_interval'] 
    ppo_params = ['learning_rate', 'gamma', 'clip_range', 'ent_coef']
    
    print("📊 PESOS DE RECOMPENSA:")
    for param in reward_params:
        if param in best_params:
            print(f"   {param:20}: {best_params[param]}")
    
    print(f"\n🎮 PARÁMETROS DE CONTROL:")
    for param in control_params:
        if param in best_params:
            print(f"   {param:20}: {best_params[param]}")
    
    print(f"\n🤖 HIPERPARÁMETROS PPO:")
    for param in ppo_params:
        if param in best_params:
            value = best_params[param]
            if param == 'learning_rate' or param == 'ent_coef':
                print(f"   {param:20}: {value:.2e}")
            else:
                print(f"   {param:20}: {value}")
    
    # Crear configuración con parámetros del mejor checkpoint global
    ultimate_cfg = yaml.safe_load(yaml.dump(cfg))
    
    # Aplicar mejores parámetros por categorías
    for param in reward_params:
        if param in best_params:
            ultimate_cfg['reward'][param] = best_params[param]
    
    for param in control_params:
        if param in best_params:
            ultimate_cfg['control'][param] = best_params[param]
    
    # Crear sección PPO optimizada
    ultimate_cfg['ppo_optimized'] = {}
    for param in ppo_params:
        if param in best_params:
            ultimate_cfg['ppo_optimized'][param] = best_params[param]
    
    # Agregar metadatos de optimización
    ultimate_cfg['optimization_metadata'] = {
        'trials_completed': len(study.trials),
        'best_trial_number': best_trial_global.number,
        'best_checkpoint_idx': best_checkpoint_idx,
        'best_score_global': best_score_global,
        'optimization_time_hours': elapsed_time / 3600,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': f'Parameters from trial {best_trial_global.number} with best checkpoint score {best_score_global:.6f}'
    }
    
    # Guardar configuración ultimate
    ultimate_config_path = os.path.join("experiments", "configs", "ultimate_optimized.yaml")
    with open(ultimate_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(ultimate_cfg, f, default_flow_style=False, indent=2)
    
    print(f"\n💾 Configuración ultimate guardada en: {ultimate_config_path}")
    
    # Guardar study completo
    import pickle
    study_path = "ultimate_autotuning_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    print(f"📁 Study completo guardado en: {study_path}")
    print(f"\n🚀 Siguiente paso:")
    print(f"   python scripts/ultimate_training.py --config {ultimate_config_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
