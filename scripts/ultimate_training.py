#!/usr/bin/env python3
"""
Entrenamiento ULTIMATE con los mejores hiperparámetros encontrados.
Entrena el modelo definitivo usando la configuración optimizada.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from src.rl.callbacks.tensorboard_callback import TensorboardKpiCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


def load_cfg(config_path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_ultimate_env(cfg: dict):
    """Crear entorno optimizado."""
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
    )
    env_core = TrafficLightGymEnv(ec)
    env = Monitor(env_core)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def create_ultimate_model(env, cfg: dict, use_optimized: bool = True):
    """Crear modelo PPO con hiperparámetros optimizados."""
    
    if use_optimized and 'ppo_optimized' in cfg:
        # Usar hiperparámetros optimizados
        ppo_params = cfg['ppo_optimized']
        print("🚀 Usando hiperparámetros optimizados:")
        for key, value in ppo_params.items():
            if key in ['learning_rate', 'ent_coef']:
                print(f"   {key:20}: {value:.2e}")
            else:
                print(f"   {key:20}: {value}")
    else:
        # Usar hiperparámetros base
        ppo_params = cfg['ppo']
        ppo_params['learning_rate'] = ppo_params.pop('lr', 0.0003)
        print("⚠️  Usando hiperparámetros base (no optimizados)")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_params.get('learning_rate', 0.0003),
        gamma=ppo_params.get('gamma', 0.99),
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        clip_range=ppo_params.get('clip_range', 0.2),
        ent_coef=ppo_params.get('ent_coef', 0.01),
        device="cpu",
        verbose=1
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento ULTIMATE del modelo definitivo")
    parser.add_argument("--config", required=True, help="Ruta de configuración optimizada")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps (default: 100K)")
    parser.add_argument("--model-name", default="ppo_ultimate_v1", help="Nombre del modelo")
    parser.add_argument("--no-optimized", action="store_true", help="No usar hiperparámetros optimizados")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Error: Configuración no encontrada: {args.config}")
        return 1
    
    # Cargar configuración
    cfg = load_cfg(args.config)
    
    print("🚀 ENTRENAMIENTO ULTIMATE DEL MODELO DEFINITIVO")
    print("=" * 60)
    print(f"📁 Configuración: {args.config}")
    print(f"⏱️  Total timesteps: {args.timesteps:,}")
    print(f"🏷️  Nombre del modelo: {args.model_name}")
    
    # Mostrar metadatos de optimización si existen
    if 'optimization_metadata' in cfg:
        meta = cfg['optimization_metadata']
        print(f"\n📊 ORIGEN DE LA OPTIMIZACIÓN:")
        print(f"   Score obtenido: {meta.get('best_score_global', 'N/A')}")
        print(f"   Trial origen: {meta.get('best_trial_number', 'N/A')}")
        print(f"   Checkpoint: {meta.get('best_checkpoint_idx', 'N/A')}")
        print(f"   Tiempo optimización: {meta.get('optimization_time_hours', 0):.2f}h")
    
    # Crear entorno
    print(f"\n🌍 Creando entorno...")
    env = make_ultimate_env(cfg)
    
    # Crear modelo con hiperparámetros optimizados
    print(f"\n🤖 Creando modelo PPO...")
    model = create_ultimate_model(env, cfg, use_optimized=not args.no_optimized)
    
    # Configurar callbacks
    print(f"\n📊 Configurando callbacks...")
    
    # TensorBoard callback
    tb_log_name = f"ULTIMATE_{args.model_name}"
    tb_callback = TensorboardKpiCallback()
    
    # Solo usar TensorBoard callback (EvalCallback causa conflictos)
    callbacks = [tb_callback]
    
    print(f"✅ Configuración completada")
    print(f"\n🎯 INICIANDO ENTRENAMIENTO ULTIMATE...")
    print("-" * 60)
    
    # Entrenar modelo
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name
    )
    
    # Guardar modelo final
    model_path = f"models/{args.model_name}.zip"
    model.save(model_path)
    
    print(f"\n✅ ¡ENTRENAMIENTO COMPLETADO!")
    print("=" * 60)
    print(f"💾 Modelo guardado en: {model_path}")
    print(f"📈 TensorBoard logs: logs/tensorboard/{tb_log_name}/")
    print(f"📊 Eval logs: logs/eval/")
    
    # Guardar metadatos del entrenamiento
    training_metadata = {
        'model_name': args.model_name,
        'config_used': args.config,
        'total_timesteps': args.timesteps,
        'used_optimized_hyperparams': not args.no_optimized,
        'training_completed': True
    }
    
    if 'optimization_metadata' in cfg:
        training_metadata['optimization_source'] = cfg['optimization_metadata']
    
    metadata_path = f"models/{args.model_name}_metadata.yaml"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(training_metadata, f, default_flow_style=False, indent=2)
    
    print(f"📋 Metadatos guardados en: {metadata_path}")
    
    print(f"\n🚀 SIGUIENTE PASO - EVALUACIÓN:")
    print(f"   python scripts/eval.py --model models/{args.model_name}.zip")
    
    return 0


if __name__ == "__main__":
    exit(main())


