#!/usr/bin/env python3
"""
Entrenamiento ULTIMATE SIMPLIFICADO - Sin callbacks problemáticos
Versión de diagnóstico para evitar congelamientos
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


def load_cfg(config_path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_simple_env(cfg: dict):
    """Crear entorno simplificado."""
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


def create_simple_model(env, cfg: dict):
    """Crear modelo PPO simplificado."""
    
    # Usar hiperparámetros optimizados si existen
    if 'ppo_optimized' in cfg:
        ppo_params = cfg['ppo_optimized']
        print("🚀 Usando hiperparámetros optimizados:")
        for key, value in ppo_params.items():
            if key in ['learning_rate', 'ent_coef']:
                print(f"   {key:20}: {value:.2e}")
            else:
                print(f"   {key:20}: {value}")
                
        lr = ppo_params.get('learning_rate', 0.0003)
        gamma = ppo_params.get('gamma', 0.99) 
        clip = ppo_params.get('clip_range', 0.2)
        ent = ppo_params.get('ent_coef', 0.01)
    else:
        # Usar hiperparámetros base
        ppo_params = cfg['ppo']
        lr = ppo_params.get('lr', 0.0003)
        gamma = ppo_params.get('gamma', 0.99)
        clip = ppo_params.get('clip_range', 0.2) 
        ent = ppo_params.get('ent_coef', 0.01)
        print("⚠️  Usando hiperparámetros base")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        gamma=gamma,
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        clip_range=clip,
        ent_coef=ent,
        device="cpu",
        verbose=1,
        tensorboard_log="logs/tensorboard"  # Solo TensorBoard básico
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento ULTIMATE simplificado")
    parser.add_argument("--config", required=True, help="Ruta de configuración")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps (default: 50K)")
    parser.add_argument("--model-name", default="ppo_ultimate_simple", help="Nombre del modelo")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Error: Configuración no encontrada: {args.config}")
        return 1
    
    # Cargar configuración
    cfg = load_cfg(args.config)
    
    print("🚀 ENTRENAMIENTO ULTIMATE SIMPLIFICADO")
    print("=" * 50)
    print(f"📁 Configuración: {args.config}")
    print(f"⏱️  Total timesteps: {args.timesteps:,}")
    print(f"🏷️  Nombre: {args.model_name}")
    
    # Mostrar origen de optimización
    if 'optimization_metadata' in cfg:
        meta = cfg['optimization_metadata']
        print(f"\n📊 ORIGEN:")
        print(f"   Score: {meta.get('best_score_global', 'N/A')}")
        print(f"   Trial: {meta.get('best_trial_number', 'N/A')}")
    
    try:
        # Crear entorno
        print(f"\n🌍 Creando entorno...")
        env = make_simple_env(cfg)
        print(f"✅ Entorno creado correctamente")
        
        # Crear modelo
        print(f"\n🤖 Creando modelo PPO...")
        model = create_simple_model(env, cfg)
        print(f"✅ Modelo creado correctamente")
        
        # Probar el entorno primero
        print(f"\n🧪 Probando entorno...")
        obs, info = env.reset()
        print(f"✅ Reset exitoso - obs shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"✅ Step exitoso - reward: {reward}")
        
        print(f"\n🎯 INICIANDO ENTRENAMIENTO...")
        print("-" * 50)
        
        # Entrenar sin callbacks complejos
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=f"SIMPLE_{args.model_name}"
        )
        
        # Guardar modelo
        model_path = f"models/{args.model_name}.zip"
        model.save(model_path)
        
        print(f"\n✅ ¡ENTRENAMIENTO COMPLETADO!")
        print(f"💾 Modelo guardado: {model_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        try:
            env.close()
        except:
            pass


if __name__ == "__main__":
    exit(main())
