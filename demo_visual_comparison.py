#!/usr/bin/env python3
"""
Demo visual para comparar baseline fijo vs modelo PPO entrenado.
Muestra ambos escenarios con SUMO GUI para observar las diferencias.
"""

import os
import sys
import time
import yaml
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from stable_baselines3 import PPO


def setup_sumo_env():
    """Configurar variables de entorno de SUMO."""
    if not os.environ.get("SUMO_HOME"):
        os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
    
    sumo_bin = os.path.join(os.environ["SUMO_HOME"], "bin")
    if sumo_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] += f";{sumo_bin}"
    
    print(f"SUMO_HOME: {os.environ.get('SUMO_HOME')}")
    print(f"SUMO en PATH: {sumo_bin}")


def load_cfg(path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, use_gui: bool = True) -> TrafficLightGymEnv:
    """Crear entorno con GUI habilitada."""
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
        use_gui=use_gui,
    )
    return TrafficLightGymEnv(ec)


def demo_baseline_fixed(cfg: dict, steps: int = 20):
    """Demo de baseline con semáforo fijo."""
    print("\n" + "="*60)
    print("🚦 DEMO: SEMÁFORO FIJO (BASELINE - SIN IA)")
    print("="*60)
    print("Estrategia: Ciclo fijo de 30s verde E-W, 30s verde N-S")
    print("Presiona ENTER para continuar...")
    input()
    
    env = make_env(cfg, use_gui=True)
    obs, info = env.reset()
    
    # Estrategia fija: alternar cada 6 intervalos (30 segundos)
    fixed_cycle = [0] * 6 + [1] * 6  # 30s verde E-W, 30s verde N-S
    action_idx = 0
    
    total_served = 0.0
    total_queues = 0.0
    
    for step in range(1, steps + 1):
        # Acción fija cíclica
        action = fixed_cycle[action_idx % len(fixed_cycle)]
        action_idx += 1
        
        obs, reward, term, trunc, info = env.step(action)
        
        served = info.get("served_cnt", 0.0)
        queues = info.get("queues", {})
        queue_sum = sum(queues.values())
        
        total_served += served
        total_queues += queue_sum
        
        print(f"Paso {step:02d}: Acción={action} (fija), Servidos={served:.1f}, Colas={queue_sum:.1f}, Recompensa={reward:.3f}")
        
        if term or trunc:
            obs, info = env.reset()
        
        time.sleep(0.5)  # Pausa para observar
    
    print(f"\n📊 RESULTADOS BASELINE:")
    print(f"  Total servidos: {total_served:.1f}")
    print(f"  Promedio colas: {total_queues/steps:.2f}")
    
    try:
        env.core.close()
    except:
        pass
    
    print("\nPresiona ENTER para continuar con el modelo PPO...")
    input()


def demo_ppo_model(cfg: dict, model_path: str, steps: int = 20):
    """Demo del modelo PPO entrenado."""
    print("\n" + "="*60)
    print("🤖 DEMO: MODELO PPO ENTRENADO (CON IA)")
    print("="*60)
    print("Estrategia: Control inteligente adaptativo")
    print("Presiona ENTER para continuar...")
    input()
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado: {model_path}")
        return
    
    # Cargar modelo
    model = PPO.load(model_path)
    
    env = make_env(cfg, use_gui=True)
    obs, info = env.reset()
    
    total_served = 0.0
    total_queues = 0.0
    
    for step in range(1, steps + 1):
        # Predicción inteligente del modelo
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        obs, reward, term, trunc, info = env.step(action)
        
        served = info.get("served_cnt", 0.0)
        queues = info.get("queues", {})
        queue_sum = sum(queues.values())
        
        total_served += served
        total_queues += queue_sum
        
        action_name = "MANTENER" if action == 0 else "CAMBIAR"
        print(f"Paso {step:02d}: Acción={action} ({action_name}), Servidos={served:.1f}, Colas={queue_sum:.1f}, Recompensa={reward:.3f}")
        
        if term or trunc:
            obs, info = env.reset()
        
        time.sleep(0.5)  # Pausa para observar
    
    print(f"\n📊 RESULTADOS PPO:")
    print(f"  Total servidos: {total_served:.1f}")
    print(f"  Promedio colas: {total_queues/steps:.2f}")
    
    try:
        env.core.close()
    except:
        pass


def main():
    """Función principal del demo."""
    setup_sumo_env()
    
    # Cargar configuración
    cfg_path = "experiments/configs/base.yaml"
    if not os.path.exists(cfg_path):
        print(f"❌ Error: Configuración no encontrada: {cfg_path}")
        return
    
    cfg = load_cfg(cfg_path)
    
    print("🎯 DEMO VISUAL: COMPARACIÓN SEMÁFORO FIJO vs PPO")
    print("="*60)
    print("Este demo te mostrará:")
    print("1. 🚦 Semáforo fijo tradicional (sin IA)")
    print("2. 🤖 Control inteligente con PPO (con IA)")
    print("\n⚠️  Asegúrate de que SUMO GUI se abra correctamente")
    print("⚠️  Si no se abre, verifica la instalación de SUMO")
    
    # Verificar modelo
    model_path = "models/ppo_production_100k.zip"
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado: {model_path}")
        print("Modelos disponibles:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith(".zip"):
                    print(f"  - {f}")
        return
    
    print(f"\n✅ Modelo encontrado: {model_path}")
    print("\nPresiona ENTER para comenzar el demo...")
    input()
    
    try:
        # Demo 1: Baseline fijo
        demo_baseline_fixed(cfg, steps=15)
        
        # Demo 2: Modelo PPO
        demo_ppo_model(cfg, model_path, steps=15)
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETADO")
        print("="*60)
        print("¿Notaste las diferencias?")
        print("• El semáforo fijo cambia en ciclos regulares")
        print("• El modelo PPO cambia basado en el tráfico real")
        print("• PPO debería servir más vehículos y reducir colas")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el demo: {e}")


if __name__ == "__main__":
    main()

