#!/usr/bin/env python3

import os
import pandas as pd
import argparse

# Asegurar PYTHONPATH
if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = os.getcwd()

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_fixed_demand_env(cfg: dict) -> TrafficLightGymEnv:
    """Crear entorno con demanda FIJA (usando routes originales de SUMO)."""
    
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
        # CLAVE: Sin demanda dinámica, usa flows fijos de SUMO
        demand_profile_path=None,
        dynamic_demand=False
    )
    
    env = TrafficLightGymEnv(ec)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def evaluate_cross_domain(model_path: str, config_path: str, model_name: str, steps: int = 20) -> dict:
    """Evaluar modelo en dominio cruzado."""
    
    print(f"\n🧪 EVALUACIÓN CRUZADA: {model_name}")
    print(f"   Modelo: {model_path}")
    print(f"   Config: {config_path}")
    print(f"   Duración: {steps} intervalos")
    
    cfg = load_cfg(config_path)
    env = make_fixed_demand_env(cfg)
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    
    # Métricas acumuladas
    total_served = 0.0
    total_switches = 0
    total_queue_time = 0.0
    total_spills = 0
    rewards = []
    
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        
        served = info.get("served_cnt", 0.0)
        switch = int(info.get("switch", 0))
        queues = info.get("queues", {})
        
        total_served += served
        total_switches += switch
        total_queue_time += sum(queues.values())
        rewards.append(reward)
        
        # Contar spillbacks
        if any(q > 5 for q in queues.values()):
            total_spills += 1
            
        if step % 5 == 0:
            print(f"  Paso {step:2d}: servidos={served:4.1f}, colas={sum(queues.values()):4.1f}, recompensa={reward:6.3f}")
    
    # Cerrar entorno
    try:
        env.core.close()
    except Exception:
        pass
    
    # Calcular métricas
    duration_minutes = (steps * cfg['control']['control_interval']) / 60.0
    throughput = total_served / duration_minutes
    avg_reward = sum(rewards) / len(rewards)
    avg_queue = total_queue_time / steps
    switch_rate = total_switches / steps
    spill_rate = total_spills / steps
    
    results = {
        "model": model_name,
        "config_type": "Fixed" if "base.yaml" in config_path else "Variable",
        "domain": "Fixed Demand",
        "total_served": total_served,
        "throughput_veh_per_min": throughput,
        "total_switches": total_switches,
        "switch_rate": switch_rate,
        "avg_queue": avg_queue,
        "avg_reward": avg_reward,
        "spill_rate": spill_rate,
        "duration_min": duration_minutes
    }
    
    print(f"   📊 Throughput: {throughput:.1f} veh/min")
    print(f"   📊 Colas: {avg_queue:.2f} veh promedio")
    print(f"   📊 Cambios: {total_switches} ({switch_rate:.2f}/intervalo)")
    print(f"   📊 Recompensa: {avg_reward:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluación cruzada entre dominios")
    parser.add_argument("--steps", type=int, default=20, help="Pasos de evaluación")
    args = parser.parse_args()
    
    print("🔄 EVALUACIÓN CRUZADA: GENERALISTA VS ESPECIALISTA")
    print("=" * 60)
    print("Probando cómo se desempeña cada modelo en demanda FIJA")
    
    # Verificar que existan los modelos
    models_to_test = [
        ("models/ppo_tls.zip", "experiments/configs/base.yaml", "PPO Especialista (Fijo)"),
        ("models/ppo_variable_OPTIMIZED.zip", "experiments/configs/variable_demand_optimized.yaml", "PPO Generalista (Variable)")
    ]
    
    results = []
    
    for model_path, config_path, model_name in models_to_test:
        if not os.path.exists(model_path):
            print(f"⚠️  Saltando {model_name}: no existe {model_path}")
            continue
            
        if not os.path.exists(config_path):
            print(f"⚠️  Saltando {model_name}: no existe {config_path}")
            continue
        
        try:
            result = evaluate_cross_domain(model_path, config_path, model_name, args.steps)
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluando {model_name}: {e}")
    
    if results:
        # Análisis comparativo
        df = pd.DataFrame(results)
        df.to_csv("cross_domain_evaluation.csv", index=False)
        
        print(f"\n📊 COMPARACIÓN EN DEMANDA FIJA:")
        print("=" * 60)
        print(f"{'Modelo':<30} {'Throughput':<12} {'Colas':<8} {'Cambios':<8} {'Reward':<8}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            print(f"{row['model']:<30} {row['throughput_veh_per_min']:8.1f} veh/min "
                  f"{row['avg_queue']:6.2f}   {row['total_switches']:6.0f}   {row['avg_reward']:6.3f}")
        
        # Análisis específico
        if len(results) == 2:
            especialista = df[df['model'].str.contains('Especialista')].iloc[0]
            generalista = df[df['model'].str.contains('Generalista')].iloc[0]
            
            throughput_diff = ((generalista['throughput_veh_per_min'] / especialista['throughput_veh_per_min'] - 1) * 100)
            queue_diff = ((generalista['avg_queue'] / especialista['avg_queue'] - 1) * 100)
            
            print(f"\n🎯 ANÁLISIS CROSS-DOMAIN:")
            print(f"   Especialista en su dominio: {especialista['throughput_veh_per_min']:.1f} veh/min")
            print(f"   Generalista en dominio ajeno: {generalista['throughput_veh_per_min']:.1f} veh/min")
            print(f"   Diferencia: {throughput_diff:+.1f}%")
            
            if abs(throughput_diff) < 5:
                print(f"   ✅ GENERALISTA ES COMPETITIVO (diferencia <5%)")
            elif throughput_diff > 0:
                print(f"   🔥 GENERALISTA SUPERA AL ESPECIALISTA EN SU PROPIO DOMINIO!")
            else:
                print(f"   📊 Especialista mantiene ventaja en su dominio")
            
            print(f"\n   Control de colas:")
            print(f"   Especialista: {especialista['avg_queue']:.2f} veh")
            print(f"   Generalista: {generalista['avg_queue']:.2f} veh ({queue_diff:+.1f}%)")
            
            print(f"\n   Estrategia de cambios:")
            print(f"   Especialista: {especialista['switch_rate']:.2f} cambios/intervalo")
            print(f"   Generalista: {generalista['switch_rate']:.2f} cambios/intervalo")
    
        print(f"\n💾 Resultados guardados en: cross_domain_evaluation.csv")
    
    return 0


if __name__ == "__main__":
    exit(main())


