#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from typing import Dict, List
import yaml

# Asegurar PYTHONPATH
if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = os.getcwd()

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_with_demand(cfg: dict, demand_profile: str) -> TrafficLightGymEnv:
    """Crear entorno con perfil de demanda específico."""
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
        demand_profile_path=demand_profile,  # ¡CLAVE!
        dynamic_demand=True                   # ¡CLAVE!
    )
    env = TrafficLightGymEnv(ec)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def evaluate_scenario(model_path: str, cfg: dict, demand_profile: str, scenario_name: str, steps: int = 30) -> Dict:
    """Evaluar modelo con un escenario específico de demanda."""
    
    print(f"\n🎯 Evaluando escenario: {scenario_name}")
    print(f"📁 Perfil de demanda: {demand_profile}")
    print(f"⏱️  Duración: {steps} intervalos ({steps * cfg['control']['control_interval']}s)")
    
    env = make_env_with_demand(cfg, demand_profile)
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    
    # Métricas acumuladas
    total_served = 0.0
    total_switches = 0
    total_queue_time = 0.0
    total_spills = 0
    rewards = []
    served_per_step = []
    queue_per_step = []
    
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
        served_per_step.append(served)
        queue_per_step.append(sum(queues.values()))
        
        # Contar spillbacks (colas > 5 veh)
        if any(q > 5 for q in queues.values()):
            total_spills += 1
            
        if step % 10 == 0:
            print(f"  Paso {step:2d}: servidos={served:4.1f}, colas={sum(queues.values()):4.1f}, recompensa={reward:6.3f}")
    
    # Cerrar entorno
    try:
        env.core.close()
    except Exception:
        pass
        
    # Calcular métricas finales
    duration_minutes = (steps * cfg['control']['control_interval']) / 60.0
    throughput = total_served / duration_minutes  # veh/min
    avg_reward = sum(rewards) / len(rewards)
    avg_queue = total_queue_time / steps
    switch_rate = total_switches / steps
    spill_rate = total_spills / steps
    
    results = {
        "scenario": scenario_name,
        "demand_profile": demand_profile,
        "total_served": total_served,
        "throughput_veh_per_min": throughput,
        "total_switches": total_switches,
        "switch_rate": switch_rate,
        "avg_queue": avg_queue,
        "avg_reward": avg_reward,
        "spill_rate": spill_rate,
        "duration_min": duration_minutes
    }
    
    print(f"   Resultados {scenario_name}:")
    print(f"     Throughput: {throughput:.1f} veh/min")  
    print(f"     Colas promedio: {avg_queue:.2f} veh")
    print(f"     Cambios de fase: {total_switches} ({switch_rate:.2f}/intervalo)")
    print(f"     Spillbacks: {total_spills} ({spill_rate:.2f} rate)")
    print(f"     Recompensa promedio: {avg_reward:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluación con demanda variable")
    parser.add_argument("--model", default="models/ppo_tls.zip", help="Modelo PPO a evaluar")
    parser.add_argument("--steps", type=int, default=30, help="Pasos de evaluación por escenario")
    parser.add_argument("--output", default="variable_demand_results.csv", help="Archivo de resultados")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: No se encontró el modelo {args.model}")
        return 1
        
    # Cargar configuración
    cfg = load_cfg(os.path.join("experiments", "configs", "base.yaml"))
    
    # Escenarios de demanda variable a evaluar
    scenarios = [
        ("rush_morning.yaml", "Pico Matutino"),
        ("rush_evening.yaml", "Pico Vespertino"),
        ("midday_moderate.yaml", "Mediodia"),
        ("off_peak_low.yaml", "Valle Nocturno"),
        ("peak_am_pm.yaml", "AM/PM Dinamico"),
        ("static_medium.yaml", "Estatico Medio")
    ]
    
    print("EVALUACION CON DEMANDA VARIABLE")
    print("=" * 50)
    print(f"   Modelo: {args.model}")
    print(f"   {len(scenarios)} escenarios x {args.steps} pasos cada uno")
    print()
    
    all_results = []
    
    for demand_file, scenario_name in scenarios:
        demand_path = os.path.join("experiments", "demands", demand_file)
        
        if not os.path.exists(demand_path):
            print(f"   SALTANDO {scenario_name}: no existe {demand_path}")
            continue
            
        try:
            results = evaluate_scenario(args.model, cfg, demand_path, scenario_name, args.steps)
            all_results.append(results)
        except Exception as e:
            print(f"   ERROR en {scenario_name}: {e}")
            continue
    
    # Guardar resultados
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"\n   Resultados guardados en: {args.output}")
        
        # Resumen comparativo
        print(f"\nRESUMEN COMPARATIVO:")
        print("=" * 60)
        print(f"{'Escenario':<20} {'Throughput':<12} {'Colas':<8} {'Cambios':<8} {'Reward':<8}")
        print("-" * 60)
        for _, row in df.iterrows():
            print(f"{row['scenario']:<20} {row['throughput_veh_per_min']:8.1f} veh/min "
                  f"{row['avg_queue']:6.2f}   {row['total_switches']:6.0f}   {row['avg_reward']:6.3f}")
        
        # Identificar mejor y peor escenario
        best_idx = df['throughput_veh_per_min'].idxmax()
        worst_idx = df['throughput_veh_per_min'].idxmin()
        
        print(f"\nMEJOR rendimiento: {df.iloc[best_idx]['scenario']} "
              f"({df.iloc[best_idx]['throughput_veh_per_min']:.1f} veh/min)")
        print(f"PEOR rendimiento: {df.iloc[worst_idx]['scenario']} "
              f"({df.iloc[worst_idx]['throughput_veh_per_min']:.1f} veh/min)")
    
    print(f"\nEvaluacion completada!")
    return 0


if __name__ == "__main__":
    exit(main())
