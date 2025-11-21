import os
import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rl.env.traffic_light_env import TrafficLightGymEnv, EnvConfig


def evaluate_with_switches(model, env, num_episodes=3, name="Agent", max_steps=720):
    """Evaluate agent and count phase switches."""
    print(f"\n--- Evaluating {name} ---")
    
    metrics = {
        "throughput": [],
        "avg_queue": [],
        "switches": [],
        "avg_green_duration": []
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        total_served = 0
        total_queue = 0
        steps = 0
        switches = 0
        prev_phase = None
        green_durations = []
        green_start = 0
        
        pbar = tqdm(total=max_steps, desc=f"Episode {ep+1}/{num_episodes}")
        
        # Get start time
        start_time = env.core._traci.simulation.getTime()
        
        while not (done or truncated) and steps < max_steps:
            if model == "fixed":
                if (steps * 5) % 30 == 0 and steps > 0:
                    action = 1
                else:
                    action = 0
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            
            total_served += info.get('served_cnt', 0)
            queues_dict = info.get('queues', {})
            total_queue += sum(queues_dict.values())
            
            # Track switches and green duration
            current_phase, _ = env.core.tls_state()
            if prev_phase is not None and current_phase != prev_phase and current_phase in (0, 2):
                switches += 1
                # Calculate duration of previous green
                if prev_phase in (0, 2):
                    # Note: This duration estimation is rough, but good enough for stats
                    duration = (steps - green_start) * 5  
                    green_durations.append(duration)
                green_start = steps
            prev_phase = current_phase
            
            steps += 1
            pbar.update(1)
            
        pbar.close()
        
        # Calculate metrics using REAL simulation time
        end_time = env.core._traci.simulation.getTime()
        total_duration_s = end_time - start_time
        duration_hours = total_duration_s / 3600.0
        
        throughput = total_served / duration_hours if duration_hours > 0 else 0
        avg_q = total_queue / steps if steps > 0 else 0
        avg_green = np.mean(green_durations) if green_durations else 0
        
        metrics["throughput"].append(throughput)
        metrics["avg_queue"].append(avg_q)
        metrics["switches"].append(switches)
        metrics["avg_green_duration"].append(avg_green)
        
        print(f"Episode {ep+1}: Thr={throughput:.1f}, Queue={avg_q:.1f}, Switches={switches}, Avg Green={avg_green:.1f}s, Duration={total_duration_s:.1f}s")
    
    # Summary
    mean_thr = np.mean(metrics["throughput"])
    mean_queue = np.mean(metrics["avg_queue"])
    mean_switches = np.mean(metrics["switches"])
    mean_green_dur = np.mean(metrics["avg_green_duration"])
    
    print(f"\nResult {name}:")
    print(f"  Throughput: {mean_thr:.1f} veh/h")
    print(f"  Queue: {mean_queue:.1f} veh")
    print(f"  Switches: {mean_switches:.1f}")
    print(f"  Avg Green Duration: {mean_green_dur:.1f}s")
    
    return {
        "thr": mean_thr,
        "queue": mean_queue,
        "switches": mean_switches,
        "green_dur": mean_green_dur
    }


def plot_comparison(results, output_dir):
    """Genera gráficas comparativas para la tesis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = list(results.keys())
    # Definir métricas y sus títulos
    metrics_map = [
        ("thr", "Throughput (veh/h)", "higher"),
        ("queue", "Cola Promedio (veh)", "lower"),
        ("switches", "Cambios de Fase", "lower"),
        ("green_dur", "Duración Verde Promedio (s)", "higher")
    ]
    
    # Colores profesionales para tesis (azul, rojo, verde, naranja)
    colors = ['#4c72b0', '#c44e52', '#55a868', '#dd8452']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de Estrategias de Control', fontsize=16, y=0.95)
    
    axes = axes.flatten()
    
    for i, (key, title, goal) in enumerate(metrics_map):
        ax = axes[i]
        values = [results[c][key] for c in configs]
        
        # Crear barras
        bars = ax.bar(configs, values, color=colors[i], alpha=0.8, edgecolor='black')
        
        # Títulos y etiquetas
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (height*0.01),
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        # Resaltar el mejor valor (opcional)
        if goal == "higher":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_alpha(1.0)
        bars[best_idx].set_linewidth(2)
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    save_path = output_dir / "comparativa_tesis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[GRAFICAS] Gráfica guardada exitosamente en: {save_path}")
    plt.close()


import argparse

def main():
    """Compare PPO models with different min_green settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-model", type=str, default=None, help="Path to an extra model to evaluate")
    parser.add_argument("--extra-name", type=str, default="Extra_Model", help="Name for the extra model")
    args = parser.parse_args()
    
    MODEL_MIN5 = PROJECT_ROOT / "models" / "ppo_optimized_final"
    MODEL_MIN10 = PROJECT_ROOT / "models" / "ppo_min10_final"
    MODEL_MIN15 = PROJECT_ROOT / "models" / "ppo_min15_final"
    SUMO_CFG = PROJECT_ROOT / "data" / "sumo" / "cfg" / "four_way_dynamic.sumo.cfg"
    
    print("\n" + "="*80)
    print("EVALUACIÓN: Impacto de min_green en comportamiento del agente")
    print("="*80)
    
    results = {}
    
    # Test min_green=5
    print("\n" + "="*80)
    print("MODELO 1: min_green=5s (cambios rápidos)")
    print("="*80)
    env_config_5 = EnvConfig(
        sumo_cfg_path=str(SUMO_CFG),
        min_green=5,
        max_green=180,
        w_served=1.0,
        w_queue=1.0,
        w_unbalance=0.37,
        use_gui=False,
        randomize_sumo_seed=True
    )
    env_5 = TrafficLightGymEnv(env_config_5)
    model_5 = PPO.load(MODEL_MIN5)
    results["min5"] = evaluate_with_switches(model_5, env_5, num_episodes=3, name="PPO_min5")
    env_5.close()
    
    # Test min_green=10
    if MODEL_MIN10.with_suffix(".zip").exists():
        print("\n" + "="*80)
        print("MODELO 1.5: min_green=10s (balanceado)")
        print("="*80)
        env_config_10 = EnvConfig(
            sumo_cfg_path=str(SUMO_CFG),
            min_green=10,
            max_green=180,
            w_served=1.0,
            w_queue=1.0,
            w_unbalance=0.37,
            use_gui=False,
            randomize_sumo_seed=True
        )
        env_10 = TrafficLightGymEnv(env_config_10)
        model_10 = PPO.load(MODEL_MIN10)
        results["min10"] = evaluate_with_switches(model_10, env_10, num_episodes=3, name="PPO_min10")
        env_10.close()
    else:
        print(f"\n[WARNING] Modelo min_green=10 no encontrado en {MODEL_MIN10}")

    # Test Extra Model
    if args.extra_model:
        extra_path = Path(args.extra_model)
        if extra_path.exists() or extra_path.with_suffix('.zip').exists():
            print("\n" + "="*80)
            print(f"MODELO EXTRA: {args.extra_name}")
            print("="*80)
            # Assume min_green=10 for extra model unless specified otherwise (could be improved)
            # Using 10s as a safe default for evaluation if unknown
            env_config_extra = EnvConfig(
                sumo_cfg_path=str(SUMO_CFG),
                min_green=10, 
                max_green=180,
                w_served=1.0,
                w_queue=1.0,
                w_unbalance=0.37,
                use_gui=False,
                randomize_sumo_seed=True
            )
            env_extra = TrafficLightGymEnv(env_config_extra)
            model_extra = PPO.load(extra_path)
            results["extra"] = evaluate_with_switches(model_extra, env_extra, num_episodes=3, name=args.extra_name)
            env_extra.close()
        else:
             print(f"\n[WARNING] Modelo extra no encontrado en {extra_path}")

    # Test min_green=15
    print("\n" + "="*80)
    print("MODELO 2: min_green=15s (cambios controlados)")
    print("="*80)
    env_config_15 = EnvConfig(
        sumo_cfg_path=str(SUMO_CFG),
        min_green=15,
        max_green=180,
        w_served=1.0,
        w_queue=1.0,
        w_unbalance=0.37,
        use_gui=False,
        randomize_sumo_seed=True
    )
    env_15 = TrafficLightGymEnv(env_config_15)
    model_15 = PPO.load(MODEL_MIN15)
    results["min15"] = evaluate_with_switches(model_15, env_15, num_episodes=3, name="PPO_min15")
    env_15.close()
    
    # Fixed time for reference
    print("\n" + "="*80)
    print("BASELINE: Fixed Time 30s")
    print("="*80)
    env_fixed = TrafficLightGymEnv(env_config_15)
    results["fixed"] = evaluate_with_switches("fixed", env_fixed, num_episodes=3, name="Fixed_30s")
    env_fixed.close()
    
    # Final comparison
    print("\n" + "="*80)
    print("COMPARACIÓN FINAL")
    print("="*80)
    print(f"\n{'Config':<15} {'Throughput':<15} {'Queue':<12} {'Switches':<12} {'Avg Green'}")
    print("-" * 75)
    
    for config, metrics in results.items():
        print(f"{config:<15} {metrics['thr']:<15.1f} {metrics['queue']:<12.1f} {metrics['switches']:<12.1f} {metrics['green_dur']:.1f}s")
    
    # Analysis
    print("\n" + "="*80)
    print("ANÁLISIS")
    print("="*80)
    
    # Compare min15 vs min5
    switches_reduction = ((results["min5"]["switches"] - results["min15"]["switches"]) / results["min5"]["switches"]) * 100
    green_increase = ((results["min15"]["green_dur"] - results["min5"]["green_dur"]) / results["min5"]["green_dur"]) * 100
    thr_change = ((results["min15"]["thr"] - results["min5"]["thr"]) / results["min5"]["thr"]) * 100
    queue_change = ((results["min15"]["queue"] - results["min5"]["queue"]) / results["min5"]["queue"]) * 100
    
    print(f"\nmin_green=15 vs min_green=5:")
    print(f"  Reducción de cambios: {switches_reduction:+.1f}%")
    print(f"  Aumento duración verde: {green_increase:+.1f}%")
    print(f"  Cambio throughput: {thr_change:+.1f}%")
    print(f"  Cambio cola: {queue_change:+.1f}%")
    
    if "min10" in results:
        # Compare min10 vs min5
        switches_reduction_10 = ((results["min5"]["switches"] - results["min10"]["switches"]) / results["min5"]["switches"]) * 100
        thr_change_10 = ((results["min10"]["thr"] - results["min5"]["thr"]) / results["min5"]["thr"]) * 100
        queue_change_10 = ((results["min10"]["queue"] - results["min5"]["queue"]) / results["min5"]["queue"]) * 100
        
        print(f"\nmin_green=10 vs min_green=5:")
        print(f"  Reducción de cambios: {switches_reduction_10:+.1f}%")
        print(f"  Cambio throughput: {thr_change_10:+.1f}%")
        print(f"  Cambio cola: {queue_change_10:+.1f}%")
        
        # Compare min10 vs min15
        queue_change_10_15 = ((results["min10"]["queue"] - results["min15"]["queue"]) / results["min15"]["queue"]) * 100
        print(f"\nmin_green=10 vs min_green=15:")
        print(f"  Cambio cola: {queue_change_10_15:+.1f}% (Negativo es mejor)")

    if "extra" in results:
        print(f"\nANÁLISIS EXTRA ({args.extra_name}):")
        # Compare extra vs min10 (if available) or min5
        base = "min10" if "min10" in results else "min5"
        switches_reduction_extra = ((results[base]["switches"] - results["extra"]["switches"]) / results[base]["switches"]) * 100
        thr_change_extra = ((results["extra"]["thr"] - results[base]["thr"]) / results[base]["thr"]) * 100
        queue_change_extra = ((results["extra"]["queue"] - results[base]["queue"]) / results[base]["queue"]) * 100
        
        print(f"\n{args.extra_name} vs {base}:")
        print(f"  Reducción de cambios: {switches_reduction_extra:+.1f}%")
        print(f"  Cambio throughput: {thr_change_extra:+.1f}%")
        print(f"  Cambio cola: {queue_change_extra:+.1f}%")

    if switches_reduction > 20 and thr_change >= 0:
        print("\n✅ min_green=15s es MEJOR: Menos cambios y throughput similar/mejor")
    elif switches_reduction > 20 and thr_change < -2:
        print("\n⚠️  min_green=15s reduce cambios pero sacrifica throughput")
    else:
        print("\n❌ min_green=5s es mejor para este escenario")
    
    print("="*80)
    
    # Generar gráficas
    plot_comparison(results, PROJECT_ROOT / "outputs" / "plots")
    
    # Plot results
    output_dir = PROJECT_ROOT / "results" / "plots"
    plot_comparison(results, output_dir)


if __name__ == "__main__":
    main()
