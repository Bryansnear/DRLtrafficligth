#!/usr/bin/env python3
"""Evaluación general
------------------
Evalúa el baseline de ciclo fijo y cualquier modelo PPO entrenado con este proyecto.

Uso:
    # Evaluar solo baseline
    python scripts/evaluate.py --baseline --episodes 5

    # Evaluar modelo DRL
    python scripts/evaluate.py --model models/mi_modelo.zip --config configs/reactive_random_500k.yaml --episodes 5

    # Comparar ambos
    python scripts/evaluate.py --model models/mi_modelo.zip --config configs/reactive_random_500k.yaml --baseline --episodes 5"""

import argparse
import csv
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
from stable_baselines3 import PPO

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv

DEFAULT_BASELINE_CSV = Path("outputs") / "baseline" / "baseline_results.csv"


def load_config(config_path: str) -> dict:
    """Carga archivo de configuración YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_env(config_dict: dict, use_gui: bool = False):
    """Crea ambiente de evaluación"""
    
    ec = EnvConfig(
        sumo_cfg_path=config_dict["sumo"]["cfg_path"],
        step_length=float(config_dict["sumo"]["step_length"]),
        control_interval=int(config_dict["control"]["control_interval"]),
        min_green=int(config_dict["control"]["min_green"]),
        yellow=int(config_dict["control"]["yellow"]),
        all_red=int(config_dict["control"]["all_red"]),
        e2_ids=tuple(config_dict["detectors"]["e2_ids"]),
        v_free=float(config_dict["detectors"]["v_free"]),
        jam_length_thr_m=float(config_dict["detectors"]["jam_length_thr_m"]),
        e2_capacity_per_lane=int(config_dict["detectors"]["e2_capacity_per_lane"]),
        w_served=float(config_dict["reward"]["w_served"]),
        w_queue=float(config_dict["reward"]["w_queue"]),
        w_backlog=float(config_dict["reward"]["w_backlog"]),
        w_switch=float(config_dict["reward"]["w_switch"]),
        w_spill=float(config_dict["reward"]["w_spill"]),
        w_invalid_action=float(config_dict["reward"].get("w_invalid_action", 0.0)),
        w_unbalance=float(config_dict["reward"].get("w_unbalance", 0.0)),
        w_select=float(config_dict["reward"].get("w_select", 0.0)),
        kappa_backlog=int(config_dict["reward"]["kappa_backlog"]),
        sat_headway_s=float(config_dict["reward"]["sat_headway_s"]),
        use_gui=use_gui,
        max_green=int(config_dict["control"].get("max_green", 60)),
        randomize_sumo_seed=bool(config_dict.get("randomize_sumo_seed", True)),
    )
    return TrafficLightGymEnv(ec)


def evaluate_baseline(
    env,
    num_episodes: int = 5,
    fixed_cycle_time: int = 50,
    verbose: bool = True,
    phase_durations: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Dict:
    """Evalúa baseline de tiempo fijo controlando la duración de cada fase."""
    control_interval = env.cfg.control_interval  # segundos por paso

    if phase_durations is None:
        phase_seconds = (fixed_cycle_time, fixed_cycle_time)
    else:
        if len(phase_durations) != 2:
            raise ValueError("phase_durations debe tener dos elementos (EW, NS)")
        phase_seconds = tuple(int(max(1, v)) for v in phase_durations)

    hold_steps = tuple(max(1, int(round(p / control_interval))) for p in phase_seconds)
    phase_labels = ("E/O", "N/S")

    if verbose:
        print("\n" + "=" * 80)
        print("EVALUANDO BASELINE (TIEMPO FIJO)")
        print("=" * 80)
        print(f"Intervalo de control: {control_interval}s")
        print(f"Fase {phase_labels[0]}: {phase_seconds[0]}s  -> {hold_steps[0]} pasos")
        print(f"Fase {phase_labels[1]}: {phase_seconds[1]}s  -> {hold_steps[1]} pasos")
        print(f"Episodios: {num_episodes}")

    results = []

    for ep in range(1, num_episodes + 1):
        if verbose:
            print(f"\n--- Episodio {ep}/{num_episodes} ---")

        obs, info = env.reset()

        stats = {
            'total_reward': 0.0,
            'served_vehicles': 0.0,
            'avg_queue': 0.0,
            'switches': 0,
            'steps': 0,
            'queue_history': [],
        }

        prev_phase = None
        done = False
        current_phase_idx = 0
        steps_in_phase = 0

        while not done:
            target_steps = hold_steps[current_phase_idx]
            if steps_in_phase >= target_steps:
                action = 1  # SWITCH
                steps_in_phase = 0
                current_phase_idx = 1 - current_phase_idx
            else:
                action = 0  # HOLD
                steps_in_phase += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            stats['total_reward'] += reward
            stats['served_vehicles'] += info.get('served_cnt', 0.0)
            stats['steps'] += 1

            queues = info.get('queues', {})
            avg_queue = np.mean(list(queues.values())) if queues else 0.0
            stats['queue_history'].append(avg_queue)

            current_phase, _ = env.core.tls_state()
            if prev_phase is not None and current_phase != prev_phase:
                stats['switches'] += 1
            prev_phase = current_phase

            if verbose and (stats['steps'] + 1) % 120 == 0:
                mins = (stats['steps'] + 1) * control_interval / 60
                print(f"  {mins:.0f}min: Servidos={stats['served_vehicles']:.0f}, "
                      f"Cola={avg_queue:.2f}, Switches={stats['switches']}")

        stats['avg_queue'] = np.mean(stats['queue_history']) if stats['queue_history'] else 0.0
        stats['avg_reward'] = stats['total_reward'] / stats['steps'] if stats['steps'] > 0 else 0.0
        stats['throughput'] = stats['served_vehicles'] / stats['steps'] if stats['steps'] > 0 else 0.0

        results.append(stats)

        if verbose:
            print(f"\nResultado Episodio {ep}:")
            print(f"  Vehículos servidos: {stats['served_vehicles']:.0f}")
            print(f"  Cola promedio: {stats['avg_queue']:.3f}")
            print(f"  Recompensa promedio: {stats['avg_reward']:.3f}")
            print(f"  Switches: {stats['switches']}")

    metrics = summarize_episode_metrics(results)
    summary = {
        'type': 'baseline',
        'episodes': results,
        'config': {
            'phase_durations_s': phase_seconds,
            'hold_steps': hold_steps,
            'control_interval': control_interval,
        },
        'metrics': metrics,
    }

    if save_path:
        save_baseline_summary(summary, save_path, run_tag=run_tag)

    return summary


def summarize_episode_metrics(episodes: List[Dict]) -> Dict[str, float]:
    """Calcula promedios agregados para comparar modelos vs baseline."""
    if not episodes:
        return {
            'avg_reward': 0.0,
            'avg_queue': 0.0,
            'avg_served': 0.0,
            'avg_switches': 0.0,
            'avg_steps': 0.0,
            'avg_throughput': 0.0,
        }

    def _mean(key: str) -> float:
        values = [ep.get(key, 0.0) for ep in episodes]
        return float(np.mean(values)) if values else 0.0

    return {
        'avg_reward': _mean('avg_reward'),
        'avg_queue': _mean('avg_queue'),
        'avg_served': _mean('served_vehicles'),
        'avg_switches': _mean('switches'),
        'avg_steps': _mean('steps'),
        'avg_throughput': _mean('throughput'),
    }


def save_baseline_summary(summary: Dict, path: str, run_tag: Optional[str] = None) -> None:
    """Guarda los resultados agregados del baseline en un CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = summary.get('config', {})
    metrics = summary.get('metrics', {})
    phase_seconds = config.get('phase_durations_s', (0, 0))
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'tag': run_tag or 'baseline',
        'phase_ew_s': phase_seconds[0],
        'phase_ns_s': phase_seconds[1],
        'control_interval_s': config.get('control_interval', 0),
        'episodes': len(summary.get('episodes', [])),
        'avg_reward': metrics.get('avg_reward', 0.0),
        'avg_queue': metrics.get('avg_queue', 0.0),
        'avg_served': metrics.get('avg_served', 0.0),
        'avg_switches': metrics.get('avg_switches', 0.0),
        'avg_steps': metrics.get('avg_steps', 0.0),
        'avg_throughput': metrics.get('avg_throughput', 0.0),
    }

    file_exists = output_path.exists()
    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
def evaluate_drl_model(
    env,
    model_path: str,
    num_episodes: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Evalúa modelo DRL entrenado.
    
    Args:
        env: Ambiente de tráfico
        model_path: Ruta al modelo .zip
        num_episodes: Número de episodios a evaluar
        verbose: Imprimir progreso
    
    Returns:
        Diccionario con resultados
    """
    if verbose:
        print("\n" + "=" * 80)
        print("EVALUANDO MODELO DRL")
        print("=" * 80)
        print(f"Modelo: {model_path}")
        print(f"Episodios: {num_episodes}")
    
    # Cargar modelo
    model = PPO.load(model_path)
    
    results = []
    control_interval = env.cfg.control_interval
    
    for ep in range(1, num_episodes + 1):
        if verbose:
            print(f"\n--- Episodio {ep}/{num_episodes} ---")
        
        obs, info = env.reset()
        
        stats = {
            'total_reward': 0.0,
            'served_vehicles': 0.0,
            'avg_queue': 0.0,
            'switches': 0,
            'steps': 0,
            'queue_history': [],
            'actions': [],
        }
        
        prev_phase = None
        done = False
        
        while not done:
            # Modelo decide acción
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            stats['actions'].append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Acumular estadísticas
            stats['total_reward'] += reward
            stats['served_vehicles'] += info.get('served_cnt', 0.0)
            stats['steps'] += 1
            
            # Colas
            queues = info.get('queues', {})
            avg_queue = np.mean(list(queues.values())) if queues else 0.0
            stats['queue_history'].append(avg_queue)
            
            # Contar switches
            current_phase, _ = env.core.tls_state()
            if prev_phase is not None and current_phase != prev_phase:
                stats['switches'] += 1
            prev_phase = current_phase
            
            # Progreso cada 10 minutos
            if verbose and (stats['steps'] + 1) % 120 == 0:
                mins = (stats['steps'] + 1) * control_interval / 60
                print(f"  {mins:.0f}min: Servidos={stats['served_vehicles']:.0f}, "
                      f"Cola={avg_queue:.2f}, Switches={stats['switches']}")
        
        # Calcular promedios
        stats['avg_queue'] = np.mean(stats['queue_history']) if stats['queue_history'] else 0.0
        stats['avg_reward'] = stats['total_reward'] / stats['steps'] if stats['steps'] > 0 else 0.0
        stats['throughput'] = stats['served_vehicles'] / stats['steps'] if stats['steps'] > 0 else 0.0
        
        # Análisis de acciones
        actions_array = np.array(stats['actions'])
        stats['hold_rate'] = np.sum(actions_array == 0) / len(actions_array) if len(actions_array) > 0 else 0.0
        stats['switch_rate'] = np.sum(actions_array == 1) / len(actions_array) if len(actions_array) > 0 else 0.0
        
        results.append(stats)
        
        if verbose:
            print(f"\nResultado Episodio {ep}:")
            print(f"  Vehículos servidos: {stats['served_vehicles']:.0f}")
            print(f"  Cola promedio: {stats['avg_queue']:.3f}")
            print(f"  Recompensa promedio: {stats['avg_reward']:.3f}")
            print(f"  Switches: {stats['switches']}")
            print(f"  Acciones - HOLD: {stats['hold_rate']*100:.1f}%, SWITCH: {stats['switch_rate']*100:.1f}%")
    
    return {
        'type': 'drl',
        'model_path': model_path,
        'episodes': results,
    }


def print_comparison(baseline_results: Dict, drl_results: Dict):
    """Imprime comparación entre baseline y DRL"""
    print("\n" + "=" * 80)
    print("COMPARACIÓN FINAL")
    print("=" * 80)
    
    # Extraer métricas
    def extract_metrics(results: Dict) -> Tuple:
        episodes = results['episodes']
        served = [ep['served_vehicles'] for ep in episodes]
        queues = [ep['avg_queue'] for ep in episodes]
        rewards = [ep['avg_reward'] for ep in episodes]
        switches = [ep['switches'] for ep in episodes]
        
        return (
            np.mean(served), np.std(served),
            np.mean(queues), np.std(queues),
            np.mean(rewards), np.std(rewards),
            np.mean(switches), np.std(switches),
        )
    
    b_served_m, b_served_s, b_queue_m, b_queue_s, b_rew_m, b_rew_s, b_sw_m, b_sw_s = extract_metrics(baseline_results)
    d_served_m, d_served_s, d_queue_m, d_queue_s, d_rew_m, d_rew_s, d_sw_m, d_sw_s = extract_metrics(drl_results)
    
    # Calcular diferencias
    served_diff = ((d_served_m - b_served_m) / b_served_m * 100) if b_served_m > 0 else 0.0
    queue_diff = ((d_queue_m - b_queue_m) / b_queue_m * 100) if b_queue_m > 0 else 0.0
    
    print(f"\n{'Métrica':<25} {'Baseline':<20} {'DRL':<20} {'Diferencia':<15}")
    print("-" * 80)
    print(f"{'Vehículos servidos':<25} {b_served_m:>8.1f} ± {b_served_s:<7.1f} {d_served_m:>8.1f} ± {d_served_s:<7.1f} {served_diff:>+6.1f}%")
    print(f"{'Cola promedio':<25} {b_queue_m:>8.3f} ± {b_queue_s:<7.3f} {d_queue_m:>8.3f} ± {d_queue_s:<7.3f} {queue_diff:>+6.1f}%")
    print(f"{'Recompensa promedio':<25} {b_rew_m:>8.3f} ± {b_rew_s:<7.3f} {d_rew_m:>8.3f} ± {d_rew_s:<7.3f}")
    print(f"{'Switches':<25} {b_sw_m:>8.0f} ± {b_sw_s:<7.0f} {d_sw_m:>8.0f} ± {d_sw_s:<7.0f}")
    
    print("\n" + "=" * 80)
    if served_diff > 0 and queue_diff < 0:
        print("✅ DRL MEJOR: Más vehículos servidos y menos colas")
    elif served_diff > 0:
        print("⚠️  DRL MIXTO: Más vehículos servidos pero más colas")
    else:
        print("❌ BASELINE MEJOR: Baseline supera a DRL")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación general de baseline y modelos DRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model', type=str, help='Ruta al modelo DRL (.zip)')
    parser.add_argument('--config', type=str, default='configs/reactive_random_500k.yaml', help='Ruta al archivo de configuracion YAML')
    parser.add_argument('--baseline', action='store_true', help='Evaluar baseline de tiempo fijo')
    parser.add_argument('--episodes', type=int, default=5, help='Número de episodios (default: 5)')
    parser.add_argument('--routes', type=str, default='data/sumo/routes/flows_cross_1h_variable.rou.xml',
                        help='Archivo de rutas SUMO')
    parser.add_argument('--cycle-time', type=int, default=60,
                        help='Tiempo de ciclo fijo para baseline en segundos (default: 60)')
    parser.add_argument('--phase-ew', type=int, help='Tiempo verde fijo (s) para la fase Este/Oeste')
    parser.add_argument('--phase-ns', type=int, help='Tiempo verde fijo (s) para la fase Norte/Sur')
    parser.add_argument(
        '--baseline-save',
        type=str,
        default=str(DEFAULT_BASELINE_CSV),
        help=f'Archivo CSV para guardar el baseline (default: {DEFAULT_BASELINE_CSV})',
    )
    parser.add_argument(
        '--baseline-tag',
        type=str,
        help='Etiqueta opcional para identificar este baseline en el CSV',
    )
    parser.add_argument('--no-randomize', action='store_true',
                        help='Desactivar randomización de semilla SUMO')
    parser.add_argument('--quiet', action='store_true', help='Modo silencioso')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.baseline and not args.model:
        parser.error("Debe especificar --baseline y/o --model")
    
    if args.model and not args.config:
        parser.error("--model requiere --config")
    
    # Resultados
    baseline_results = None
    drl_results = None
    
    # Cargar config
    if args.config:
        config_dict = load_config(args.config)
    elif args.baseline:
        # Usar config base para baseline standalone
        config_path = 'configs/reactive_random_500k.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Config no encontrado: {config_path}")
            print("   Especifique --config")
            return 1
        config_dict = load_config(config_path)
    else:
        parser.error("Requiere --config")
    
    try:
        # Evaluar baseline
        if args.baseline:
            env = create_env(config_dict, use_gui=False)
            phase_ew = args.phase_ew if args.phase_ew is not None else args.cycle_time
            phase_ns = args.phase_ns if args.phase_ns is not None else args.cycle_time
            save_path = args.baseline_save.strip() if args.baseline_save else None
            baseline_results = evaluate_baseline(
                env, 
                num_episodes=args.episodes,
                fixed_cycle_time=args.cycle_time,
                verbose=not args.quiet,
                phase_durations=(phase_ew, phase_ns),
                save_path=save_path,
                run_tag=args.baseline_tag,
            )
            env.core.close()
        
        # Evaluar modelo DRL
        if args.model:
            if not os.path.exists(args.model):
                print(f"❌ Modelo no encontrado: {args.model}")
                return 1
            
            env = create_env(config_dict, use_gui=False)
            drl_results = evaluate_drl_model(
                env,
                args.model,
                num_episodes=args.episodes,
                verbose=not args.quiet
            )
            env.core.close()
        
        # Comparación si hay ambos
        if baseline_results and drl_results:
            print_comparison(baseline_results, drl_results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluación interrumpida por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

