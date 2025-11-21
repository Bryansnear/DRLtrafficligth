#!/usr/bin/env python3
"""Búsqueda de hiperparámetros (Fase 1) con Optuna.

Optimiza parámetros de control y recompensa sobre la config base
`configs/reactive_random_500k.yaml`, entrenando modelos cortos y
evaluándolos en el mismo escenario SUMO.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuplewqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq

import numpy as np
import optuna
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).parent.parent
BASE_CONFIG = PROJECT_ROOT / "configs" / "reactive_random_500k.yaml"
MODELS_DIR = PROJECT_ROOT / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv  # noqa: E402


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_cfg(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def make_env(cfg_dict: dict, *, use_gui: bool = False) -> TrafficLightGymEnv:
    ec = EnvConfig(
        sumo_cfg_path=cfg_dict["sumo"]["cfg_path"],
        step_length=float(cfg_dict["sumo"]["step_length"]),
        control_interval=int(cfg_dict["control"]["control_interval"]),
        min_green=int(cfg_dict["control"]["min_green"]),
        yellow=int(cfg_dict["control"]["yellow"]),
        all_red=int(cfg_dict["control"]["all_red"]),
        max_green=int(cfg_dict["control"].get("max_green", 60)),
        e2_ids=tuple(cfg_dict["detectors"]["e2_ids"]),
        v_free=float(cfg_dict["detectors"]["v_free"]),
        jam_length_thr_m=float(cfg_dict["detectors"]["jam_length_thr_m"]),
        e2_capacity_per_lane=int(cfg_dict["detectors"]["e2_capacity_per_lane"]),
        w_served=float(cfg_dict["reward"]["w_served"]),
        w_queue=float(cfg_dict["reward"]["w_queue"]),
        w_backlog=float(cfg_dict["reward"]["w_backlog"]),
        w_switch=float(cfg_dict["reward"]["w_switch"]),
        w_spill=float(cfg_dict["reward"]["w_spill"]),
        w_invalid_action=float(cfg_dict["reward"].get("w_invalid_action", 0.0)),
        w_unbalance=float(cfg_dict["reward"].get("w_unbalance", 0.0)),
        w_select=float(cfg_dict["reward"].get("w_select", 0.0)),
        kappa_backlog=int(cfg_dict["reward"]["kappa_backlog"]),
        sat_headway_s=float(cfg_dict["reward"]["sat_headway_s"]),
        use_gui=use_gui,
        randomize_sumo_seed=bool(cfg_dict.get("randomize_sumo_seed", False)),
        always_compute_metrics=True,  # FORCE metrics computation for evaluation
    )
    return TrafficLightGymEnv(ec)


def evaluate_model(
    cfg_dict: dict,
    model_path: Path,
    episodes: int,
) -> Dict[str, float]:
    """Evalúa un modelo PPO en el entorno especificado por cfg_dict."""

    max_steps = int(cfg_dict["train"]["max_episode_steps"])

    env = make_env(cfg_dict, use_gui=False)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=max_steps)

    model = PPO.load(model_path)

    results: List[Dict[str, float]] = []
    control_interval = cfg_dict["control"]["control_interval"]

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            stats = {
                "total_reward": 0.0,
                "served_vehicles": 0.0,
                "avg_queue": 0.0,
                "switches": 0.0,
                "steps": 0,
            }
            queue_hist: List[float] = []
            prev_phase = None
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                stats["total_reward"] += float(reward)
                stats["served_vehicles"] += float(info.get("served_cnt", 0.0))
                stats["steps"] += 1

                queues = info.get("queues", {})
                avg_queue = float(np.mean(list(queues.values()))) if queues else 0.0
                queue_hist.append(avg_queue)

                current_phase, _ = env.core.tls_state()
                if prev_phase is not None and current_phase != prev_phase:
                    stats["switches"] += 1.0
                prev_phase = current_phase

            stats["avg_queue"] = float(np.mean(queue_hist)) if queue_hist else 0.0
            stats["avg_reward"] = (
                stats["total_reward"] / stats["steps"] if stats["steps"] > 0 else 0.0
            )
            stats["throughput"] = (
                stats["served_vehicles"] / stats["steps"] if stats["steps"] > 0 else 0.0
            )
            results.append(stats)
    finally:
        try:
            env.close()
        except Exception:
            pass

    def mean(key: str) -> float:
        vals = [ep[key] for ep in results]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "avg_reward": mean("avg_reward"),
        "avg_queue": mean("avg_queue"),
        "avg_served": mean("served_vehicles"),
        "avg_switches": mean("switches"),
        "avg_steps": mean("steps"),
        "avg_throughput": mean("throughput"),
    }


def build_trial_config(base_cfg: dict, trial: optuna.Trial, timesteps: int) -> dict:
    """Aplica las sugerencias de Optuna sobre una copia de la config base."""

    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy

    # Control: FIXED max_green=180s (don't optimize)
    cfg["control"]["min_green"] = 5  # Keep fixed
    cfg["control"]["max_green"] = 180  # FIXED - don't optimize
    
    # Reward: Keep w_unbalance=0.37 FIXED (already optimized)
    r = cfg["reward"]
    r["w_served"] = 0.0
    r["w_queue"] = 1.0  # MaxPressure base
    r["w_backlog"] = 0.0
    r["w_switch"] = 0.0
    r["w_spill"] = 0.0
    r["w_unbalance"] = 0.37  # FIXED - already optimized
    
    # PPO HYPERPARAMETERS: Optimize these now
    ppo = cfg.setdefault("ppo", {})
    ppo["learning_rate"] = float(trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True))
    ppo["ent_coef"] = float(trial.suggest_float("ent_coef", 0.001, 0.1, log=True))
    ppo["gamma"] = float(trial.suggest_float("gamma", 0.95, 0.999))
    
    # Network architecture
    net_size = trial.suggest_categorical("net_size", ["small", "medium", "large"])
    if net_size == "small":
        ppo["net_arch"] = [[64, 64]]
    elif net_size == "medium":
        ppo["net_arch"] = [[128, 128]]
    else:  # large
        ppo["net_arch"] = [[256, 128, 64]]

    # Train section: timesteps + n_envs pequeño para esta fase
    t = cfg.setdefault("train", {})
    t["total_timesteps"] = int(timesteps)
    t["max_episode_steps"] = 512  # Longer episodes to test hyperparams
    t["n_envs"] = 10  # Parallel envs

    # Keep randomization ON for better generalization
    cfg["randomize_sumo_seed"] = True
    
    # Use dynamic scenario for better generalization
    cfg["sumo"]["cfg_path"] = "data/sumo/cfg/four_way_dynamic.sumo.cfg"

    return cfg


def run_train(config_path: Path, timesteps: int, model_name: str, n_envs: int) -> bool:
    """Lanza scripts/train.py para entrenar un modelo corto."""

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--config",
        str(config_path),
        "--timesteps",
        str(timesteps),
        "--model-name",
        model_name,
        "--n-envs",
        str(n_envs),
    ]

    print("\n[OPTUNA] Lanzando entrenamiento:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[OPTUNA] Entrenamiento falló: {exc}")
        return False


def make_objective(timesteps: int, episodes: int) -> optuna.trial.Trial:
    base_cfg = load_cfg(BASE_CONFIG)

    def objective(trial: optuna.Trial) -> float:
        cfg = build_trial_config(base_cfg, trial, timesteps)

        trial_id = trial.number
        trial_cfg_path = PROJECT_ROOT / "configs" / f"optuna_trial_{trial_id}.yaml"
        save_cfg(cfg, trial_cfg_path)

        model_name = f"optuna_trial_{trial_id}"
        n_envs = int(cfg["train"].get("n_envs", 4))

        ok = run_train(trial_cfg_path, timesteps, model_name, n_envs)
        if not ok:
            raise optuna.TrialPruned()

        model_path = MODELS_DIR / f"{model_name}.zip"
        if not model_path.exists():
            print(f"[OPTUNA] Modelo no encontrado: {model_path}")
            raise optuna.TrialPruned()

        metrics = evaluate_model(cfg, model_path, episodes=episodes)
        avg_thr = metrics["avg_throughput"]
        avg_queue = metrics["avg_queue"]

        # Objetivo: maximizar throughput penalizando colas altas
        penalty = 0.0
        queue_threshold = 4.2
        if avg_queue > queue_threshold:
            penalty = (avg_queue - queue_threshold) * 0.5

        score = avg_thr - penalty

        trial.set_user_attr("avg_throughput", avg_thr)
        trial.set_user_attr("avg_queue", avg_queue)
        trial.set_user_attr("avg_reward", metrics["avg_reward"])
        trial.set_user_attr("avg_served", metrics["avg_served"])

        print(
            f"[OPTUNA] Trial {trial_id} -> score={score:.4f}, "
            f"thr={avg_thr:.4f}, queue={avg_queue:.4f}"
        )

        return score

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna Fase 1 - búsqueda sobre reward/control")
    p.add_argument("--trials", type=int, default=20, help="Número de trials de Optuna")
    p.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Timesteps de entrenamiento por trial (corto)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodios de evaluación por trial",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="optuna_study",
        help="Nombre del estudio (para DB sqlite)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Iniciando Optuna con {args.trials} trials...")
    print(f"Timesteps por trial: {args.timesteps}")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{args.study_name}.db",
        load_if_exists=True,
    )

    objective = make_objective(args.timesteps, args.episodes)
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "=" * 60)
    print("RESULTADOS OPTUNA")
    print("=" * 60)
    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor valor (score): {study.best_value:.4f}")
    print("Mejores parámetros:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
