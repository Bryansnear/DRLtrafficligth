#!/usr/bin/env python3
"""Entrenamiento PPO para el cruce de 2 semaforos."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.rl.callbacks.tensorboard_callback import TensorboardKpiCallback
from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv

DEFAULT_CONFIG = project_root / "configs" / "reactive_random_500k.yaml"


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def create_env(cfg_dict: dict, *, use_gui: bool, demand: str | None, dynamic: bool) -> TrafficLightGymEnv:
    """Devuelve una instancia de TrafficLightGymEnv parametrizada."""

    env_cfg = EnvConfig(
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
        demand_profile_path=demand,
        dynamic_demand=dynamic,
        use_gui=use_gui,
        randomize_sumo_seed=bool(cfg_dict.get("randomize_sumo_seed", False)),
    )
    return TrafficLightGymEnv(env_cfg)


def build_env(
    cfg: dict,
    *,
    gui: bool,
    demand: str | None,
    dynamic: bool,
    base_seed: int,
    n_envs: int,
):
    """Crea entornos secuenciales o paralelos según n_envs."""

    max_steps = cfg["train"]["max_episode_steps"]

    if n_envs > 1:
        print("\n" + "=" * 60)
        print("CONFIGURANDO ENTORNOS PARALELOS")
        print("=" * 60)
        print(f"Número de entornos: {n_envs}")
        print(f"Seeds: {base_seed} .. {base_seed + n_envs - 1}")
        print("=" * 60 + "\n")

        def make_env_fn(rank: int):
            def _init():
                env_seed = base_seed + rank
                np.random.seed(env_seed)
                env = create_env(cfg, use_gui=False, demand=demand, dynamic=dynamic)
                env = Monitor(env)
                env = TimeLimit(env, max_episode_steps=max_steps)
                env.reset(seed=env_seed)
                return env

            return _init

        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)], start_method="spawn")

    print("\nCreando entorno único (sin paralelización)\n")
    env = create_env(cfg, use_gui=gui, demand=demand, dynamic=dynamic)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento PPO principal")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config YAML del entorno")
    parser.add_argument("--timesteps", type=int, default=None, help="Timesteps totales (override)")
    parser.add_argument("--fast", action="store_true", help="Ejecuta un entrenamiento corto de 2048 pasos")
    parser.add_argument("--gui", action="store_true", help="Activa SUMO GUI")
    parser.add_argument("--demand", default=None, help="Archivo YAML para demanda personalizada")
    parser.add_argument("--dynamic", action="store_true", help="Usar demanda dinámica según YAML")
    parser.add_argument("--model-name", default="ppo_tls", help="Nombre del modelo a guardar")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto", help="cpu, cuda o auto")
    parser.add_argument("--n-envs", type=int, default=None, help="Entornos paralelos (por defecto usa el YAML)")
    args = parser.parse_args()

    models_dir = project_root / "models"
    tensorboard_dir = project_root / "logs" / "tensorboard"
    models_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config)
    if args.n_envs is None:
        args.n_envs = cfg["train"].get("n_envs", 1)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Auto-detected device: {args.device.upper()}")
        if args.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   VRAM: {vram_gb:.2f} GB")

    seeds = cfg.get("seeds", {})
    base_seed = seeds.get("env", 42)
    np.random.seed(seeds.get("numpy", 42))

    env = build_env(
        cfg,
        gui=args.gui,
        demand=args.demand,
        dynamic=args.dynamic,
        base_seed=base_seed,
        n_envs=args.n_envs,
    )

    model = PPO(
        policy=cfg["ppo"]["policy"],
        env=env,
        learning_rate=cfg["ppo"].get("learning_rate", cfg["ppo"].get("lr", 3e-4)),
        gamma=cfg["ppo"]["gamma"],
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        n_epochs=cfg["ppo"].get("n_epochs", 10),
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        device=args.device,
    )

    try:
        if args.fast:
            print("\n" + "=" * 60)
            print("ENTRENAMIENTO RAPIDO (2048 steps)")
            print("=" * 60 + "\n")
            model.learn(total_timesteps=2048, callback=TensorboardKpiCallback(), progress_bar=True)
            model.save(models_dir / f"{args.model_name}_fast")
        else:
            total_ts = args.timesteps or cfg["train"]["total_timesteps"]
            print("\n" + "=" * 60)
            print("ENTRENAMIENTO PPO")
            print("=" * 60)
            print(f"Timesteps totales: {total_ts:,}")
            print(f"Device: {args.device.upper()}")
            print(f"N envs: {args.n_envs}")
            print(f"Steps por update: {cfg['ppo']['n_steps'] * args.n_envs:,}")
            print(f"Batch size: {cfg['ppo']['batch_size']}")
            print(f"N epochs: {cfg['ppo'].get('n_epochs', 10)}")
            print("=" * 60 + "\n")

            model.learn(total_timesteps=total_ts, callback=TensorboardKpiCallback(), progress_bar=True)
            model.save(models_dir / args.model_name)
            print(f"\n[OK] Modelo guardado en {models_dir / args.model_name}\n")
    finally:
        print("Cerrando entornos...")
        try:
            env.close()
            time.sleep(0.5)
        except Exception:
            pass
        print("Entornos cerrados.")


if __name__ == "__main__":
    main()
