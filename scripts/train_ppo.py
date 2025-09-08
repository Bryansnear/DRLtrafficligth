import os
import yaml
import numpy as np
import argparse

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg_dict: dict) -> TrafficLightGymEnv:
    ec = EnvConfig(
        sumo_cfg_path=cfg_dict["sumo"]["cfg_path"],
        step_length=float(cfg_dict["sumo"]["step_length"]),
        control_interval=int(cfg_dict["control"]["control_interval"]),
        min_green=int(cfg_dict["control"]["min_green"]),
        yellow=int(cfg_dict["control"]["yellow"]),
        all_red=int(cfg_dict["control"]["all_red"]),
        e2_ids=tuple(cfg_dict["detectors"]["e2_ids"]),
        v_free=float(cfg_dict["detectors"]["v_free"]),
        jam_length_thr_m=float(cfg_dict["detectors"]["jam_length_thr_m"]),
        e2_capacity_per_lane=int(cfg_dict["detectors"]["e2_capacity_per_lane"]),
        w_served=float(cfg_dict["reward"]["w_served"]),
        w_queue=float(cfg_dict["reward"]["w_queue"]),
        w_backlog=float(cfg_dict["reward"]["w_backlog"]),
        w_switch=float(cfg_dict["reward"]["w_switch"]),
        w_spill=float(cfg_dict["reward"]["w_spill"]),
        kappa_backlog=int(cfg_dict["reward"]["kappa_backlog"]),
        sat_headway_s=float(cfg_dict["reward"]["sat_headway_s"]),
        use_gui=False,
    )
    return TrafficLightGymEnv(ec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Entrenamiento rápido sin evaluación (2048 timesteps)")
    parser.add_argument("--no-eval", action="store_true", help="Desactiva evaluación durante el entrenamiento")
    parser.add_argument("--timesteps", type=int, default=None, help="Total de timesteps a entrenar (si no se especifica, usa config)")
    args = parser.parse_args()

    cfg_path = os.path.join("experiments", "configs", "base.yaml")
    cfg = load_cfg(cfg_path)

    # Entrenamiento PPO (básico)
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from gymnasium.wrappers import TimeLimit
    from src.rl.callbacks.tensorboard_callback import TensorboardKpiCallback

    # Crear env (single) con Monitor y TimeLimit para episodios finitos
    env = Monitor(make_env(cfg))
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])

    # Modelo
    model = PPO(
        policy=cfg["ppo"]["policy"],
        env=env,
        learning_rate=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
        verbose=1,
        tensorboard_log=os.path.join("logs", "tensorboard"),
        device="cpu",
    )

    if args.fast:
        # Train sin evaluación
        model.learn(total_timesteps=2048, callback=TensorboardKpiCallback(), progress_bar=True)
        model.save(os.path.join("models", "ppo_tls_fast"))
    else:
        total_ts = args.timesteps if args.timesteps is not None else cfg["train"]["total_timesteps"]
        if args["no-eval"] if isinstance(args, dict) else args.no_eval:
            model.learn(total_timesteps=total_ts, callback=TensorboardKpiCallback(), progress_bar=True)
        else:
            # Eval callback
            eval_env = Monitor(make_env(cfg))
            eval_env = TimeLimit(eval_env, max_episode_steps=cfg["train"]["max_episode_steps"])
            eval_cb = EvalCallback(
                eval_env=eval_env,
                best_model_save_path=os.path.join("models"),
                log_path=os.path.join("logs", "eval"),
                eval_freq=cfg["train"]["eval_freq"],
                deterministic=True,
                n_eval_episodes=1,
                render=False,
            )
            # Train con evaluación
            model.learn(total_timesteps=total_ts, callback=[TensorboardKpiCallback(), eval_cb], progress_bar=True)
        model.save(os.path.join("models", "ppo_tls"))
    model.save(os.path.join("models", "ppo_tls"))

    # Cierre limpio
    try:
        # Intenta cerrar TraCI si existe
        if hasattr(env, "envs"):
            for e in env.envs:
                if hasattr(e, "env") and hasattr(e.env, "core"):
                    e.env.core.close()
        elif hasattr(env, "core"):
            env.core.close()
    except Exception:
        pass
    # Cerrar eval_env si se creó
    try:
        if (not args.fast) and (not (args.no_eval)) and 'eval_env' in locals():
            if hasattr(eval_env, "core"):
                eval_env.core.close()
    except Exception:
        pass


