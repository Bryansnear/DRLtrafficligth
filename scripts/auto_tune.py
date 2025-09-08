import os
import argparse
import yaml
import optuna
from typing import Dict

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, overrides: Dict) -> TrafficLightGymEnv:
    rc = cfg["reward"]
    rc = {**rc, **overrides}
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
        w_served=float(rc["w_served"]),
        w_queue=float(rc["w_queue"]),
        w_backlog=float(rc["w_backlog"]),
        w_switch=float(rc["w_switch"]),
        w_spill=float(rc["w_spill"]),
        kappa_backlog=int(rc["kappa_backlog"]),
        sat_headway_s=float(rc["sat_headway_s"]),
        use_gui=False,
    )
    env = TrafficLightGymEnv(ec)
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=cfg["train"]["max_episode_steps"])
    return env


def objective(trial: optuna.Trial, cfg: dict) -> float:
    # Espacio de búsqueda expandido: pesos de recompensa, min_green y control_interval
    overrides = {
        "w_served": trial.suggest_float("w_served", 0.1, 0.6),        # Expandido
        "w_queue": trial.suggest_float("w_queue", 0.2, 0.9),          # Expandido  
        "w_backlog": trial.suggest_float("w_backlog", 0.1, 0.8),      # Expandido
        "w_switch": trial.suggest_float("w_switch", 0.01, 0.20),      # Expandido
        "w_spill": trial.suggest_float("w_spill", 0.05, 0.5),         # Expandido
        "kappa_backlog": trial.suggest_int("kappa_backlog", 5, 15),   # Nuevo parámetro
    }
    # Optimizar también parámetros de control
    cfg2 = yaml.safe_load(yaml.dump(cfg))
    cfg2["control"]["min_green"] = int(trial.suggest_int("min_green", 5, 15))
    cfg2["control"]["control_interval"] = int(trial.suggest_int("control_interval", 3, 8))

    env = make_env(cfg2, overrides)
    model = PPO(policy=cfg["ppo"]["policy"], env=env, learning_rate=cfg["ppo"]["lr"], gamma=cfg["ppo"]["gamma"],
                n_steps=cfg["ppo"]["n_steps"], batch_size=cfg["ppo"]["batch_size"], clip_range=cfg["ppo"]["clip_range"],
                ent_coef=cfg["ppo"]["ent_coef"], device="cpu", verbose=0)
    # Entrenamiento más largo para mejor evaluación por trial
    model.learn(total_timesteps=6144)  # ~3 epochs para mejor convergencia

    # Evaluación más robusta: 20 intervalos (100s de simulación)
    obs, info = env.reset()
    served_total = 0.0
    switches = 0
    qsum = 0.0
    reward_sum = 0.0
    spill_count = 0
    
    for step in range(20):  # 20 pasos = 100s de simulación  
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        served_total += info.get("served_cnt", 0.0)
        switches += int(info.get("switch", 0))
        qsum += sum(info.get("queues", {}).values())
        reward_sum += reward
        # Contar spillbacks
        if any(q > 5 for q in info.get("queues", {}).values()):
            spill_count += 1
            
    # Función objetivo mejorada: balancea throughput, colas, spillback y estabilidad
    throughput_score = served_total / 20.0  # veh/intervalo 
    stability_penalty = (switches / 20.0) * 0.3  # penalizar cambios excesivos
    queue_penalty = (qsum / 400.0) * 0.4  # normalizado por capacidad total estimada
    spill_penalty = (spill_count / 20.0) * 0.6  # penalizar congestión severa
    
    score = throughput_score - stability_penalty - queue_penalty - spill_penalty
    try:
        env.core.close()
    except Exception:
        pass
    return float(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tuning avanzado con Optuna")
    parser.add_argument("--trials", type=int, default=25, help="Número de trials (default: 25)")  
    parser.add_argument("--jobs", type=int, default=1, help="Jobs paralelos (default: 1)")
    parser.add_argument("--sampler", choices=["tpe", "random", "cmaes"], default="tpe", 
                       help="Sampler de Optuna (default: tpe)")
    args = parser.parse_args()

    cfg = load_cfg(os.path.join("experiments", "configs", "base.yaml"))
    
    # Configurar sampler más sofisticado
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.sampler == "random":  
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:  # cmaes
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    
    print(f"🎯 Iniciando autotuning con {args.trials} trials usando {args.sampler.upper()} sampler...")
    print("📊 Parámetros a optimizar:")
    print("  - Pesos de recompensa (w_served, w_queue, w_backlog, w_switch, w_spill)")
    print("  - Parámetros de control (min_green, control_interval)")
    print("  - Factor backlog (kappa_backlog)")
    
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, cfg), n_trials=args.trials, n_jobs=args.jobs)

    print("\n" + "="*70)
    print("🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
    print("="*70)
    for key, value in study.best_params.items():
        print(f"  {key:20}: {value}")
    print(f"\n🎯 Mejor score: {study.best_value:.4f}")
    
    # Mostrar top 3 trials
    print("\n📊 TOP 3 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -float('inf'), reverse=True)[:3]
    for i, trial in enumerate(trials, 1):
        print(f"\n#{i} (Score: {trial.value:.4f}):")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    print(f"\n💾 Para aplicar los mejores parámetros, ejecuta:")
    print(f"   python scripts/apply_best_params.py")
    
    # Guardar study para análisis posterior
    import pickle
    with open("autotuning_study.pkl", "wb") as f:
        pickle.dump(study, f)
    print(f"📁 Study guardado en: autotuning_study.pkl")





