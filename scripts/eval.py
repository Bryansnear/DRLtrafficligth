import os
import yaml
import argparse
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, use_gui: bool, tripinfo: str | None = None, demand: str | None = None, dynamic: bool = False) -> TrafficLightGymEnv:
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
        tripinfo_output=tripinfo,
        demand_profile_path=demand,
        dynamic_demand=dynamic,
    )
    return TrafficLightGymEnv(ec)


def main():
    parser = argparse.ArgumentParser(description="Evaluación del modelo PPO sobre SUMO")
    parser.add_argument("--model", default=os.path.join("models", "ppo_tls.zip"), help="Ruta al modelo .zip")
    parser.add_argument("--steps", type=int, default=20, help="Pasos (intervalos de control) a evaluar")
    parser.add_argument("--gui", action="store_true", help="Usar SUMO GUI")
    parser.add_argument("--tripinfo", default=None, help="Ruta para exportar tripinfo.xml (opcional)")
    parser.add_argument("--demand", default=None, help="Perfil YAML de demanda (experiments/demands/*.yaml)")
    parser.add_argument("--dynamic", action="store_true", help="Usar inyección dinámica (TraCI) según perfil")
    args = parser.parse_args()

    cfg = load_cfg(os.path.join("experiments", "configs", "base.yaml"))
    env = make_env(cfg, use_gui=args.gui, tripinfo=args.tripinfo, demand=args.demand, dynamic=args.dynamic)

    print(f"Cargando modelo: {args.model}")
    model = PPO.load(args.model, env=env, device="cpu")

    obs, info = env.reset()
    kpi_served = 0.0
    kpi_switches = 0
    kpi_queues_sum: Dict[str, float] = {eid: 0.0 for eid in env.cfg.e2_ids}

    for i in range(1, args.steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        kpi_served += info.get("served_cnt", 0.0)
        kpi_switches += int(info.get("switch", 0))
        for eid, q in info.get("queues", {}).items():
            kpi_queues_sum[eid] += float(q)
        print(f"Paso {i:02d}: recompensa={reward:.3f} servidos={info.get('served_cnt', 0)} cambio={info.get('switch', 0)}")

    # Promedios de colas por carril
    kpi_queues_avg = {eid: v / float(args.steps) for eid, v in kpi_queues_sum.items()}

    print("-" * 60)
    print(f"KPIs en {args.steps} intervalos:")
    print(f"  Vehículos servidos totales: {kpi_served:.0f}")
    print(f"  Cambios de fase: {kpi_switches}")
    print(f"  Colas promedio por carril: {kpi_queues_avg}")

    # Cierre y conversión tripinfo
    try:
        env.core.close()
    except Exception:
        pass

    # Convertir tripinfo a CSV si se pidió
    if args.tripinfo:
        try:
            sumo_home = os.environ.get("SUMO_HOME")
            if sumo_home:
                xml2csv = os.path.join(sumo_home, "tools", "xml", "xml2csv.py")
                out_csv = os.path.splitext(args.tripinfo)[0] + ".csv"
                os.system(f"python \"{xml2csv}\" \"{args.tripinfo}\" -o \"{out_csv}\"")
                print(f"tripinfo CSV: {out_csv}")
        except Exception:
            pass


if __name__ == "__main__":
    # PYTHONPATH
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.getcwd()
    main()


