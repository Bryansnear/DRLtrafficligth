import os
from pprint import pprint

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv


def main():
    cfg = EnvConfig(
        sumo_cfg_path=os.path.join("data", "sumo", "cfg", "four_way_1h.sumo.cfg"),
        step_length=1.0,
        control_interval=5,
        min_green=5,
        yellow=3,
        all_red=1,
        e2_ids=("E_in_0", "E_in_1", "S_in_0", "S_in_1"),
        v_free=13.9,
        jam_length_thr_m=20.0,
        e2_capacity_per_lane=6,
    )
    env = TrafficLightGymEnv(cfg)
    obs, info = env.reset(seed=42)
    print("Obs shape:", obs.shape)
    # Secuencia corta: mantener, mantener, cambiar, mantener
    actions = [0, 0, 1, 0]
    for i, a in enumerate(actions, 1):
        phase, tip = env.core.tls_state()
        print(f"Step {i} | fase={phase} tiempo_en_fase={tip:.1f}s | action={'hold' if a==0 else 'switch'}")
        obs, rew, term, trunc, info = env.step(a)
        print(f"  reward={rew:.3f}  served={info['served_cnt']:.0f}  queues={info['queues']}")
    env.core.close()


if __name__ == "__main__":
    # Asegura PYTHONPATH si corres fuera de IDE
    if os.getcwd() not in (os.environ.get("PYTHONPATH", "")):
        os.environ["PYTHONPATH"] = os.getcwd()
    main()






