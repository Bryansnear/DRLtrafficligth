import os
import sys
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set


def ensure_sumo_tools_on_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME no está definido. Configúralo para usar TraCI."
        )
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)


def main():
    parser = argparse.ArgumentParser(description="Control de tiempo fijo con KPIs y tripinfo")
    parser.add_argument("--cfg", default=os.path.join("data", "sumo", "cfg", "four_way_1h.sumo.cfg"))
    parser.add_argument("--duration", type=int, default=120, help="Duración total (s)")
    parser.add_argument("--greenA", type=int, default=20, help="Verde Fase A (Este) en s")
    parser.add_argument("--greenB", type=int, default=20, help="Verde Fase B (Sur) en s")
    parser.add_argument("--yellow", type=int, default=3, help="Amarillo en s")
    parser.add_argument("--gui", action="store_true", help="Usar SUMO GUI")
    parser.add_argument("--tripinfo", default=os.path.join("data", "sumo", "cfg", "tripinfo_fixed.xml"))
    args = parser.parse_args()

    ensure_sumo_tools_on_path()
    from sumolib import checkBinary  # type: ignore
    import traci  # type: ignore

    sumo_bin = checkBinary("sumo-gui" if args.gui else "sumo")
    traci.start(
        [
            sumo_bin,
            "-c", args.cfg,
            "--step-length", "1.0",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--tripinfo-output", args.tripinfo,
        ],
        label="fixed",
    )
    traci.switch("fixed")

    try:
        # IDs de TLS y detectores
        tls = traci.trafficlight.getIDList()[0]
        e2_ids: List[str] = list(traci.lanearea.getIDList())
        e2_ids.sort()

        # KPIs
        served_total = 0
        switches = 0
        queues_sum: Dict[str, float] = defaultdict(float)
        interval_len = 5
        next_report = interval_len

        # Track inside sets para entered/exited
        prev_inside: Dict[str, Set[str]] = {eid: set() for eid in e2_ids}

        sim_t = traci.simulation.getTime()
        end_t = sim_t + args.duration

        # Iniciar en Fase A (0: verde Este), luego amarillo (1), Fase B (2), amarillo (3)
        current_phase = 0
        traci.trafficlight.setPhase(tls, current_phase)
        phase_ends_at = sim_t + args.greenA

        print(f"Detectors: {e2_ids}")
        print(f"Tripinfo: {args.tripinfo}")

        while traci.simulation.getTime() < end_t:
            traci.simulationStep()
            sim_t = traci.simulation.getTime()

            # KPIs entered/exited y colas
            for eid in e2_ids:
                now_inside = set(traci.lanearea.getLastStepVehicleIDs(eid))
                entered = len(now_inside - prev_inside[eid])
                exited = len(prev_inside[eid] - now_inside)
                prev_inside[eid] = now_inside
                served_total += exited
                queues_sum[eid] += traci.lanearea.getLastStepHaltingNumber(eid)

            # Cambios de fase por tiempo fijo
            if sim_t >= phase_ends_at:
                if current_phase == 0:  # pasar a amarillo Este
                    traci.trafficlight.setPhase(tls, 1)
                    current_phase = 1
                    phase_ends_at = sim_t + args.yellow
                elif current_phase == 1:  # pasar a verde Sur
                    traci.trafficlight.setPhase(tls, 2)
                    current_phase = 2
                    phase_ends_at = sim_t + args.greenB
                    switches += 1
                elif current_phase == 2:  # pasar a amarillo Sur
                    traci.trafficlight.setPhase(tls, 3)
                    current_phase = 3
                    phase_ends_at = sim_t + args.yellow
                else:  # 3 -> volver a verde Este
                    traci.trafficlight.setPhase(tls, 0)
                    current_phase = 0
                    phase_ends_at = sim_t + args.greenA
                    switches += 1

            # Reporte simple cada intervalo
            if sim_t >= next_report:
                print(f"t={int(sim_t)}s | fase={current_phase} servidos={served_total}")
                next_report += interval_len

        # Promedios de colas por carril
        steps = max(1, int(args.duration))
        queues_avg = {eid: round(queues_sum[eid] / steps, 2) for eid in e2_ids}

        print("-" * 60)
        print("KPIs tiempo fijo:")
        print(f"  Vehículos servidos totales: {served_total}")
        print(f"  Cambios de fase: {switches}")
        print(f"  Colas promedio por carril: {queues_avg}")
        print(f"  tripinfo guardado en: {args.tripinfo}")

    finally:
        try:
            traci.close(False)
        except Exception:
            pass


if __name__ == "__main__":
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.getcwd()
    main()


