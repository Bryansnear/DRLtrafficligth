import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def ensure_sumo_tools_on_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME no está definido. Configúralo para usar TraCI (p.ej., C:/Program Files (x86)/Eclipse/Sumo)."
        )
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspección de métricas TraCI (E2 y TLS) para verificar observables"
    )
    parser.add_argument(
        "--cfg",
        default=os.path.join("data", "sumo", "cfg", "four_way_1h.sumo.cfg"),
        help="Ruta al archivo .sumocfg",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Segundos a simular (step_length=1.0)",
    )
    parser.add_argument(
        "--control-interval",
        type=int,
        default=5,
        help="Intervalo (s) para agregar métricas y mostrar resumen",
    )
    parser.add_argument(
        "--jam-thr",
        type=float,
        default=20.0,
        help="Umbral de jamLength (m) para marcar spillback",
    )
    parser.add_argument(
        "--vfree",
        type=float,
        default=13.9,
        help="Velocidad libre (m/s) para normalizar meanSpeed",
    )
    args = parser.parse_args()

    ensure_sumo_tools_on_path()
    from sumolib import checkBinary  # type: ignore
    import traci  # type: ignore

    sumo_binary = checkBinary("sumo")
    traci.start(
        [
            sumo_binary,
            "-c",
            args.cfg,
            "--step-length",
            "1.0",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
    )

    try:
        # Detectores E2 disponibles (laneareas)
        e2_ids: List[str] = list(traci.lanearea.getIDList())
        # Si existen, ordena para estabilidad (E primero, luego S)
        e2_ids.sort()

        # TLS
        tls_ids = list(traci.trafficlight.getIDList())
        tls_id = tls_ids[0] if tls_ids else None

        print(f"Detectores E2: {e2_ids}")
        print(f"Semáforos (TLS): {tls_ids}")

        # Track de vehículos dentro de cada E2 para entered/exited
        prev_inside: Dict[str, Set[str]] = {eid: set() for eid in e2_ids}

        # Track para tiempo en fase
        last_phase = None
        last_phase_change_time = 0.0

        # Acumuladores por ΔTc
        acc_entered: Dict[str, int] = defaultdict(int)
        acc_exited: Dict[str, int] = defaultdict(int)

        def print_interval_summary(sim_time: float) -> None:
            # TLS info
            phase_idx = traci.trafficlight.getPhase(tls_id) if tls_id else -1
            time_in_phase = sim_time - last_phase_change_time if tls_id else 0.0
            # Por cada E2, obtener métricas de cierre de intervalo
            rows = []
            for eid in e2_ids:
                queue = traci.lanearea.getLastStepHaltingNumber(eid)
                occ_pct = traci.lanearea.getLastStepOccupancy(eid)  # porcentaje (0-100)
                occ_frac = occ_pct / 100.0
                speed = traci.lanearea.getLastStepMeanSpeed(eid)
                jam_m = traci.lanearea.getJamLengthMeters(eid)
                spill = 1 if (occ_frac > 0.90 or jam_m > args.jam_thr) else 0
                rows.append(
                    {
                        "e2": eid,
                        "queue": int(queue),
                        "occ": round(occ_pct, 2),
                        "meanV": round(speed, 2),
                        "jam_m": round(jam_m, 1),
                        "entered": acc_entered[eid],
                        "exited": acc_exited[eid],
                        "spill": spill,
                    }
                )
            print("-" * 80)
            print(
                f"t={int(sim_time)}s | TLS fase={phase_idx} tiempo_en_fase={time_in_phase:.1f}s"
            )
            for r in rows:
                print(
                    f"{r['e2']:>10} | cola={r['queue']:>2} ocupacion={r['occ']:.2f} "
                    f"vel={r['meanV']:>4.1f}m/s atasco_m={r['jam_m']:>4.1f}m "
                    f"entraron={r['entered']:>2} salieron={r['exited']:>2} desborde={r['spill']}"
                )

        # Simular
        end_time = float(args.duration)
        next_report = float(args.control_interval)

        while traci.simulation.getTime() < end_time:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # TLS tracking
            if tls_id is not None:
                phase_now = traci.trafficlight.getPhase(tls_id)
                if last_phase is None:
                    last_phase = phase_now
                    last_phase_change_time = sim_time
                elif phase_now != last_phase:
                    last_phase = phase_now
                    last_phase_change_time = sim_time

            # Por E2: calc entered/exited vía dif de conjuntos
            for eid in e2_ids:
                now_inside = set(traci.lanearea.getLastStepVehicleIDs(eid))
                entered = len(now_inside - prev_inside[eid])
                exited = len(prev_inside[eid] - now_inside)
                acc_entered[eid] += entered
                acc_exited[eid] += exited
                prev_inside[eid] = now_inside

            # Reporte por intervalo
            if sim_time >= next_report or abs(sim_time - next_report) < 1e-6:
                print_interval_summary(sim_time)
                # reset acumuladores del intervalo
                acc_entered = defaultdict(int)
                acc_exited = defaultdict(int)
                next_report += float(args.control_interval)

        # Resumen final
        print("=" * 80)
        print("Simulación finalizada.")

    finally:
        traci.close(False)


if __name__ == "__main__":
    main()


