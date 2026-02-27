"""
Evaluación de la intersección 667932004 - Baseline y DRL.
Usa los 4 detectores configurados igual que el escenario de entrenamiento.
"""
import os
import sys
import csv
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import traci
import sumolib
from stable_baselines3 import PPO

# Configuración
NET_FILE = PROJECT_ROOT / "data" / "sumo" / "network" / "ibarra_mayorista_10cuadras.net.xml"
ROUTES_FILE = PROJECT_ROOT / "data" / "sumo" / "routes" / "ibarra_mayorista_routes.rou.xml"
CONFIG_FILE = PROJECT_ROOT / "data" / "sumo" / "cfg" / "intersection_667932004.sumocfg"
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_ibarra_v1.zip"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "intersection_667932004"

# Intersección y detectores (alineados con fases: Ph0=E+S, Ph2=N+W)
TARGET_TLS = "667932004"
# Monitoreamos Lane 0 y Lane 1 de cada acceso
# Fase 0: E+S
E2_GROUPS = [
    ["E_in_0"], # Input 0 (Group E)
    ["S_in_0"], # Input 1 (Group S)
    ["N_in_0"], # Input 2 (Group N)
    ["W_in_0"]  # Input 3 (Group W)
]
ALL_E2_IDS = [det for group in E2_GROUPS for det in group]

# Semilla fija
FIXED_SEED = 42

# Parámetros de control (igual que entrenamiento)
CONTROL_INTERVAL = 5
MIN_GREEN = 10
YELLOW = 3
ALL_RED = 1
MAX_GREEN = 60
E2_CAPACITY = 12
V_FREE = 13.9
JAM_THR_M = 20.0

class BaselineController:
    """Control de tiempo fijo."""
    def __init__(self, tls_id, cycle_time=60):
        self.tls_id = tls_id
        self.cycle_time = cycle_time
        self.time_in_phase = 0
        self.switch_count = 0
    
    def step(self, sim_time):
        self.time_in_phase += 1
        # Cambiar cada cycle_time/2 segundos
        if self.time_in_phase >= self.cycle_time // 2:
            try:
                current = traci.trafficlight.getPhase(self.tls_id)
                num_phases = len(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases)
                next_phase = (current + 1) % num_phases
                traci.trafficlight.setPhase(self.tls_id, next_phase)
                # Mantener la fase fija para evitar actuated logic
                traci.trafficlight.setPhaseDuration(self.tls_id, 1000000)
                self.time_in_phase = 0
                self.switch_count += 1
            except:
                pass

class DRLController:
    """Control DRL con el modelo PPO."""
    def __init__(self, model_path, tls_id):
        self.model = PPO.load(str(model_path))
        self.tls_id = tls_id
        self.time_in_phase = 0
        self.switch_count = 0
        self._backlog = {eid: 0.0 for eid in ALL_E2_IDS}
        self._prev_inside = {eid: set() for eid in ALL_E2_IDS}
        self._last_phase = -1
        
        # Bloquear fase inicial
        try:
             traci.trafficlight.setPhaseDuration(self.tls_id, 1000000)
        except:
             pass
    
    def get_observation(self):
        """Construye observación de 24 features agregando carriles."""
        obs = []
        max_backlog = float(10 * E2_CAPACITY)
        
        # Para cada GRUPO (Input 0..3), tomamos el MAX de los carriles
        # Esto simula un carril "crítico" o saturado
        for group in E2_GROUPS:
            # Colectar métricas del grupo
            g_queue = []
            g_occ = []
            g_vel = []
            g_back = []
            
            for eid in group:
                try:
                    q = traci.lanearea.getLastStepHaltingNumber(eid)
                    o = traci.lanearea.getLastStepOccupancy(eid)
                    v = traci.lanearea.getLastStepMeanSpeed(eid)
                    b = self._backlog[eid]
                    
                    g_queue.append(min(1.0, float(q) / float(E2_CAPACITY)))
                    g_occ.append(float(o) / 100.0)
                    g_vel.append(0.0 if v < 0 else min(1.0, float(v) / V_FREE))
                    g_back.append(min(1.0, b / max(1e-6, max_backlog)))
                except:
                    g_queue.append(0.0)
                    g_occ.append(0.0)
                    g_vel.append(0.0)
                    g_back.append(0.0)
            
            # Usar MAX para ser conservador (si un carril está lleno, alerta)
            # O AVG? MAX es mejor para semáforos (un carril bloqueado requiere verde)
            obs.extend([max(g_queue), max(g_occ), max(g_vel), max(g_back)])
        
        # TLS: one-hot (2) + tiempo_en_fase_norm (1) = 3
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            one_hot = [1.0, 0.0] if phase in (0, 1) else [0.0, 1.0]
            tip_norm = min(1.0, self.time_in_phase / float(MIN_GREEN))
            obs.extend(one_hot + [tip_norm])
        except:
            obs.extend([0.0, 0.0, 0.0])
        
        # Spillback por grupo (usando MAX del grupo)
        for group in E2_GROUPS:
            g_spill = []
            for eid in group:
                try:
                    occ = traci.lanearea.getLastStepOccupancy(eid)
                    jam = traci.lanearea.getJamLengthMeters(eid)
                    s = 1.0 if (occ > 90 or jam > JAM_THR_M) else 0.0
                    g_spill.append(s)
                except:
                    g_spill.append(0.0)
            obs.append(max(g_spill))
        
        # served_mask_sum (1)
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            mask_sum = 1.0 if phase in (0, 2) else 0.5
            obs.append(mask_sum)
        except:
            obs.append(0.0)
        
        return np.array(obs[:24], dtype=np.float32)
    
    def update_backlog(self):
        """Actualiza backlog por carril."""
        for eid in ALL_E2_IDS:
            try:
                now_inside = set(traci.lanearea.getLastStepVehicleIDs(eid))
                entered = len(now_inside - self._prev_inside[eid])
                exited = len(self._prev_inside[eid] - now_inside)
                self._backlog[eid] += entered - exited
                self._backlog[eid] = max(0.0, self._backlog[eid])
                self._prev_inside[eid] = now_inside
            except:
                pass
    
    def step(self, sim_time):
        """Ejecuta paso de control DRL."""
        # Detectar cambio de fase REAL (completado transicion amarillo->verde)
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            # Si cambió de fase VERDE a VERDE (ej 0->2), reseteamos
            # Pero SUMO pasa por amarillo. 0 -> 1 -> 2.
            # Resetear al entrar a una NUEVA fase verde estable (0 o 2)
            if self._last_phase != -1 and current_phase != self._last_phase:
                 if current_phase in (0, 2):
                     self.time_in_phase = 0
            self._last_phase = current_phase
        except:
            pass
            
        self.time_in_phase += 1
        self.update_backlog()
        
        # Solo decidir cada CONTROL_INTERVAL
        if sim_time % CONTROL_INTERVAL != 0:
            return None
        
        obs = self.get_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Aplicar acción
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            
            # Action 1 = Switch (Si pasó min green)
            # Action 0 = Keep
            
            # Forzar mantener fase actual si no cambiamos
            # Esto sobreescribe la lógica actuated de SUMO
            traci.trafficlight.setPhaseDuration(self.tls_id, 1000000)

            if action == 1 and self.time_in_phase >= MIN_GREEN and phase in (0, 2):
                yellow_phase = 1 if phase == 0 else 3
                
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                traci.trafficlight.setPhaseDuration(self.tls_id, YELLOW + ALL_RED)
                
                # NO resetear timer aquí, se reseteará cuando llegue a la nueva fase verde
                self.switch_count += 1
                
            elif self.time_in_phase >= MAX_GREEN and phase in (0, 2):
                # Timeout forzado
                yellow_phase = 1 if phase == 0 else 3
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                traci.trafficlight.setPhaseDuration(self.tls_id, YELLOW + ALL_RED)
                self.switch_count += 1
        except:
            pass
        
        return action

def generate_routes(duration=7200, veh_per_hour=2000, seed=FIXED_SEED):
    """Genera rutas DIRECCIONADAS para saturar la intersección 667932004."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Edges de la intersección 667932004
    # Incoming: N=-1228099685#0, S=-756028261, E=1228099684, W=51962697#2
    # Outgoing: S=-1228099684, N=1228099685#0, W=-51962697#2, E=756028261 (Ojo: IDs de salida pueden variar, verificar conexiones)
    # Usando inspección previa:
    # N_in -> S_out (-1228099684), W_out (-51962697#2), E_out (756028261 - giro)
    
    in_edges = ["-1228099685#0", "-756028261", "1228099684", "51962697#2"]
    out_edges = ["-1228099684", "1228099685#0", "-51962697#2", "756028261"]
    
    root = ET.Element("routes")
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6")
    vtype.set("decel", "4.5")
    vtype.set("sigma", "0.5")
    vtype.set("length", "4.5")
    vtype.set("maxSpeed", "15")
    
    # Generar tráfico pesado
    n_veh = int(veh_per_hour * (duration / 3600))
    interval = 3600 / veh_per_hour
    veh_id = 0
    
    print(f"  Inyectando {n_veh} vehículos en flujos dirigidos...")
    
    for t in range(0, duration):
        # Probabilidad de inserción por segundo para alcanzar rate deseado
        if random.random() < (veh_per_hour / 3600.0):
            # Elegir origen y destino válidos
            src = random.choice(in_edges)
            dst = random.choice(out_edges)
            
            # Evitar U-turns inmediatos obvios si coinciden IDs base (simple heurística)
            if src.strip("-") in dst or dst.strip("-") in src:
                 dst = random.choice([e for e in out_edges if e != dst])

            trip = ET.SubElement(root, "trip")
            trip.set("id", f"veh_{veh_id}")
            trip.set("type", "car")
            trip.set("depart", str(t))
            trip.set("from", src)
            trip.set("to", dst)
            veh_id += 1
            
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(ROUTES_FILE, encoding="utf-8", xml_declaration=True)
    return veh_id

def run_evaluation(mode="baseline", duration=7200, cycle_time=60, gui=False, seed=FIXED_SEED):
    """Ejecuta evaluación."""
    print("=" * 70)
    print(f"EVALUACIÓN {mode.upper()} - INTERSECCIÓN {TARGET_TLS}")
    print("=" * 70)
    print(f"Semilla: {seed}, Duración: {duration}s ({duration/3600:.1f}h)")
    
    # Generar rutas
    print("\n[1/3] Generando rutas...")
    n_veh = generate_routes(duration=duration, seed=seed)
    print(f"  Vehículos: {n_veh}")
    
    # Iniciar SUMO
    print("\n[2/3] Iniciando simulación...")
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", str(CONFIG_FILE), "--seed", str(seed),
                "--start", "true" if not gui else "false", "--quit-on-end", "true"]
    
    traci.start(sumo_cmd)
    
    # Métricas
    metrics = {'time': [], 'queue': [], 'queue_m': [], 'throughput': [], 
               'speed': [], 'waiting': []}
    total_arrived = 0
    
    try:
        # Verificar semáforo
        if TARGET_TLS not in traci.trafficlight.getIDList():
            print(f"  ERROR: Semáforo {TARGET_TLS} no encontrado!")
            return None, None
        
        # Crear controlador
        if mode == "baseline":
            controller = BaselineController(TARGET_TLS, cycle_time)
            print(f"  Baseline: ciclo {cycle_time}s")
        else:
            controller = DRLController(MODEL_PATH, TARGET_TLS)
            print(f"  DRL: {MODEL_PATH.name}")
        
        step = 0
        last_progress = 0
        
        while step < duration:
            traci.simulationStep()
            controller.step(step)
            
            # Recopilar cada 5s
            if step % 5 == 0:
                queue = 0
                queue_m = 0
                speed_sum = 0
                speed_n = 0
                waiting = 0
                
                for eid in ALL_E2_IDS:
                    try:
                        queue += traci.lanearea.getJamLengthVehicle(eid)
                        queue_m += traci.lanearea.getJamLengthMeters(eid)
                        s = traci.lanearea.getLastStepMeanSpeed(eid)
                        if s >= 0:
                            speed_sum += s
                            speed_n += 1
                        if s < 0.5:
                            waiting += traci.lanearea.getLastStepVehicleNumber(eid)
                    except:
                        pass
                
                total_arrived += traci.simulation.getArrivedNumber()
                
                metrics['time'].append(step)
                metrics['queue'].append(queue)
                metrics['queue_m'].append(queue_m)
                metrics['throughput'].append(total_arrived)
                metrics['speed'].append(speed_sum / speed_n if speed_n > 0 else 0)
                metrics['waiting'].append(waiting)
            
            # Progreso
            progress = int(step / duration * 100)
            if progress >= last_progress + 10:
                print(f"  [{progress}%] Queue={metrics['queue'][-1]}, Throughput={total_arrived}")
                last_progress = progress
            
            step += 1
        
        switches = controller.switch_count
        
    finally:
        traci.close()
    
    # Resultados
    results = {
        'mode': mode,
        'seed': seed,
        'duration_s': duration,
        'throughput': total_arrived,
        'throughput_per_hour': total_arrived / (duration / 3600),
        'avg_queue': np.mean(metrics['queue']),
        'max_queue': max(metrics['queue']),
        'avg_queue_m': np.mean(metrics['queue_m']),
        'max_queue_m': max(metrics['queue_m']),
        'avg_speed': np.mean(metrics['speed']),
        'avg_waiting': np.mean(metrics['waiting']),
        'switches': switches,
    }
    
    return results, metrics

def print_results(results):
    """Imprime resultados."""
    print("\n" + "=" * 70)
    print(f"RESULTADOS - {results['mode'].upper()}")
    print("=" * 70)
    print(f"  Throughput: {results['throughput']} veh ({results['throughput_per_hour']:.1f}/h)")
    print(f"  Cola promedio: {results['avg_queue']:.2f} veh")
    print(f"  Cola máxima: {results['max_queue']:.0f} veh")
    print(f"  Largo cola promedio: {results['avg_queue_m']:.2f} m")
    print(f"  Largo cola máximo: {results['max_queue_m']:.2f} m")
    print(f"  Velocidad promedio: {results['avg_speed']:.2f} m/s ({results['avg_speed']*3.6:.1f} km/h)")
    print(f"  Cambios de fase: {results['switches']}")

def save_results(results, metrics, mode):
    """Guarda resultados."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = OUTPUT_DIR / f"{mode}_summary_{ts}.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    metrics_file = OUTPUT_DIR / f"{mode}_metrics_{ts}.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        for i in range(len(metrics['time'])):
            writer.writerow([metrics[k][i] for k in metrics.keys()])
    
    print(f"\nGuardado en: {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "drl", "both"], default="both")
    parser.add_argument("--duration", type=int, default=7200)
    parser.add_argument("--cycle-time", type=int, default=60)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    
    if args.mode in ["baseline", "both"]:
        results_bl, metrics_bl = run_evaluation("baseline", args.duration, args.cycle_time, args.gui, args.seed)
        if results_bl:
            print_results(results_bl)
            save_results(results_bl, metrics_bl, "baseline")
    
    if args.mode in ["drl", "both"]:
        results_drl, metrics_drl = run_evaluation("drl", args.duration, args.cycle_time, args.gui, args.seed)
        if results_drl:
            print_results(results_drl)
            save_results(results_drl, metrics_drl, "drl")
    
    # Comparación
    if args.mode == "both" and results_bl and results_drl:
        print("\n" + "=" * 70)
        print("COMPARACIÓN BASELINE vs DRL")
        print("=" * 70)
        for key in ['throughput', 'avg_queue', 'max_queue', 'avg_queue_m', 'avg_speed']:
            bl = results_bl[key]
            drl = results_drl[key]
            diff = ((drl - bl) / bl * 100) if bl != 0 else 0
            symbol = "✅" if (key in ['throughput', 'avg_speed'] and diff > 0) or \
                            (key in ['avg_queue', 'max_queue', 'avg_queue_m'] and diff < 0) else "❌"
            print(f"  {key}: BL={bl:.2f}, DRL={drl:.2f}, Diff={diff:+.1f}% {symbol}")

if __name__ == "__main__":
    main()
