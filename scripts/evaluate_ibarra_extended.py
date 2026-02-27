"""
Evaluación EXTENDIDA del escenario Ibarra Mayorista.
Modo Híbrido:
- Intersección 667932004: Controlada por DRL (PPO) optimizado (con sensores y phase locking).
- Resto de la Red: Controlado por lógica estándar de SUMO (Actuated/Static).
- Tráfico: Generado para toda la red (Global) con semilla fija.
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

# ==========================================
# CONFIGURACIÓN
# ==========================================
NET_FILE = PROJECT_ROOT / "data" / "sumo" / "network" / "ibarra_mayorista_10cuadras.net.xml"
ROUTES_FILE = PROJECT_ROOT / "data" / "sumo" / "routes" / "ibarra_mayorista_global_routes.rou.xml"
CONFIG_FILE = PROJECT_ROOT / "data" / "sumo" / "cfg" / "ibarra_mayorista.sumocfg"

# Archivos adicionales específicos
ADDITIONAL_DRL = PROJECT_ROOT / "data" / "sumo" / "additional" / "intersection_667932004_e2.add.xml"

# Modelo DRL
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_ibarra_v1.zip"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ibarra_full_network"

# Objetivo DRL
TARGET_TLS = "667932004"
FIXED_SEED = 42

# Parámetros DRL (mismos que evaluate_intersection_667932004.py)
CONTROL_INTERVAL = 5
MIN_GREEN = 10
YELLOW = 3
ALL_RED = 1
MAX_GREEN = 60
E2_CAPACITY = 12
V_FREE = 13.9
JAM_THR_M = 20.0

# Detectores para DRL (Solo Lane 0)
E2_GROUPS = [
    ["E_in_0"], # Input 0 (Group E)
    ["S_in_0"], # Input 1 (Group S)
    ["N_in_0"], # Input 2 (Group N)
    ["W_in_0"]  # Input 3 (Group W)
]
ALL_DRL_DETECTORS = [det for group in E2_GROUPS for det in group]

# ==========================================
# CLASES DE CONTROL
# ==========================================
class DRLController:
    """Control DRL con el modelo PPO (Copiado de evaluate_intersection_667932004.py)."""
    def __init__(self, model_path, tls_id):
        self.model = PPO.load(str(model_path))
        self.tls_id = tls_id
        self.time_in_phase = 0
        self.switch_count = 0
        self._backlog = {eid: 0.0 for eid in ALL_DRL_DETECTORS}
        self._prev_inside = {eid: set() for eid in ALL_DRL_DETECTORS}
        self._last_phase = -1
        
        # Bloquear fase inicial para tomar control exclusivo
        try:
             traci.trafficlight.setPhaseDuration(self.tls_id, 1000000)
        except:
             pass
    
    def get_observation(self):
        """Construye observación de 24 features."""
        obs = []
        max_backlog = float(10 * E2_CAPACITY)
        
        # Features por grupo (Max Aggregation)
        for group in E2_GROUPS:
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
                    g_queue.append(0.0); g_occ.append(0.0); g_vel.append(0.0); g_back.append(0.0)
            
            obs.extend([max(g_queue), max(g_occ), max(g_vel), max(g_back)])
        
        # TLS State
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            one_hot = [1.0, 0.0] if phase in (0, 1) else [0.0, 1.0]
            tip_norm = min(1.0, self.time_in_phase / float(MIN_GREEN))
            obs.extend(one_hot + [tip_norm])
        except:
            obs.extend([0.0, 0.0, 0.0])
        
        # Spillback
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
        
        # Served mask
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            mask_sum = 1.0 if phase in (0, 2) else 0.5
            obs.append(mask_sum)
        except:
            obs.append(0.0)
        
        return np.array(obs[:24], dtype=np.float32)
    
    def update_backlog(self):
        for eid in ALL_DRL_DETECTORS:
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
        # Detectar cambio fase real
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            if self._last_phase != -1 and current_phase != self._last_phase:
                 if current_phase in (0, 2):
                     self.time_in_phase = 0
            self._last_phase = current_phase
        except:
            pass
            
        self.time_in_phase += 1
        self.update_backlog()
        
        if sim_time % CONTROL_INTERVAL != 0:
            return None
        
        obs = self.get_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray): action = action.item()
        
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            # Re-Lock phase to fight SUMO actuated logic
            traci.trafficlight.setPhaseDuration(self.tls_id, 1000000)

            if action == 1 and self.time_in_phase >= MIN_GREEN and phase in (0, 2):
                yellow_phase = 1 if phase == 0 else 3
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                traci.trafficlight.setPhaseDuration(self.tls_id, YELLOW + ALL_RED)
                self.switch_count += 1
                
            elif self.time_in_phase >= MAX_GREEN and phase in (0, 2):
                yellow_phase = 1 if phase == 0 else 3
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)
                traci.trafficlight.setPhaseDuration(self.tls_id, YELLOW + ALL_RED)
                self.switch_count += 1
        except:
            pass
        return action

# ==========================================
# GENERACIÓN DE TRÁFICO GLOBAL
# ==========================================
def generate_global_routes(net_file, output_file, duration=7200, veh_per_hour=3000, seed=42):
    """Genera tráfico aleatorio en TODO el mapa."""
    random.seed(seed)
    np.random.seed(seed)
    
    net = sumolib.net.readNet(str(net_file))
    edges = net.getEdges()
    # Solo edges válidos para autos
    valid_edges = [e for e in edges if e.allows("passenger") and e.getLength() > 20] # >20m para evitar errores insercion
    
    print(f"Generando rutas GLOBALES con semilla {seed}...")
    print(f"  Edges válidos: {len(valid_edges)}")
    
    root = ET.Element("routes")
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6"); vtype.set("decel", "4.5"); vtype.set("maxSpeed", "15")
    
    # Generar viajes
    n_veh = int(veh_per_hour * (duration / 3600))
    interval = 3600 / veh_per_hour
    veh_id = 0
    
    for t in range(0, duration):
        if random.random() < (veh_per_hour / 3600.0):
            src = random.choice(valid_edges)
            dst = random.choice(valid_edges)
            
            attempts = 0
            while (src == dst or not net.getShortestPath(src, dst)) and attempts < 20:
                dst = random.choice(valid_edges)
                attempts += 1
            
            if src != dst and attempts < 20:
                trip = ET.SubElement(root, "trip")
                trip.set("id", f"veh_{veh_id}")
                trip.set("type", "car")
                trip.set("depart", str(t))
                trip.set("from", src.getID())
                trip.set("to", dst.getID())
                veh_id += 1
                
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"  Vehículos generados: {veh_id}")
    return veh_id

# ==========================================
# EVALUACIÓN
# ==========================================
def run_full_evaluation(mode="baseline", duration=7200, gui=False, seed=42, volume=1000):
    print("=" * 70)
    print(f"EVALUACIÓN RED COMPLETA - MODO {mode.upper()}")
    print("=" * 70)
    
    # 1. Generar Rutas Globales
    generate_global_routes(NET_FILE, ROUTES_FILE, duration, veh_per_hour=volume, seed=seed)
    
    # 2. Configurar SUMO
    sumo_binary = "sumo-gui" if gui else "sumo"
    
    # Argumentos base
    sumo_cmd = [
        sumo_binary,
        "-n", str(NET_FILE),
        "-r", str(ROUTES_FILE),
        "--step-length", "1.0",
        "--seed", str(seed),
        "--start", "true" if not gui else "false",
        "--quit-on-end", "true",
        "--no-step-log", "true",
        "--ignore-route-errors", "true"
    ]
    
    # Si es DRL, añadir detectores adicionales solo para ese nodo
    additional_files = []
    if mode == "drl":
        additional_files.append(str(ADDITIONAL_DRL))
    
    if additional_files:
        sumo_cmd.extend(["-a", ",".join(additional_files)])
        
    print(f"\nIniciando SUMO ({mode})...")
    traci.start(sumo_cmd)
    
    # 3. Inicializar Controlador DRL (si aplica)
    drl_controller = None
    if mode == "drl":
        try:
            drl_controller = DRLController(MODEL_PATH, TARGET_TLS)
            print(f"  Controlador DRL activado para {TARGET_TLS}")
        except Exception as e:
            print(f"  ERROR iniciando DRL: {e}")
            traci.close()
            return None
            
    # 4. Loop Simulación
    metrics = {
        'time': [],
        'running_veh': [],
        'mean_speed': [],
        'mean_waiting': [], # Vehículos detenidos
        'total_arrived': [],
        'drl_queue': [] # Cola en intersección DRL (si aplica)
    }
    
    step = 0
    total_arrived = 0
    
    try:
        while step < duration:
            traci.simulationStep()
            
            # --- Lógica DRL Híbrida ---
            if mode == "drl" and drl_controller:
                 # Solo controla TARGET_TLS, el resto es automático
                 drl_controller.step(step)
            
            # --- Recolección Métricas Globales ---
            if step % 5 == 0:
                # Métricas globales de la red
                running = traci.vehicle.getIDCount()
                arrived = traci.simulation.getArrivedNumber()
                total_arrived += arrived
                
                # Velocidad promedio de todos los vehículos en la red
                # Costoso computacionalmente pedir datos de todos, usaremos muestra o agregados si es posible?
                # traci.vehicle.getIDList() es lento con miles de autos.
                # Mejor usar métricas de sistema si existen, o sampling.
                # Para simplificar, usaremos sensores DRL para métrica local y Arrived para global.
                
                # Vamos a obtener wait time global acumulado (approx)
                # minExpectedNumber=0 nos da statisticas de todos los que estan cargados
                
                metrics['time'].append(step)
                metrics['running_veh'].append(running)
                metrics['total_arrived'].append(total_arrived)
                
                # Métrica local DRL (para verificar que no explote la intersección)
                q_local = 0
                if mode == "drl":
                    for det in ALL_DRL_DETECTORS:
                         try: q_local += traci.lanearea.getLastStepHaltingNumber(det)
                         except: pass
                    metrics['drl_queue'].append(q_local)
                else:
                    metrics['drl_queue'].append(0)

            step += 1
            if step % 1000 == 0:
                print(f"  Step {step}/{duration}: Arrived={total_arrived}, Running={traci.vehicle.getIDCount()}")
                
    finally:
        traci.close()
        
    # Resultados finales
    results = {
        'mode': mode,
        'seed': seed,
        'total_throughput': total_arrived,
        'avg_running': np.mean(metrics['running_veh']),
        'drl_switches': drl_controller.switch_count if drl_controller else 0
    }
    return results, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "drl"], required=True)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--volume", type=int, default=1000, help="Vehicles per hour for global generation")
    args = parser.parse_args()
    
    results, metrics = run_full_evaluation(args.mode, args.duration, args.gui, args.seed, args.volume)
    
    print("\n" + "="*60)
    print(f"RESULTADOS {args.mode.upper()}")
    print("="*60)
    print(f"  Throughput Global: {results['total_throughput']} veh")
    print(f"  Avg Running Vehs: {results['avg_running']:.1f}")
    if args.mode == "drl":
        print(f"  DRL Phase Switches: {results['drl_switches']}")
        
    # Guardar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"global_{args.mode}_{ts}.csv", 'w') as f:
        w = csv.writer(f)
        w.writerow(results.keys())
        w.writerow(results.values())

if __name__ == "__main__":
    main()
