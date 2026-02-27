"""
Evaluación DINÁMICA del escenario Ibarra Mayorista (Perfil Rush Hour).
Soporta demanda variable en el tiempo (Low -> Peak -> Recovery).
"""
import os
import sys
import csv
import random
import argparse
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import traci
import sumolib
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ==========================================
# CONFIGURACIÓN
# ==========================================
NET_FILE = PROJECT_ROOT / "data" / "sumo" / "network" / "ibarra_mayorista_10cuadras.net.xml"
ROUTES_FILE = PROJECT_ROOT / "data" / "sumo" / "routes" / "ibarra_global_dynamic.rou.xml"
CONFIG_FILE = PROJECT_ROOT / "data" / "sumo" / "cfg" / "ibarra_mayorista.sumocfg"
ADDITIONAL_DRL = PROJECT_ROOT / "data" / "sumo" / "additional" / "intersection_667932004_e2.add.xml"
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_ibarra_v1.zip"
TARGET_TLS = "667932004"

# Parameters
CONTROL_INTERVAL = 5
MIN_GREEN = 10
YELLOW = 3
ALL_RED = 1
MAX_GREEN = 60
E2_CAPACITY = 12
V_FREE = 13.9
JAM_THR_M = 20.0
E2_GROUPS = [["E_in_0"], ["S_in_0"], ["N_in_0"], ["W_in_0"]]
ALL_DRL_DETECTORS = [det for group in E2_GROUPS for det in group]

class DRLController:
    """Control DRL (Misma lógica que extended)."""
    def __init__(self, model_path, tls_id):
        self.model = PPO.load(str(model_path))
        self.tls_id = tls_id
        self.time_in_phase = 0
        self.switch_count = 0
        self._backlog = {eid: 0.0 for eid in ALL_DRL_DETECTORS}
        self._prev_inside = {eid: set() for eid in ALL_DRL_DETECTORS}
        self._last_phase = -1
        try: traci.trafficlight.setPhaseDuration(self.tls_id, 1000000)
        except: pass
    
    def get_observation(self):
        obs = []
        max_backlog = float(10 * E2_CAPACITY)
        for group in E2_GROUPS:
            g_queue, g_occ, g_vel, g_back = [], [], [], []
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
        
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            one_hot = [1.0, 0.0] if phase in (0, 1) else [0.0, 1.0]
            tip_norm = min(1.0, self.time_in_phase / float(MIN_GREEN))
            obs.extend(one_hot + [tip_norm])
        except: obs.extend([0.0, 0.0, 0.0])
        
        for group in E2_GROUPS:
            g_spill = []
            for eid in group:
                try:
                    occ = traci.lanearea.getLastStepOccupancy(eid)
                    jam = traci.lanearea.getJamLengthMeters(eid)
                    s = 1.0 if (occ > 90 or jam > JAM_THR_M) else 0.0
                    g_spill.append(s)
                except: g_spill.append(0.0)
            obs.append(max(g_spill))
            
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
            mask_sum = 1.0 if phase in (0, 2) else 0.5
            obs.append(mask_sum)
        except: obs.append(0.0)
        
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
            except: pass
            
    def step(self, sim_time):
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            if self._last_phase != -1 and current_phase != self._last_phase:
                 if current_phase in (0, 2): self.time_in_phase = 0
            self._last_phase = current_phase
        except: pass
        self.time_in_phase += 1
        self.update_backlog()
        
        if sim_time % CONTROL_INTERVAL != 0: return None
        
        obs = self.get_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray): action = action.item()
        
        try:
            phase = traci.trafficlight.getPhase(self.tls_id)
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
        except: pass
        return action

def generate_dynamic_routes(net_file, output_file, seed=42):
    """Genera perfil Rush Hour."""
    random.seed(seed)
    net = sumolib.net.readNet(str(net_file))
    valid_edges = [e for e in net.getEdges() if e.allows("passenger") and e.getLength() > 20]
    
    # PERFIL DEFINIDO (Duration seconds, Volume veh/h)
    # Total 6 horas (21600s)
    phases = [
        (3600, 1000),   # 1h: Warmup (Bajo)
        (7200, 2200),   # 2h: RUSH HOUR (Pico Extremo)
        (3600, 1500),   # 1h: Cooldown 1 (Medio)
        (7200, 1000)    # 2h: Recovery (Bajo)
    ]
    
    print(f"Generando rutas DINÁMICAS (Rush Hour) con semilla {seed}...")
    root = ET.Element("routes")
    vtype = ET.SubElement(root, "vType"); vtype.set("id", "car"); vtype.set("accel", "2.6"); vtype.set("decel", "4.5"); vtype.set("maxSpeed", "15")
    
    veh_id = 0
    current_time = 0
    
    for duration, volume in phases:
        print(f"  Fase: {duration}s @ {volume} veh/h")
        for t in range(current_time, current_time + duration):
            if random.random() < (volume / 3600.0):
                src = random.choice(valid_edges)
                dst = random.choice(valid_edges)
                if src != dst and src.getID() != dst.getID():
                    trip = ET.SubElement(root, "trip")
                    trip.set("id", f"veh_{veh_id}")
                    trip.set("type", "car")
                    trip.set("depart", str(t))
                    trip.set("from", src.getID())
                    trip.set("to", dst.getID())
                    veh_id += 1
        current_time += duration
        
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"  Total Vehículos: {veh_id}")
    return current_time

def run_dynamic_eval(mode="baseline", gui=False, seed=42):
    end_time = generate_dynamic_routes(NET_FILE, ROUTES_FILE, seed=seed)
    
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [sumo_binary, "-n", str(NET_FILE), "-r", str(ROUTES_FILE), "--step-length", "1.0", "--seed", str(seed), "--no-step-log", "true", "--waiting-time-memory", "10000", "--quit-on-end", "true", "--start", "true", "--ignore-route-errors", "true"]
    
    if mode == "drl": sumo_cmd.extend(["-a", str(ADDITIONAL_DRL)])
    
    traci.start(sumo_cmd)
    
    drl = DRLController(MODEL_PATH, TARGET_TLS) if mode == "drl" else None
    
    step = 0
    metrics = {'running': [], 'arrived': 0}
    
    while step < end_time:
        traci.simulationStep()
        if drl: drl.step(step)
        
        if step % 60 == 0:
            metrics['running'].append(traci.vehicle.getIDCount())
            
        metrics['arrived'] += traci.simulation.getArrivedNumber()
        step += 1
    
    # metrics['arrived'] is already total
    traci.close()
    
    print(f"RESULTADOS {mode.upper()}: Arrived={metrics['arrived']}, MeanRunning={np.mean(metrics['running']):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    run_dynamic_eval(args.mode, args.gui)
