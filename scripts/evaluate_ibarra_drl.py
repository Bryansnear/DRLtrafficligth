"""
Evaluación con modelo DRL para el escenario Ibarra Mayorista.
Controla el semáforo 667932004 en la calle 13 de Abril con el modelo PPO entrenado.
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
ADDITIONAL_FILE = PROJECT_ROOT / "data" / "sumo" / "additional" / "ibarra_13abril_e2.add.xml"
CONFIG_FILE = PROJECT_ROOT / "data" / "sumo" / "cfg" / "ibarra_mayorista_baseline.sumocfg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ibarra_drl"
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_optimized_final.zip"

# Semilla fija (misma que baseline para comparación)
FIXED_SEED = 42

# Semáforo objetivo
TARGET_TLS = "667932004"

class DRLController:
    """Controlador DRL para un semáforo específico."""
    
    def __init__(self, model_path, tls_id, e2_detector_ids, control_interval=5, min_green=10, yellow=3):
        self.model = PPO.load(model_path)
        self.tls_id = tls_id
        self.e2_ids = e2_detector_ids
        self.control_interval = control_interval
        self.min_green = min_green
        self.yellow = yellow
        
        self.current_phase = 0
        self.time_in_phase = 0
        self.last_action = 0
        self.switch_count = 0
        
        print(f"DRL Controller inicializado para TLS: {tls_id}")
        print(f"  Modelo: {model_path}")
        print(f"  Detectores E2: {len(e2_detector_ids)}")
    
    def get_observation(self):
        """Construye la observación del entorno para el modelo (24 features).
        
        Observación:
        - Por carril (4 detectores): [cola_norm, ocupacion, vel_norm, backlog_norm] → 16
        - TLS: fase one-hot (2) + tiempo_en_fase_norm (1) → 3
        - Spillback por carril (4) → 4
        - served_mask_sum (1) → 1
        Total: 24 features
        """
        obs = []
        
        # Datos de 4 detectores (como en el entrenamiento)
        e2_capacity = 12
        jam_thr_m = 20.0
        
        for i, det_id in enumerate(self.e2_ids[:4]):
            try:
                queue = traci.lanearea.getLastStepHaltingNumber(det_id)
                occ_pct = traci.lanearea.getLastStepOccupancy(det_id)
                mean_v = traci.lanearea.getLastStepMeanSpeed(det_id)
                jam_m = traci.lanearea.getJamLengthMeters(det_id)
                
                cola_norm = min(1.0, float(queue) / float(e2_capacity))
                occ_frac = float(occ_pct) / 100.0
                vel_norm = 0.0 if mean_v < 0 else min(1.0, float(mean_v) / 13.9)
                backlog_norm = 0.0  # Simplificado para inferencia
                
                obs.extend([cola_norm, occ_frac, vel_norm, backlog_norm])
            except:
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # Padding si hay menos de 4 detectores
        while len(obs) < 16:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # TLS: fase one-hot (2) + tiempo_en_fase_norm (1)
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            # One-hot para fase (0,1 = Este, 2,3 = Sur)
            if current_phase in (0, 1):
                one_hot = [1.0, 0.0]
            else:
                one_hot = [0.0, 1.0]
            tip_norm = min(1.0, self.time_in_phase / 10.0)  # min_green=10
            obs.extend(one_hot + [tip_norm])
        except:
            obs.extend([0.0, 0.0, 0.0])
        
        # Spillback por carril (4)
        for i, det_id in enumerate(self.e2_ids[:4]):
            try:
                occ_pct = traci.lanearea.getLastStepOccupancy(det_id)
                jam_m = traci.lanearea.getJamLengthMeters(det_id)
                spill = 1.0 if (occ_pct > 90 or jam_m > jam_thr_m) else 0.0
                obs.append(spill)
            except:
                obs.append(0.0)
        
        # Padding spillback
        while len(obs) < 23:
            obs.append(0.0)
        
        # served_mask_sum (1)
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            mask_sum = 1.0 if current_phase in (0, 2) else 0.5
            obs.append(mask_sum)
        except:
            obs.append(0.0)
        
        return np.array(obs[:24], dtype=np.float32)
    
    def step(self, sim_time):
        """Ejecuta un paso del controlador DRL."""
        self.time_in_phase += 1
        
        # Solo tomar decisión cada control_interval
        if sim_time % self.control_interval != 0:
            return None
        
        # Obtener observación
        obs = self.get_observation()
        
        # Obtener acción del modelo
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Interpretar acción: 0 = mantener, 1 = cambiar
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Aplicar acción
        if action == 1 and self.time_in_phase >= self.min_green:
            # Cambiar de fase
            try:
                current_phase = traci.trafficlight.getPhase(self.tls_id)
                num_phases = len(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases)
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(self.tls_id, next_phase)
                self.time_in_phase = 0
                self.switch_count += 1
            except:
                pass
        
        self.last_action = action
        return action

def get_e2_detector_ids(additional_file):
    """Lee IDs de detectores E2."""
    tree = ET.parse(additional_file)
    root = tree.getroot()
    return [det.get('id') for det in root.findall('laneAreaDetector')]

def generate_routes_with_seed(net_file, output_file, duration=7200, veh_per_hour=1500, seed=FIXED_SEED):
    """Genera rutas con semilla fija."""
    random.seed(seed)
    np.random.seed(seed)
    
    net = sumolib.net.readNet(str(net_file))
    edges = net.getEdges()
    valid_edges = [e for e in edges if e.allows("passenger") and e.getLength() > 10]
    
    root = ET.Element("routes")
    
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6")
    vtype.set("decel", "4.5")
    vtype.set("sigma", "0.5")
    vtype.set("length", "4.5")
    vtype.set("maxSpeed", "15")
    
    interval = 3600 / veh_per_hour
    veh_id = 0
    
    for t in range(0, duration, max(1, int(interval))):
        from_edge = random.choice(valid_edges)
        to_edge = random.choice(valid_edges)
        
        attempts = 0
        while from_edge == to_edge and attempts < 10:
            to_edge = random.choice(valid_edges)
            attempts += 1
        
        if from_edge != to_edge:
            trip = ET.SubElement(root, "trip")
            trip.set("id", f"veh_{veh_id}")
            trip.set("type", "car")
            trip.set("depart", str(t + random.uniform(0, interval)))
            trip.set("from", from_edge.getID())
            trip.set("to", to_edge.getID())
            veh_id += 1
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"  Vehículos generados: {veh_id}")
    return veh_id

def run_drl_evaluation(duration=7200, gui=False, seed=FIXED_SEED, model_path=MODEL_PATH):
    """Ejecuta evaluación con modelo DRL."""
    print("=" * 70)
    print("EVALUACIÓN DRL - IBARRA MAYORISTA (CALLE 13 DE ABRIL)")
    print("=" * 70)
    print(f"Semilla fija: {seed}")
    print(f"Duración: {duration}s ({duration/3600:.1f} horas)")
    print(f"Modelo: {model_path}")
    print(f"Semáforo objetivo: {TARGET_TLS}")
    
    # Regenerar rutas con la semilla fija
    print("\n[1/4] Generando rutas con semilla fija...")
    generate_routes_with_seed(NET_FILE, ROUTES_FILE, duration=duration, seed=seed)
    
    # Obtener detectores E2
    print("\n[2/4] Cargando detectores E2...")
    e2_ids = get_e2_detector_ids(ADDITIONAL_FILE)
    print(f"  Detectores: {len(e2_ids)}")
    
    # Iniciar SUMO
    print("\n[3/4] Iniciando simulación...")
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", str(CONFIG_FILE),
        "--seed", str(seed),
        "--start", "true" if not gui else "false",
        "--quit-on-end", "true"
    ]
    
    traci.start(sumo_cmd)
    
    # Métricas
    metrics = {
        'time': [],
        'queue_13abril': [],
        'queue_length_m': [],
        'throughput': [],
        'avg_speed_13abril': [],
        'waiting_13abril': [],
        'vehicle_count_13abril': [],
        'drl_action': [],
    }
    
    total_arrived = 0
    
    try:
        # Verificar que el semáforo existe
        tls_ids = traci.trafficlight.getIDList()
        print(f"  Semáforos disponibles: {len(tls_ids)}")
        
        if TARGET_TLS not in tls_ids:
            print(f"  ADVERTENCIA: Semáforo {TARGET_TLS} no encontrado!")
            print(f"  Semáforos disponibles: {list(tls_ids)[:10]}...")
            # Usar el primer semáforo disponible como alternativa
            target_tls = tls_ids[0] if tls_ids else None
            print(f"  Usando alternativa: {target_tls}")
        else:
            target_tls = TARGET_TLS
            print(f"  Semáforo {TARGET_TLS} encontrado ✓")
        
        # Crear controlador DRL
        controller = DRLController(
            model_path=str(model_path),
            tls_id=target_tls,
            e2_detector_ids=e2_ids,
            control_interval=5,
            min_green=10
        )
        
        step = 0
        last_progress = 0
        
        while step < duration:
            traci.simulationStep()
            
            # Aplicar control DRL
            action = controller.step(step)
            
            # Recopilar métricas cada 5 segundos
            if step % 5 == 0:
                queue_total = 0
                queue_length_m = 0
                speed_sum = 0
                speed_count = 0
                vehicle_count = 0
                waiting = 0
                
                for det_id in e2_ids:
                    try:
                        jam = traci.lanearea.getJamLengthVehicle(det_id)
                        jam_m = traci.lanearea.getJamLengthMeters(det_id)
                        speed = traci.lanearea.getLastStepMeanSpeed(det_id)
                        veh = traci.lanearea.getLastStepVehicleNumber(det_id)
                        
                        queue_total += jam
                        queue_length_m += jam_m
                        if speed >= 0:
                            speed_sum += speed
                            speed_count += 1
                        vehicle_count += veh
                        if speed < 0.5:
                            waiting += veh
                    except:
                        pass
                
                arrived = traci.simulation.getArrivedNumber()
                total_arrived += arrived
                
                metrics['time'].append(step)
                metrics['queue_13abril'].append(queue_total)
                metrics['queue_length_m'].append(queue_length_m)
                metrics['throughput'].append(total_arrived)
                metrics['avg_speed_13abril'].append(speed_sum / speed_count if speed_count > 0 else 0)
                metrics['waiting_13abril'].append(waiting)
                metrics['vehicle_count_13abril'].append(vehicle_count)
                metrics['drl_action'].append(action if action is not None else -1)
            
            # Progreso
            progress = int(step / duration * 100)
            if progress >= last_progress + 10:
                print(f"  [{progress}%] t={step}s, Queue={metrics['queue_13abril'][-1]:.1f}, "
                      f"Throughput={total_arrived}, Switches={controller.switch_count}")
                last_progress = progress
            
            step += 1
        
        switch_count = controller.switch_count
        
    finally:
        traci.close()
    
    # Calcular estadísticas
    print("\n[4/4] Calculando estadísticas...")
    
    results = {
        'seed': seed,
        'duration_s': duration,
        'model': str(model_path.name),
        'target_tls': TARGET_TLS,
        'total_throughput': total_arrived,
        'throughput_per_hour': total_arrived / (duration / 3600),
        'avg_queue_13abril': np.mean(metrics['queue_13abril']),
        'max_queue_13abril': max(metrics['queue_13abril']),
        'std_queue_13abril': np.std(metrics['queue_13abril']),
        'avg_queue_length_m': np.mean(metrics['queue_length_m']),
        'max_queue_length_m': max(metrics['queue_length_m']),
        'avg_speed_13abril': np.mean(metrics['avg_speed_13abril']),
        'min_speed_13abril': min(metrics['avg_speed_13abril']),
        'avg_waiting_13abril': np.mean(metrics['waiting_13abril']),
        'avg_vehicles_13abril': np.mean(metrics['vehicle_count_13abril']),
        'total_switches': switch_count,
    }
    
    return results, metrics

def save_results(results, metrics, output_dir, run_name="drl_extended"):
    """Guarda resultados."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = output_dir / f"{run_name}_summary_{timestamp}.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    metrics_file = output_dir / f"{run_name}_metrics_{timestamp}.csv"
    with open(metrics_file, 'w', newline='') as f:
        fieldnames = list(metrics.keys())
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(metrics['time'])):
            writer.writerow([metrics[k][i] for k in fieldnames])
    
    print(f"\nResultados guardados:")
    print(f"  - {summary_file}")
    print(f"  - {metrics_file}")
    
    return summary_file, metrics_file

def print_summary(results):
    """Imprime resumen."""
    print("\n" + "=" * 70)
    print("RESUMEN - DRL CALLE 13 DE ABRIL")
    print("=" * 70)
    print(f"  Semilla: {results['seed']}")
    print(f"  Duración: {results['duration_s']/3600:.1f} horas")
    print(f"  Modelo: {results['model']}")
    print(f"  Semáforo: {results['target_tls']}")
    print("-" * 70)
    print(f"  THROUGHPUT:")
    print(f"    Total: {results['total_throughput']} vehículos")
    print(f"    Por hora: {results['throughput_per_hour']:.1f} veh/h")
    print(f"  COLAS (13 de Abril):")
    print(f"    Promedio: {results['avg_queue_13abril']:.2f} vehículos")
    print(f"    Máxima: {results['max_queue_13abril']:.0f} vehículos")
    print(f"    Desv. Std: {results['std_queue_13abril']:.2f}")
    print(f"  LARGO DE COLA (metros):")
    print(f"    Promedio: {results['avg_queue_length_m']:.2f} m")
    print(f"    Máximo: {results['max_queue_length_m']:.2f} m")
    print(f"  VELOCIDAD (13 de Abril):")
    print(f"    Promedio: {results['avg_speed_13abril']:.2f} m/s ({results['avg_speed_13abril']*3.6:.1f} km/h)")
    print(f"    Mínima: {results['min_speed_13abril']:.2f} m/s")
    print(f"  CONTROL DRL:")
    print(f"    Cambios de fase: {results['total_switches']}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Evaluación DRL Ibarra - 13 de Abril")
    parser.add_argument("--duration", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    
    args = parser.parse_args()
    
    results, metrics = run_drl_evaluation(
        duration=args.duration,
        gui=args.gui,
        seed=args.seed,
        model_path=Path(args.model)
    )
    
    print_summary(results)
    save_results(results, metrics, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
