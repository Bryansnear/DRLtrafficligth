"""
Evaluación Baseline para el escenario Ibarra Mayorista.
Corre simulación con control de tiempo fijo y recopila métricas.
"""
import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

# Agregar project root al path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import traci
import sumolib

# Configuración por defecto
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "sumo" / "cfg" / "ibarra_mayorista_baseline.sumocfg"
DEFAULT_NET = PROJECT_ROOT / "data" / "sumo" / "network" / "ibarra_mayorista_10cuadras.net.xml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ibarra_baseline"

class BaselineController:
    """Controlador de tiempo fijo para semáforos."""
    
    def __init__(self, tls_ids, cycle_time=60, phase_split=0.5):
        """
        Args:
            tls_ids: Lista de IDs de semáforos a controlar
            cycle_time: Duración total del ciclo en segundos
            phase_split: Proporción del tiempo para la fase principal (0-1)
        """
        self.tls_ids = tls_ids
        self.cycle_time = cycle_time
        self.phase_split = phase_split
        self.phase_times = {}
        
        # Inicializar tiempos de fase para cada semáforo
        for tls_id in tls_ids:
            self.phase_times[tls_id] = {
                'current_phase': 0,
                'time_in_phase': 0
            }
    
    def step(self, current_time):
        """Ejecuta un paso del controlador."""
        for tls_id in self.tls_ids:
            try:
                state = self.phase_times[tls_id]
                state['time_in_phase'] += 1
                
                # Obtener número de fases
                num_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
                
                # Calcular duración de cada fase
                if state['current_phase'] % 2 == 0:  # Fases principales (verde)
                    phase_duration = int(self.cycle_time * self.phase_split / (num_phases // 2))
                else:  # Fases de transición (amarillo/rojo)
                    phase_duration = 4  # 4 segundos para amarillo
                
                # Cambiar de fase si es necesario
                if state['time_in_phase'] >= phase_duration:
                    state['current_phase'] = (state['current_phase'] + 1) % num_phases
                    state['time_in_phase'] = 0
                    traci.trafficlight.setPhase(tls_id, state['current_phase'])
                    
            except Exception as e:
                pass  # Ignorar errores de semáforos no válidos

class MetricsCollector:
    """Recopila métricas de la simulación."""
    
    def __init__(self, e2_detector_ids):
        self.e2_ids = e2_detector_ids
        self.metrics = {
            'time': [],
            'total_vehicles': [],
            'total_waiting': [],
            'avg_speed': [],
            'throughput': [],
            'queue_lengths': [],
            'detector_data': []
        }
        self.total_departed = 0
        self.total_arrived = 0
    
    def collect(self, sim_time):
        """Recopila métricas del paso actual."""
        # Vehículos en simulación
        num_vehicles = traci.vehicle.getIDCount()
        
        # Vehículos esperando
        waiting = sum(1 for v in traci.vehicle.getIDList() 
                     if traci.vehicle.getSpeed(v) < 0.1)
        
        # Velocidad promedio
        speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Throughput (vehículos que llegaron)
        arrived = traci.simulation.getArrivedNumber()
        self.total_arrived += arrived
        
        # Departed
        departed = traci.simulation.getDepartedNumber()
        self.total_departed += departed
        
        # Datos de detectores E2
        detector_data = {}
        total_queue = 0
        for det_id in self.e2_ids:
            try:
                jam_length = traci.lanearea.getJamLengthVehicle(det_id)
                vehicle_count = traci.lanearea.getLastStepVehicleNumber(det_id)
                mean_speed = traci.lanearea.getLastStepMeanSpeed(det_id)
                detector_data[det_id] = {
                    'jam_length': jam_length,
                    'vehicle_count': vehicle_count,
                    'mean_speed': mean_speed
                }
                total_queue += jam_length
            except:
                pass
        
        # Guardar métricas
        self.metrics['time'].append(sim_time)
        self.metrics['total_vehicles'].append(num_vehicles)
        self.metrics['total_waiting'].append(waiting)
        self.metrics['avg_speed'].append(avg_speed)
        self.metrics['throughput'].append(self.total_arrived)
        self.metrics['queue_lengths'].append(total_queue)
        self.metrics['detector_data'].append(detector_data)
        
        return {
            'vehicles': num_vehicles,
            'waiting': waiting,
            'avg_speed': avg_speed,
            'throughput': self.total_arrived,
            'queue': total_queue
        }
    
    def get_summary(self):
        """Retorna resumen de métricas."""
        import numpy as np
        return {
            'total_throughput': self.total_arrived,
            'total_departed': self.total_departed,
            'avg_queue': np.mean(self.metrics['queue_lengths']) if self.metrics['queue_lengths'] else 0,
            'max_queue': max(self.metrics['queue_lengths']) if self.metrics['queue_lengths'] else 0,
            'avg_speed': np.mean(self.metrics['avg_speed']) if self.metrics['avg_speed'] else 0,
            'avg_waiting': np.mean(self.metrics['total_waiting']) if self.metrics['total_waiting'] else 0,
        }

def get_e2_detector_ids(additional_file):
    """Lee los IDs de detectores E2 del archivo additional."""
    tree = ET.parse(additional_file)
    root = tree.getroot()
    return [det.get('id') for det in root.findall('laneAreaDetector')]

def run_baseline_simulation(config_file, duration=3600, cycle_time=60, gui=False, verbose=True):
    """
    Ejecuta simulación baseline con control de tiempo fijo.
    """
    # Obtener paths
    config_path = Path(config_file)
    config_dir = config_path.parent
    
    # Parsear config para obtener archivos
    tree = ET.parse(config_file)
    root = tree.getroot()
    
    additional_file = None
    for elem in root.findall('.//additional-files'):
        rel_path = elem.get('value')
        additional_file = (config_dir / rel_path).resolve()
        break
    
    # Obtener detectores E2
    e2_ids = get_e2_detector_ids(additional_file) if additional_file else []
    print(f"Detectores E2 encontrados: {len(e2_ids)}")
    
    # Iniciar SUMO
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", str(config_file), "--start", "--quit-on-end", "true"]
    
    print(f"\nIniciando simulación baseline...")
    print(f"  Duración: {duration}s")
    print(f"  Ciclo de tiempo fijo: {cycle_time}s")
    
    traci.start(sumo_cmd)
    
    try:
        # Obtener IDs de semáforos
        tls_ids = traci.trafficlight.getIDList()
        print(f"  Semáforos encontrados: {len(tls_ids)}")
        
        # Crear controlador baseline
        controller = BaselineController(tls_ids, cycle_time=cycle_time)
        
        # Crear recolector de métricas
        collector = MetricsCollector(e2_ids)
        
        # Ejecutar simulación
        step = 0
        while step < duration:
            traci.simulationStep()
            
            # Aplicar control baseline
            controller.step(step)
            
            # Recopilar métricas cada 5 segundos
            if step % 5 == 0:
                metrics = collector.collect(step)
                
                if verbose and step % 300 == 0:  # Cada 5 minutos
                    print(f"  [{step//60}min] Veh: {metrics['vehicles']}, "
                          f"Queue: {metrics['queue']:.1f}, "
                          f"Throughput: {metrics['throughput']}")
            
            step += 1
        
        # Obtener resumen
        summary = collector.get_summary()
        
    finally:
        traci.close()
    
    return summary, collector.metrics

def save_results(summary, metrics, output_dir, run_name="baseline"):
    """Guarda resultados en archivos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar resumen
    summary_file = output_dir / f"{run_name}_summary_{timestamp}.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    
    # Guardar métricas temporales
    metrics_file = output_dir / f"{run_name}_metrics_{timestamp}.csv"
    with open(metrics_file, 'w', newline='') as f:
        fieldnames = ['time', 'total_vehicles', 'total_waiting', 'avg_speed', 'throughput', 'queue_lengths']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(metrics['time'])):
            writer.writerow({
                'time': metrics['time'][i],
                'total_vehicles': metrics['total_vehicles'][i],
                'total_waiting': metrics['total_waiting'][i],
                'avg_speed': metrics['avg_speed'][i],
                'throughput': metrics['throughput'][i],
                'queue_lengths': metrics['queue_lengths'][i]
            })
    
    print(f"\nResultados guardados en:")
    print(f"  - Resumen: {summary_file}")
    print(f"  - Métricas: {metrics_file}")
    
    return summary_file, metrics_file

def main():
    parser = argparse.ArgumentParser(description="Evaluación Baseline Ibarra Mayorista")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                       help="Archivo de configuración SUMO")
    parser.add_argument("--duration", type=int, default=3600,
                       help="Duración de la simulación en segundos")
    parser.add_argument("--cycle-time", type=int, default=60,
                       help="Tiempo de ciclo fijo en segundos")
    parser.add_argument("--gui", action="store_true",
                       help="Usar SUMO-GUI")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                       help="Directorio de salida")
    parser.add_argument("--quiet", action="store_true",
                       help="Modo silencioso")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVALUACIÓN BASELINE - IBARRA MAYORISTA")
    print("=" * 60)
    
    # Ejecutar simulación
    summary, metrics = run_baseline_simulation(
        args.config,
        duration=args.duration,
        cycle_time=args.cycle_time,
        gui=args.gui,
        verbose=not args.quiet
    )
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"  Throughput total: {summary['total_throughput']} vehículos")
    print(f"  Cola promedio: {summary['avg_queue']:.2f} vehículos")
    print(f"  Cola máxima: {summary['max_queue']} vehículos")
    print(f"  Velocidad promedio: {summary['avg_speed']:.2f} m/s")
    print(f"  Vehículos esperando (promedio): {summary['avg_waiting']:.1f}")
    
    # Guardar resultados
    save_results(summary, metrics, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
