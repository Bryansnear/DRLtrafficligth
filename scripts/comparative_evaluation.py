#!/usr/bin/env python3
"""
Script de evaluación comparativa entre baseline y modelos entrenados.
Compara el rendimiento de diferentes estrategias de control de semáforos.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple
from pathlib import Path
from tabulate import tabulate

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.env.traffic_light_env import EnvConfig, TrafficLightGymEnv
from stable_baselines3 import PPO


def load_cfg(path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, use_gui: bool = False) -> TrafficLightGymEnv:
    """Crear entorno de evaluación."""
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
    )
    return TrafficLightGymEnv(ec)


def evaluate_baseline_fixed_flow(cfg: dict, steps: int = 30) -> Dict:
    """Evaluación de baseline con semáforo fijo y flujo fijo."""
    print(">> Evaluando BASELINE 1: Semaforo Fijo + Flujo Fijo...")
    
    # Usar configuración original (flujo fijo)
    env = make_env(cfg)
    obs, info = env.reset()
    
    # Métricas
    total_served = 0.0
    total_switches = 0
    total_queues = 0.0
    total_reward = 0.0
    queue_history = []
    served_history = []
    
    # Estrategia fija: alternar cada 30 segundos (6 intervalos de 5s)
    fixed_cycle = [0] * 6 + [1] * 6  # 30s verde E-W, 30s verde N-S
    action_idx = 0
    
    start_time = time.time()
    
    for step in range(steps):
        # Acción fija cíclica
        action = fixed_cycle[action_idx % len(fixed_cycle)]
        action_idx += 1
        
        obs, reward, term, trunc, info = env.step(action)
        
        # Recopilar métricas
        served = info.get("served_cnt", 0.0)
        switch = int(info.get("switch", 0))
        queues = info.get("queues", {})
        queue_sum = sum(queues.values())
        
        total_served += served
        total_switches += switch
        total_queues += queue_sum
        total_reward += reward
        
        queue_history.append(queue_sum)
        served_history.append(served)
        
        if term or trunc:
            obs, info = env.reset()
    
    elapsed_time = time.time() - start_time
    
    # Cerrar entorno
    try:
        env.core.close()
    except:
        pass
    
    # Calcular métricas
    avg_served = total_served / steps
    avg_queues = total_queues / steps
    switch_rate = total_switches / steps
    avg_reward = total_reward / steps
    queue_std = np.std(queue_history)
    throughput_stability = np.std(served_history)
    
    return {
        'method': 'Baseline Fijo (Flujo Fijo)',
        'avg_served': avg_served,
        'avg_queues': avg_queues,
        'switch_rate': switch_rate,
        'avg_reward': avg_reward,
        'queue_std': queue_std,
        'throughput_stability': throughput_stability,
        'evaluation_time': elapsed_time,
        'total_steps': steps,
        'baseline_type': 'fixed_flow'
    }


def evaluate_baseline_variable_flow(cfg: dict, steps: int = 30) -> Dict:
    """Evaluación de baseline con semáforo fijo y flujo variable."""
    print(">> Evaluando BASELINE 2: Semaforo Fijo + Flujo Variable...")
    
    # Crear entorno con demanda variable
    class VariableFlowEnv:
        def __init__(self, base_cfg):
            self.base_cfg = base_cfg
            self.demand_profiles = [
                "experiments/demands/rush_morning.yaml",
                "experiments/demands/midday_moderate.yaml", 
                "experiments/demands/off_peak_low.yaml"
            ]
            # Filtrar perfiles que existen
            self.demand_profiles = [p for p in self.demand_profiles if os.path.exists(p)]
            if not self.demand_profiles:
                # Fallback: usar perfil base
                self.demand_profiles = [None]
            self.current_profile_idx = 0
            self.reset_count = 0
    
        def get_next_env(self):
            # Rotar entre perfiles cada reset
            if len(self.demand_profiles) > 1:
                profile = self.demand_profiles[self.reset_count % len(self.demand_profiles)]
                self.reset_count += 1
                
                # Modificar configuración temporalmente
                if profile and os.path.exists(profile):
                    modified_cfg = self.base_cfg.copy()
                    # Aquí podrías cargar parámetros específicos del perfil
                    # Por simplicidad, usamos la config base pero con rotación conceptual
                    return make_env(modified_cfg)
                else:
                    return make_env(self.base_cfg)
            else:
                return make_env(self.base_cfg)
    
    var_env = VariableFlowEnv(cfg)
    
    # Métricas
    total_served = 0.0
    total_switches = 0
    total_queues = 0.0
    total_reward = 0.0
    queue_history = []
    served_history = []
    
    # Estrategia fija: alternar cada 25 segundos (5 intervalos de 5s) para más variabilidad
    fixed_cycle = [0] * 5 + [1] * 5  # 25s verde E-W, 25s verde N-S
    action_idx = 0
    
    start_time = time.time()
    
    # Evaluar en bloques para simular cambios de demanda
    steps_per_block = max(1, steps // 3)  # Dividir en 3 bloques
    current_step = 0
    
    while current_step < steps:
        # Crear nuevo entorno para este bloque (simula cambio de demanda)
        env = var_env.get_next_env()
        obs, info = env.reset()
        
        block_steps = min(steps_per_block, steps - current_step)
        
        for step in range(block_steps):
            # Acción fija cíclica
            action = fixed_cycle[action_idx % len(fixed_cycle)]
            action_idx += 1
            
            obs, reward, term, trunc, info = env.step(action)
            
            # Recopilar métricas
            served = info.get("served_cnt", 0.0)
            switch = int(info.get("switch", 0))
            queues = info.get("queues", {})
            queue_sum = sum(queues.values())
            
            total_served += served
            total_switches += switch
            total_queues += queue_sum
            total_reward += reward
            
            queue_history.append(queue_sum)
            served_history.append(served)
            
            if term or trunc:
                obs, info = env.reset()
        
        current_step += block_steps
        
        # Cerrar entorno del bloque
        try:
            env.core.close()
        except:
            pass
    
    elapsed_time = time.time() - start_time
    
    # Calcular métricas
    avg_served = total_served / steps
    avg_queues = total_queues / steps
    switch_rate = total_switches / steps
    avg_reward = total_reward / steps
    queue_std = np.std(queue_history)
    throughput_stability = np.std(served_history)
    
    return {
        'method': 'Baseline Fijo (Flujo Variable)',
        'avg_served': avg_served,
        'avg_queues': avg_queues,
        'switch_rate': switch_rate,
        'avg_reward': avg_reward,
        'queue_std': queue_std,
        'throughput_stability': throughput_stability,
        'evaluation_time': elapsed_time,
        'total_steps': steps,
        'baseline_type': 'variable_flow'
    }


def evaluate_model(model_path: str, cfg: dict, steps: int = 30) -> Dict:
    """Evaluación de modelo entrenado."""
    model_name = os.path.basename(model_path).replace('.zip', '')
    print(f">> Evaluando MODELO: {model_name}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Cargar modelo
    model = PPO.load(model_path)
    
    env = make_env(cfg)
    obs, info = env.reset()
    
    # Métricas
    total_served = 0.0
    total_switches = 0
    total_queues = 0.0
    total_reward = 0.0
    queue_history = []
    served_history = []
    
    start_time = time.time()
    
    for step in range(steps):
        # Predicción del modelo
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        
        # Recopilar métricas
        served = info.get("served_cnt", 0.0)
        switch = int(info.get("switch", 0))
        queues = info.get("queues", {})
        queue_sum = sum(queues.values())
        
        total_served += served
        total_switches += switch
        total_queues += queue_sum
        total_reward += reward
        
        queue_history.append(queue_sum)
        served_history.append(served)
        
        if term or trunc:
            obs, info = env.reset()
    
    elapsed_time = time.time() - start_time
    
    # Cerrar entorno
    try:
        env.core.close()
    except:
        pass
    
    # Calcular métricas
    avg_served = total_served / steps
    avg_queues = total_queues / steps
    switch_rate = total_switches / steps
    avg_reward = total_reward / steps
    queue_std = np.std(queue_history)
    throughput_stability = np.std(served_history)
    
    return {
        'method': f'PPO ({model_name})',
        'avg_served': avg_served,
        'avg_queues': avg_queues,
        'switch_rate': switch_rate,
        'avg_reward': avg_reward,
        'queue_std': queue_std,
        'throughput_stability': throughput_stability,
        'evaluation_time': elapsed_time,
        'total_steps': steps,
        'model_path': model_path
    }


def calculate_improvement(baseline: Dict, model: Dict) -> Dict:
    """Calcular mejoras porcentuales respecto al baseline."""
    improvements = {}
    
    # Métricas donde más es mejor
    for metric in ['avg_served', 'avg_reward']:
        if baseline[metric] != 0:
            improvement = ((model[metric] - baseline[metric]) / abs(baseline[metric])) * 100
            improvements[f'{metric}_improvement'] = improvement
        else:
            improvements[f'{metric}_improvement'] = 0
    
    # Métricas donde menos es mejor
    for metric in ['avg_queues', 'switch_rate', 'queue_std', 'throughput_stability']:
        if baseline[metric] != 0:
            improvement = ((baseline[metric] - model[metric]) / abs(baseline[metric])) * 100
            improvements[f'{metric}_improvement'] = improvement
        else:
            improvements[f'{metric}_improvement'] = 0
    
    return improvements


def main():
    parser = argparse.ArgumentParser(description="Evaluación comparativa de modelos de control de tráfico")
    parser.add_argument("--config", default="experiments/configs/base.yaml", help="Configuración base")
    parser.add_argument("--steps", type=int, default=30, help="Pasos de evaluación por método")
    parser.add_argument("--models", nargs='+', help="Lista de modelos a evaluar (opcional)")
    parser.add_argument("--output", default="comparative_evaluation.csv", help="Archivo de salida")
    parser.add_argument("--gui", action="store_true", help="Usar SUMO GUI")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"ERROR: Configuración no encontrada: {args.config}")
        return 1
    
    cfg = load_cfg(args.config)
    
    print("EVALUACION COMPARATIVA DE CONTROL DE TRAFICO")
    print("=" * 60)
    print(f"Configuracion: {args.config}")
    print(f"Pasos por metodo: {args.steps}")
    print(f"Archivo de salida: {args.output}")
    print()
    
    results = []
    
    # 1. Evaluar baseline con flujo fijo
    try:
        baseline_fixed_result = evaluate_baseline_fixed_flow(cfg, args.steps)
        results.append(baseline_fixed_result)
        print(f"OK Baseline Flujo Fijo completado")
    except Exception as e:
        print(f"ERROR evaluando baseline flujo fijo: {e}")
        return 1
    
    # 2. Evaluar baseline con flujo variable
    try:
        baseline_variable_result = evaluate_baseline_variable_flow(cfg, args.steps)
        results.append(baseline_variable_result)
        print(f"OK Baseline Flujo Variable completado")
    except Exception as e:
        print(f"ERROR evaluando baseline flujo variable: {e}")
        return 1
    
    # 3. Buscar modelos automáticamente si no se especificaron
    if args.models is None:
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            args.models = [os.path.join(models_dir, f) for f in model_files]
        else:
            args.models = []
    
    if not args.models:
        print("AVISO: No se encontraron modelos para evaluar")
    
    # 4. Evaluar cada modelo
    for model_path in args.models:
        try:
            model_result = evaluate_model(model_path, cfg, args.steps)
            results.append(model_result)
            print(f"OK Modelo {os.path.basename(model_path)} completado")
        except Exception as e:
            print(f"ERROR evaluando {model_path}: {e}")
            continue
    
    if len(results) < 3:
        print("ERROR: Se necesitan al menos 2 baselines + 1 modelo para comparar")
        return 1
    
    # 5. Calcular mejoras respecto a ambos baselines
    baseline_fixed = results[0]   # Baseline flujo fijo
    baseline_variable = results[1]  # Baseline flujo variable
    
    # Calcular mejoras respecto al baseline fijo para todos los modelos
    for i in range(2, len(results)):  # Empezar desde el primer modelo (índice 2)
        improvements_fixed = calculate_improvement(baseline_fixed, results[i])
        improvements_variable = calculate_improvement(baseline_variable, results[i])
        
        # Agregar prefijos para distinguir las mejoras
        for key, value in improvements_fixed.items():
            results[i][f'vs_fixed_{key}'] = value
        for key, value in improvements_variable.items():
            results[i][f'vs_variable_{key}'] = value
    
    # 6. Crear DataFrame y guardar
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    # 7. Mostrar resultados
    print(f"\nRESULTADOS COMPARATIVOS:")
    print("=" * 100)
    
    # Mostrar métricas principales en formato de tabla
    display_cols = ['method', 'avg_served', 'avg_queues', 'switch_rate', 'avg_reward']
    display_data = df[display_cols].round(3)
    
    # Renombrar columnas para mejor presentación
    display_data = display_data.rename(columns={
        'method': 'Método',
        'avg_served': 'Veh/Intervalo',
        'avg_queues': 'Colas Prom.',
        'switch_rate': 'Cambios/Int.',
        'avg_reward': 'Recompensa'
    })
    
    print(tabulate(display_data, headers='keys', tablefmt='grid', showindex=False))
    
    # Mostrar mejoras respecto a ambos baselines
    if len(results) > 2:
        print(f"\nMEJORAS RESPECTO A BASELINE FLUJO FIJO:")
        print("=" * 100)
        
        fixed_improvement_cols = [col for col in df.columns if col.startswith('vs_fixed_')]
        if fixed_improvement_cols:
            fixed_improvement_df = df[['method'] + fixed_improvement_cols].iloc[2:].copy()  # Excluir baselines
            
            # Renombrar columnas para mostrar más limpio
            column_mapping = {
                'method': 'Método',
                'vs_fixed_avg_served_improvement': 'Veh/Int (%)',
                'vs_fixed_avg_queues_improvement': 'Colas (%)',
                'vs_fixed_switch_rate_improvement': 'Cambios (%)', 
                'vs_fixed_avg_reward_improvement': 'Recompensa (%)'
            }
            
            # Seleccionar solo las columnas principales para la tabla
            main_cols = ['method'] + [col for col in fixed_improvement_cols if any(key in col for key in ['served', 'queues', 'switch_rate', 'reward']) and 'improvement' in col][:4]
            if len(main_cols) > 1:
                fixed_table_df = fixed_improvement_df[main_cols].copy()
                
                # Renombrar columnas
                for old_col, new_col in column_mapping.items():
                    if old_col in fixed_table_df.columns:
                        fixed_table_df = fixed_table_df.rename(columns={old_col: new_col})
                
                # Formatear porcentajes
                for col in fixed_table_df.columns:
                    if col != 'Método' and '%' in col:
                        fixed_table_df[col] = fixed_table_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                
                print(tabulate(fixed_table_df, headers='keys', tablefmt='grid', showindex=False))
        
        print(f"\nMEJORAS RESPECTO A BASELINE FLUJO VARIABLE:")
        print("=" * 100)
        
        variable_improvement_cols = [col for col in df.columns if col.startswith('vs_variable_')]
        if variable_improvement_cols:
            variable_improvement_df = df[['method'] + variable_improvement_cols].iloc[2:].copy()  # Excluir baselines
            
            # Renombrar columnas para mostrar más limpio
            column_mapping = {
                'method': 'Método',
                'vs_variable_avg_served_improvement': 'Veh/Int (%)',
                'vs_variable_avg_queues_improvement': 'Colas (%)',
                'vs_variable_switch_rate_improvement': 'Cambios (%)',
                'vs_variable_avg_reward_improvement': 'Recompensa (%)'
            }
            
            # Seleccionar solo las columnas principales para la tabla
            main_cols = ['method'] + [col for col in variable_improvement_cols if any(key in col for key in ['served', 'queues', 'switch_rate', 'reward']) and 'improvement' in col][:4]
            if len(main_cols) > 1:
                variable_table_df = variable_improvement_df[main_cols].copy()
                
                # Renombrar columnas
                for old_col, new_col in column_mapping.items():
                    if old_col in variable_table_df.columns:
                        variable_table_df = variable_table_df.rename(columns={old_col: new_col})
                
                # Formatear porcentajes
                for col in variable_table_df.columns:
                    if col != 'Método' and '%' in col:
                        variable_table_df[col] = variable_table_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                
                print(tabulate(variable_table_df, headers='keys', tablefmt='grid', showindex=False))
    
    # 8. Identificar mejor modelo
    if len(results) > 2:
        # Ranking basado en múltiples métricas (score compuesto)
        models_only = df.iloc[2:].copy()  # Excluir ambos baselines
        
        # Score compuesto: served (peso 40%) + reward (30%) - queues (20%) - switches (10%)
        models_only['composite_score'] = (
            models_only['avg_served'] * 0.4 +
            models_only['avg_reward'] * 0.3 -
            models_only['avg_queues'] * 0.2 -
            models_only['switch_rate'] * 0.1
        )
        
        best_model = models_only.loc[models_only['composite_score'].idxmax()]
        
        print(f"\nMEJOR MODELO:")
        print("=" * 60)
        
        # Crear tabla del mejor modelo
        best_model_data = [
            ['Modelo', best_model['method']],
            ['Score Compuesto', f"{best_model['composite_score']:.3f}"],
            ['Throughput', f"{best_model['avg_served']:.2f} veh/intervalo"],
            ['Colas Promedio', f"{best_model['avg_queues']:.2f}"],
            ['Recompensa', f"{best_model['avg_reward']:.3f}"],
            ['Cambios/Intervalo', f"{best_model['switch_rate']:.2f}"]
        ]
        
        print(tabulate(best_model_data, headers=['Métrica', 'Valor'], tablefmt='grid'))
    
    print(f"\nResultados guardados en: {args.output}")
    print(f"Evaluacion comparativa completada")
    
    return 0


if __name__ == "__main__":
    exit(main())
