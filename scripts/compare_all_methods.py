#!/usr/bin/env python3

import os
import pandas as pd
import subprocess
import sys

def run_baseline_comparison():
    """Ejecutar baseline fijo con diferentes configuraciones de demanda."""
    print("🔧 COMPARACIÓN BASELINE CON DEMANDA VARIABLE")
    print("=" * 60)
    
    # Simular diferentes intensidades de tráfico con baseline
    scenarios = [
        (120, "Valle Nocturno (baseline)", 8, 8),      # Verde corto, demanda baja
        (120, "Mediodia (baseline)", 15, 15),          # Verde medio, demanda media
        (120, "Pico Matutino (baseline)", 25, 15),     # Verde largo Este, pico direccional
        (120, "Pico Vespertino (baseline)", 15, 25),   # Verde largo Sur, pico direccional
    ]
    
    baseline_results = []
    
    for duration, scenario_name, greenA, greenB in scenarios:
        print(f"\n▶️ Ejecutando {scenario_name}...")
        
        cmd = [
            "python", "scripts/baseline_fixed.py",
            "--duration", str(duration),
            "--greenA", str(greenA), 
            "--greenB", str(greenB)
        ]
        
        try:
            # Ejecutar y capturar salida
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # Parsear salida simple
                lines = result.stdout.split('\n')
                served = 0
                switches = 0
                avg_queues = 0.0
                
                for line in lines:
                    if "Vehículos servidos totales:" in line:
                        served = int(line.split(':')[1].strip())
                    elif "Cambios de fase:" in line:
                        switches = int(line.split(':')[1].strip())
                    elif "Colas promedio por carril:" in line:
                        # Extraer promedio simple
                        queue_str = line.split('{')[1].split('}')[0]
                        queues = [float(v.strip()) for v in queue_str.split(',') if ':' in v]
                        avg_queues = sum([float(q.split(':')[1].strip()) for q in queue_str.split(',') if ':' in q]) / 4
                
                throughput = (served / (duration / 60.0))  # veh/min
                
                baseline_results.append({
                    "scenario": scenario_name,
                    "method": "Baseline Fijo",
                    "throughput_veh_per_min": throughput,
                    "total_switches": switches,
                    "avg_queue": avg_queues,
                    "duration_s": duration
                })
                
                print(f"   ✅ Throughput: {throughput:.1f} veh/min, Colas: {avg_queues:.2f}, Cambios: {switches}")
            else:
                print(f"   ❌ Error ejecutando {scenario_name}")
                
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
    
    return baseline_results


def load_ppo_results():
    """Cargar resultados previos de PPO."""
    ppo_results = []
    
    # Resultados del modelo original (valores promedio de evaluación anterior)
    original_scenarios = [
        ("Pico Matutino", 46.5, 10, 5.7),
        ("Pico Vespertino", 42.8, 10, 7.4),
        ("Mediodia", 45.0, 10, 5.3),
        ("Valle Nocturno", 45.0, 10, 3.7),
        ("AM/PM Dinamico", 47.2, 10, 6.7),
        ("Estatico Medio", 45.8, 10, 8.3)
    ]
    
    for scenario, throughput, switches, queues in original_scenarios:
        ppo_results.append({
            "scenario": scenario,
            "method": "PPO Original (Demanda Fija)",
            "throughput_veh_per_min": throughput,
            "total_switches": switches,
            "avg_queue": queues,
            "duration_s": 160
        })
    
    # Resultados del modelo con demanda variable (de la evaluación reciente)
    variable_scenarios = [
        ("Pico Matutino", 37.5, 3, 10.58),
        ("Pico Vespertino", 35.6, 3, 10.83),
        ("Mediodia", 42.5, 6, 7.17),
        ("Valle Nocturno", 34.4, 3, 7.83),
        ("AM/PM Dinamico", 40.6, 3, 10.92),
        ("Estatico Medio", 44.4, 3, 11.17)
    ]
    
    for scenario, throughput, switches, queues in variable_scenarios:
        ppo_results.append({
            "scenario": scenario,
            "method": "PPO Variable",
            "throughput_veh_per_min": throughput,
            "total_switches": switches,
            "avg_queue": queues,
            "duration_s": 96
        })
    
    return ppo_results


def main():
    print("🚦 COMPARACIÓN COMPLETA DE MÉTODOS DE CONTROL")
    print("=" * 70)
    
    # Ejecutar comparaciones baseline
    print("📊 Ejecutando baselines con tiempo fijo...")
    baseline_results = run_baseline_comparison()
    
    # Cargar resultados PPO
    ppo_results = load_ppo_results()
    
    # Combinar todos los resultados
    all_results = baseline_results + ppo_results
    
    if all_results:
        # Crear DataFrame para análisis
        df = pd.DataFrame(all_results)
        
        # Guardar resultados completos
        df.to_csv("complete_comparison.csv", index=False)
        print(f"\n💾 Resultados completos guardados en: complete_comparison.csv")
        
        # Análisis por método
        print(f"\n📈 RESUMEN POR MÉTODO:")
        print("=" * 70)
        
        method_summary = df.groupby('method').agg({
            'throughput_veh_per_min': ['mean', 'std', 'min', 'max'],
            'avg_queue': 'mean',
            'total_switches': 'mean'
        }).round(2)
        
        print(method_summary)
        
        # Top performers por throughput
        print(f"\n🏆 TOP PERFORMERS (Throughput):")
        print("-" * 50)
        top_throughput = df.nlargest(3, 'throughput_veh_per_min')
        for _, row in top_throughput.iterrows():
            print(f"{row['method']:25} | {row['scenario']:15} | {row['throughput_veh_per_min']:6.1f} veh/min")
        
        # Peores performers (más colas)
        print(f"\n🔻 PEORES COLAS:")
        print("-" * 50)
        worst_queues = df.nlargest(3, 'avg_queue')
        for _, row in worst_queues.iterrows():
            print(f"{row['method']:25} | {row['scenario']:15} | {row['avg_queue']:6.2f} veh cola")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        print("-" * 50)
        
        ppo_original = df[df['method'] == 'PPO Original (Demanda Fija)']['throughput_veh_per_min'].mean()
        ppo_variable = df[df['method'] == 'PPO Variable']['throughput_veh_per_min'].mean()
        
        if len(baseline_results) > 0:
            baseline_avg = pd.DataFrame(baseline_results)['throughput_veh_per_min'].mean()
            print(f"1. PPO Original supera baseline en {((ppo_original/baseline_avg - 1) * 100):+.1f}%")
        
        print(f"2. PPO Variable es {((ppo_variable/ppo_original - 1) * 100):+.1f}% vs PPO Original")
        print(f"3. La demanda variable requiere re-tuning de hiperparámetros")
        print(f"4. PPO Original sigue siendo el mejor para demanda conocida")
        
        if ppo_variable < ppo_original:
            print(f"5. ⚠️  RECOMENDACIÓN: Usar PPO Original hasta optimizar variable")
        
    else:
        print("❌ No se pudieron obtener resultados para comparación")
    
    print(f"\n✅ Comparación completa finalizada!")
    return 0


if __name__ == "__main__":
    # Establecer PYTHONPATH
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.getcwd()
    
    exit(main())


