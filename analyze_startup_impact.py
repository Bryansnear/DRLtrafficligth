#!/usr/bin/env python3
"""
Análisis del impacto del tiempo de arranque en el control de semáforos.
Compara diferentes configuraciones y su efecto en el throughput real.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

def analyze_startup_impact():
    """Analizar el impacto del tiempo de arranque en diferentes configuraciones."""
    
    print("🚗 ANÁLISIS: IMPACTO DEL TIEMPO DE ARRANQUE EN SEMÁFOROS")
    print("="*70)
    
    print("\n📊 CONFIGURACIONES COMPARADAS:")
    print("="*50)
    
    configs = {
        "Original (100k)": {
            "min_green": 5,
            "sat_headway_s": 2.0,
            "w_switch": 0.12,
            "startup_loss": "No considerado"
        },
        "Seguridad Peatonal": {
            "min_green": 20,
            "sat_headway_s": 2.5,
            "w_switch": 0.15,
            "startup_loss": "Parcialmente considerado"
        },
        "Arranque Realista": {
            "min_green": 25,
            "sat_headway_s": 3.0,
            "w_switch": 0.20,
            "startup_loss": "Completamente modelado"
        }
    }
    
    for name, config in configs.items():
        print(f"\n🔧 {name}:")
        print(f"  • Tiempo mín. verde: {config['min_green']}s")
        print(f"  • Headway saturación: {config['sat_headway_s']}s/veh")
        print(f"  • Penalización cambios: {config['w_switch']}")
        print(f"  • Tiempo de arranque: {config['startup_loss']}")
    
    print("\n🧮 CÁLCULO TEÓRICO DE THROUGHPUT:")
    print("="*50)
    
    # Calcular throughput teórico para diferentes configuraciones
    control_interval = 8  # segundos
    
    for name, config in configs.items():
        # Vehículos por intervalo sin pérdidas
        theoretical_max = control_interval / config["sat_headway_s"]
        
        # Considerar pérdida por tiempo de arranque
        if "No considerado" in config["startup_loss"]:
            startup_loss = 0  # No se considera
            effective_green = config["min_green"]
        elif "Parcialmente" in config["startup_loss"]:
            startup_loss = 1.5  # Pérdida parcial
            effective_green = max(0, config["min_green"] - startup_loss)
        else:  # Completamente modelado
            startup_loss = 2.5  # Pérdida completa modelada
            effective_green = max(0, config["min_green"] - startup_loss)
        
        # Throughput real considerando pérdidas
        if effective_green > 0:
            real_throughput = min(theoretical_max, effective_green / config["sat_headway_s"])
        else:
            real_throughput = 0
        
        efficiency = (real_throughput / theoretical_max) * 100 if theoretical_max > 0 else 0
        
        print(f"\n📈 {name}:")
        print(f"  • Throughput teórico: {theoretical_max:.2f} veh/intervalo")
        print(f"  • Pérdida por arranque: {startup_loss:.1f}s")
        print(f"  • Verde efectivo: {effective_green:.1f}s")
        print(f"  • Throughput real: {real_throughput:.2f} veh/intervalo")
        print(f"  • Eficiencia: {efficiency:.1f}%")

def show_startup_science():
    """Mostrar la ciencia detrás del tiempo de arranque."""
    
    print("\n🔬 CIENCIA DEL TIEMPO DE ARRANQUE")
    print("="*50)
    
    print("\n⏱️  COMPONENTES DEL TIEMPO PERDIDO:")
    print("  1. Tiempo de reacción del conductor: 1.0-2.0s")
    print("     • Percepción visual del cambio: 0.5-1.0s")
    print("     • Procesamiento cognitivo: 0.5-1.0s")
    
    print("\n  2. Tiempo de aceleración: 0.5-1.5s")
    print("     • Liberación del freno: 0.2-0.5s")
    print("     • Aceleración inicial: 0.3-1.0s")
    
    print("\n  3. Efecto cascada (vehículos siguientes): +0.3-0.5s/veh")
    print("     • Cada vehículo adicional en cola")
    print("     • Aumenta el tiempo total perdido")
    
    print("\n📊 VALORES TÍPICOS EN INGENIERÍA:")
    print("  • Tiempo perdido total: 2-4 segundos")
    print("  • Headway de saturación: 2.0-2.5s (sin pérdidas)")
    print("  • Headway real efectivo: 2.5-3.5s (con pérdidas)")
    
    print("\n🎯 IMPLICACIONES PARA RL:")
    print("  • Cambios frecuentes → Mayor pérdida acumulada")
    print("  • Tiempos verdes cortos → Baja eficiencia")
    print("  • Necesidad de penalizar cambios innecesarios")
    print("  • Modelar headway realista en función de recompensa")

def compare_when_ready():
    """Comparar modelos cuando estén disponibles."""
    
    print("\n🔄 COMPARACIÓN DE MODELOS")
    print("="*40)
    
    models_to_compare = [
        "models/ppo_production_100k.zip",
        "models/ppo_realistic_startup.zip"
    ]
    
    available_models = [m for m in models_to_compare if os.path.exists(m)]
    
    if len(available_models) >= 2:
        print("✅ Modelos disponibles para comparación:")
        for model in available_models:
            print(f"  • {os.path.basename(model)}")
        
        # Ejecutar comparación
        cmd = f"""python scripts/comparative_evaluation.py \\
            --models {' '.join(available_models)} \\
            --steps 40 \\
            --output startup_impact_comparison.csv"""
        
        print(f"\n🚀 Ejecutando comparación...")
        print(f"Comando: {cmd}")
        os.system(cmd)
        
        # Analizar resultados si están disponibles
        if os.path.exists("startup_impact_comparison.csv"):
            analyze_comparison_results()
    else:
        print("⏳ Esperando que termine el entrenamiento...")
        print(f"Modelos encontrados: {len(available_models)}/2")
        for model in models_to_compare:
            status = "✅" if os.path.exists(model) else "⏳"
            print(f"  {status} {os.path.basename(model)}")

def analyze_comparison_results():
    """Analizar resultados de la comparación."""
    
    print("\n📊 ANÁLISIS DE RESULTADOS")
    print("="*40)
    
    try:
        df = pd.read_csv("startup_impact_comparison.csv")
        
        # Filtrar solo modelos PPO
        ppo_models = df[df['method'].str.contains('PPO')]
        
        print("\n🤖 RENDIMIENTO DE MODELOS:")
        for _, row in ppo_models.iterrows():
            model_name = row['method']
            print(f"\n{model_name}:")
            print(f"  • Vehículos/intervalo: {row['avg_served']:.2f}")
            print(f"  • Cambios/intervalo: {row['switch_rate']:.2f}")
            print(f"  • Colas promedio: {row['avg_queues']:.2f}")
            print(f"  • Recompensa: {row['avg_reward']:.3f}")
        
        # Calcular mejoras
        if len(ppo_models) >= 2:
            original = ppo_models.iloc[0]
            realistic = ppo_models.iloc[1]
            
            throughput_change = ((realistic['avg_served'] - original['avg_served']) / original['avg_served']) * 100
            switch_change = ((realistic['switch_rate'] - original['switch_rate']) / original['switch_rate']) * 100
            
            print(f"\n📈 IMPACTO DEL MODELADO DE ARRANQUE:")
            print(f"  • Cambio en throughput: {throughput_change:+.1f}%")
            print(f"  • Cambio en frecuencia de cambios: {switch_change:+.1f}%")
            print(f"  • Estabilidad mejorada: {'✅' if switch_change < 0 else '⚠️'}")
    
    except Exception as e:
        print(f"❌ Error analizando resultados: {e}")

if __name__ == "__main__":
    analyze_startup_impact()
    show_startup_science()
    compare_when_ready()

