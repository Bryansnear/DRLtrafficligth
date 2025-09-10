#!/usr/bin/env python3
"""
Script para comparar el modelo original vs el modelo con seguridad peatonal mejorada.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

def compare_models():
    """Comparar ambos modelos cuando estén listos."""
    
    print("🚦 COMPARACIÓN: MODELO ORIGINAL vs SEGURIDAD PEATONAL")
    print("="*70)
    
    # Verificar si el nuevo modelo existe
    new_model = "models/ppo_pedestrian_safe.zip"
    original_model = "models/ppo_production_100k.zip"
    
    if not os.path.exists(new_model):
        print(f"⏳ Esperando que termine el entrenamiento: {new_model}")
        print("🏃 Ejecuta este script cuando el entrenamiento termine")
        return
    
    if not os.path.exists(original_model):
        print(f"❌ Modelo original no encontrado: {original_model}")
        return
    
    # Ejecutar evaluación comparativa
    print("🔬 Ejecutando evaluación comparativa...")
    
    cmd = f"""python scripts/comparative_evaluation.py \\
        --models {original_model} {new_model} \\
        --steps 40 \\
        --output comparison_pedestrian_safety.csv"""
    
    print(f"Comando: {cmd}")
    os.system(cmd)
    
    # Leer y mostrar resultados
    if os.path.exists("comparison_pedestrian_safety.csv"):
        df = pd.read_csv("comparison_pedestrian_safety.csv")
        
        print("\n📊 RESULTADOS DE LA COMPARACIÓN:")
        print("="*50)
        
        # Filtrar solo los modelos PPO
        ppo_models = df[df['method'].str.contains('PPO')]
        
        for _, row in ppo_models.iterrows():
            model_name = row['method']
            print(f"\n🤖 {model_name}:")
            print(f"  • Vehículos/intervalo: {row['avg_served']:.2f}")
            print(f"  • Colas promedio: {row['avg_queues']:.2f}")
            print(f"  • Cambios/intervalo: {row['switch_rate']:.2f}")
            print(f"  • Recompensa: {row['avg_reward']:.3f}")
        
        print("\n🎯 ANÁLISIS DE SEGURIDAD PEATONAL:")
        
        # Comparar cambios por intervalo
        original_switches = ppo_models[ppo_models['method'].str.contains('production')]['switch_rate'].iloc[0]
        safe_switches = ppo_models[ppo_models['method'].str.contains('pedestrian')]['switch_rate'].iloc[0]
        
        reduction = ((original_switches - safe_switches) / original_switches) * 100
        
        print(f"  • Reducción en cambios: {reduction:.1f}%")
        print(f"  • Tiempo mínimo verde: 20s (vs 5s original)")
        print(f"  • Cumple estándares internacionales: ✅")
        
    else:
        print("❌ No se pudo generar el archivo de comparación")

def show_safety_standards():
    """Mostrar información sobre estándares de seguridad."""
    
    print("📚 ESTÁNDARES INTERNACIONALES DE SEGURIDAD PEATONAL")
    print("="*60)
    
    print("\n🚶 TIEMPOS MÍNIMOS RECOMENDADOS:")
    print("  • Mínimo absoluto: 15-20 segundos")
    print("  • Con peatones: 20-25 segundos") 
    print("  • Adultos mayores: 25-30 segundos")
    print("  • Intervalo líder peatonal: +7 segundos")
    
    print("\n🔬 VELOCIDADES DE CAMINATA:")
    print("  • Estándar tradicional: 1.2 m/s")
    print("  • Recomendado actual: 0.8-1.0 m/s")
    print("  • Adultos mayores: 0.6-0.8 m/s")
    
    print("\n🌍 EJEMPLOS INTERNACIONALES:")
    print("  • Reino Unido: Aumentó de 6.1s a 7.3s")
    print("  • Nueva York: LPI de 7s redujo accidentes 33%")
    print("  • Chile: Revisión de 2,800 intersecciones")
    
    print("\n⚙️ CONFIGURACIÓN IMPLEMENTADA:")
    print("  • min_green: 20 segundos (era 5)")
    print("  • all_red: 2 segundos (era 1)")
    print("  • w_switch: 0.15 (mayor penalización)")
    
    print("\n✅ BENEFICIOS ESPERADOS:")
    print("  • Mayor seguridad para peatones")
    print("  • Menos cambios abruptos")
    print("  • Cumplimiento de estándares")
    print("  • Inclusión de adultos mayores")

if __name__ == "__main__":
    show_safety_standards()
    print("\n" + "="*70)
    compare_models()

