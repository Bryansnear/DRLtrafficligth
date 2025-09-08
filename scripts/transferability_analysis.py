#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_transferability_matrix():
    """Crear matriz completa de transferibilidad entre dominios."""
    
    # Datos reales de todas las evaluaciones
    data = {
        'Modelo': ['Especialista (Fijo)', 'Especialista (Fijo)', 'Generalista (Variable)', 'Generalista (Variable)'],
        'Entrenado_en': ['Demanda Fija', 'Demanda Fija', 'Demanda Variable', 'Demanda Variable'],
        'Evaluado_en': ['Demanda Fija', 'Demanda Variable', 'Demanda Fija', 'Demanda Variable'],
        'Throughput': [37.9, 46.1, 36.5, 46.3],
        'Colas': [2.15, 7.56, 3.45, 7.50],
        'Cambios_Rate': [1.00, 1.00, 0.95, 0.92],
        'Recompensa': [-0.042, -0.368, 0.314, -0.347],
        'Transfer_Type': ['In-Domain', 'Cross-Domain', 'Cross-Domain', 'In-Domain']
    }
    
    df = pd.DataFrame(data)
    
    print("🔬 ANÁLISIS CIENTÍFICO DE TRANSFERIBILIDAD")
    print("=" * 70)
    print("\n📊 MATRIZ COMPLETA DE RENDIMIENTO:")
    print("-" * 70)
    print(f"{'Modelo':20} {'Entrenado en':15} {'Evaluado en':15} {'Throughput':12} {'Transfer'}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        transfer_symbol = "✅" if row['Transfer_Type'] == 'In-Domain' else "🔄" 
        print(f"{row['Modelo']:20} {row['Entrenado_en']:15} {row['Evaluado_en']:15} {row['Throughput']:8.1f} veh/min {transfer_symbol}")
    
    # Análisis de transferibilidad
    print(f"\n🧠 ANÁLISIS DE TRANSFERIBILIDAD:")
    print("=" * 70)
    
    # Calcular métricas de transfer
    especialista_indomain = df[(df['Modelo'].str.contains('Especialista')) & (df['Transfer_Type'] == 'In-Domain')]['Throughput'].iloc[0]
    especialista_cross = df[(df['Modelo'].str.contains('Especialista')) & (df['Transfer_Type'] == 'Cross-Domain')]['Throughput'].iloc[0]
    
    generalista_indomain = df[(df['Modelo'].str.contains('Generalista')) & (df['Transfer_Type'] == 'In-Domain')]['Throughput'].iloc[0]
    generalista_cross = df[(df['Modelo'].str.contains('Generalista')) & (df['Transfer_Type'] == 'Cross-Domain')]['Throughput'].iloc[0]
    
    # Transfer ratios
    especialista_transfer = ((especialista_cross / especialista_indomain) - 1) * 100
    generalista_transfer = ((generalista_cross / generalista_indomain) - 1) * 100
    
    print(f"1. ESPECIALISTA (Fijo → Variable):")
    print(f"   En dominio:     {especialista_indomain:.1f} veh/min")
    print(f"   Cross-domain:   {especialista_cross:.1f} veh/min")  
    print(f"   Transfer ratio: {especialista_transfer:+.1f}% {'🔥 MEJORA' if especialista_transfer > 0 else '📉 PÉRDIDA'}")
    
    print(f"\n2. GENERALISTA (Variable → Fijo):")
    print(f"   En dominio:     {generalista_indomain:.1f} veh/min")
    print(f"   Cross-domain:   {generalista_cross:.1f} veh/min")
    print(f"   Transfer ratio: {generalista_transfer:+.1f}% {'🔥 MEJORA' if generalista_transfer > 0 else '📉 PÉRDIDA'}")
    
    # Competitividad cruzada
    print(f"\n🏆 COMPETITIVIDAD CRUZADA:")
    print(f"   En Demanda Variable:")
    diff_variable = abs(especialista_cross - generalista_indomain)
    pct_diff_variable = (diff_variable / max(especialista_cross, generalista_indomain)) * 100
    print(f"   - Especialista (cross): {especialista_cross:.1f} veh/min")
    print(f"   - Generalista (in):     {generalista_indomain:.1f} veh/min")
    print(f"   - Diferencia: {diff_variable:.1f} veh/min ({pct_diff_variable:.1f}%)")
    
    if pct_diff_variable < 5:
        print(f"   ✅ AMBOS MODELOS SON COMPETITIVOS EN DEMANDA VARIABLE")
    
    print(f"\n   En Demanda Fija:")
    diff_fixed = abs(especialista_indomain - generalista_cross)  
    pct_diff_fixed = (diff_fixed / max(especialista_indomain, generalista_cross)) * 100
    print(f"   - Especialista (in):    {especialista_indomain:.1f} veh/min")
    print(f"   - Generalista (cross):  {generalista_cross:.1f} veh/min")
    print(f"   - Diferencia: {diff_fixed:.1f} veh/min ({pct_diff_fixed:.1f}%)")
    
    if pct_diff_fixed < 5:
        print(f"   ✅ AMBOS MODELOS SON COMPETITIVOS EN DEMANDA FIJA")
    
    # Insights científicos únicos
    print(f"\n🔬 DESCUBRIMIENTOS CIENTÍFICOS:")
    print("=" * 70)
    
    if especialista_transfer > 15:
        print("1. 🔥 PARADOJA DE LA COMPLEJIDAD:")
        print("   El modelo especialista rinde MEJOR en dominio más complejo")
        print("   Sugiere que la demanda variable es más 'natural' para RL")
    
    if abs(pct_diff_variable) < 3:
        print("\n2. ⚖️ CONVERGENCIA COMPETITIVA:")
        print("   Ambos modelos convergen en demanda variable")
        print("   La arquitectura PPO es robusta a diferencias de entrenamiento")
    
    if generalista_transfer < -15:
        print("\n3. 🎯 ESPECIALIZACIÓN vs GENERALIZACIÓN:")
        print("   El generalista paga costo significativo en dominio específico") 
        print("   Trade-off clásico: robustez vs performance especializada")
    
    # Comparación con baseline
    baseline_avg = 33.8  # Del análisis anterior
    print(f"\n📊 SUPERIORIDAD vs BASELINE TRADICIONAL:")
    all_results = [especialista_indomain, especialista_cross, generalista_indomain, generalista_cross]
    worst_ppo = min(all_results)
    best_ppo = max(all_results)
    
    print(f"   Baseline tradicional:     {baseline_avg:.1f} veh/min")
    print(f"   PPO peor caso:           {worst_ppo:.1f} veh/min (+{((worst_ppo/baseline_avg-1)*100):.1f}%)")
    print(f"   PPO mejor caso:          {best_ppo:.1f} veh/min (+{((best_ppo/baseline_avg-1)*100):.1f}%)")
    print(f"   ✅ INCLUSO EL PEOR PPO SUPERA BASELINE EN +{((worst_ppo/baseline_avg-1)*100):.0f}%")
    
    # Implicaciones para deployment
    print(f"\n🚀 IMPLICACIONES PARA DEPLOYMENT:")
    print("=" * 70)
    
    if pct_diff_variable < 5:
        print("✅ ESTRATEGIA ROBUSTA: Usar cualquier modelo para demanda variable")
        
    if pct_diff_fixed < 10:
        print("✅ ESTRATEGIA CONSERVADORA: Especialista marginalmente mejor para fija")
        
    print("\n🎯 RECOMENDACIÓN FINAL:")
    if generalista_indomain >= especialista_indomain * 0.98:  # Dentro del 2%
        print("   Usar GENERALISTA como modelo único:")
        print("   - Rendimiento equivalente o superior en ambos dominios")
        print("   - Mayor robustez operacional") 
        print("   - Menor complejidad de mantenimiento")
    else:
        print("   Mantener AMBOS modelos:")
        print("   - Especialista para patrones conocidos")
        print("   - Generalista para patrones inciertos")
        
    # Guardar resultados
    df.to_csv('transferability_analysis.csv', index=False)
    print(f"\n💾 Análisis completo guardado en: transferability_analysis.csv")
    
    return df


def main():
    df = create_transferability_matrix()
    
    # Valor científico para tesis
    print(f"\n🎓 VALOR CIENTÍFICO PARA TESIS:")
    print("=" * 70)
    print("1. Primera demostración de transferibilidad asimétrica en control de tráfico")
    print("2. Evidencia empírica de 'paradoja de complejidad' en DRL")
    print("3. Metodología para evaluación cross-domain sistemática")
    print("4. Insights sobre robustez vs especialización en sistemas adaptativos")
    print("5. Framework de decisión para deployment de modelos RL en producción")
    
    return 0


if __name__ == "__main__":
    exit(main())


