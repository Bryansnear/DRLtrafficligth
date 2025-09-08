#!/usr/bin/env python3
"""
Ejemplo de uso del script de evaluación comparativa con doble baseline.
Muestra cómo usar los nuevos baselines (flujo fijo y variable).
"""

import os
import subprocess
import sys

def run_evaluation_example():
    """Ejecutar ejemplo de evaluación comparativa."""
    
    print("🔬 EJEMPLO: EVALUACIÓN COMPARATIVA CON DOBLE BASELINE")
    print("=" * 60)
    print("Este ejemplo ejecuta:")
    print("1. 🚦 Baseline Fijo (Flujo Fijo) - Semáforo fijo con demanda constante")
    print("2. 🚦 Baseline Fijo (Flujo Variable) - Semáforo fijo con demanda variable")
    print("3. 🤖 Todos los modelos PPO entrenados")
    print()
    
    # Verificar que SUMO esté configurado
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        print("⚠️  Configurando SUMO_HOME...")
        os.environ["SUMO_HOME"] = "C:\\Program Files (x86)\\Eclipse\\Sumo"
        os.environ["PATH"] += f";{os.environ['SUMO_HOME']}\\bin"
    
    # Comando de evaluación
    cmd = [
        sys.executable, "scripts/comparative_evaluation.py",
        "--steps", "20",  # 20 pasos para ejemplo rápido
        "--output", "example_comparison.csv"
    ]
    
    print("🚀 Ejecutando evaluación...")
    print(f"Comando: {' '.join(cmd)}")
    print()
    
    try:
        # Ejecutar comando
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Mostrar output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Evaluación completada exitosamente!")
            
            # Verificar que se creó el archivo
            if os.path.exists("example_comparison.csv"):
                print(f"📊 Resultados guardados en: example_comparison.csv")
                
                # Mostrar primeras líneas del CSV
                try:
                    with open("example_comparison.csv", "r") as f:
                        lines = f.readlines()[:5]  # Primeras 5 líneas
                    print("\n📋 Vista previa del CSV:")
                    print("".join(lines))
                except Exception as e:
                    print(f"Error leyendo CSV: {e}")
            else:
                print("⚠️  Archivo de resultados no encontrado")
        else:
            print(f"❌ Error en la evaluación (código: {result.returncode})")
            
    except Exception as e:
        print(f"❌ Error ejecutando evaluación: {e}")

def show_baseline_comparison():
    """Mostrar explicación de los baselines."""
    
    print("\n📊 EXPLICACIÓN DE LOS BASELINES:")
    print("=" * 50)
    
    print("\n🚦 BASELINE 1: Semáforo Fijo + Flujo Fijo")
    print("   - Estrategia: Ciclo fijo de 30s verde E-W, 30s verde N-S")
    print("   - Demanda: Constante según configuración base")
    print("   - Propósito: Referencia básica de semáforo tradicional")
    
    print("\n🚦 BASELINE 2: Semáforo Fijo + Flujo Variable")
    print("   - Estrategia: Ciclo fijo de 25s verde E-W, 25s verde N-S")
    print("   - Demanda: Variable (rota entre perfiles de demanda)")
    print("   - Propósito: Evaluar robustez ante cambios de tráfico")
    
    print("\n🤖 MODELOS PPO:")
    print("   - Estrategia: Aprendizaje por refuerzo adaptativo")
    print("   - Demanda: Según entrenamiento (fija o variable)")
    print("   - Propósito: Control inteligente optimizado")
    
    print("\n📈 MÉTRICAS DE COMPARACIÓN:")
    print("   - Throughput (vehículos servidos)")
    print("   - Longitud de colas")
    print("   - Frecuencia de cambios de semáforo")
    print("   - Recompensa del sistema RL")
    print("   - Estabilidad del sistema")

if __name__ == "__main__":
    show_baseline_comparison()
    
    response = input("\n¿Ejecutar evaluación de ejemplo? (y/n): ")
    if response.lower() in ['y', 'yes', 's', 'si']:
        run_evaluation_example()
    else:
        print("👋 Ejemplo cancelado. Puedes ejecutar manualmente:")
        print("python scripts/comparative_evaluation.py --steps 20 --output results.csv")
