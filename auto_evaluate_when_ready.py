#!/usr/bin/env python3
"""
Script que espera a que termine el entrenamiento y automáticamente evalúa el modelo.
Monitorea la carpeta models/ hasta que aparezca ppo_production_100k.zip
"""

import os
import time
import subprocess
import sys
from pathlib import Path

def wait_for_model(model_name="ppo_production_100k", check_interval=30):
    """Espera hasta que el modelo esté listo."""
    model_path = f"models/{model_name}.zip"
    metadata_path = f"models/{model_name}_metadata.yaml"
    
    print(f"🔍 Monitoreando entrenamiento del modelo: {model_name}")
    print(f"📁 Esperando archivo: {model_path}")
    print(f"⏱️  Revisando cada {check_interval} segundos...")
    print()
    
    start_time = time.time()
    
    while True:
        # Verificar si el modelo existe
        if os.path.exists(model_path):
            # Verificar también los metadatos para asegurar que terminó completamente
            if os.path.exists(metadata_path):
                elapsed = time.time() - start_time
                print(f"✅ ¡Modelo encontrado después de {elapsed/60:.1f} minutos!")
                print(f"📦 Archivo: {model_path}")
                print(f"📋 Metadatos: {metadata_path}")
                return model_path
            else:
                print(f"⏳ Modelo encontrado pero esperando metadatos...")
        else:
            elapsed = time.time() - start_time
            print(f"⏳ Esperando... ({elapsed/60:.1f} min transcurridos)")
        
        time.sleep(check_interval)

def run_evaluation(model_path, steps=50):
    """Ejecutar evaluación comparativa del modelo."""
    
    print(f"\n🚀 INICIANDO EVALUACIÓN DEL MODELO ENTRENADO")
    print("=" * 60)
    
    # Configurar SUMO
    os.environ["SUMO_HOME"] = "C:\\Program Files (x86)\\Eclipse\\Sumo"
    os.environ["PATH"] += f";{os.environ['SUMO_HOME']}\\bin"
    
    # Comando de evaluación comparativa
    output_file = f"evaluation_{Path(model_path).stem}.csv"
    cmd = [
        sys.executable, "scripts/comparative_evaluation.py",
        "--steps", str(steps),
        "--output", output_file,
        "--models", model_path  # Evaluar específicamente este modelo
    ]
    
    print(f"📊 Ejecutando evaluación comparativa...")
    print(f"🎯 Pasos de evaluación: {steps}")
    print(f"📁 Archivo de salida: {output_file}")
    print(f"🤖 Modelo: {model_path}")
    print()
    
    try:
        # Ejecutar evaluación
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ¡EVALUACIÓN COMPLETADA EXITOSAMENTE!")
            print("=" * 60)
            print(result.stdout)
            
            # Mostrar archivo de resultados
            if os.path.exists(output_file):
                print(f"\n📊 Resultados guardados en: {output_file}")
                
                # Mostrar resumen del CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(output_file)
                    print(f"\n📈 RESUMEN DE RESULTADOS:")
                    print(f"   Métodos evaluados: {len(df)}")
                    
                    # Encontrar el modelo entrenado
                    model_name = Path(model_path).stem
                    model_row = df[df['method'].str.contains(model_name, na=False)]
                    if not model_row.empty:
                        row = model_row.iloc[0]
                        print(f"   🏆 Modelo {model_name}:")
                        print(f"      Throughput: {row['avg_served']:.2f} veh/intervalo")
                        print(f"      Colas: {row['avg_queues']:.2f}")
                        print(f"      Recompensa: {row['avg_reward']:.3f}")
                        
                        # Buscar mejoras si existen
                        improvement_cols = [col for col in df.columns if 'improvement' in col and 'vs_fixed' in col]
                        if improvement_cols and not model_row.empty:
                            served_improvement = model_row[f'vs_fixed_avg_served_improvement'].iloc[0] if f'vs_fixed_avg_served_improvement' in model_row.columns else None
                            if served_improvement is not None:
                                print(f"      Mejora throughput: {served_improvement:+.1f}% vs baseline fijo")
                        
                except Exception as e:
                    print(f"   (Error leyendo CSV: {e})")
            
        else:
            print("❌ ERROR EN LA EVALUACIÓN:")
            print(result.stderr)
            print(result.stdout)
            
    except Exception as e:
        print(f"❌ Error ejecutando evaluación: {e}")

def main():
    print("🤖 MONITOR AUTOMÁTICO DE ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 60)
    print("Este script:")
    print("1. 👀 Monitorea el progreso del entrenamiento")
    print("2. ⏳ Espera hasta que termine completamente")
    print("3. 🚀 Ejecuta evaluación comparativa automáticamente")
    print("4. 📊 Muestra resultados en tablas formateadas")
    print()
    
    # Esperar a que termine el entrenamiento
    model_path = wait_for_model("ppo_production_100k")
    
    # Evaluar el modelo
    run_evaluation(model_path, steps=50)
    
    print(f"\n🎉 ¡PROCESO COMPLETO!")
    print("=" * 60)
    print("✅ Entrenamiento terminado")
    print("✅ Evaluación completada")
    print("✅ Resultados disponibles")

if __name__ == "__main__":
    main()

