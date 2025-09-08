#!/usr/bin/env python3
"""
Demostración del nuevo formato de tablas en la evaluación comparativa.
Ejecuta una evaluación rápida para mostrar las tablas formateadas.
"""

import os
import subprocess
import sys

def main():
    print("🎨 DEMOSTRACIÓN: NUEVO FORMATO DE TABLAS")
    print("=" * 60)
    print("Este demo ejecuta una evaluación comparativa con:")
    print("✅ Formato de tablas mejorado con 'tabulate'")
    print("✅ Doble baseline (flujo fijo y variable)")
    print("✅ Comparaciones claras y organizadas")
    print("✅ Mejor modelo destacado en tabla")
    print()
    
    # Configurar SUMO si no está configurado
    if not os.environ.get("SUMO_HOME"):
        print("⚙️  Configurando SUMO...")
        os.environ["SUMO_HOME"] = "C:\\Program Files (x86)\\Eclipse\\Sumo"
        os.environ["PATH"] += f";{os.environ['SUMO_HOME']}\\bin"
    
    # Comando de demostración
    cmd = [
        sys.executable, "scripts/comparative_evaluation.py",
        "--steps", "8",  # 8 pasos para demo rápido pero informativo
        "--output", "demo_tables_results.csv"
    ]
    
    print("🚀 Ejecutando demostración...")
    print(f"Comando: {' '.join(cmd)}")
    print("=" * 60)
    print()
    
    try:
        # Ejecutar con output en tiempo real
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            universal_newlines=True
        )
        
        # Mostrar output en tiempo real
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ DEMOSTRACIÓN COMPLETADA!")
            print("📊 Observa cómo las tablas están perfectamente formateadas")
            print("📁 Resultados detallados en: demo_tables_results.csv")
            
            # Mostrar beneficios del nuevo formato
            print("\n🎯 BENEFICIOS DEL NUEVO FORMATO:")
            print("• 📋 Tablas con bordes y alineación perfecta")
            print("• 🔢 Columnas numéricas bien alineadas")
            print("• 📈 Porcentajes de mejora claramente visibles")
            print("• 🏆 Mejor modelo destacado en tabla estructurada")
            print("• 👀 Fácil lectura y comprensión visual")
            
        else:
            print(f"❌ Error en la demostración (código: {process.returncode})")
            
    except Exception as e:
        print(f"❌ Error ejecutando demostración: {e}")

if __name__ == "__main__":
    main()

