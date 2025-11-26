# Guía de Archivos: Entregables de Tesis

Este documento sirve como índice y guía explicativa para navegar por los resultados y documentos generados durante el desarrollo de la tesis sobre **Optimización de Semáforos con Deep Reinforcement Learning (PPO)**.

## 1. Informes y Documentación

### 📄 `THESIS_FULL_REPORT.md` (Informe Consolidado)
*   **¿Qué es?**: Es el documento central y más importante. Contiene la versión final y estructurada de la investigación.
*   **Contenido Clave**:
    *   **Resumen Ejecutivo**: Logros principales (reducción de colas, mejora de flujo).
    *   **Arquitectura**: Diagramas y explicación del sistema (SUMO + PPO).
    *   **Configuración Experimental**: Detalles del perfil de tráfico dinámico usado para las pruebas.
    *   **Resultados**: Gráficos comparativos, análisis de aprendizaje y métricas de desempeño.
    *   **Ecuación de Recompensa**: Explicación matemática detallada de la función de optimización.
*   **Uso**: Este archivo debe ser la base para redactar los capítulos de la tesis final.

### 📄 `REPORTE_DESARROLLO.md` (Bitácora de Desarrollo)
*   **¿Qué es?**: Un historial técnico cronológico de cómo se construyó el sistema.
*   **Contenido Clave**:
    *   **Problemas y Soluciones**: Documentación de obstáculos críticos (ej. el problema de "HOLD infinito" o el "flickering") y cómo se resolvieron mediante código o diseño de recompensas.
    *   **Evolución**: Cómo cambiaron las métricas a medida que se ajustaban los hiperparámetros.
*   **Uso**: Útil para justificar decisiones de diseño en la defensa de tesis o secciones de metodología.

### 📄 `TRAINING_PLAN.md` (Plan de Entrenamiento)
*   **¿Qué es?**: El plan estratégico original para el entrenamiento de los modelos.
*   **Contenido Clave**:
    *   **Fases de Optimización**: Definición de las etapas de búsqueda de hiperparámetros con Optuna.
    *   **Espacio de Búsqueda**: Rangos de valores probados para Learning Rate, Gamma, etc.
*   **Uso**: Referencia para entender la rigurosidad del proceso de experimentación.

---

## 2. Visualizaciones y Gráficos (`/graficos`)

Esta carpeta contiene las imágenes en alta resolución utilizadas en los reportes.

### Comparativas y Rendimiento
*   **`comparativa_tesis.png`**: Gráfico de barras principal. Compara el throughput y las colas del sistema propuesto (PPO) contra el sistema tradicional (Fijo).
*   **`queue_distribution.png`**: Histograma que demuestra cómo PPO logra mantener los carriles vacíos con mayor frecuencia que el baseline.
*   **`eval_boxplot_queues.png`**: Diagrama de caja que muestra la estabilidad y baja varianza del modelo propuesto.

### Entrenamiento y Aprendizaje
*   **`training_reward.png`**: Curva fundamental que muestra cómo el agente aprende a obtener más recompensa con el tiempo.
*   **`training_entropy.png`**: Muestra cómo el agente reduce su aleatoriedad (exploración) a medida que gana confianza en su política.
*   **`training_value_loss.png` / `policy_loss.png` / `kl_div.png`**: Gráficos técnicos para validar que la red neuronal convergió correctamente y no hubo inestabilidad numérica.

### Análisis del Sistema
*   **`traffic_profile.png`**: Visualización de las 5 fases de tráfico (Calentamiento, Picos, Ráfagas) usadas para estresar al modelo.
*   **`action_frequency.png`**: Compara qué tan frecuentemente cambia de luz el agente vs el sistema fijo.
*   **`reward_composition.png`**: Desglose visual de cómo la función de recompensa penaliza colas y desequilibrios en tiempo real.

---

## 3. Código Fuente (Referencia)

Aunque el código está en la carpeta raíz, estos son los scripts principales que generaron estos resultados:

*   `scripts/train.py`: Script de entrenamiento del agente.
*   `scripts/evaluate.py`: Script de evaluación y generación de métricas.
*   `scripts/generate_thesis_plots.py`: Script de Python que genera todos los gráficos de esta carpeta.
