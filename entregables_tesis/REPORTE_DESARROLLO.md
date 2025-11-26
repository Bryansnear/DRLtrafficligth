# Reporte de Desarrollo: Optimización de Agente RL para Control de Semáforos

**Fecha:** 20 de Noviembre, 2025  
**Proyecto:** Control de Tráfico Adaptativo mediante PPO y SUMO  
**Autor:** (Tu Nombre / Asistente AI)

---

## 1. Introducción y Definición del Problema

El objetivo del proyecto fue desarrollar un agente de Aprendizaje por Refuerzo (RL) utilizando el algoritmo PPO (*Proximal Policy Optimization*) para controlar un cruce de semáforos en un entorno simulado (SUMO). El objetivo principal era superar el rendimiento de un controlador de tiempos fijos tradicional, maximizando el flujo vehicular (*throughput*) y minimizando las colas de espera.

### 1.1. El Problema Inicial: "HOLD" y Comportamiento Errático
En las primeras iteraciones, nos encontramos con dos problemas fundamentales en el comportamiento del agente:

1.  **El problema del "HOLD" infinito:** El agente aprendía que cambiar de fase implicaba un "costo" (tiempo perdido en amarillo/rojo). Para evitar penalizaciones a corto plazo, decidía mantener el semáforo en verde indefinidamente en una sola dirección, ignorando las colas que crecían en la dirección contraria.
2.  **El "Parpadeo" (Flickering):** En el extremo opuesto, cuando penalizábamos demasiado las colas, el agente intentaba cambiar de fase frenéticamente (cada 5 segundos), lo que generaba caos y tiempos muertos excesivos por las transiciones de seguridad (Amarillo -> Rojo).

---

## 2. Metodología de Solución

Para resolver estos problemas, adoptamos un enfoque iterativo basado en tres pilares: Ingeniería de Recompensas, Optimización de Hiperparámetros y Ajuste de Restricciones de Control.

### 2.1. Ingeniería de Recompensas (Reward Shaping)
Abandonamos las funciones de recompensa simples y adoptamos un enfoque basado en **MaxPressure** con penalizaciones de estabilidad. La función de recompensa final diseñada fue:

$$ R = - w_{queue} \cdot Q - w_{unbalance} \cdot U $$

Donde:
*   **$Q$ (Queue):** La suma de vehículos detenidos. Minimizar esto es el objetivo principal.
*   **$U$ (Unbalance):** Una penalización por la diferencia de colas entre los carriles con semáforo verde y los que tienen rojo.
    *   *Efecto:* Esto solucionó el problema del "HOLD". Si el agente deja una cola crecer demasiado en el carril rojo mientras el verde está vacío, el término de desequilibrio ($U$) crece, castigando al agente y forzándolo a cambiar.

### 2.2. Optimización con Optuna (Fase de Tuning)
Utilizamos **Optuna** para encontrar los hiperparámetros óptimos que el ajuste manual no lograba descubrir. Realizamos múltiples "trials" variando:
*   **Learning Rate:** Encontramos que un valor cercano a `4.65e-4` funcionaba mejor.
*   **Gamma (Factor de descuento):** Se ajustó a `0.951`, priorizando recompensas a mediano plazo.
*   **Coeficiente de Entropía:** Un valor alto (`0.05`) fue crucial para forzar al agente a explorar y no quedarse estancado en óptimos locales (como el "HOLD").
*   **Arquitectura de Red:** Se aumentó la capacidad de la red neuronal a `[256, 128, 64]` para capturar mejor la complejidad del tráfico.

### 2.3. Restricción de `min_green` (Estabilidad)
Aun con el modelo optimizado, el agente tendía a ser demasiado "nervioso". Introdujimos la variable `min_green` (tiempo mínimo de verde) como una restricción dura en el entorno para evaluar el compromiso entre reactividad y estabilidad.

---

## 3. Experimentos y Evaluación Comparativa

Entrenamos y comparamos tres variantes del modelo final optimizado, variando únicamente el tiempo mínimo de verde, contra una línea base de tiempo fijo (30s).

### 3.1. Modelos Evaluados
1.  **PPO min_green=5s:** Modelo agresivo, máxima reactividad.
2.  **PPO min_green=10s:** Modelo balanceado.
3.  **PPO min_green=15s:** Modelo conservador, mayor estabilidad.
4.  **Baseline (Fixed 30s):** Semáforo tradicional de ciclos fijos.

### 3.2. La Corrección Crítica en la Evaluación
Inicialmente, los resultados sugerían que el modelo de 5s era superior. Sin embargo, detectamos un **sesgo en la métrica de Throughput**.
*   *Error:* Calculábamos el flujo basándonos en "pasos de decisión" del agente.
*   *Realidad:* El modelo de 5s realizaba tantos cambios (194 cambios vs 119 del fijo) que la simulación real duraba mucho más tiempo debido a los segundos perdidos en fases de amarillo y rojo (tiempos muertos).
*   *Corrección:* Ajustamos el cálculo para usar el `traci.simulation.getTime()` real.

### 3.3. Resultados Finales (Corregidos)

| Configuración | Throughput (veh/h) | Cola Promedio (veh) | Cambios de Fase | Eficiencia Relativa |
| :--- | :--- | :--- | :--- | :--- |
| **min_green=5s** | 1419.0 | **1.6** | 193.7 | Baja (Mucho tiempo muerto) |
| **min_green=10s** | **1528.3** | 2.3 | 139.0 | **Alta** |
| **min_green=15s** | **1528.6** | 2.9 | 121.3 | **Alta** |
| **Fixed 30s** | 1462.0 | 5.6 | 119.0 | Media (Referencia) |

---

## 4. Conclusiones y Logros

1.  **Superación del Baseline:** Los modelos RL optimizados (10s y 15s) lograron un **Throughput 4.5% superior** al control de tiempo fijo.
2.  **Gestión de Colas:** Lo más impactante fue la reducción de colas. El modelo RL redujo la cola promedio en un **~50-60%** respecto al fijo (de 5.6 vehículos a ~2.3).
3.  **Estabilidad vs. Eficiencia:** Se demostró que ser "hiper-reactivo" (5s) es contraproducente. Aunque mantiene las colas al mínimo absoluto, se pierde demasiado tiempo en transiciones.
4.  **El Ganador:** El modelo con **`min_green=10s`** representa el punto óptimo. Ofrece el mismo flujo vehicular que el de 15s, pero con una gestión de colas un **21% mejor**, manteniendo una frecuencia de cambios aceptable.

Este proyecto demuestra que un agente PPO bien ajustado, con una función de recompensa consciente del desequilibrio y restricciones de tiempo mínimas, supera significativamente a los sistemas de control tradicionales en escenarios de tráfico dinámico.
