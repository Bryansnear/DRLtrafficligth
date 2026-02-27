# Informe Consolidado de Tesis: Control de Tráfico Adaptativo con Deep Reinforcement Learning

**Autor:** Bryan
**Fecha:** Noviembre 2025
**Proyecto:** Optimización de Semáforos con PPO y SUMO

---

## 1. Resumen Ejecutivo

Este proyecto desarrolla un sistema de control de semáforos inteligente capaz de adaptarse a flujos de tráfico dinámicos y altamente variables. Utilizando el algoritmo **Proximal Policy Optimization (PPO)**, el agente aprende a minimizar las colas y maximizar el flujo vehicular, superando a los controladores tradicionales de tiempo fijo en escenarios complejos.

**Logros Clave:**
*   **Reducción de Colas:** ~60% menos vehículos en espera comparado con el baseline.
*   **Mejora de Flujo:** Aumento del 4.5% en throughput.
*   **Estabilidad:** Eliminación del comportamiento errático mediante restricciones de `min_green` y penalizaciones de cambio.

---

## 2. Arquitectura del Sistema

El sistema se basa en la interacción cíclica entre un agente de aprendizaje y un entorno simulado.

### 2.1. Componentes
1.  **Entorno (SUMO):** Simula la física del tráfico, vehículos y red vial.
2.  **Interfaz (TraCI/Gymnasium):** Convierte el estado del simulador en observaciones numéricas para el agente.
3.  **Agente (Stable Baselines3 PPO):** Red neuronal que decide la fase del semáforo.

### 2.2. Diseño de la Función de Recompensa (Reward Shaping)
El núcleo del aprendizaje del agente reside en su función de recompensa. Se diseñó una función compuesta que penaliza comportamientos indeseados (colas, desequilibrio) y premia la eficiencia.

#### Ecuación General
$$ R_t = - w_{queue} \cdot Q_t - w_{unbalance} \cdot U_t - w_{switch} \cdot C_t $$

#### Componentes Detallados
1.  **Penalización por Cola ($Q_t$):**
    *   *Definición:* Suma de vehículos detenidos en todos los carriles entrantes, normalizada por la capacidad del carril.
    *   *Objetivo:* Es el término dominante ($w_{queue} \approx 1.0$). Minimizar $Q_t$ equivale a maximizar el flujo vehicular (teoría MaxPressure).
    
2.  **Penalización por Desequilibrio ($U_t$):**
    *   *Definición:* Diferencia absoluta entre la cola acumulada en los carriles con semáforo rojo y los carriles con verde.
    *   *Objetivo:* Evitar el problema del "HOLD infinito". Si el agente deja un carril en rojo demasiado tiempo, su cola crece desproporcionadamente respecto al carril verde (que se vacía), aumentando $U_t$ y forzando al agente a cambiar de fase para reducir este castigo.
    
3.  **Costo de Cambio ($C_t$):**
    *   *Definición:* Valor binario (0 o 1) que se activa si el agente decide cambiar de fase.
    *   *Objetivo:* Desincentivar el "flickering" (cambios rápidos e innecesarios). Actúa como una fricción suave.

#### Dinámica Temporal
La siguiente gráfica ilustra cómo interactúan estos componentes en un ciclo típico simulado:

![Composición de Recompensa](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/reward_composition.png)

*Figura 2: Dinámica de la recompensa. La zona roja representa el castigo base por colas. La zona naranja muestra cómo el desequilibrio crece si no se atiende una fase roja. Las barras azules son los "costos" puntuales de cambiar de fase.*

---

## 3. Configuración Experimental

Para validar la robustez del agente, se diseñó un perfil de tráfico dinámico que simula un día típico con horas pico cambiantes.

### 3.1. Perfil de Demanda (Traffic Profile)
El entrenamiento y evaluación se realizaron sobre un escenario de 1 hora con 5 fases distintas:

1.  **Calentamiento (0-15m):** Flujo medio y equilibrado.
2.  **Pico Sur (15-30m):** Alta demanda en dirección Sur-Norte.
3.  **Transición (30-40m):** Retorno a flujo medio.
4.  **Pico Este (40-55m):** Alta demanda en dirección Este-Oeste.
5.  **Ráfagas (55-60m):** Variabilidad extrema y rápida.

![Perfil de Tráfico](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/traffic_profile.png)

*Figura 1: Probabilidad de inserción de vehículos por segundo a lo largo de la simulación. Se observan claramente los picos de demanda que el agente debe gestionar.*

---

## 4. Resultados y Análisis

### 4.1. Comparativa de Rendimiento
Se comparó el agente PPO optimizado (con restricción de verde mínimo de 10s) contra un controlador de tiempo fijo (30s por fase).

![Comparativa Tesis](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/comparativa_tesis.png)

*Figura 2: El agente PPO (10s) logra un throughput superior (barra azul) manteniendo las colas significativamente más bajas (barra roja) que el sistema fijo.*

### 4.2. Gestión de Colas
La distribución de colas muestra cómo el agente PPO logra mantener la mayoría de los carriles vacíos o con muy pocos vehículos, mientras que el sistema fijo tiene una distribución más dispersa con colas largas frecuentes.

![Distribución de Colas](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/queue_distribution.png)

*Figura 3: Histograma de longitud de colas. PPO concentra la densidad cerca de 0, indicando una gestión eficiente.*

### 4.3. Comportamiento de Control (Acciones)
Uno de los desafíos fue evitar que el agente cambiara de fase demasiado rápido ("flickering"). La siguiente gráfica muestra la frecuencia de cambios por hora.

![Frecuencia de Cambios](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/action_frequency.png)

*Figura 4: El agente PPO realiza más cambios de fase que el fijo (139 vs 119), lo cual es esperado para adaptarse a la demanda, pero se mantiene dentro de un rango razonable gracias a la restricción de `min_green`.*

### 4.4. Evolución del Aprendizaje
El objetivo principal es maximizar la recompensa acumulada. La siguiente gráfica muestra cómo el agente mejora su rendimiento a lo largo de los pasos de entrenamiento.

![Curva de Aprendizaje (Recompensa)](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/training_reward.png)
*Figura 5: Evolución de la recompensa promedio por episodio. Se observa una tendencia ascendente clara, indicando que el agente está aprendiendo a controlar el tráfico eficientemente.*

Además, la entropía decrece, lo que indica que el agente gana confianza en su política:

![Entropía de Entrenamiento](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/training_entropy.png)
*Figura 6: La caída en la entropía indica que el agente reduce su exploración aleatoria.*

#### 4.4.1. Análisis de Pérdidas (Losses)
Para verificar la estabilidad del entrenamiento PPO, analizamos las funciones de pérdida del Actor (Policy) y el Crítico (Value).

![Value Loss](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/training_value_loss.png)
*Figura 6: La pérdida de valor decrece, indicando que el crítico aprende a estimar mejor el retorno esperado.*

![Policy Loss](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/training_policy_loss.png)
*Figura 7: La pérdida de política oscila pero se mantiene acotada, característico de PPO.*

#### 4.4.2. Divergencia KL
La divergencia KL aproximada nos indica cuánto cambia la política entre actualizaciones. PPO busca mantener esto bajo para evitar cambios destructivos.

![KL Divergence](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/training_kl_div.png)
*Figura 8: La divergencia se mantiene cerca del objetivo, confirmando un entrenamiento estable.*

### 4.5. Variabilidad de Resultados (Boxplot)
Más allá de los promedios, es crucial analizar la variabilidad del rendimiento.

![Boxplot Colas](/C:/Users/bryan/.gemini/antigravity/brain/51982cb2-336b-4989-9334-39d620d31838/eval_boxplot_queues.png)
*Figura 9: El diagrama de caja muestra que PPO no solo tiene una media menor, sino también una varianza mucho más baja (cajas más compactas), garantizando un servicio más consistente.*

---

## 5. Conclusiones

1.  **Adaptabilidad:** El agente PPO demostró ser capaz de manejar escenarios de tráfico asimétricos (Pico Sur/Este) donde los sistemas fijos fallan al asignar tiempo verde innecesario a carriles vacíos.
2.  **Eficiencia:** La reducción del 60% en colas implica un impacto directo en la reducción de tiempos de espera y emisiones contaminantes.
3.  **Estabilidad:** La implementación de restricciones duras (`min_green`) fue crucial para hacer el sistema viable en el mundo real, evitando oscilaciones peligrosas.

---

## 6. Referencias y Herramientas
*   **Simulador:** SUMO (Simulation of Urban MObility).
*   **Algoritmo:** PPO (Stable Baselines3).
*   **Optimización:** Optuna (Tuning de hiperparámetros).
*   **Lenguaje:** Python 3.10+.
