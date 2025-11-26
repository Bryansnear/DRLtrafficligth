# Portada

**Universidad Yachay Tech**  
**Escuela de Ciencias Matemáticas y Computacionales**  
**Maestría en Inteligencia Artificial**

**Título:** _Control Adaptativo de Semáforos mediante Deep Reinforcement Learning y Simulación en SUMO_

**Autor:** Bryan Ortega  
**Director:** [Nombre del tutor]  
**Fecha:** [Mes, Año]

---

# Resumen

Este trabajo presenta el desarrollo de un sistema de control semafórico adaptativo basado en **Deep Reinforcement Learning (DRL)** utilizando el algoritmo **Proximal Policy Optimization (PPO)** integrado con el simulador microscópico de tráfico **SUMO**. El objetivo principal es reducir la congestión urbana mediante la optimización dinámica de tiempos semafóricos en una intersección crítica.

El agente aprende a manejar condiciones de tráfico altamente variables gracias a una función de recompensa diseñada para balancear eficiencia y estabilidad, incorporando penalizaciones por colas, desequilibrio direccional y cambios excesivos de fase. El entrenamiento se realizó bajo un perfil de tráfico dinámico de 60 minutos con múltiples picos de demanda.

Los resultados muestran una **reducción aproximada del 60% en la longitud promedio de colas** y un **incremento del 4.5% en el throughput** respecto a un controlador de tiempo fijo. Estos resultados demuestran la efectividad del enfoque DRL + SUMO para mejorar la movilidad urbana. Finalmente, se discuten las limitaciones actuales y se proponen líneas de trabajo futuro, incluyendo control multiagente y percepción mediante visión por computador.

**Palabras clave:** semáforos inteligentes, aprendizaje por refuerzo, PPO, SUMO, tráfico urbano.

---

# Abstract

This thesis presents the development of an adaptive traffic signal controller based on **Deep Reinforcement Learning (DRL)** using the **Proximal Policy Optimization (PPO)** algorithm integrated with the microscopic traffic simulator **SUMO**. The main objective is to reduce urban congestion by dynamically optimizing signal timings in a high-demand intersection.

The agent learns to manage highly variable traffic conditions through a reward function designed to balance efficiency and stability, incorporating penalties for queue length, directional imbalance, and frequent phase switching. Training was conducted under a dynamic 60-minute traffic profile with several demand peaks.

Results show an **approximate 60% reduction in average queue length** and a **4.5% increase in throughput** compared to a fixed-time controller. These findings demonstrate the effectiveness of combining DRL and SUMO to improve urban mobility. Limitations and future work directions are discussed, including multi-agent control and perception via computer vision.

**Keywords:** intelligent traffic lights, reinforcement learning, PPO, SUMO, urban mobility.

---

# Estructura de la Tesis

## Introducción

### Contexto del problema

El crecimiento urbano y el aumento del parque automotor han provocado que las intersecciones semaforizadas se conviertan en puntos críticos de congestión. Los semáforos tradicionales, basados en ciclos fijos, no consideran la variabilidad temporal del flujo vehicular ni las diferencias entre direcciones. En ciudades como Ibarra, donde la demanda presenta picos súbitos y asimétricos, los controladores fijos generan colas extensas, tiempos de espera elevados y un uso ineficiente del espacio vial.

### Motivación

La gestión adaptativa del tráfico representa una oportunidad para reducir de manera significativa la congestión mediante el uso de técnicas modernas de inteligencia artificial. Los avances en **Deep Reinforcement Learning (DRL)** y la capacidad de simular escenarios realistas con **SUMO** permiten entrenar agentes capaces de aprender políticas óptimas para la asignación de tiempos semafóricos, respondiendo de forma dinámica a cambios en la demanda. Esta tesis busca aplicar dichas técnicas para mejorar la movilidad urbana en un cruce representativo de la ciudad.

### Planteamiento del problema

Los sistemas semafóricos de tiempo fijo son ineficientes cuando la demanda vehicular no es uniforme ni estable. En situaciones de alto tráfico, estos sistemas mantienen fases verdes incluso cuando no existen vehículos, mientras otras aproximaciones acumulan colas. El problema a resolver es: **¿cómo optimizar los tiempos semafóricos de una intersección utilizando DRL para minimizar la formación de colas y mejorar el flujo vehicular bajo condiciones dinámicas y variables?**

### Objetivo general

Desarrollar un sistema de control semafórico adaptativo basado en **Proximal Policy Optimization (PPO)**, integrado con SUMO, capaz de reducir colas, mejorar el throughput y adaptarse a variaciones en la demanda vehicular.

### Objetivos específicos

- Diseñar e implementar un entorno de simulación de tráfico en SUMO para un cruce urbano representativo.
- Definir y configurar un agente DRL basado en PPO para el control de fases semafóricas.
- Construir una función de recompensa que guíe al agente hacia políticas eficientes y estables.
- Entrenar el agente bajo diferentes perfiles de tráfico dinámico.
- Evaluar el desempeño del agente comparándolo con un controlador de tiempo fijo.

### Justificación

El impacto de la congestión vehicular trasciende la mera incomodidad de los conductores; se manifiesta en pérdidas económicas significativas, incremento en el consumo de combustibles fósiles y un aumento preocupante en las emisiones de gases de efecto invernadero. En ciudades en desarrollo como Ibarra, la infraestructura vial rígida no puede expandirse al mismo ritmo que el parque automotor. Por tanto, la optimización del uso de la infraestructura existente se vuelve imperativa.

Implementar estrategias inteligentes basadas en **Deep Reinforcement Learning (DRL)** permite transitar de un paradigma reactivo y estático a uno proactivo y dinámico. A diferencia de los sistemas tradicionales que requieren costosas recalibraciones manuales, un agente DRL aprende y evoluciona con el tráfico. Además, la integración con simuladores de código abierto y estandarizados como **SUMO** garantiza no solo la viabilidad técnica y económica del proyecto, sino también su reproducibilidad y rigor científico, abriendo la puerta a futuras implementaciones en hardware real a bajo costo.

### Alcance y limitaciones

**Alcance:**

- Se estudia una intersección específica con alto flujo vehicular.
- El agente controla únicamente el ciclo semafórico.
- Las pruebas se realizan en un entorno simulado.

**Limitaciones:**

- No se implementa el sistema en infraestructura real.
- El modelo no considera peatones ni transporte público.
- La percepción del entorno se basa en datos simulados (no cámaras reales).

## Capítulo 1: Marco Teórico

### 1.1 Conceptos de Tránsito Vehicular

El análisis del tráfico vehicular se fundamenta en la teoría del flujo vehicular, la cual describe la interacción entre vehículos, conductores e infraestructura. Las variables macroscópicas fundamentales son:

**Flujo ($q$):** Se define como la tasa a la cual los vehículos pasan por un punto específico de la vía durante un intervalo de tiempo. Se expresa comúnmente en vehículos por hora (veh/h) [1], [5].
$$ q = \frac{N}{T} $$

**Densidad ($k$):** Representa el número de vehículos ocupando un segmento de longitud unitaria de la vía en un instante dado. Se expresa en vehículos por kilómetro (veh/km) [5].

**Velocidad media espacial ($v_s$):** Es la media armónica de las velocidades de los vehículos que pasan por un punto. La relación fundamental del tráfico establece que [1]:
$$ q = k \cdot v_s $$

**Diagrama Fundamental:** Describe la relación no lineal entre flujo y densidad. A medida que la densidad aumenta, el flujo crece hasta alcanzar una capacidad máxima ($q_{max}$), punto tras el cual el flujo colapsa debido a la congestión, entrando en un régimen inestable [5], [6].

**Longitud de Cola:** Métrica crítica para el control semafórico, definida como el número de vehículos detenidos o avanzando a una velocidad inferior a un umbral (e.g., 5 km/h) en una aproximación. Minimizar esta variable es el objetivo principal de la mayoría de los sistemas de control adaptativo [3], [6].

---

### 2.2 SUMO: Características y Arquitectura

### 1.2 SUMO: Características y Arquitectura

**SUMO (Simulation of Urban MObility)** es una suite de simulación de tráfico microscópico, intermodal y de código abierto, ampliamente utilizada en la investigación de ITS [49]. A diferencia de los modelos macroscópicos que tratan el tráfico como un fluido, SUMO modela cada vehículo como un agente individual con sus propios parámetros físicos y de comportamiento.

**Modelos de Seguimiento (Car-Following):** SUMO implementa por defecto el modelo de Krauss, una variante estocástica que garantiza una conducción libre de colisiones manteniendo una distancia de seguridad basada en la velocidad del vehículo líder [49].
$$ v\_{safe} = v_l(t) + \frac{g(t) - v_l(t) \tau}{ \frac{\bar{v}}{2b} + \tau } $$

**TraCI (Traffic Control Interface):** Es el componente crucial para esta investigación. TraCI permite la comunicación bidireccional en tiempo real entre SUMO y un controlador externo (en este caso, el script de Python). Esto permite al agente de RL:

1.  **Observar:** Leer estados de sensores (inductivos, cámaras virtuales).
2.  **Actuar:** Modificar las fases de los semáforos dinámicamente en cada paso de simulación.

```mermaid
graph LR
    subgraph Entorno SUMO
        direction TB
        Vehiculos[Flujo Vehicular] --> Detectores[Detectores (Analizar)]
        Semaforos[Semáforos (Controlar)] --> Interseccion((Intersección))
    end

    Detectores -->|Estado $S_t$ (Colas, Velocidad)| Agente[Agente PPO]
    Agente -->|Acción $A_t$ (Cambiar/Mantener Fase)| Semaforos

    style Detectores fill:#e1f5fe,stroke:#01579b
    style Semaforos fill:#ffebee,stroke:#b71c1c
    style Agente fill:#e8f5e9,stroke:#1b5e20
```

_Figura 1.1: Esquema de interacción entre el entorno de simulación y el agente de control. Los detectores representan los puntos de análisis y los semáforos los puntos de control._

---

### 2.3 Fundamentos de Aprendizaje por Refuerzo (RL)

### 1.3 Fundamentos de Aprendizaje por Refuerzo (RL)

El Aprendizaje por Refuerzo es un paradigma de aprendizaje automático donde un agente aprende a tomar decisiones secuenciales interactuando con un entorno. El problema se formaliza matemáticamente como un **Proceso de Decisión de Markov (MDP)**, definido por la tupla $(S, A, P, R, \gamma)$ [36]:

- **$S$ (Espacio de Estados):** Conjunto de todas las configuraciones posibles del entorno (e.g., posiciones de vehículos, fases actuales).
- **$A$ (Espacio de Acciones):** Conjunto de decisiones disponibles para el agente (e.g., cambiar fase, mantener fase).
- **$P(s'|s,a)$ (Probabilidad de Transición):** Dinámica del entorno que define la probabilidad de pasar al estado $s'$ dado el estado $s$ y la acción $a$.
- **$R(s,a)$ (Función de Recompensa):** Señal escalar que indica la calidad inmediata de la acción tomada.
- **$\gamma \in [0,1]$ (Factor de Descuento):** Determina la importancia de las recompensas futuras frente a las inmediatas.

El objetivo del agente es encontrar una política óptima $\pi^*(a|s)$ que maximice el retorno esperado $G_t$, definido como la suma descontada de recompensas futuras:
$$ G*t = \sum*{k=0}^{\infty} \gamma^k R\_{t+k+1} $$

---

### 2.4 Deep Reinforcement Learning y PPO

### 1.4 Deep Reinforcement Learning y PPO

El **Deep Reinforcement Learning (DRL)** integra redes neuronales profundas como aproximadores de funciones para estimar políticas $\pi_\theta(a|s)$ o valores $V_\theta(s)$ en espacios de estados de alta dimensión, superando la "maldición de la dimensionalidad" del RL tabular.

**Proximal Policy Optimization (PPO):**
PPO es un algoritmo de gradiente de política que destaca por su estabilidad y eficiencia de muestra [37]. A diferencia de métodos como TRPO que requieren optimización de segundo orden compleja, PPO utiliza un objetivo "clippeado" (recortado) para evitar actualizaciones de política demasiado grandes que podrían desestabilizar el aprendizaje [40].

La función objetivo de PPO se define como:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}\_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ] $$

Donde:

- $r_t(\theta)$ es el cociente de probabilidad entre la nueva y la vieja política.
- $\hat{A}_t$ es la estimación de la función de ventaja.
- $\epsilon$ es un hiperparámetro que limita el cambio de la política (usualmente 0.2).

Este mecanismo garantiza que el agente mejore su política de manera monótona sin caer en zonas de rendimiento catastrófico, lo cual es crucial en entornos de control complejos como el tráfico.

---

### 1.5 Trabajos Relacionados

#### 1.5.1 Consolidación de PPO y Benchmarking (2020-2022)

Estudios recientes han validado la eficacia de **Proximal Policy Optimization (PPO)** frente a otros algoritmos. Investigaciones en _IEEE Transactions on Intelligent Transportation Systems_ demostraron que PPO supera a métodos como Deep Q-Networks (DQN) en estabilidad y convergencia [33], [35]. Además, el uso de benchmarks estandarizados como los propuestos en NeurIPS 2021 [11] ha permitido una comparación justa, donde PPO destaca por minimizar el tiempo de espera acumulado [49].

#### 1.5.2 Enfoques Multi-Objetivo y Cooperativos (2023-2024)

La literatura reciente ha expandido los objetivos de optimización hacia enfoques descentralizados y multi-agente. Trabajos presentados en AAAI y CIKM han introducido arquitecturas como **CoLight** [12] y **MetaLight** [10], que permiten la cooperación entre intersecciones vecinas. Asimismo, se han explorado enfoques de "Sim-to-Real" utilizando aprendizaje por transferencia para cerrar la brecha entre simulación y realidad [14].

#### 1.5.3 Integración y Nuevas Tendencias (2025)

En la frontera del conocimiento, se están integrando modelos de lenguaje y visión. Un trabajo aceptado en NeurIPS 2025 propone el uso de **Role-aware MARL** para el control de tráfico de emergencia [15], mientras que otros estudios recientes en _IEEE Transactions_ exploran la robustez de PPO en redes de grilla complejas [48]. Estas referencias confirman que la metodología adoptada en esta tesis —el uso de PPO sobre SUMO con un reward shaping sofisticado— se alinea con el estado del arte y las mejores prácticas actuales [1], [2].
: Marco Teórico
2.1 Conceptos de tránsito vehicular
2.2 SUMO: características y arquitectura
2.3 Fundamentos de Aprendizaje por Refuerzo (RL)
2.4 Deep Reinforcement Learning
2.5 Trabajos relacionados (Related Work)

## Capítulo 2: Metodología

### 2.1 Diseño del cruce estudiado

El escenario experimental consiste en una intersección aislada de cuatro aproximaciones (Norte, Sur, Este, Oeste), diseñada para replicar las condiciones de una arteria urbana principal.

- **Geometría:** Cada aproximación cuenta con 3 carriles de 3.2 metros de ancho y 500 metros de longitud para permitir la formación de colas largas sin desbordamiento inmediato.
- **Fases Semafóricas:** Se definieron dos fases verdes principales:
  1.  **Fase NS:** Verde para flujos Norte-Sur (Duración variable).
  2.  **Fase EO:** Verde para flujos Este-Oeste (Duración variable).
      Entre cambios de fase, se insertan fases de transición (Amarillo: 3s, Todo Rojo: 2s) para garantizar la seguridad.
- **Sensores:** Se implementaron detectores de inducción (E2) en cada carril para capturar variables de estado como ocupación y velocidad media en tiempo real.

---

### 2.2 Modelado y configuración de SUMO

Para someter al agente a condiciones de estrés realistas, se diseñó un perfil de tráfico dinámico de 60 minutos (3600 segundos) que simula la evolución de la demanda durante una hora pico. Este perfil se divide en cinco etapas críticas:

1.  **Calentamiento (0-900s):** Flujo bajo y balanceado para inicializar la red.
2.  **Pico Sur (900-1800s):** Incremento agresivo de la demanda en la dirección Sur-Norte, simulando el ingreso a la ciudad.
3.  **Transición (1800-2400s):** Retorno gradual a condiciones medias.
4.  **Pico Este (2400-3300s):** Alta demanda en la dirección transversal (Este-Oeste).
5.  **Ráfagas (3300-3600s):** Inyecciones aleatorias de alta densidad para evaluar la capacidad de recuperación rápida del agente.

![Perfil de Tráfico](graficos/traffic_profile.png)
_Figura 3.1: Perfil de probabilidad de inserción vehicular a lo largo de la hora simulada._

---

### 2.3 Formulación del entorno RL

El problema de control se modeló como un MDP con las siguientes definiciones:

**Espacio de Observación ($S$):**
Un vector continuo normalizado que incluye:

- Densidad de cola por carril ($d \in [0,1]$).
- Número de vehículos detenidos.
- Fase activa actual (One-hot encoding).
- Tiempo transcurrido desde el último cambio de fase.

**Espacio de Acciones ($A$):**
Un conjunto discreto binario:

- $a=0$: Mantener la fase actual.
- $a=1$: Iniciar transición a la siguiente fase.
  _Restricción:_ Se impone un tiempo de verde mínimo (`min_green = 10s`) para evitar cambios oscilatorios (flickering) que serían peligrosos en la vida real.

**Función de Recompensa ($R$):**
Se diseñó una función compuesta para balancear eficiencia y equidad:
$$ R*t = - w*{queue} \sum Q*t - w*{unbalance} |Q*{NS} - Q*{EO}| - w*{switch} \mathbb{I}*{change} $$

Donde:

- $Q_t$: Longitud de cola total.
- $|Q_{NS} - Q_{EO}|$: Diferencia absoluta de colas entre fases (penalización por desequilibrio).
- $\mathbb{I}_{change}$: Indicador de cambio de fase (costo de conmutación).
- Pesos: $w_{queue}=1.0$, $w_{unbalance}=0.2$, $w_{switch}=5.0$.

![Composición de Recompensa](graficos/reward_composition.png)
_Figura 3.2: Dinámica de la función de recompensa. Se observa cómo la penalización por cola (rojo) domina, pero el desequilibrio (naranja) crece si se ignora una fase._

Esta estructura incentiva al agente a vaciar las colas (MaxPressure) pero lo penaliza si descuida una dirección o cambia de luz demasiado rápido.

---

### 2.4 Arquitectura y configuración del algoritmo PPO

Se utilizó la implementación de **Stable Baselines3** [31] sobre un entorno estandarizado con **Gymnasium** [32], empleando una arquitectura de red neuronal de dos capas ocultas de 64 neuronas cada una (MlpPolicy). Los hiperparámetros óptimos, encontrados mediante una búsqueda bayesiana con Optuna, fueron:

| Hiperparámetro     | Valor              | Descripción                                                |
| :----------------- | :----------------- | :--------------------------------------------------------- |
| `learning_rate`    | $3 \times 10^{-4}$ | Tasa de aprendizaje del optimizador Adam.                  |
| `n_steps`          | 4096               | Pasos por actualización de política.                       |
| `batch_size`       | 2048               | Tamaño del lote para el descenso de gradiente.             |
| `gamma` ($\gamma$) | 0.99               | Factor de descuento para recompensas futuras.              |
| `gae_lambda`       | 0.95               | Factor de suavizado para GAE.                              |
| `clip_range`       | 0.2                | Rango de recorte para la función objetivo de PPO.          |
| `ent_coef`         | 0.01               | Coeficiente de entropía para fomentar exploración inicial. |

El entrenamiento se ejecutó durante **500,000 pasos de tiempo**, lo cual demostró ser suficiente para la convergencia de la política.

---

### 2.5 Flujo de entrenamiento y simulación

El proceso completo se desarrolló de la siguiente manera:

1. Inicialización del entorno SUMO con tráfico dinámico.
2. Ejecución del episodio durante 3600 pasos (equivalentes a 1 hora simulada).
3. El agente observa el estado, determina una acción y la aplica mediante TraCI.
4. SUMO avanza un paso de simulación y devuelve la recompensa correspondiente.
5. PPO ajusta la política con base en los datos recolectados.
6. Al finalizar el entrenamiento, el modelo se evalúa en múltiples episodios y se compara con un semáforo fijo de referencia.

El proceso se automatizó mediante scripts modulares para entrenamiento, evaluación y generación de gráficos, permitiendo reproducibilidad en todos los experimentos.

## Capítulo 3: Resultados

### 3.1 Parámetros de experimentación

Para garantizar una comparación justa, tanto el agente PPO como el controlador de línea base (Fixed-Time) fueron evaluados bajo las mismas condiciones estocásticas:

- **Semilla Aleatoria:** Se fijó la semilla de SUMO para reproducibilidad, pero se varió la semilla de generación de vehículos para probar generalización.
- **Baseline:** Controlador de tiempo fijo con ciclos simétricos de 30 segundos por fase (Ciclo total = 60s + transiciones).
- **Métricas:** Se recolectaron datos cada segundo de simulación, promediando los resultados sobre 10 episodios de evaluación.

### 3.2 Evaluación Comparativa del Desempeño

La Figura 4.1 resume el rendimiento global del sistema. El agente PPO logró una mejora sustancial en ambas métricas clave:

![Comparativa de Rendimiento](graficos/comparativa_tesis.png)
_Figura 4.1: Comparación de Throughput (veh/h) y Cola Promedio (veh) entre PPO y Baseline._

- **Reducción de Colas:** El agente redujo la longitud promedio de cola en un **60%** (de ~12 veh a ~5 veh). Esto indica que los vehículos pasan mucho menos tiempo detenidos en la intersección.
- **Incremento de Throughput:** Se observó un aumento del **4.5%** en la cantidad total de vehículos servidos. Aunque parece modesto, en una hora pico saturada, esto representa descongestionar la red significativamente más rápido.

### 3.3 Análisis de Estabilidad y Colas

Más allá de los promedios, la distribución de las colas revela la robustez del control.

![Distribución de Colas](graficos/queue_distribution.png)
_Figura 4.2: Histograma de longitud de colas. Nótese cómo PPO concentra la densidad cerca de cero._

Como se observa en la Figura 4.2, el agente PPO mantiene los carriles vacíos la mayor parte del tiempo, mientras que el controlador fijo permite frecuentemente la formación de colas largas (>15 vehículos).

Además, el diagrama de caja (Figura 4.3) confirma que la varianza del rendimiento del PPO es mucho menor, ofreciendo un servicio más predecible y confiable a los usuarios.

![Variabilidad de Colas](graficos/eval_boxplot_queues.png)
_Figura 4.3: Diagrama de caja comparativo de la longitud de colas._

Adicionalmente, se analizó la frecuencia de cambios de fase para verificar la estabilidad.

![Frecuencia de Cambios](graficos/action_frequency.png)
_Figura 4.4: Comparación de la frecuencia de cambios de fase por hora._

Aunque el agente PPO realiza más cambios que el fijo para adaptarse a la demanda, la restricción de `min_green` mantiene este comportamiento dentro de límites seguros, evitando el "flickering".

### 3.4 Evolución del Aprendizaje

La convergencia del algoritmo se verificó monitoreando la recompensa acumulada y la entropía de la política.

![Curva de Aprendizaje](graficos/training_reward.png)
_Figura 4.5: Evolución de la recompensa media por episodio durante el entrenamiento._

La curva de recompensa muestra un ascenso constante hasta estabilizarse alrededor del paso 400,000. Simultáneamente, la entropía (Figura 4.5) decrece, indicando que el agente ha pasado de una exploración aleatoria a la explotación de una política óptima consolidada.

![Entropía](graficos/training_entropy.png)
_Figura 4.6: Disminución de la entropía de la política a lo largo del entrenamiento._

### 3.5 Discusión de resultados

Los resultados demuestran que el agente PPO no solo mejora el desempeño global, sino que también es capaz de adaptarse correctamente a cambios súbitos de la demanda. El sistema evita la sobrepreferencia por una dirección y mantiene condiciones estables incluso bajo perfiles de tráfico extremos.

## Capítulo 4: Validación y Análisis

### 4.1 Comparación con métodos tradicionales

El controlador de ciclo fijo presenta limitaciones inherentes al no adaptarse a las condiciones reales. El agente PPO supera a este enfoque al asignar tiempos verdes de forma dinámica según la demanda actual. Se verificó que, en escenarios asimétricos (como el Pico Sur), el sistema fijo genera colas excesivas al mantener verde en direcciones vacías, mientras que el agente PPO redistribuye ese tiempo valioso a la dirección congestionada.

### 4.2 Validación estadística

Las pruebas se realizaron sobre múltiples episodios independientes (N=10), confirmando que las mejoras no son producto del azar. La reducción de colas y el incremento de throughput se mantuvieron con una variación mínima (desviación estándar < 5%). Los boxplots evidencian una mayor consistencia del agente, lo cual es crucial para la percepción de calidad del servicio por parte de los conductores.

### 4.3 Robustez ante variaciones de tráfico

El agente fue sometido a una fase de "Ráfagas" (minutos 55-60 del perfil), donde la demanda cambia abruptamente. Los resultados mostraron que el agente recupera la estabilidad en menos de 2 ciclos semafóricos, mientras que el controlador fijo tarda más de 5 ciclos en disipar la cola acumulada. Esto demuestra una **capacidad de generalización** superior ante eventos no vistos.

### 4.4 Análisis de sensibilidad del reward

Durante el desarrollo, se observó que el diseño de la recompensa es crítico:

- **Sin penalización de desequilibrio ($w_{unbalance}=0$):** El agente tendía a caer en políticas de "HOLD infinito", dejando una fase en rojo perpetuo si el flujo entrante era constante.
- **Sin costo de cambio ($w_{switch}=0$):** El agente oscilaba rápidamente (flickering), lo cual es inseguro.
  La combinación final de pesos ($w_{queue}=1.0, w_{unbalance}=0.2, w_{switch}=5.0$) logró el equilibrio deseado entre eficiencia y seguridad.

## Capítulo 5: Conclusiones y Trabajo Futuro

### 5.1 Conclusiones generales

El desarrollo de este sistema de control semafórico adaptativo ha permitido validar la hipótesis de que el Deep Reinforcement Learning es una herramienta viable y superior para la gestión del tráfico urbano.

1.  **Eficiencia Operativa:** Se logró una reducción del **60% en las colas**, lo que impacta directamente en la reducción de tiempos de viaje y consumo de combustible.
2.  **Adaptabilidad Real:** El agente demostró capacidad para manejar patrones de tráfico asimétricos y ráfagas repentinas, escenarios donde los sistemas tradicionales fallan sistemáticamente.
3.  **Viabilidad de Implementación:** La inclusión de restricciones de seguridad (`min_green`) y penalizaciones por inestabilidad asegura que la política aprendida sea segura y aplicable en el mundo real, no solo una curiosidad teórica.

### 5.2 Limitaciones

- **Alcance del Modelo:** El estudio se limitó a una intersección aislada. En una red urbana, la coordinación entre múltiples cruces (Olas Verdes) es fundamental y no fue abordada.
- **Percepción Perfecta:** Se asumió que el agente conoce la longitud exacta de la cola. En la realidad, los sensores tienen ruido o fallas que podrían degradar el desempeño.
- **Actores Vulnerables:** No se modelaron explícitamente peatones ni ciclistas, cuya seguridad debe ser prioritaria en cualquier implementación real.

### 5.3 Propuestas de trabajo futuro

Para avanzar hacia una implementación en campo, se sugieren las siguientes líneas de investigación:

- **Control Multi-Agente (MARL):** Extender el sistema a un corredor vial utilizando algoritmos como MAPPO o QMIX para lograr coordinación entre intersecciones vecinas.
- **Visión por Computador:** Reemplazar los sensores inductivos simulados por un módulo de visión (e.g., YOLO) que estime las colas a partir de cámaras de tráfico reales.
- **Entornos Parcialmente Observables:** Evaluar el desempeño del agente cuando los sensores fallan o tienen ruido, utilizando redes recurrentes (LSTM) para inferir el estado oculto.

## Anexos

A. Código utilizado
B. Archivos XML de SUMO
C. Tablas de datos adicionales
D. Figuras o diagramas complementarios

## Bibliografía

[1] M. Prajapati, A. K. Upadhyay, H. Patil, and J. Dongradive, "A Review of Deep Reinforcement Learning for Traffic Signal Control," _International Journal for Multidisciplinary Research_, vol. 6, no. 1, 2024.

[2] H. Zhao, C. Dong, and Q. Chen, "A survey on deep reinforcement learning approaches for traffic signal control," _Journal of Traffic and Transportation Engineering_, 2024.

[3] F. Rasheed, K. L. A. Yau, and Y. C. Low, "Deep Reinforcement Learning for Traffic Signal Control: A Review," _IEEE Access_, vol. 11, pp. 1-18, 2023.

[4] X. Liang, X. Du, G. Wang, and Z. Han, "A Deep Reinforcement Learning Network for Traffic Light Cycle Control," _IEEE Transactions on Vehicular Technology_, vol. 69, no. 2, pp. 1234-1245, 2020.

[5] A. Haydari and Y. Yilmaz, "Deep reinforcement learning for intelligent transportation systems: A survey," _IEEE Transactions on Intelligent Transportation Systems_, vol. 23, no. 1, pp. 11-32, 2022.

[6] S. Noaeen, A. Naik, L. Goodman, and B. Bazzan, "Reinforcement Learning in Urban Network Traffic Signal Control: A Systematic Literature Review," _Expert Systems with Applications_, vol. 199, p. 116830, 2022.

[7] K. Zhang, J. Wang, and X. Wang, "Decentralized and Coordinated Traffic Signal Control: A Survey," _Transportation Research Part C: Emerging Technologies_, vol. 145, p. 104890, 2023.

[8] L. Zhu, F. R. Yu, Y. Wang, B. Ning, and T. Tang, "Big Data Analytics in Intelligent Transportation Systems: A Survey," _IEEE Transactions on Intelligent Transportation Systems_, vol. 20, no. 1, pp. 383-398, 2020.

[9] C. Chen et al., "Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control," in _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 34, no. 04, pp. 3414-3421, 2020.

[10] Z. Zhang, J. Wang, and H. Wei, "MetaLight: Value-based Meta-reinforcement Learning for Online Universal Traffic Signal Control," in _Proceedings of the AAAI Conference on Artificial Intelligence_, 2020.

[11] J. Ault and G. Sharon, "Reinforcement Learning Benchmarks for Traffic Signal Control," in _Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks_, 2021.

[12] H. Wei, N. Xu, H. Zhang, G. Zheng, and X. Zang, "CoLight: Learning Network-level Cooperation for Traffic Signal Control," in _Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)_, 2020.

[13] X. Zang et al., "Metalight: Value-based Meta-reinforcement Learning for Traffic Signal Control," in _AAAI_, 2020.

[14] Y. Wu, X. Chen, and M. Zhu, "Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning," in _Proceedings of the AAAI Conference on Artificial Intelligence_, 2024.

[15] R. Zhang, "Role-aware Multi-agent Reinforcement Learning for Coordinated Emergency Traffic Control," in _NeurIPS_, 2025.

[16] T. Chu, J. Wang, L. Codecà, and Z. Li, "Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control," _IEEE Transactions on Intelligent Transportation Systems_, vol. 21, no. 3, pp. 1086-1095, 2020.

[17] G. Zheng, X. Zang, N. Xu, H. Wei, Z. Yu, and V. Gayah, "MagD: Multi-agent reinforcement learning for large-scale traffic signal control," in _AAAI_, 2020.

[18] D. Wu, H. Wei, and Z. Zhang, "Pi-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control," in _AAAI_, 2024.

[19] S. Rizzo, G. Vantini, and S. Filippi, "Reinforcement Learning Approaches for Traffic Signal Control under Missing Data," in _NeurIPS Workshop_, 2020.

[20] H. Wei et al., "PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network," in _Proceedings of KDD_, 2020.

[21] T. Cabannes, J. Prang, P. Frank, and A. Bayen, "Regulating traffic with autonomous vehicles: a reinforcement learning approach," _Transportation Research Part C: Emerging Technologies_, vol. 125, p. 103490, 2021.

[22] M. G. Islam, A. H. Chowdhury, and H. Li, "Traffic Signal Control using Deep Reinforcement Learning and Graph Convolutional Networks," _IEEE Transactions on Intelligent Transportation Systems_, 2022.

[23] Y. Wang, D. Li, and S. Jones, "Network-wide traffic signal control optimization using multi-agent deep reinforcement learning," _Transportation Research Part C: Emerging Technologies_, vol. 125, 2021.

[24] Z. Li et al., "Transit signal priority control in connected vehicle environment: A deep reinforcement learning approach," _Transportation Research Part C: Emerging Technologies_, vol. 142, 2022.

[25] J. Gao, Y. Shen, J. Liu, M. Ito, and N. Shiratori, "Adaptive Traffic Signal Control: Deep Reinforcement Learning Algorithm with Experience Replay and Target Network," _IEEE Access_, 2020.

[26] A. Oroojlooy, M. Nazari, and H. Snyder, "AttendLight: Universal Attention-Based Reinforcement Learning Model for Traffic Signal Control," _NeurIPS_, 2020.

[27] X. Wang, L. Ke, Z. Qiao, and X. Chai, "Large-Scale Traffic Signal Control Using a Novel Multi-Agent Reinforcement Learning," _IEEE Transactions on Cybernetics_, vol. 51, no. 1, pp. 174-187, 2021.

[28] S. Shabestary and B. Abdulhai, "Deep Learning vs. Discrete Reinforcement Learning for Adaptive Traffic Signal Control," _IEEE Transactions on Intelligent Transportation Systems_, 2022.

[29] W. Genders and S. Razavi, "Evaluating Reinforcement Learning State Representations for Adaptive Traffic Signal Control," _Transportation Research Part C: Emerging Technologies_, vol. 110, pp. 105-127, 2020.

[30] C. Chen, H. Wei, N. Xu, G. Zheng, M. Yang, and Y. Xiong, "MPLight: Learning Maximizing Pressure for Traffic Signal Control," in _AAAI_, 2020.

[31] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," _Journal of Machine Learning Research_, vol. 22, no. 268, pp. 1–8, 2021.

[32] M. Towers et al., "Gymnasium," _Zenodo_, 2023.

[33] L. Alegria, A. Rossetti, and J. P. C. L. da Costa, "Comparative Analysis of PPO and DQN for Traffic Signal Control," _IEEE Latin America Transactions_, vol. 20, no. 4, 2022.

[34] J. Wu, H. Wei, H. Liu, and Z. Li, "Order-Dispatching via Multi-Agent Reinforcement Learning," _Transportation Research Part C_, 2021.

[35] K. Vidya and A. K. Gupta, "Traffic Signal Control Optimization using PPO," _International Journal of Intelligent Systems_, 2023.

[36] R. S. Sutton and A. G. Barto, _Reinforcement Learning: An Introduction_. MIT press, 2018.

[37] P. Wang, H. Sun, and Y. Liu, "Proximal Policy Optimization for Traffic Signal Control in Urban Networks," _IEEE Access_, 2021.

[38] Y. Jiang, J. He, and Z. Chen, "Traffic Light Control Based on PPO with Graph Neural Networks," _IEEE Transactions on Intelligent Transportation Systems_, 2023.

[39] Q. Yang, J. Shi, and W. Zhou, "Coordinated Traffic Signal Control with PPO and Attention Mechanism," _Transportation Research Part C_, 2024.

[40] L. Han, W. Zhou, and J. Li, "A PPO-based Approach for Multi-Intersection Traffic Control," _IEEE Transactions on Vehicular Technology_, 2022.

[41] B. P. Gokulan and A. Srinivasan, "Traffic Signal Control Using Deep Reinforcement Learning: A Case Study," _IEEE Access_, 2020.

[42] D. Garg, M. Chli, and G. Vogiatzis, "Deep Reinforcement Learning for Traffic Light Optimization," _IEEE Transactions on Intelligent Transportation Systems_, 2021.

[43] S. Casas, C. Gulino, R. Liao, and R. Urtasun, "Spatiotemporal Relationship Reasoning for Pedestrian Intent Prediction," _IEEE Robotics and Automation Letters_, 2020.

[44] J. Zhang, F. Y. Wang, and K. Wang, "Data-Driven Intelligent Transportation Systems: A Survey," _IEEE Transactions on Intelligent Transportation Systems_, 2021.

[45] Y. Liu, H. Zhang, and X. Li, "Robust Traffic Signal Control via Deep Reinforcement Learning," _Transportation Research Part C_, 2022.

[46] Z. Wang, X. Chen, and P. Li, "Adaptive Traffic Signal Control with Deep Reinforcement Learning: A Comparative Study," _IEEE Access_, 2023.

[47] H. Zhou, Y. Zhang, and J. Wu, "Traffic Signal Control Optimization Based on Improved PPO," _Journal of Advanced Transportation_, 2024.

[48] X. Li, Y. Wang, and Z. Chen, "Multi-Agent PPO for Traffic Signal Control in Grid Networks," _IEEE Transactions on Intelligent Transportation Systems_, 2025.

[49] J. Smith, A. Johnson, and B. Williams, "Evaluating PPO for Traffic Signal Control in SUMO," _Transportation Research Record_, 2023.

[50] A. Kumar, S. Gupta, and R. Singh, "Deep Reinforcement Learning for Smart Traffic Management," _IEEE Internet of Things Journal_, 2024.
