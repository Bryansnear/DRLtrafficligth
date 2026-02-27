# Trabajos Relacionados (2021-2025)

Este documento recopila investigaciones recientes sobre el control de semáforos utilizando Deep Reinforcement Learning (DRL), con énfasis en PPO y SUMO.

## 2022: Consolidación de PPO y Benchmarking
En este año, varios estudios se centraron en comparar PPO con otros algoritmos estándar.
- **Superioridad de PPO:** Un estudio demostró que PPO superó a algoritmos como Actor-Critic (A2C), Deep Q-Networks (DQN) y agentes fijos/aleatorios tanto en escenarios de agente único como multi-agente. PPO destacó por minimizar el tiempo de espera acumulado promedio bajo diversas condiciones de tráfico [1].
- **Policy-based vs Value-based:** Se realizaron comparaciones exhaustivas entre métodos basados en políticas (como PPO) y basados en valor (como DQN) dentro de SUMO, entrenando agentes en diversos patrones de tráfico para validar su robustez frente a enfoques tradicionales [2].
- **Herramientas de Benchmarking:** Se propuso el toolkit **RESCO** (REinforced Signal COntrol) para estandarizar la comparación de controladores basados en RL en SUMO, facilitando la evaluación de nuevos algoritmos [4].

## 2023: Enfoques Multi-Objetivo y Cooperativos
La investigación evolucionó hacia objetivos más complejos más allá de solo reducir colas.
- **MOMA-DDPG:** Se introdujo una arquitectura cooperativa multi-objetivo (Multi-Objective Multi-Agent Deep Deterministic Policy Gradient) que optimiza simultáneamente el tiempo de espera y las emisiones de carbono. Aunque usa DDPG, establece un precedente importante para funciones de recompensa compuestas como la utilizada en esta tesis [5].

## 2024: Integración y Nuevas Arquitecturas
Se exploraron combinaciones de técnicas y mejoras en la infraestructura de simulación.
- **Lógica Difusa + DRL:** Un estudio propuso combinar Deep Q-Networks con lógica difusa para mejorar la eficiencia, modificando la función de recompensa para considerar el tiempo de espera específico por carril [7].
- **Control Multi-Modal:** Investigación sobre el uso de DRL multi-agente para coordinar múltiples semáforos en entornos de tráfico urbano multi-modal (vehículos, transporte público, etc.) usando SUMO [8].
- **SUMO-RL y Gymnasium:** Se popularizó el uso de librerías estandarizadas como SUMO-RL y Gymnasium (la misma stack tecnológica usada en esta tesis) para implementar algoritmos como DQN y optimizar el throughput frente a sistemas de tiempo fijo [10, 12].

## 2025 (y finales 2024): Tendencias Recientes
- **EP-D3QN con PPO:** Un trabajo reciente propuso un método de optimización basado en "Double Dueling Deep Q-Network" (D3QN) combinado con MaxPressure, pero adoptando específicamente **PPO para mejorar la velocidad de convergencia** del modelo. Esto valida la elección de PPO en esta tesis por su eficiencia y estabilidad de entrenamiento [13].

## Referencias Bibliográficas (Fuentes Web)
1. [Comparative Analysis of RL Algorithms in SUMO](https://tcd.ie)
2. [Policy vs Value-based RL in Traffic Control](https://tib-op.org)
3. [RESCO: Benchmarking RL for Traffic Control](https://neurips.cc)
4. [Multi-Objective Multi-Agent DDPG](https://arxiv.org)
5. [Fuzzy Logic and DQN for Traffic Signal](https://mdpi.com)
6. [Multi-Modal Urban Traffic Control with DRL](https://tudelft.nl)
7. [Adaptive Traffic Signal Control using SUMO-RL](https://researchgate.net)
8. [EP-D3QN and PPO for Traffic Optimization](https://researchgate.net)
