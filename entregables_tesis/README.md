# Entregables de Tesis: Optimización de Semáforos con DRL

Este directorio contiene todos los documentos y resultados generados durante el desarrollo de la tesis.

## Contenido

### Reportes Principales
*   **`THESIS_FULL_REPORT.md`**: Informe consolidado final. Incluye resumen ejecutivo, arquitectura, configuración experimental, resultados detallados y conclusiones. **Este es el documento principal.**
*   **`REPORTE_DESARROLLO.md`**: Bitácora histórica del desarrollo. Detalla los problemas encontrados (ej. "HOLD infinito"), las soluciones implementadas y la evolución de las métricas.
*   **`TRAINING_PLAN.md`**: Plan de entrenamiento original, detallando las fases de optimización con Optuna.

### Gráficos (`/graficos`)
Carpeta con todas las visualizaciones generadas en alta resolución:
*   `comparativa_tesis.png`: Barras comparando Baseline vs PPO.
*   `training_reward.png`: Curva de aprendizaje (Recompensa).
*   `training_entropy.png`: Evolución de la entropía.
*   `queue_distribution.png`: Histograma de colas.
*   `traffic_profile.png`: Perfil de demanda de tráfico (5 fases).
*   `action_frequency.png`: Frecuencia de cambios de fase.
*   `reward_composition.png`: Dinámica de la función de recompensa.
*   `training_value_loss.png`, `training_policy_loss.png`, `training_kl_div.png`: Métricas técnicas de entrenamiento.
*   `eval_boxplot_queues.png`: Diagrama de caja de variabilidad de colas.

## Instrucciones
Para reproducir estos resultados, referirse a los scripts en la carpeta raíz del proyecto:
1.  `scripts/train.py`: Para entrenar nuevos modelos.
2.  `scripts/evaluate.py`: Para evaluar modelos existentes.
3.  `scripts/generate_thesis_plots.py`: Para regenerar los gráficos.
