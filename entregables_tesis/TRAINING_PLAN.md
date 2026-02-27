# Plan de Entrenamientos y Optuna (DRL Semáforos 2×2)

Este documento resume el plan de experimentos para mejorar el desempeño del agente DRL frente al baseline fijo de 25 s por fase.

## Fase 0 – Punto de partida (lo que ya tenemos)

- **Escenario:** `configs/reactive_random_500k.yaml`  
  - SUMO cfg: `data/sumo/cfg/four_way_1h_variable.sumo.cfg` (flujos variables + picos).
  - `train.max_episode_steps = 720` (1 h de simulación por episodio).
- **Baseline fijo seleccionado:** ciclo simétrico **25 s / 25 s**
  - Resultados guardados en `outputs/baseline/baseline_results.csv` (fila `sweep_25s`).
- **Métrica principal para comparar:**  
  - `avg_throughput` (avg_served / steps)  
  - Con restricción de colas: idealmente `avg_queue` ≤ ~3.8–4.0.

Scripts clave:

- `scripts/train.py` – Entrenamiento PPO de un modelo.
- `scripts/evaluate.py` – Evaluación baseline + DRL (con CSV de baseline).
- `scripts/pipeline.py` – Entrenamiento + evaluación en un solo comando.

Ejemplo de pipeline actual (modelo “simétrico”):

```bash
python scripts/pipeline.py \
  --config configs/reactive_random_500k.yaml \
  --timesteps 500000 \
  --episodes 10 \
  --run-name symm_reactive_500k
```

---

## Fase 1 – Optuna sobre recompensa y control (búsqueda rápida)

**Objetivo:** encontrar combinaciones prometedoras de parámetros de _control_ y _reward_ que aumenten throughput sin disparar las colas.

### Configuración de los trials

- **Timesteps por trial:** `100k` (rápido, aprox 20 % de un entrenamiento completo).
- **Entornos paralelos (n_envs):** 4–6 (menos ruido pero manteniendo estabilidad).
- **Episodios de evaluación por trial:** 5–7 episodios.
- **Aleatorización de tráfico (fase 1):**
  - En esta fase se puede usar `randomize_sumo_seed: false` para reducir ruido en la métrica.
  - Seeds fijos (`seeds.numpy`, `seeds.env`, `seeds.torch`) para reproducibilidad.

### Espacio de búsqueda sugerido (sobre el YAML)

Modificar vía Optuna estos campos de `configs/reactive_random_500k.yaml`:

- Bloque `control:`  
  - `min_green` ∈ [5, 15]  
  - `max_green` ∈ [30, 60]

- Bloque `reward:`  
  - `w_served` ∈ [1.0, 2.0]  
  - `w_queue` ∈ [0.8, 1.6]  
  - `w_backlog` ∈ [0.4, 0.8]  
  - `w_switch` ∈ [0.0005, 0.02]  
  - `w_spill` ∈ [0.6, 1.2]  
  - `w_unbalance` ∈ [0.0, 0.5] (opcional, equidad Este/Sur)

- Bloque `ppo:` (solo pequeños ajustes en esta fase, o fijar y dejar para Fase 3).

### Función objetivo (idea)

1. Para cada trial:
   - Generar un YAML temporal con los parámetros propuestos.
   - Entrenar `100k` timesteps (`scripts/train.py`).  
   - Evaluar el modelo resultante (`scripts/evaluate.py`) con:
     - `--baseline --phase-ew 25 --phase-ns 25 --episodes 5 --quiet`
   - Leer métricas desde el CSV y calcular objetivo:
     - Objetivo = `avg_throughput` – penalización si `avg_queue` > 4.0.

2. Guardar la mejor configuración de cada trial en `experiments/configs/` o directamente en `configs/` como, por ejemplo:
   - `configs/optuna_candidate_01.yaml`
   - `configs/optuna_candidate_02.yaml`

Número de trials recomendado para esta fase: **30–40**.

**Resultado de la fase 1:** lista de 3–5 configuraciones candidatas ordenadas por objetivo.

---

## Fase 2 – Reentrenamiento largo y robustez

**Objetivo:** comprobar que las mejores configs de Optuna siguen siendo buenas con más timesteps y tráfico aleatorio.

Para cada configuración candidata (por ejemplo `configs/optuna_candidate_01.yaml`):

1. **Ajustar YAML para entrenamiento real:**
   - `randomize_sumo_seed: true` (activar aleatorización de SUMO).
   - `train.total_timesteps` = `500k` o `1M` (según tiempo disponible).
   - `train.n_envs`: 8–12 (si la máquina lo permite).

2. **Entrenar con pipeline:**

   ```bash
   python scripts/pipeline.py \
     --config configs/optuna_candidate_01.yaml \
     --timesteps 500000 \
     --episodes 10 \
     --run-name optuna_cand1_500k
   ```

   Esto crea:
   - `models/ppo_optuna_cand1_500k.zip`
   - `outputs/optuna_cand1_500k/REPORT.md` + `config_used.yaml`

3. **Evaluar contra baseline 25 s:**

   ```bash
   python scripts/evaluate.py \
     --model models/ppo_optuna_cand1_500k.zip \
     --config configs/optuna_candidate_01.yaml \
     --episodes 10 \
     --baseline \
     --phase-ew 25 --phase-ns 25 \
     --baseline-tag fixed25_optuna_cand1
   ```

4. **Criterios para elegir “campeón” de Fase 2:**
   - Mayor `avg_throughput`.  
   - `avg_queue` no mucho más alta que el baseline (ej. < +10 %).  
   - Número de `switches` razonable (sin parpadeos excesivos).

**Resultado de la fase 2:** 1–2 configuraciones “campeonas” ya entrenadas largo y con resultados robustos.

---

## Fase 3 – Afinar PPO alrededor del campeón

**Objetivo:** exprimir un poco más el rendimiento ajustando solo hiperparámetros de PPO.

Partiendo de la mejor configuración de Fase 2 (por ejemplo `configs/optuna_best.yaml`):

- Congelar `sumo`, `control`, `detectors`, `reward`.  
- Variar únicamente:
  - `ppo.learning_rate` ∈ [1e-5, 5e-4] (log-uniform).  
  - `ppo.n_steps` ∈ {2048, 4096, 8192} (según VRAM).  
  - `ppo.batch_size` ∈ {1024, 2048, 4096}.  
  - `ppo.ent_coef` ∈ [0.0, 0.05].

Flujo de trabajo:

1. Crear un script Optuna específico para PPO (o reusar `optuna_optimize.py` con otro `study_name` y otro conjunto de parámetros).
2. Usar timesteps medios (`200k`) por trial para no gastar demasiado tiempo.
3. Tomar las 1–2 mejores combinaciones, reentrenar largo (`500k–1M`), y evaluar de nuevo con `evaluate.py` + baseline 25 s.

**Resultado de la fase 3:** modelo(s) final(es) listo(s) para comparar contra el baseline fijo y para reporte en la tesis (tablas, gráficos, etc.).

---

## Resumen rápido de comandos clave

- **Baseline fijo 25 s / 25 s (ya medido, repetir si es necesario):**

```bash
python scripts/evaluate.py --baseline \
  --episodes 5 \
  --phase-ew 25 --phase-ns 25 \
  --baseline-tag fixed25
```

- **Entrenamiento estándar con configuración actual:**

```bash
python scripts/pipeline.py \
  --config configs/reactive_random_500k.yaml \
  --timesteps 500000 \
  --episodes 10 \
  --run-name symm_reactive_500k
```

- **Evaluar un modelo concreto contra baseline:**

```bash
python scripts/evaluate.py \
  --model models/ppo_symm_reactive_500k.zip \
  --config configs/reactive_random_500k.yaml \
  --episodes 10 \
  --baseline \
  --phase-ew 25 --phase-ns 25 \
  --baseline-tag fixed25_symm_reactive
```

Con este plan, puedes ir marcando qué fase ya completaste y qué configuraciones fueron ganando, sin perderte entre tantos experimentos. Cuando estés listo, el siguiente paso será adaptar `scripts/optuna_optimize.py` para implementar exactamente este flujo de Optuna.
