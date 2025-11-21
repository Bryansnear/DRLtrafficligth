# Traffic Signal DRL (minimal)

Repositorio limpio para entrenar y evaluar un único agente PPO que controla dos semáforos en SUMO.  
Se eliminaron modelos históricos, logs, experimentos y documentación antigua para dejar solo lo esencial.

## Estructura

- `configs/` – YAML con los hiperparámetros del entorno (`reactive_random_500k.yaml`).
- `data/sumo/` – Escenario SUMO (red, rutas, detectores).
- `scripts/` – Tres entradas principales:
  - `train.py` – entrenamiento PPO.
  - `evaluate.py` – evaluación de baseline y modelo.
  - `pipeline.py` – orquesta entrenamiento + evaluación.
- `src/` – implementación del entorno Gym y utilidades.
- `requirements.txt` – dependencias.

## Instalación rápida

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 1. Entrenar

```bash
python scripts/train.py --config configs/reactive_random_500k.yaml --timesteps 500000
```

El modelo se guarda en `models/` (el script crea la carpeta automáticamente).

## 2. Evaluar

```bash
python scripts/evaluate.py --model models/ppo_tls.zip --config configs/reactive_random_500k.yaml --baseline --episodes 5
```

## Baseline fijo de referencia

Antes de comparar cualquier modelo nuevo, ejecuta el baseline de tiempo fijo para obtener el punto de referencia y guardarlo automáticamente en `outputs/baseline/baseline_results.csv`.

```bash
# Ejemplo: 35 s para la fase Este/Oeste y 25 s para la fase Norte/Sur
python scripts/evaluate.py --baseline --episodes 5 --phase-ew 35 --phase-ns 25 --baseline-tag baseline_35_25
```

Puedes repetir el comando con otras combinaciones de fases; cada corrida se anexa al CSV con métricas promedio.

## 3. Pipeline completo

```bash
python scripts/pipeline.py --config configs/reactive_random_500k.yaml --timesteps 500000 --episodes 10
```

### Parámetros útiles

- `--skip-training --model models/mi_modelo.zip` para evaluar un modelo existente.
- `--run-name custom_run` para nombrar carpetas y el modelo generado.

## Notas

- `models/`, `logs/` y `outputs/` se crean nuevamente cuando se ejecutan los scripts.
- El escenario SUMO se mantiene en `data/sumo`. Ajusta `configs/*.yaml` para cambiar recompensas, tiempos o rutas.
- Si necesitas nuevos experimentos o documentación extra, crea las carpetas correspondientes desde cero; el repositorio comienza ligero a propósito.
