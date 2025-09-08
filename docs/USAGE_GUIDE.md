# 📚 GUÍA DE USO - Control de Tráfico con RL

## 🚀 Scripts Principales

### 1. Autotuning Optimizado (`ultimate_autotuning.py`)
**Optimiza hiperparámetros y mantiene solo la mejor configuración**

```bash
# Configurar SUMO
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"

# Ejecutar autotuning (50 trials, 4 horas max)
python scripts/ultimate_autotuning.py --trials 50 --timeout 14400

# Autotuning rápido (20 trials, 1 hora)
python scripts/ultimate_autotuning.py --trials 20 --timeout 3600 --sampler tpe
```

**Características:**
- ✅ **Auto-limpieza**: Elimina configuraciones y archivos .pkl antiguos
- ✅ **Solo guarda el mejor**: `experiments/configs/best_optimized.yaml`
- ✅ **Sin archivos basura**: Mantiene el proyecto limpio

---

### 2. Entrenamiento Ultimate (`ultimate_training.py`)
**Entrena con la mejor configuración encontrada**

```bash
# Entrenamiento completo (100K timesteps)
python scripts/ultimate_training.py --config experiments/configs/best_optimized.yaml --timesteps 100000 --model-name ppo_production

# Entrenamiento rápido para pruebas
python scripts/ultimate_training.py --config experiments/configs/base.yaml --timesteps 10000 --model-name ppo_test
```

---

### 3. Evaluación Comparativa (`comparative_evaluation.py`)
**Compara 2 baselines fijos vs modelos entrenados**

```bash
# Evaluación automática (encuentra todos los modelos)
python scripts/comparative_evaluation.py --steps 30 --output results.csv

# Evaluación específica de modelos
python scripts/comparative_evaluation.py --models models/ppo_production.zip models/ppo_test.zip --steps 50

# Con interfaz gráfica SUMO
python scripts/comparative_evaluation.py --gui --steps 20
```

**Baselines incluidos:**
- 🚦 **Baseline Fijo (Flujo Fijo)**: Semáforo fijo con demanda constante
- 🚦 **Baseline Fijo (Flujo Variable)**: Semáforo fijo con demanda variable

**Métricas evaluadas:**
- 🚗 **Throughput**: Vehículos servidos por intervalo
- 🚦 **Colas**: Longitud promedio de colas
- 🔄 **Switches**: Frecuencia de cambios de semáforo
- 🎯 **Recompensa**: Score del modelo RL
- 📊 **Estabilidad**: Variabilidad en el rendimiento

---

### 4. Evaluación Simple (`eval.py`)
**Evaluación individual de un modelo**

```bash
# Evaluar modelo específico
python scripts/eval.py --model models/ppo_production.zip --steps 25

# Con GUI para visualización
python scripts/eval.py --model models/ppo_production.zip --gui --steps 15
```

---

## 🔄 Flujo de Trabajo Completo

### Paso 1: Optimización de Hiperparámetros
```bash
# Buscar los mejores hiperparámetros
python scripts/ultimate_autotuning.py --trials 30 --timeout 7200
```
**Resultado**: `experiments/configs/best_optimized.yaml` (solo el mejor)

### Paso 2: Entrenamiento del Modelo Final
```bash
# Entrenar con la mejor configuración
python scripts/ultimate_training.py --config experiments/configs/best_optimized.yaml --timesteps 100000 --model-name ppo_final
```
**Resultado**: `models/ppo_final.zip` + metadatos

### Paso 3: Evaluación Comparativa
```bash
# Comparar con baseline y otros modelos
python scripts/comparative_evaluation.py --steps 50 --output final_comparison.csv
```
**Resultado**: Tabla comparativa con mejoras porcentuales

---

## 📊 Interpretación de Resultados

### Métricas Clave:
- **avg_served**: ↗️ Más alto es mejor (throughput)
- **avg_queues**: ↘️ Más bajo es mejor (menos congestión)
- **switch_rate**: ↘️ Más bajo es mejor (menos cambios innecesarios)
- **avg_reward**: ↗️ Más alto es mejor (optimización RL)

### Ejemplo de Output:
```
📊 RESULTADOS COMPARATIVOS:
====================================================================================================
+--------------------------------+-----------------+---------------+----------------+--------------+
| Método                         |   Veh/Intervalo |   Colas Prom. |   Cambios/Int. |   Recompensa |
+================================+=================+===============+================+==============+
| Baseline Fijo (Flujo Fijo)     |             8.5 |          12.3 |           0.15 |       -0.234 |
+--------------------------------+-----------------+---------------+----------------+--------------+
| Baseline Fijo (Flujo Variable) |             7.2 |          15.1 |           0.15 |       -0.298 |
+--------------------------------+-----------------+---------------+----------------+--------------+
| PPO (ppo_final)                |            11.2 |           6.8 |           0.08 |        0.456 |
+--------------------------------+-----------------+---------------+----------------+--------------+

📈 MEJORAS RESPECTO A BASELINE FLUJO FIJO:
====================================================================================================
+------------------------------+---------------+------------------+-------------+---------------+
| Método                       | Veh/Int (%)   | Recompensa (%)   | Colas (%)   | Cambios (%)   |
+==============================+===============+==================+=============+===============+
| PPO (ppo_final)              | +31.8%        | +195.3%          | +44.7%      | +47.1%        |
+------------------------------+---------------+------------------+-------------+---------------+

📈 MEJORAS RESPECTO A BASELINE FLUJO VARIABLE:
====================================================================================================
+------------------------------+---------------+------------------+-------------+---------------+
| Método                       | Veh/Int (%)   | Recompensa (%)   | Colas (%)   | Cambios (%)   |
+==============================+===============+==================+=============+===============+
| PPO (ppo_final)              | +55.6%        | +252.7%          | +55.0%      | +46.7%        |
+------------------------------+---------------+------------------+-------------+---------------+

🏆 MEJOR MODELO:
============================================================
+-------------------+--------------------+
| Métrica           | Valor              |
+===================+====================+
| Modelo            | PPO (ppo_final)    |
+-------------------+--------------------+
| Score Compuesto   | 8.456              |
+-------------------+--------------------+
| Throughput        | 11.20 veh/intervalo|
+-------------------+--------------------+
| Colas Promedio    | 6.80               |
+-------------------+--------------------+
| Recompensa        | 0.456              |
+-------------------+--------------------+
| Cambios/Intervalo | 0.08               |
+-------------------+--------------------+
```

---

## 🛠️ Configuración del Sistema

### Requisitos Previos:
1. **SUMO instalado** en `C:\Program Files (x86)\Eclipse\Sumo`
2. **Python 3.8+** con dependencias instaladas
3. **Variables de entorno** configuradas:
   ```bash
   $env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
   $env:PATH += ";$env:SUMO_HOME\bin"
   ```

### Estructura de Archivos Importante:
```
Cursor-gpt5/
├── experiments/configs/
│   ├── base.yaml              # Configuración base
│   └── best_optimized.yaml    # Mejor configuración (auto-generada)
├── models/                    # Modelos entrenados (.zip)
├── scripts/                   # Scripts principales
└── data/sumo/                 # Datos de simulación SUMO
```

---

## 🎯 Tips de Uso

1. **Autotuning**: Ejecuta con `--trials 20-50` para balance tiempo/calidad
2. **Entrenamiento**: Usa 50K-100K timesteps para modelos de producción
3. **Evaluación**: 30-50 steps dan resultados estables
4. **Limpieza**: Los scripts se auto-limpian, no acumulan archivos basura
5. **Reproducibilidad**: Los metadatos guardan toda la info necesaria

---

## 🚨 Solución de Problemas

### Error: "SUMO_HOME no está definido"
```bash
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
```

### Error: "No module named 'gymnasium'"
```bash
pip install -r requirements.txt
```

### Error: "peer shutdown" (SUMO)
- Es normal al final de la simulación, no afecta los resultados
- El modelo se guarda correctamente antes del error

### Sin modelos para evaluar:
- Primero entrena al menos un modelo con `ultimate_training.py`
- Los modelos se guardan en `models/` con extensión `.zip`
