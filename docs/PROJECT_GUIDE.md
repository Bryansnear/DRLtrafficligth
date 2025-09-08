## Guía del proyecto: Control de semáforos con DRL + SUMO (estado y estructura)

Este documento resume la estructura del proyecto, qué hace cada carpeta/archivo, cómo ejecutar el escenario en SUMO y qué está implementado vs. pendiente. Úsalo como referencia rápida al retomar el trabajo o al abrir un nuevo chat.

### 1) Estructura de directorios (alto nivel)

```
data/
  sumo/
    network/
      four_way_cross.net.xml          # Red actual (intersección en cruz) lista para simular
      plain/                          # Fuente editable (nodos/edges) si se quiere reconstruir la red
        cross.nodes.nod.xml
        cross.edges.edg.xml
    routes/
      flows_cross_1h.rou.xml          # Rutas con <flow> (vehículos por hora) para 1h
    cfg/
      four_way_1h.sumo.cfg            # Configuración principal de SUMO (apunta a red/rutas/additional)
      run.log                         # (opcional) logs de ejecución cuando se corre con --log
    additional/
      detectors.add.xml               # Detectores E2 ("cámaras") 50 m antes del alto
      e2_*.xml                        # Salidas de los detectores (se generan al simular)

src/
  rl/
    env/                              # (pendiente) Entorno Gym (TrafficLightEnv)
    agents/                           # (pendiente) Definiciones PPO/algoritmos
    callbacks/                        # (pendiente) Callbacks de entrenamiento/evaluación
    utils/                            # (pendiente) Utilidades (TraCI helpers, mapeos, etc.)

experiments/
  configs/                            # (pendiente) YAMLs de hiperparámetros/escenarios
  runs/                               # (pendiente) Salidas de ejecuciones/experimentos

logs/
  tensorboard/                        # (pendiente) Logs de TB
  eval/                               # (pendiente) Resultados de evaluación

models/                               # (pendiente) Pesos entrenados (PPO)
outputs/
  reports/                            # (pendiente) Gráficos/tablas para el paper

notebooks/                            # (pendiente) Análisis exploratorio
scripts/                              # (pendiente) Utilidades CLI (train/eval/make_scenario)

docs/
  PROJECT_GUIDE.md                    # Este documento (fuente de verdad de estructura/estado)
  reward_spec.md                      # Especificación de recompensa/entrenamiento (detalle técnico)
```

### 2) Escenario SUMO actual (qué hay y cómo correr)

- Red: `data/sumo/network/four_way_cross.net.xml`
  - Intersección en cruz con TLS `C`.
  - 2 carriles por aproximación en E y S (entrantes), salidas hacia N y W.
  - Fases definidas en la red (static) y controlables vía TraCI (se usará con DRL).

- Rutas: `data/sumo/routes/flows_cross_1h.rou.xml`
  - Cuatro flujos (vehículos por hora) con comentarios de movimientos:
    - E → N (giro derecha), E → W (recto), S → N (recto), S → W (giro izquierda).
  - Puedes editar `vehsPerHour` o duplicar líneas con nuevas ventanas `begin/end`.

- Detectores (cámaras E2): `data/sumo/additional/detectors.add.xml`
  - Colocados ~50 m antes del alto en cada carril entrante: `E_to_C_0/1`, `S_to_C_0/1`.
  - Frecuencia 1 s. Salidas en `data/sumo/additional/e2_*.xml`.

- Config principal: `data/sumo/cfg/four_way_1h.sumo.cfg`
  - Apunta a red, rutas y additional.

Ejecutar:

```powershell
# GUI
sumo-gui -c data/sumo/cfg/four_way_1h.sumo.cfg

# Headless (prueba rápida 10 s)
sumo -c data/sumo/cfg/four_way_1h.sumo.cfg -Q --end 10
```

Visualizar salidas de detectores (ejemplo):

```powershell
Get-Content data\sumo\additional\e2_E_in_0.xml -Tail 20
# Convertir a CSV (requiere SUMO tools en PATH/SUMO_HOME)
python "%SUMO_HOME%/tools/xml/xml2csv.py" data/sumo/additional/e2_E_in_0.xml -o data/sumo/additional/e2_E_in_0.csv
```

Re‑generar la red desde los fuentes (si cambias nodes/edges):

```powershell
netconvert --node-files data/sumo/network/plain/cross.nodes.nod.xml `
          --edge-files data/sumo/network/plain/cross.edges.edg.xml `
          --output-file data/sumo/network/four_way_cross.net.xml --tls.guess
```

Re‑generar rutas con otra demanda (si prefieres trips aleatorios en vez de flows):

```powershell
python "%SUMO_HOME%/tools/randomTrips.py" -n data/sumo/network/four_way_cross.net.xml `
  -o data/sumo/routes/flows_cross_1h.rou.xml -b 0 -e 3600 --period 1.5 --seed 456 --fringe-factor 5
```

### 3) Estado del proyecto (hasta dónde vamos)

- Listo:
  - Estructura de carpetas organizada para SUMO y RL.
  - Red en cruz funcional y rutas por flujo (<flow>), con comentarios por movimiento.
  - Detectores E2 operativos y referenciados en `.sumocfg`.
  - Documento técnico de recompensa/entrenamiento: `docs/reward_spec.md`.

- Pendiente inmediato:
  - Implementar entorno Gym `TrafficLightEnv` (TraCI) con observación/acción/recompensa según la guía.
  - Script de entrenamiento `train_ppo.py` (Stable-Baselines3) + callbacks.
  - YAML de configuración de hiperparámetros/umbrales.

- Evoluciones opcionales del escenario:
  - Añadir aproximaciones N y W entrantes/salientes para cruz plenamente simétrica.
  - Programas TLS adicionales (yellow/all‑red explícitos en `additional`) si se requiere.
  - Inyección dinámica de demanda vía TraCI (Poisson por carril) para picos en tiempo real.

### 4) Decisiones y convenciones

- Referencia de carriles/edges:
  - Entrantes: `E_to_C_0/1`, `S_to_C_0/1`.
  - Salidas: `C_to_N_0/1`, `C_to_W_0/1`.
  - Los flows usan `departLane='random'` para distribuir en ambos carriles.

- Tiempos operativos recomendados:
  - `step_length`=1 s, `control_interval`=5 s, `min_green`≥5 s, `yellow`=3 s, `all_red`=1 s.

### 5) Enlace a especificación de recompensa

Para la fórmula exacta, normalizaciones, pseudocódigo y parámetros iniciales, ver:

- `docs/reward_spec.md` (mantener allí solo lo funcional de recompensa/entrenamiento).

### 6) Próximos pasos sugeridos (checklist)

- [ ] Crear `src/rl/env/traffic_light_env.py` (Gym + TraCI) siguiendo `reward_spec.md`.
- [ ] Añadir `experiments/configs/base.yaml` con pesos y tiempos.
- [ ] Implementar `scripts/train_ppo.py` y `scripts/eval.py`.
- [ ] Registrar KPIs (detectors + tripinfo) y visualización básica en `notebooks/`.



