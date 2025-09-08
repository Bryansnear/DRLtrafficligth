## DemandManager (demanda estática y dinámica)

Objetivo: separar la definición de demanda del escenario y soportar variaciones realistas en el tiempo.

### Modos
- Estático (flows): tasas fijas en `.rou.xml` (más rápido y muy reproducible).
- Dinámico (TraCI): tasas por movimiento que cambian en el tiempo (picos, ondas, ruido, incidentes). Inyección Poisson por segundo.

### Movimientos soportados (red actual)
- E→W (recto), E→N (giro), S→N (recto), S→W (giro). IDs de rutas: `E2W`, `E2N`, `S2N`, `S2W`.

### Formato YAML de demanda
```yaml
# experiments/demands/static_medium.yaml
seed: 42
type: static          # static | dynamic
vph:                   # vehículos por hora por movimiento
  E2W: 900
  E2N: 400
  S2N: 800
  S2W: 400

# experiments/demands/peak_am_pm.yaml
seed: 42
type: dynamic         # usa ventanas
base_vph:
  E2W: 700
  E2N: 300
  S2N: 700
  S2W: 300
windows:              # escalones temporales (segundos)
  - begin: 0
    end: 900
    delta_vph:
      E2W: 0
      E2N: 0
      S2N: 0
      S2W: 0
  - begin: 900        # pico AM
    end: 1800
    delta_vph:
      E2W: 400
      E2N: 200
      S2N: 400
      S2W: 200
  - begin: 1800       # valle
    end: 2700
    delta_vph:
      E2W: -200
      E2N: -100
      S2N: -200
      S2W: -100
```

### Funcionamiento
- Cada segundo \(\Delta t=1\,s\): para cada movimiento `rid` se generan \(k \sim \mathrm{Poisson}(\lambda \Delta t)\), con \(\lambda=\mathrm{vph}/3600\).
- Se añaden \(k\) vehículos con `traci.vehicle.add(vid, rid, departLane='random', departSpeed='max')`.
- `vph` proviene de `base_vph` + sumatoria de `delta_vph` según la ventana vigente (en `type: dynamic`).

### API (src/rl/utils/demand.py)
- `DemandManager(profile_path: str, rng_seed: int)`
  - `get_vph(t_s: int) -> dict[rid, float]`
  - `maybe_inject(traci, t_s: int) -> dict[rid, int]`  (retorna cuántos se inyectaron por movimiento)
- El entorno llama a `maybe_inject(...)` en cada paso de simulación si `dynamic_demand=True`.

### Integración en el entorno
- `EnvConfig` incluye `demand_profile_path` y `dynamic_demand`.
- En `start()`: se crean rutas `E2W`, `E2N`, `S2N`, `S2W` si no existen.
- En `step()`: dentro del bucle de \(\Delta T_c\) se llama `demand_manager.maybe_inject(...)` en cada segundo.

### Buenas prácticas
- Mantener `seed` fijo para reproducibilidad.
- Escalar `vph` según la longitud del episodio; evitar saturaciones irrealistas.
- Registrar `vph` efectivos por ventana en el log del experimento.









