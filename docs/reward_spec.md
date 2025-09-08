## Especificación de la señal de entrenamiento y política de control (PPO + SUMO)

### 1) Contexto y objetivo
- Red: intersección en cruz con un TLS central (`C`) y 2 carriles por aproximación.
- Restricción realista: solo “cámara” por aproximación upstream (detectores E2 a ~50 m de la línea de alto).
- Objetivo: minimizar acumulación de cola y atascos, maximizando el servicio efectivo, sin parpadeo de fases.

### 2) Sensores y variables medibles (por carril entrante l)
- Desde E2 (freq=1 s):
  - `entraron_l(t)`: vehículos que entran al área en el paso t.
  - `salieron_l(t)`: vehículos que salen del área en el paso t (aprox. los que cruzan la línea de alto).
  - `cola_l(t)`: vehículos detenidos dentro del área.
  - `ocupacion_l(t)`, `longitud_atrasco_m_l(t)`, `velocidad_media_l(t)`.
- Capacidad geométrica por carril: \(cap_l = \left\lfloor \frac{L_{E2}}{L_{veh}} \right\rfloor\), con \(L_{E2} \approx 50\,m\), \(L_{veh} \approx 7.5\,m\) ⇒ \(cap_l \approx 6\).

### 3) Estado (observación) del entorno
Vector concatenado por carril entrante (orden fijo p.ej., `E_to_C_0, E_to_C_1, S_to_C_0, S_to_C_1`) más el estado del TLS:
- \(q^{norm}_l = \min\{1, \frac{cola_l}{cap_l}\}\)
- \(occ_l = ocupacion_l\) (en [0,1])
- \(v_l = \mathrm{clip}(velocidad\_media_l / v_{free}, 0, 1)\)
- One‑hot de la fase actual y \(t_{en\_fase}/t_{min\_green}\) (clipped a [0,1])
- Indicador de spillback: \(spill_l = \mathbb{1}[occ_l > 0.9 \;\lor\; longitud\_atrasco\_m_l > J_{thr}]\)

### 4) Acción
- Discreta: seleccionar bloque de fase en cada intervalo de control \(\Delta T_c\) (p.ej., 5 s):
  - Fase A: movimientos desde Este (recto + giro permitido).
  - Fase B: movimientos desde Sur (recto + giro permitido).
- Restricciones: \(t_{min\_green} \ge 5\,s\), \(t_{yellow}=3\,s\), \(t_{all\text{-}red}=1\,s\). Cambios respetan amarillo + all‑red.

### 5) Recompensa (solo con upstream)
Se usa una “presión sustituta” basada en backlog y servicio, sin requerir colas downstream.

Acumulado por carril: \(backlog_l(t) = backlog_l(t-1) + entraron_l(t) - salieron_l(t)\).

En cada intervalo de control (agregando los 1 s internos):
- \(S = \sum_l salieron_l\) (servicio total en el intervalo)
- \(Q = \sum_l cola_l\) (colas al final del intervalo)
- \(B = \sum_l backlog_l\) (acumulación)
- \(SW = \mathbb{1}[\text{hubo cambio de fase}]\)
- \(SP = \sum_l spill_l\)

Normalizaciones:
- \(S^{norm} = S / S_{max}\), con \(S_{max} = N_{carriles\_con\_verde} \cdot \Delta T_c / h_{sat}\). Aprox \(1\) si se sirve “saturado”.
- \(Q^{norm} = Q / (\sum_l cap_l)\).
- \(B^{norm} = B / (\kappa \sum_l cap_l)\) con \(\kappa\) ventana de escala (p.ej., 10 intervalos).

Fórmula de recompensa:
\[ R = + w_s\, S^{norm} \; - \; w_q\, Q^{norm} \; - \; w_b\, B^{norm} \; - \; w_{sw}\, SW \; - \; w_{sp}\, SP \]

Pesos iniciales (tunable):
- \(w_s=0.25,\; w_q=0.40,\; w_b=0.30,\; w_{sw}=0.05,\; w_{sp}=0.20\) con clipping de \(R\) a \([-1,1]\).

Razonamiento:
- Premia el servicio efectivo (vehículos que realmente cruzan la línea de alto).
- Castiga colas remanentes y acumulación (proxy de presión), evita parpadeo y penaliza spillback severo.

### 6) Temporalidad y agregación
- `step_length` = 1 s en SUMO; `control_interval` \(\Delta T_c\) = 5 s (aplicar acción y recompensa).
- Dentro de \(\Delta T_c\), acumular `entered/exited` y tomar `queue/occupancy/jam` del último step.

### 7) Pseudocódigo del cálculo por intervalo
```python
# inputs: per-lane time series within ΔTc → entered_l[], exited_l[], queue_l_end, occupancy_l_end, jam_l_end
served = sum(sum(exited_l_i for exited_l_i in exited_l[l]) for l in lanes)
queue = sum(queue_l_end[l] for l in lanes)
for l in lanes:
    backlog[l] += sum(entered_l[l]) - sum(exited_l[l])
backlog_sum = sum(backlog[l] for l in lanes)
spill = sum(int(occupancy_l_end[l] > 0.9 or jam_l_end[l] > J_thr) for l in lanes)

S_norm = served / S_max
Q_norm = queue / sum(cap_l[l] for l in lanes)
B_norm = backlog_sum / (kappa * sum(cap_l[l] for l in lanes))

R = + w_s*S_norm - w_q*Q_norm - w_b*B_norm - w_sw*switch - w_sp*spill
R = clip(R, -1.0, 1.0)
```

### 8) Espacio de observación (detalle)
- Por carril: \([q^{norm}_l,\; occupancy_l,\; v_l]\).
- TLS: one‑hot de fase actual, \(t_{en\_fase}/t_{min\_green}\).
- Indicadores: \(spill_l\) por carril.

### 9) PPO y entrenamiento
- Política MLP: [64, 128, 64], ReLU.
- Hiperparámetros iniciales: `lr=3e-4`, `gamma=0.99`, `n_steps=2048`, `batch_size=64`, `clip=0.2`, `ent_coef=0.01`.
- `control_interval=5 s`, `min_green=5 s`, `yellow=3 s`, `all_red=1 s`.
- Callbacks: evaluación periódica, early stopping por no‑mejora (p. ej., 10 evals).

### 10) Métricas para el paper
- Detector E2: `servicio` (salieron), `cola` (detenidos), `ocupacion`, `longitud_atrasco_m`, `velocidad_media` por carril y total.
- SUMO outputs (opcionales): `tripinfo.xml` (tiempo/retardo), `emission.xml` (CO2/NOx), throughput total.
- Baselines: plan fijo (ciclos iguales) y/o actuado por demanda simple.

### 11) Reproducibilidad y configuración
- Exponer en YAML: pesos \(w_s, w_q, w_b, w_{sw}, w_{sp}\), umbral `J_thr`, `control_interval`, `min_green`, `yellow`.
- Guardar semillas (`numpy`, `torch`, `SUMO`) y versiones (`SUMO_VERSION`, `sb3_version`).

### 12) Consideraciones prácticas
- Ubicar el E2 de modo que termine en la línea de alto para que `exited` ≈ vehículos servidos.
- Ajustar \(cap_l\) si cambia la longitud del E2.
- Si se agregan aproximaciones N/W y movimientos adicionales, ampliar el mapeo de carriles y mantener la misma fórmula.

### 13) Referencias (para redactar luego)
- Max‑Pressure/PressLight/MPLight; estudios con medición upstream‑only (añadir citas específicas).



