import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración de estilo
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Datos (Recopilados de las pruebas recientes)
volumes = [1000, 1500, 2000]
baseline_throughput = [405, 560, 388]
drl_throughput = [367, 548, 439]

baseline_running = [36.2, 84.4, 663.8]
drl_running = [34.6, 101.5, 631.0]

output_dir = r"c:\Users\bryan\Maestria\Tesis\Desarollo\Cursor gpt5\entregables_tesis\graficos"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# GRÁFICO 1: Throughput Global vs Demanda
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
width = 0.35
x = np.arange(len(volumes))

plt.bar(x - width/2, baseline_throughput, width, label='Baseline (Semáforos Fijos)', color='#95a5a6')
plt.bar(x + width/2, drl_throughput, width, label='PPO Optimizado (Inteligente)', color='#e74c3c')

# Etiquetas de valor
for i, v in enumerate(baseline_throughput):
    plt.text(i - width/2, v + 10, str(v), ha='center', fontsize=10)
for i, v in enumerate(drl_throughput):
    diff = ((v - baseline_throughput[i]) / baseline_throughput[i]) * 100
    color = 'green' if diff > 0 else 'red'
    label = f"{v}\n({diff:+.1f}%)"
    plt.text(i + width/2, v + 10, label, ha='center', fontsize=10, color='black', fontweight='bold')

plt.xlabel('Demanda de Tráfico (Vehículos/Hora)')
plt.ylabel('Throughput (Vehículos que completaron ruta)')
plt.title('Impacto del PPO Optimizado en el Flujo Vehicular Global')
plt.xticks(x, [f"{v} veh/h\n(Fluido)", f"{v} veh/h\n(Denso)", f"{v} veh/h\n(Saturado)"])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "throughput_comparison.png"), dpi=300)
print("Generado: throughput_comparison.png")

# ---------------------------------------------------------
# GRÁFICO 2: Congestión
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(volumes, baseline_running, 'o--', label='Baseline', color='#95a5a6', linewidth=2)
plt.plot(volumes, drl_running, 'o-', label='PPO Optimizado', color='#3498db', linewidth=3)

# Anotación en el punto crítico
plt.annotate('Recuperación ante Saturación\n(-5% Autos Atascados)', 
             xy=(2000, 631), xytext=(1500, 500),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlabel('Demanda de Tráfico (Vehículos/Hora)')
plt.ylabel('Promedio de Vehículos en Red (Congestión)')
plt.title('Gestión de la Congestión Extrema')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "congestion_levels.png"), dpi=300)
print("Generado: congestion_levels.png")

# ---------------------------------------------------------
# GRÁFICO 3: Eficiencia Relativa
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
improvements = [((d - b)/b)*100 for b, d in zip(baseline_throughput, drl_throughput)]

colors = ['red' if x < 0 else 'green' for x in improvements]
plt.bar(x, improvements, color=colors, width=0.5)
plt.axhline(0, color='black', linewidth=1)

for i, v in enumerate(improvements):
    plt.text(i, v + (1 if v > 0 else -2), f"{v:+.1f}%", ha='center', fontweight='bold')

plt.xticks(x, [str(v) for v in volumes])
plt.xlabel('Demanda de Tráfico (Veh/h)')
plt.ylabel('% Cambio en Throughput vs Baseline')
plt.title('El Agente PPO Optimizado como "Gestor de Crisis"')
plt.suptitle('Mejora relativa del rendimiento a medida que aumenta el estrés', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "relative_improvement.png"), dpi=300)
print("Generado: relative_improvement.png")
