"""
Visualización en GUI de SUMO: Fixed Time vs PPO Optimized
Permite ver cómo cada agente controla el semáforo en tiempo real
"""
import os
import sys
from pathlib import Path
from stable_baselines3 import PPO
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rl.env.traffic_light_env import TrafficLightGymEnv, EnvConfig


def visualize_agent(model, env, name="Agent", max_steps=720):
    """
    Visualize agent controlling traffic in SUMO GUI.
    
    Args:
        model: PPO model or "fixed" for fixed-time controller
        env: TrafficLightGymEnv with GUI enabled
        name: Agent name for display
        max_steps: Maximum steps to run (default 720 = 1 hour)
    """
    print("\n" + "="*80)
    print(f"VISUALIZANDO: {name}")
    print("="*80)
    print(f"Control: SUMO GUI abierto")
    print(f"Duración: {max_steps} pasos (~{max_steps*5/60:.0f} minutos simulados)")
    print(f"Presiona Ctrl+C para detener")
    print("="*80 + "\n")
    
    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    switches = 0
    prev_phase = None
    
    total_served = 0
    total_queue = 0
    
    try:
        while not (done or truncated) and steps < max_steps:
            # Get action
            if model == "fixed":
                # Fixed time: Switch every 30s (6 steps of 5s)
                if (steps * 5) % 30 == 0 and steps > 0:
                    action = 1  # SWITCH
                else:
                    action = 0  # HOLD
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Track metrics
            total_served += info.get('served_cnt', 0)
            queues_dict = info.get('queues', {})
            total_queue += sum(queues_dict.values())
            
            # Count switches
            current_phase, _ = env.core.tls_state()
            if prev_phase is not None and current_phase != prev_phase and current_phase in (0, 2):
                switches += 1
            prev_phase = current_phase
            
            steps += 1
            
            # Print progress every 60 steps (5 minutes)
            if steps % 60 == 0:
                elapsed_min = steps * 5 / 60
                avg_queue = total_queue / steps
                print(f"[{elapsed_min:.0f} min] Switches: {switches}, Avg Queue: {avg_queue:.1f} veh")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Deteniendo simulación...")
    
    # Final stats
    duration_hours = (steps * 5) / 3600.0
    throughput = total_served / duration_hours if duration_hours > 0 else 0
    avg_queue = total_queue / steps if steps > 0 else 0
    
    print("\n" + "="*80)
    print(f"RESUMEN - {name}")
    print("="*80)
    print(f"Pasos simulados: {steps} ({steps*5/60:.1f} minutos)")
    print(f"Cambios de fase: {switches}")
    print(f"Throughput: {throughput:.1f} veh/h")
    print(f"Cola promedio: {avg_queue:.1f} veh")
    print("="*80 + "\n")


def main():
    """Run visualization for both agents sequentially."""
    
    SUMO_CFG = PROJECT_ROOT / "data" / "sumo" / "cfg" / "four_way_dynamic.sumo.cfg"
    MODEL_PATH = PROJECT_ROOT / "models" / "ppo_optimized_final"
    
    print("\n" + "="*80)
    print("VISUALIZACIÓN COMPARATIVA: Fixed Time vs PPO Optimized")
    print("="*80)
    print("\nEste script te mostrará:")
    print("1. Fixed Time (30s por fase)")
    print("2. PPO Optimized (Optuna-tuned)")
    print("\nAmbos en el escenario dinámico con SUMO GUI.")
    print("="*80)
    
    # Ask user which to run first
    print("\n¿Qué agente quieres ver primero?")
    print("1. Fixed Time 30s")
    print("2. PPO Optimized")
    print("3. Ambos (Fixed primero, luego PPO)")
    
    choice = input("\nElige (1/2/3): ").strip()
    
    agents_to_run = []
    if choice == "1":
        agents_to_run = [("fixed", "Fixed Time 30s")]
    elif choice == "2":
        agents_to_run = [(MODEL_PATH, "PPO Optimized")]
    else:  # Default to both
        agents_to_run = [("fixed", "Fixed Time 30s"), (MODEL_PATH, "PPO Optimized")]
    
    # Run each agent
    for model_or_fixed, name in agents_to_run:
        # Create environment with GUI
        env_config = EnvConfig(
            sumo_cfg_path=str(SUMO_CFG),
            min_green=5,
            max_green=180,
            w_served=1.0,
            w_queue=1.0,
            w_unbalance=0.37,
            always_compute_metrics=True,
            use_gui=True,  # ENABLE GUI
            randomize_sumo_seed=False  # Same scenario for fair comparison
        )
        
        env = TrafficLightGymEnv(env_config)
        
        try:
            # Load model if PPO
            if model_or_fixed != "fixed":
                print(f"\nCargando modelo {name}...")
                model = PPO.load(model_or_fixed)
            else:
                model = "fixed"
            
            # Visualize
            visualize_agent(model, env, name=name, max_steps=360)  # 30 min default
            
        finally:
            env.close()
            time.sleep(1)  # Brief pause before next agent
        
        # Ask if continue to next agent
        if len(agents_to_run) > 1 and agents_to_run.index((model_or_fixed, name)) < len(agents_to_run) - 1:
            input("\nPresiona Enter para continuar con el siguiente agente...")
    
    print("\n✅ Visualización completada!")


if __name__ == "__main__":
    main()
