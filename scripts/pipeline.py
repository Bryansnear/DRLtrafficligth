import argparse
import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Ejecuta un comando en la terminal y maneja errores."""
    print(f"\n[PIPELINE] Ejecutando: {command}")
    try:
        # shell=True permite usar comandos como en la terminal
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] El comando falló con código de salida {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento y Evaluación de Tráfico")
    
    # Argumentos para entrenamiento
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML (ej: configs/final_optimized_ppo_min10.yaml)")
    parser.add_argument("--model-name", type=str, help="Nombre del modelo a guardar/cargar (ej: ppo_min10_final)")
    
    # Nuevos argumentos interactivos/opcionales
    parser.add_argument("--timesteps", type=int, help="Total de timesteps para entrenar (sobreescribe config)")
    parser.add_argument("--n-envs", type=int, help="Número de entornos paralelos (sobreescribe config)")
    
    # Modos de ejecución
    parser.add_argument("--steps", type=str, default="train,eval", 
                        help="Pasos a ejecutar separados por coma. Opciones: train, eval. Default: train,eval")
    
    args = parser.parse_args()
    
    # Rutas base
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    train_script = scripts_dir / "train.py"
    eval_script = scripts_dir / "evaluate_min_green.py"
    
    steps = args.steps.split(",")
    
    # 1. ENTRENAMIENTO
    if "train" in steps:
        if not args.config or not args.model_name:
            print("[ERROR] Para entrenar necesitas especificar --config y --model-name")
            sys.exit(1)
            
        print("\n" + "="*60)
        print(" 🚀 INICIANDO FASE DE ENTRENAMIENTO")
        print("="*60)
        
        # Interactivo: Si no se pasaron argumentos, preguntar (opcional)
        timesteps = args.timesteps
        n_envs = args.n_envs
        
        if timesteps is None:
            try:
                print(f"¿Timesteps totales? (Enter para usar default del config): ", end="", flush=True)
                # Usamos sys.stdin.readline para evitar problemas en algunos entornos
                user_input = sys.stdin.readline().strip()
                if user_input:
                    timesteps = int(user_input)
            except ValueError:
                print("Entrada inválida, usando default.")

        if n_envs is None:
            try:
                print(f"¿Número de entornos (n_envs)? (Enter para usar default del config): ", end="", flush=True)
                user_input = sys.stdin.readline().strip()
                if user_input:
                    n_envs = int(user_input)
            except ValueError:
                print("Entrada inválida, usando default.")

        cmd_train = f"python \"{train_script}\" --config \"{args.config}\" --model-name \"{args.model_name}\""
        
        if timesteps:
            cmd_train += f" --timesteps {timesteps}"
        if n_envs:
            cmd_train += f" --n-envs {n_envs}"
            
        run_command(cmd_train)
        
    # 2. EVALUACIÓN
    if "eval" in steps:
        print("\n" + "="*60)
        print(" 📊 INICIANDO FASE DE EVALUACIÓN")
        print("="*60)
        
        cmd_eval = f"python \"{eval_script}\""
        
        # Si se entrenó un modelo o se pasó un nombre, pasarlo como extra para comparar
        if args.model_name:
             model_path = project_root / "models" / args.model_name
             cmd_eval += f" --extra-model \"{model_path}\" --extra-name \"{args.model_name}\""
        
        run_command(cmd_eval)

if __name__ == "__main__":
    main()
