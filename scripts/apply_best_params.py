#!/usr/bin/env python3

import os
import pickle
import yaml
import argparse
from typing import Dict


def load_study(study_path: str):
    """Cargar study de Optuna desde archivo pickle."""
    with open(study_path, "rb") as f:
        return pickle.load(f)


def update_config(config_path: str, best_params: Dict, backup: bool = True) -> None:
    """Actualizar archivo de configuración con mejores parámetros."""
    
    # Backup del archivo original
    if backup:
        backup_path = config_path.replace('.yaml', '_backup.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            with open(backup_path, 'w', encoding='utf-8') as bf:
                bf.write(f.read())
        print(f"✅ Backup guardado en: {backup_path}")
    
    # Cargar configuración actual
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Aplicar mejores parámetros
    reward_params = ['w_served', 'w_queue', 'w_backlog', 'w_switch', 'w_spill', 'kappa_backlog']
    control_params = ['min_green', 'control_interval']
    
    print("🔧 Aplicando parámetros optimizados:")
    
    # Actualizar parámetros de recompensa
    for param in reward_params:
        if param in best_params:
            old_value = cfg['reward'].get(param, 'N/A')
            cfg['reward'][param] = best_params[param]
            print(f"  reward.{param:15}: {old_value} → {best_params[param]:.6f}")
    
    # Actualizar parámetros de control
    for param in control_params:
        if param in best_params:
            old_value = cfg['control'].get(param, 'N/A')
            cfg['control'][param] = best_params[param]
            print(f"  control.{param:13}: {old_value} → {best_params[param]}")
    
    # Agregar comentario de cuando fue optimizado
    import datetime
    cfg['# AUTO_TUNED'] = f"Optimizado automáticamente el {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Guardar configuración actualizada
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"✅ Configuración actualizada guardada en: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Aplicar mejores parámetros del autotuning")
    parser.add_argument("--study", default="autotuning_study.pkl", 
                       help="Archivo del study de Optuna (default: autotuning_study.pkl)")
    parser.add_argument("--config", default="experiments/configs/base.yaml",
                       help="Archivo de configuración a actualizar")
    parser.add_argument("--no-backup", action="store_true", 
                       help="No crear backup del archivo original")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.study):
        print(f"❌ Error: No se encontró el archivo {args.study}")
        print("   Ejecuta primero: python scripts/auto_tune.py")
        return 1
        
    if not os.path.exists(args.config):
        print(f"❌ Error: No se encontró el archivo de configuración {args.config}")
        return 1
    
    # Cargar study y obtener mejores parámetros
    study = load_study(args.study)
    best_params = study.best_params
    best_score = study.best_value
    
    print("🏆 APLICANDO MEJORES PARÁMETROS ENCONTRADOS")
    print("=" * 50)
    print(f"📊 Mejor score obtenido: {best_score:.6f}")
    print(f"🔧 Total de parámetros: {len(best_params)}")
    print(f"🧪 Evaluado en {len(study.trials)} trials")
    print()
    
    # Actualizar configuración
    update_config(args.config, best_params, backup=not args.no_backup)
    
    print()
    print("✅ ¡Parámetros aplicados exitosamente!")
    print(f"💡 Ahora puedes entrenar con: python scripts/train_ppo.py")
    
    return 0


if __name__ == "__main__":
    exit(main())




