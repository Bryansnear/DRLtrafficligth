#!/usr/bin/env python3
"""
Analizar el mejor checkpoint de todos los trials, no solo el mejor trial completo.
"""

import pickle
import os
import yaml
import argparse
from typing import Dict, Tuple, Optional


def load_study(study_path: str):
    """Cargar study de Optuna desde archivo pickle."""
    with open(study_path, "rb") as f:
        return pickle.load(f)


def find_best_checkpoint_across_all_trials(study) -> Tuple[Optional[object], float, int]:
    """
    Encontrar el mejor checkpoint de TODOS los trials (podados y completados).
    
    Returns:
        tuple: (best_trial, best_score, best_checkpoint_idx)
    """
    best_trial = None
    best_score = float('-inf')
    best_checkpoint_idx = -1
    
    print("🔍 ANALIZANDO TODOS LOS CHECKPOINTS...")
    print("=" * 60)
    
    for trial in study.trials:
        print(f"\nTrial {trial.number:2d} ({trial.state.name}):")
        
        # Analizar checkpoints intermedios
        if trial.intermediate_values:
            for checkpoint_idx, score in trial.intermediate_values.items():
                print(f"  Checkpoint {checkpoint_idx}: {score:.6f}")
                
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    best_checkpoint_idx = checkpoint_idx
                    print(f"    🏆 NUEVO MEJOR CHECKPOINT!")
        
        # Si es un trial completado, también verificar el valor final
        if hasattr(trial, 'value') and trial.value is not None:
            print(f"  Final score: {trial.value:.6f}")
            if trial.value > best_score:
                best_score = trial.value
                best_trial = trial
                best_checkpoint_idx = -1  # -1 indica score final, no checkpoint
                print(f"    🏆 NUEVO MEJOR SCORE FINAL!")
    
    return best_trial, best_score, best_checkpoint_idx


def create_ultimate_config_from_best_checkpoint(
    study, 
    cfg: dict, 
    best_trial, 
    best_score: float, 
    checkpoint_idx: int,
    output_path: str = "experiments/configs/ultimate_best_checkpoint.yaml"
) -> None:
    """Crear configuración usando parámetros del mejor checkpoint."""
    
    print(f"\n🎯 CREANDO CONFIGURACIÓN DEFINITIVA")
    print("=" * 50)
    print(f"🏆 Mejor score encontrado: {best_score:.6f}")
    print(f"📊 Trial origen: {best_trial.number}")
    print(f"⏱️ Checkpoint: {checkpoint_idx if checkpoint_idx >= 0 else 'Final'}")
    
    # Usar parámetros del mejor trial
    best_params = best_trial.params
    
    # Categorizar parámetros
    reward_params = ['w_served', 'w_queue', 'w_backlog', 'w_switch', 'w_spill', 'kappa_backlog', 'sat_headway_s']
    control_params = ['min_green', 'control_interval'] 
    ppo_params = ['learning_rate', 'gamma', 'clip_range', 'ent_coef']
    
    print("\n📊 MEJORES PARÁMETROS (del mejor checkpoint):")
    print("=" * 50)
    
    print("🎯 PESOS DE RECOMPENSA:")
    for param in reward_params:
        if param in best_params:
            print(f"   {param:20}: {best_params[param]}")
    
    print(f"\n🎮 PARÁMETROS DE CONTROL:")
    for param in control_params:
        if param in best_params:
            print(f"   {param:20}: {best_params[param]}")
    
    print(f"\n🤖 HIPERPARÁMETROS PPO:")
    for param in ppo_params:
        if param in best_params:
            value = best_params[param]
            if param == 'learning_rate' or param == 'ent_coef':
                print(f"   {param:20}: {value:.2e}")
            else:
                print(f"   {param:20}: {value}")
    
    # Crear configuración optimizada
    ultimate_cfg = yaml.safe_load(yaml.dump(cfg))
    
    # Aplicar mejores parámetros por categorías
    for param in reward_params:
        if param in best_params:
            ultimate_cfg['reward'][param] = best_params[param]
    
    for param in control_params:
        if param in best_params:
            ultimate_cfg['control'][param] = best_params[param]
    
    # Crear sección PPO optimizada
    ultimate_cfg['ppo_optimized'] = {}
    for param in ppo_params:
        if param in best_params:
            ultimate_cfg['ppo_optimized'][param] = best_params[param]
    
    # Agregar metadatos de optimización
    ultimate_cfg['optimization_metadata'] = {
        'source': 'best_checkpoint_analysis',
        'best_trial_number': best_trial.number,
        'best_checkpoint_idx': checkpoint_idx,
        'best_score': best_score,
        'total_trials_analyzed': len(study.trials),
        'note': f'Parameters from trial {best_trial.number} checkpoint {checkpoint_idx} with score {best_score:.6f}'
    }
    
    # Guardar configuración
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(ultimate_cfg, f, default_flow_style=False, indent=2)
    
    print(f"\n💾 Configuración definitiva guardada en: {output_path}")
    print(f"🚀 Listo para entrenar con los MEJORES parámetros encontrados!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analizar mejor checkpoint de todos los trials")
    parser.add_argument("--study", default="ultimate_autotuning_study.pkl", 
                       help="Archivo del study de Optuna")
    parser.add_argument("--config", default="experiments/configs/base.yaml",
                       help="Configuración base")
    parser.add_argument("--output", default="experiments/configs/ultimate_best_checkpoint.yaml",
                       help="Archivo de salida")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.study):
        print(f"❌ Error: No se encontró {args.study}")
        return 1
    
    if not os.path.exists(args.config):
        print(f"❌ Error: No se encontró {args.config}")
        return 1
    
    try:
        # Cargar study y configuración
        study = load_study(args.study)
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        print("🚀 ANÁLISIS DEL MEJOR CHECKPOINT DE TODOS LOS TRIALS")
        print("=" * 60)
        print(f"📁 Study: {args.study}")
        print(f"🧪 Total trials: {len(study.trials)}")
        
        # Encontrar mejor checkpoint global
        best_trial, best_score, best_checkpoint_idx = find_best_checkpoint_across_all_trials(study)
        
        if best_trial is None:
            print("❌ No se encontraron checkpoints válidos")
            return 1
        
        # Crear configuración con mejores parámetros
        output_path = create_ultimate_config_from_best_checkpoint(
            study, cfg, best_trial, best_score, best_checkpoint_idx, args.output
        )
        
        print(f"\n✅ ¡ANÁLISIS COMPLETADO!")
        print(f"🎯 Mejor score global: {best_score:.6f}")
        print(f"📊 Trial ganador: {best_trial.number}")
        print(f"🚀 Siguiente paso: python scripts/ultimate_training.py --config {output_path}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
