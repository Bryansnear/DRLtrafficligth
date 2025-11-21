from typing import Dict, Any

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardKpiCallback(BaseCallback):
    """Registra KPIs simples en TensorBoard a partir de info del env.

    Espera que info contenga claves como 'served_cnt', 'queues', 'switch'.
    Calcula promedios de TODOS los envs paralelos.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.switch_buffer = []  # Acumula switches para calcular tasa promedio
        self.buffer_size = 100   # Promedio de últimos 100 steps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos or len(infos) == 0:
            return True
        
        try:
            # Calcular promedios de TODOS los envs (no solo el primero)
            n_envs = len(infos)
            
            # Served count promedio
            served_values = [float(info.get("served_cnt", 0)) for info in infos if "served_cnt" in info]
            if served_values:
                avg_served = sum(served_values) / len(served_values)
                self.logger.record("kpi/served", avg_served)
            
            # Switch rate promedio
            switch_values = [float(info.get("switch", 0)) for info in infos if "switch" in info]
            if switch_values:
                # Promedio instantáneo de este step
                instant_switch = sum(switch_values) / len(switch_values)
                
                # Acumular en buffer para tasa promedio
                self.switch_buffer.append(instant_switch)
                if len(self.switch_buffer) > self.buffer_size:
                    self.switch_buffer.pop(0)
                
                # Registrar ambos: instantáneo y promedio móvil
                avg_switch_rate = sum(self.switch_buffer) / len(self.switch_buffer)
                self.logger.record("kpi/switch_instant", instant_switch)
                self.logger.record("kpi/switch", avg_switch_rate)
            
            # Queue sum promedio
            queue_sums = []
            for info in infos:
                if "queues" in info and isinstance(info["queues"], dict):
                    qsum = sum(float(v) for v in info["queues"].values())
                    queue_sums.append(qsum)
            if queue_sums:
                avg_queue_sum = sum(queue_sums) / len(queue_sums)
                self.logger.record("kpi/queues_sum", avg_queue_sum)
                
        except Exception:
            pass
        return True





