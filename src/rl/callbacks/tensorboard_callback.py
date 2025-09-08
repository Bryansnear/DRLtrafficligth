from typing import Dict, Any

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardKpiCallback(BaseCallback):
    """Registra KPIs simples en TensorBoard a partir de info del env.

    Espera que info contenga claves como 'served_cnt', 'queues', 'switch'.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos or len(infos) == 0:
            return True
        info = infos[0]  # DummyVecEnv: primer env
        try:
            if "served_cnt" in info:
                self.logger.record("kpi/served", float(info["served_cnt"]))
            if "switch" in info:
                self.logger.record("kpi/switch", float(info["switch"]))
            if "queues" in info and isinstance(info["queues"], dict):
                qsum = sum(float(v) for v in info["queues"].values())
                self.logger.record("kpi/queues_sum", qsum)
        except Exception:
            pass
        return True





