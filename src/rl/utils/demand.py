from __future__ import annotations

import math
import yaml
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class DemandProfile:
    seed: int
    type: str  # 'static' | 'dynamic'
    vph: Optional[Dict[str, float]] = None
    base_vph: Optional[Dict[str, float]] = None
    windows: Optional[list] = None


class DemandManager:
    def __init__(self, profile_path: str):
        with open(profile_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.profile = DemandProfile(**data)
        self.rng = np.random.default_rng(self.profile.seed)

    def get_vph(self, t_s: int) -> Dict[str, float]:
        if self.profile.type == "static":
            return dict(self.profile.vph or {})
        # dynamic
        v = dict(self.profile.base_vph or {})
        for w in (self.profile.windows or []):
            if int(w["begin"]) <= t_s < int(w["end"]):
                dv = w.get("delta_vph", {})
                for rid, delta in dv.items():
                    v[rid] = max(0.0, float(v.get(rid, 0.0)) + float(delta))
        return v

    def maybe_inject(self, traci, t_s: int) -> Dict[str, int]:
        """Inyecta vehículos según Poisson por movimiento. Devuelve conteos por rid.
        Requiere que existan rutas 'E2W','E2N','S2N','S2W'.
        """
        vph = self.get_vph(t_s)
        counts: Dict[str, int] = {}
        for rid, rate in vph.items():
            lam = max(0.0, float(rate)) / 3600.0  # por segundo
            k = self.rng.poisson(lam)
            counts[rid] = int(k)
            for j in range(int(k)):
                vid = f"{rid}_{t_s}_{j}"
                try:
                    traci.vehicle.add(vid, rid, typeID="car", departLane="random", departSpeed="max")
                except Exception:
                    pass
        return counts





































