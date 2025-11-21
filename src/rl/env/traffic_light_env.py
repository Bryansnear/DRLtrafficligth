from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore
from src.rl.utils.demand import DemandManager


@dataclass
class EnvConfig:
    sumo_cfg_path: str
    step_length: float = 1.0
    control_interval: int = 5
    min_green: int = 5
    yellow: int = 3
    all_red: int = 1
    max_green: int = 60
    e2_ids: Sequence[str] = ("E_in_0", "E_in_1", "S_in_0", "S_in_1")
    v_free: float = 13.9
    jam_length_thr_m: float = 20.0
    e2_capacity_per_lane: int = 6
    use_gui: bool = False
    tripinfo_output: Optional[str] = None
    demand_profile_path: Optional[str] = None
    dynamic_demand: bool = False
    sumo_port: Optional[int] = None  # Puerto único para entornos paralelos
    randomize_sumo_seed: bool = False  # ⭐ Aleatorizar seed de SUMO en cada reset
    # Recompensa
    w_served: float = 0.25
    w_queue: float = 0.40
    w_backlog: float = 0.30
    w_switch: float = 0.05
    w_spill: float = 0.20
    kappa_backlog: int = 10
    sat_headway_s: float = 2.0  # s/veh a saturación
    # Pesos adicionales
    w_invalid_action: float = 0.0
    w_unbalance: float = 0.0
    w_select: float = 0.0


def ensure_sumo_tools_on_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME no está definido. Configúralo para usar TraCI."
        )
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)


class TrafficLightEnv:
    """Entorno mínimo para inspección y posterior integración con Gym.

    - Observación (por carril): cola_norm, ocupacion, vel_norm, spillback + TLS (fase, tiempo_en_fase).
    - Acción: hold/switch (no implementada aquí todavía).
    - Recompensa: cálculo según docs (se añadirá en integración Gym).
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._traci = None
        self._label: str = f"env-{os.getpid()}-{id(self)}"
        self._tls_id: Optional[str] = None
        self._phase_index: Optional[int] = None
        self._last_phase_change: float = 0.0
        self._demand: Optional[DemandManager] = None
        self._current_sumo_seed: Optional[int] = None  # ⭐ Seed actual de SUMO

    def start(self) -> None:
        ensure_sumo_tools_on_path()
        from sumolib import checkBinary  # type: ignore
        import traci  # type: ignore

        sumo_binary = checkBinary("sumo-gui" if self.cfg.use_gui else "sumo")
        args = [
            sumo_binary,
            "-c",
            self.cfg.sumo_cfg_path,
            "--step-length",
            str(self.cfg.step_length),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--no-warnings",
            "true",
        ]
        if self.cfg.tripinfo_output:
            args += ["--tripinfo-output", self.cfg.tripinfo_output]
        
        # Usar puerto único si está especificado
        if self.cfg.sumo_port is not None:
            args += ["--remote-port", str(self.cfg.sumo_port)]
        
        # ⭐ Agregar seed aleatorio si está habilitado
        if self._current_sumo_seed is not None:
            args += ["--seed", str(self._current_sumo_seed)]
        
        traci.start(args, label=self._label)
        self._traci = traci
        traci.switch(self._label)
        tls_ids = list(traci.trafficlight.getIDList())
        self._tls_id = tls_ids[0] if tls_ids else None
        self._phase_index = traci.trafficlight.getPhase(self._tls_id) if self._tls_id else None
        self._last_phase_change = traci.simulation.getTime()
        # Demand profile
        if self.cfg.demand_profile_path:
            self._demand = DemandManager(self.cfg.demand_profile_path)
            # Asegura rutas
            try:
                self._traci.route.add("E2W", ["E_to_C", "C_to_W"])  # type: ignore
                self._traci.route.add("E2N", ["E_to_C", "C_to_N"])  # type: ignore
                self._traci.route.add("S2N", ["S_to_C", "C_to_N"])  # type: ignore
                self._traci.route.add("S2W", ["S_to_C", "C_to_W"])  # type: ignore
            except Exception:
                pass

    def close(self) -> None:
        if self._traci is not None:
            try:
                self._traci.switch(self._label)
                # Intentar cerrar limpiamente
                try:
                    self._traci.close()
                except Exception:
                    # Si falla close normal, intentar close(False)
                    try:
                        self._traci.close(False)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                self._traci = None

    def step_until(self, until_time: float) -> None:
        assert self._traci is not None
        self._traci.switch(self._label)
        while self._traci.simulation.getTime() < until_time:
            self._traci.simulationStep()
            if self._tls_id is not None:
                phase_now = self._traci.trafficlight.getPhase(self._tls_id)
                if self._phase_index is None:
                    self._phase_index = phase_now
                    self._last_phase_change = self._traci.simulation.getTime()
                elif phase_now != self._phase_index:
                    self._phase_index = phase_now
                    self._last_phase_change = self._traci.simulation.getTime()

    def read_e2_features(self) -> Dict[str, Dict[str, float]]:
        """Lee métricas E2 por carril."""
        assert self._traci is not None
        self._traci.switch(self._label)
        feats: Dict[str, Dict[str, float]] = {}
        for eid in self.cfg.e2_ids:
            queue = self._traci.lanearea.getLastStepHaltingNumber(eid)
            occ_pct = self._traci.lanearea.getLastStepOccupancy(eid)
            occ_frac = float(occ_pct) / 100.0
            mean_v = self._traci.lanearea.getLastStepMeanSpeed(eid)
            jam_m = self._traci.lanearea.getJamLengthMeters(eid)
            cola_norm = min(1.0, float(queue) / float(self.cfg.e2_capacity_per_lane))
            vel_norm = 0.0 if mean_v < 0 else min(1.0, float(mean_v) / self.cfg.v_free)
            spill = 1.0 if (occ_frac > 0.90 or jam_m > self.cfg.jam_length_thr_m) else 0.0
            feats[eid] = {
                "cola_norm": cola_norm,
                "ocupacion": occ_frac,
                "vel_norm": vel_norm,
                "spill": spill,
            }
        return feats

    def tls_state(self) -> Tuple[int, float]:
        assert self._traci is not None
        self._traci.switch(self._label)
        if self._tls_id is None:
            return -1, 0.0
        phase = self._traci.trafficlight.getPhase(self._tls_id)
        t = self._traci.simulation.getTime() - self._last_phase_change
        return phase, t


class TrafficLightGymEnv(gym.Env):
    """Entorno Gym con acción hold(0)/switch(1), observación y recompensa mejorada.

    Observación MEJORADA (24):
      - Por carril (E0,E1,S0,S1): [cola_norm, ocupacion, vel_norm, backlog_norm] → 16
      - TLS: fase one-hot (2) + tiempo_en_fase_norm (1) → 3
      - Spillback por carril (4) → 4
      - served_mask_sum (1) → 1
      Total: 24 features con información completa del estado
    """

    metadata = {"render.modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.core = TrafficLightEnv(cfg)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(24,), dtype=np.float32)
        # Estado interno
        self._prev_inside: Dict[str, set] = {eid: set() for eid in cfg.e2_ids}
        self._backlog: Dict[str, float] = {eid: 0.0 for eid in cfg.e2_ids}
        self._last_phase_idx: Optional[int] = None
        self._last_switch: int = 0
        self._phase_elapsed: float = 0.0

    def _served_lane_ids(self, phase_idx: int) -> List[str]:
        # 0: verde Este, 2: verde Sur (según red actual)
        if phase_idx == 0:
            return ["E_in_0", "E_in_1"]
        if phase_idx == 2:
            return ["S_in_0", "S_in_1"]
        # Si amarillo, asumimos la última verde
        if self._last_phase_idx in (0, 2):
            return self._served_lane_ids(self._last_phase_idx)  # type: ignore
        return []

    def _obs(self) -> np.ndarray:
        feats = self.core.read_e2_features()
        phase, tip = self.core.tls_state()
        # Por carril en orden fijo - AHORA CON BACKLOG
        order = list(self.cfg.e2_ids)
        vec: List[float] = []
        
        # Calcular capacidad máxima para normalización del backlog
        max_backlog = float(self.cfg.kappa_backlog * self.cfg.e2_capacity_per_lane)
        
        for eid in order:
            f = feats[eid]
            # Normalizar backlog por carril
            backlog_norm = min(1.0, self._backlog[eid] / max(1e-6, max_backlog))
            vec += [f["cola_norm"], f["ocupacion"], f["vel_norm"], backlog_norm]
        
        # TLS
        one_hot = [1.0, 0.0] if phase in (0, 1) else [0.0, 1.0]  # A(este) vs B(sur)
        tip_norm = min(1.0, tip / max(1e-6, float(self.cfg.min_green)))
        vec += one_hot + [tip_norm]
        
        # Spillback
        for eid in order:
            vec.append(feats[eid]["spill"])
        
        # served_mask_sum (ayuda a saber quién está en verde)
        served = self._served_lane_ids(phase)
        mask_sum = float(len(served)) / 2.0  # 0, 0.5, 1.0
        vec.append(mask_sum)
        
        return np.asarray(vec, dtype=np.float32)

    def _reward(self, served_cnt: float, queues: Dict[str, float], spills: Dict[str, float], switch_flag: int, phase_idx: int) -> float:
        # Normalizaciones
        served_lanes = self._served_lane_ids(phase_idx)
        n_served = max(1, len(served_lanes))
        s_max = (self.cfg.control_interval / max(1e-6, self.cfg.sat_headway_s)) * n_served
        s_norm = min(1.0, served_cnt / max(1e-6, s_max))
        q_sum = sum(queues.values())
        q_norm = min(1.0, q_sum / float(self.cfg.e2_capacity_per_lane * len(queues)))
        b_sum = sum(self._backlog.values())
        b_norm = min(1.0, b_sum / float(self.cfg.kappa_backlog * self.cfg.e2_capacity_per_lane * len(queues)))
        sp_sum = sum(spills.values())
        
        # Desequilibrio entre grupos (E vs S) normalizado
        group_e = [eid for eid in self.cfg.e2_ids if eid.startswith("E_")]
        group_s = [eid for eid in self.cfg.e2_ids if eid.startswith("S_")]
        backlog_e = sum(self._backlog.get(eid, 0.0) for eid in group_e)
        backlog_s = sum(self._backlog.get(eid, 0.0) for eid in group_s)
        denom_backlog = float(self.cfg.kappa_backlog * self.cfg.e2_capacity_per_lane * max(1, len(self.cfg.e2_ids)))
        unbalance = 0.0
        if denom_backlog > 0:
            unbalance = min(1.0, abs(backlog_e - backlog_s) / denom_backlog)
        
        # Selección por presión: premiar si el grupo servido tiene mayor backlog relativo
        selected_backlog = sum(self._backlog.get(eid, 0.0) for eid in served_lanes)
        other_lanes = [eid for eid in self.cfg.e2_ids if eid not in served_lanes]
        other_backlog = sum(self._backlog.get(eid, 0.0) for eid in other_lanes)
        select_pressure = 0.0
        if denom_backlog > 0:
            select_pressure = float(np.clip((selected_backlog - other_backlog) / denom_backlog, -1.0, 1.0))

        R = (
            + self.cfg.w_served * s_norm
            - self.cfg.w_queue * q_norm
            - self.cfg.w_backlog * b_norm
            - self.cfg.w_switch * float(switch_flag)
            - self.cfg.w_spill * sp_sum
            # Nuevos términos
            - self.cfg.w_unbalance * unbalance
            + self.cfg.w_select * select_pressure
        )
        return float(np.clip(R, -1.0, 1.0))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # ⭐ Generar seed aleatorio para SUMO si está habilitado
        if self.cfg.randomize_sumo_seed:
            # Usar np_random del gym environment para reproducibilidad
            self.core._current_sumo_seed = self.np_random.integers(0, 2**31 - 1)
        else:
            self.core._current_sumo_seed = None
        
        # Reiniciar SUMO
        try:
            self.core.close()
        except Exception:
            pass
        self.core.start()
        # Reiniciar tracking
        self._prev_inside = {eid: set() for eid in self.cfg.e2_ids}
        self._backlog = {eid: 0.0 for eid in self.cfg.e2_ids}
        self._last_phase_idx = self.core.tls_state()[0]
        self._last_switch = 0
        self._phase_elapsed = 0.0
        # Simular hasta el próximo límite de control para inicializar
        self.core.step_until(self.core._traci.simulation.getTime() + self.cfg.control_interval)
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action: int):
        # Decidir cambio (hold/switch) con min_green
        switch_flag = 0
        phase, _ = self.core.tls_state()
        elapsed = self._phase_elapsed if phase in (0, 2) else 0.0
        caution_duration = float(self.cfg.yellow + self.cfg.all_red)
        # Forzar cambio si se excede max_green o si la acción lo solicita (respetando min_green)
        do_switch = False
        invalid_action_flag = 0
        if phase in (0, 2):
            if elapsed >= self.cfg.max_green:
                do_switch = True
            elif action == 1 and elapsed >= self.cfg.min_green:
                do_switch = True
            elif action == 1 and elapsed < self.cfg.min_green:
                # Penalizar intento de cambio antes de min_green
                invalid_action_flag = 1

        if do_switch:
            # Secuencia segura: amarillo (fase 1 o 3) durante cfg.yellow, luego verde destino (0 o 2)
            target_green = 2 if phase == 0 else 0
            yellow_phase = 1 if phase in (0, 1) else 3
            try:
                # Forzar amarillo
                self.core._traci.trafficlight.setPhase(self.core._tls_id, yellow_phase)
                self.core._traci.trafficlight.setPhaseDuration(self.core._tls_id, 1e6)
                # Avanzar simulación durante el tiempo de amarillo + all_red
                end_caution = self.core._traci.simulation.getTime() + caution_duration
                while self.core._traci.simulation.getTime() < end_caution:
                    self.core._traci.simulationStep()
                # Cambiar a la fase verde objetivo
                self.core._traci.trafficlight.setPhase(self.core._tls_id, target_green)
                self.core._traci.trafficlight.setPhaseDuration(self.core._tls_id, 1e6)
                switch_flag = 1
            except Exception:
                pass
        else:
            # Mantener la fase actual fijando explícitamente duración del verde
            try:
                self.core._traci.trafficlight.setPhase(self.core._tls_id, phase)
                self.core._traci.trafficlight.setPhaseDuration(self.core._tls_id, 1e6)
            except Exception:
                pass

        # Acumular métricas por ΔTc
        acc_entered = {eid: 0 for eid in self.cfg.e2_ids}
        acc_exited = {eid: 0 for eid in self.cfg.e2_ids}
        end_t = self.core._traci.simulation.getTime() + self.cfg.control_interval
        while self.core._traci.simulation.getTime() < end_t:
            self.core._traci.simulationStep()
            # Inyección dinámica de demanda
            if self.cfg.dynamic_demand and self.core._traci is not None and self.core._demand is not None:
                t_s = int(self.core._traci.simulation.getTime())
                self.core._demand.maybe_inject(self.core._traci, t_s)
            # entered/exited por dif de conjuntos
            for eid in self.cfg.e2_ids:
                now_inside = set(self.core._traci.lanearea.getLastStepVehicleIDs(eid))
                entered = len(now_inside - self._prev_inside[eid])
                exited = len(self._prev_inside[eid] - now_inside)
                acc_entered[eid] += entered
                acc_exited[eid] += exited
                self._prev_inside[eid] = now_inside

        new_phase, _ = self.core.tls_state()
        if new_phase in (0, 2):
            if switch_flag:
                self._phase_elapsed = max(0.0, self.cfg.control_interval - caution_duration)
            else:
                self._phase_elapsed += self.cfg.control_interval
        else:
            self._phase_elapsed = 0.0
        # Al cierre del intervalo
        feats = self.core.read_e2_features()
        queues = {eid: self.core._traci.lanearea.getLastStepHaltingNumber(eid) for eid in self.cfg.e2_ids}
        spills = {eid: feats[eid]["spill"] for eid in self.cfg.e2_ids}
        # Actualizar backlog
        for eid in self.cfg.e2_ids:
            self._backlog[eid] += acc_entered[eid] - acc_exited[eid]
            self._backlog[eid] = max(0.0, self._backlog[eid])
        served_cnt = float(sum(acc_exited.values()))
        phase_now, _ = self.core.tls_state()
        reward = self._reward(served_cnt, queues, spills, switch_flag, phase_now) - self.cfg.w_invalid_action * float(invalid_action_flag)
        obs = self._obs()
        terminated = False
        truncated = False
        info = {
            "served_cnt": served_cnt,
            "queues": queues,
            "switch": switch_flag,
        }
        self._last_phase_idx = phase_now
        self._last_switch = switch_flag
        return obs, reward, terminated, truncated, info
    
    def close(self) -> None:
        """Cierra el environment y la conexión SUMO de forma limpia"""
        if hasattr(self, 'core') and self.core is not None:
            try:
                self.core.close()
            except Exception:
                pass

if __name__ == "__main__":
    # Demo mínima: correr 15s y leer una observación
    cfg = EnvConfig(sumo_cfg_path=os.path.join("data", "sumo", "cfg", "four_way_1h.sumo.cfg"))
    env = TrafficLightEnv(cfg)
    env.start()
    try:
        env.step_until(5)
        feats = env.read_e2_features()
        phase, tip = env.tls_state()
        print("Fase:", phase, "tiempo_en_fase:", round(tip, 1))
        for k, v in feats.items():
            print(k, v)
    finally:
        env.close()
