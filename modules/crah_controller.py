"""
modules/crah_controller.py
--------------------------
CRAH unit control with AI / SUPERVISED / LOCAL_AUTO mode switching.

Changes from v1:
  + ControlMode enum (AI, SUPERVISED, LOCAL_AUTO)
  + Per-unit mode tracking
  + revert_to_local_auto() fail-safe method
  + local_auto_step() -- simple hysteresis control when AI is offline
  + set_mode() / get_mode() interface for dashboard toggle
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from config import (
    DISCHARGE_TEMP_SETPOINT,
    FAN_SPEED_STEP,
    LOCAL_AUTO_AIRFLOW_CFM,
    LOCAL_AUTO_DISCHARGE_C,
    LOCAL_AUTO_FAN_SPEED_PCT,
    MAX_AIRFLOW_CFM,
    MAX_FAN_SPEED,
    MIN_AIRFLOW_CFM,
    MIN_FAN_SPEED,
    NUM_CRAH_UNITS,
    DEFAULT_CONTROL_MODE,
    TEMP_OPTIMAL_HIGH,
)


class ControlMode(str, Enum):
    AI          = "AI"          # Fully autonomous; AI issues all commands
    SUPERVISED  = "SUPERVISED"  # AI proposes; human approves before execution
    LOCAL_AUTO  = "LOCAL_AUTO"  # AI offline; each unit runs simple local control


class CRAHMode(str, Enum):
    NORMAL  = "NORMAL"
    BOOST   = "BOOST"
    ECONOMY = "ECONOMY"
    STANDBY = "STANDBY"


@dataclass
class CRAHState:
    unit_id:          int
    fan_speed_pct:    float
    airflow_cfm:      float
    discharge_temp_c: float
    setpoint_c:       float
    mode:             CRAHMode
    control_mode:     ControlMode = ControlMode.AI
    cumulative_kwh:   float = 0.0
    last_alarm:       str   = ""


class CRAHController:
    """
    Manages NUM_CRAH_UNITS CRAH units with mode-aware control.

    Usage
    -----
    ctrl = CRAHController()
    # Normal AI operation:
    states = ctrl.apply(target_airflows, target_discharges)
    # Fail-safe revert:
    ctrl.revert_to_local_auto()
    states = ctrl.apply_local_auto()
    """

    def __init__(self):
        nominal_speed = self._airflow_to_fan_speed(
            (MIN_AIRFLOW_CFM + MAX_AIRFLOW_CFM) / 2
        )
        self._states: list[CRAHState] = [
            CRAHState(
                unit_id=i,
                fan_speed_pct=nominal_speed,
                airflow_cfm=(MIN_AIRFLOW_CFM + MAX_AIRFLOW_CFM) / 2,
                discharge_temp_c=DISCHARGE_TEMP_SETPOINT,
                setpoint_c=DISCHARGE_TEMP_SETPOINT,
                mode=CRAHMode.NORMAL,
                control_mode=ControlMode(DEFAULT_CONTROL_MODE),
            )
            for i in range(NUM_CRAH_UNITS)
        ]
        self._global_mode = ControlMode(DEFAULT_CONTROL_MODE)

    # -- Conversion helpers ---------------------------------------------------

    @staticmethod
    def _airflow_to_fan_speed(airflow_cfm: float) -> float:
        ratio = (airflow_cfm - MIN_AIRFLOW_CFM) / max(MAX_AIRFLOW_CFM - MIN_AIRFLOW_CFM, 1)
        ratio = max(0.0, min(ratio, 1.0))
        pct = math.sqrt(ratio) * (MAX_FAN_SPEED - MIN_FAN_SPEED) + MIN_FAN_SPEED
        return round(min(MAX_FAN_SPEED, max(MIN_FAN_SPEED, pct)), 1)

    @staticmethod
    def _fan_speed_to_airflow(fan_speed_pct: float) -> float:
        norm = (fan_speed_pct - MIN_FAN_SPEED) / max(MAX_FAN_SPEED - MIN_FAN_SPEED, 1)
        norm = max(0.0, min(norm, 1.0))
        return round(MIN_AIRFLOW_CFM + (MAX_AIRFLOW_CFM - MIN_AIRFLOW_CFM) * norm ** 2, 1)

    @staticmethod
    def _classify_crah_mode(fan_speed_pct: float) -> CRAHMode:
        if fan_speed_pct >= 80: return CRAHMode.BOOST
        if fan_speed_pct <= 40: return CRAHMode.ECONOMY
        return CRAHMode.NORMAL

    def _ramp_fan_speed(self, current: float, target: float) -> float:
        diff = target - current
        if abs(diff) <= FAN_SPEED_STEP:
            return round(target, 1)
        return round(current + math.copysign(FAN_SPEED_STEP, diff), 1)

    # -- Mode Management ------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch all units to a new control mode. mode: 'AI' | 'SUPERVISED' | 'LOCAL_AUTO'"""
        try:
            new_mode = ControlMode(mode.upper())
        except ValueError:
            print(f"[CRAHController] Unknown mode: {mode}")
            return
        self._global_mode = new_mode
        for s in self._states:
            s.control_mode = new_mode
        print(f"[CRAHController] Mode changed to: {new_mode.value}")

    def get_mode(self) -> str:
        return self._global_mode.value

    def revert_to_local_auto(self) -> None:
        """
        Fail-safe: immediately revert all units to LOCAL_AUTO mode.
        Each unit will maintain safe fixed setpoints until AI recovers.
        """
        print("[CRAHController] FAIL-SAFE: Reverting all units to LOCAL_AUTO mode.")
        self.set_mode("LOCAL_AUTO")

    # -- AI / Supervised Control ----------------------------------------------

    def apply(
        self,
        target_airflows:   np.ndarray,
        target_discharges: np.ndarray,
        dt_seconds:        float = 3.0,
    ) -> list[CRAHState]:
        """Apply AI-generated targets to all units (ramp-limited)."""
        for i, state in enumerate(self._states):
            target_speed  = self._airflow_to_fan_speed(target_airflows[i])
            new_speed     = self._ramp_fan_speed(state.fan_speed_pct, target_speed)
            new_speed     = max(MIN_FAN_SPEED, min(MAX_FAN_SPEED, new_speed))
            actual_airflow= self._fan_speed_to_airflow(new_speed)

            target_dt = target_discharges[i]
            new_dt    = state.discharge_temp_c + (target_dt - state.discharge_temp_c) * 0.3

            power_kw   = ((new_speed / MAX_FAN_SPEED) ** 3) * 2.0
            energy_kwh = power_kw * (dt_seconds / 3600)

            state.fan_speed_pct    = new_speed
            state.airflow_cfm      = actual_airflow
            state.discharge_temp_c = round(new_dt, 2)
            state.setpoint_c       = round(target_dt, 2)
            state.mode             = self._classify_crah_mode(new_speed)
            state.cumulative_kwh   = round(state.cumulative_kwh + energy_kwh, 4)
        return self._states

    # -- Local Auto (Fail-Safe) -----------------------------------------------

    def apply_local_auto(
        self,
        rack_avg_temps: Optional[dict[int, float]] = None,
        dt_seconds: float = 3.0,
    ) -> list[CRAHState]:
        """
        Simple hysteresis-based local auto control for each unit.
        If rack_avg_temps provided: react to zone temperature.
        Otherwise: hold safe fixed setpoints.
        """
        for i, state in enumerate(self._states):
            if rack_avg_temps and i in rack_avg_temps:
                zone_avg = rack_avg_temps[i]
                # Simple P-controller: increase airflow if too hot
                if zone_avg > TEMP_OPTIMAL_HIGH + 2:
                    target_speed = min(state.fan_speed_pct + FAN_SPEED_STEP, MAX_FAN_SPEED)
                elif zone_avg < TEMP_OPTIMAL_HIGH - 2:
                    target_speed = max(state.fan_speed_pct - FAN_SPEED_STEP, MIN_FAN_SPEED)
                else:
                    target_speed = state.fan_speed_pct
            else:
                target_speed = LOCAL_AUTO_FAN_SPEED_PCT

            new_speed      = self._ramp_fan_speed(state.fan_speed_pct, target_speed)
            actual_airflow = self._fan_speed_to_airflow(new_speed)
            power_kw       = ((new_speed / MAX_FAN_SPEED) ** 3) * 2.0

            state.fan_speed_pct    = new_speed
            state.airflow_cfm      = actual_airflow
            state.discharge_temp_c = LOCAL_AUTO_DISCHARGE_C
            state.setpoint_c       = LOCAL_AUTO_DISCHARGE_C
            state.mode             = self._classify_crah_mode(new_speed)
            state.control_mode     = ControlMode.LOCAL_AUTO
            state.cumulative_kwh   = round(
                state.cumulative_kwh + power_kw * (dt_seconds / 3600), 4
            )
        return self._states

    # -- Accessors ------------------------------------------------------------

    def get_states(self) -> list[CRAHState]:
        return list(self._states)

    def states_as_dicts(self) -> list[dict]:
        return [
            {
                "unit_id":           s.unit_id,
                "fan_speed_pct":     round(s.fan_speed_pct,    1),
                "airflow_cfm":       round(s.airflow_cfm,      1),
                "discharge_temp_c":  round(s.discharge_temp_c, 2),
                "setpoint_c":        round(s.setpoint_c,       2),
                "mode":              s.mode.value,
                "control_mode":      s.control_mode.value,
                "cumulative_kwh":    round(s.cumulative_kwh,   3),
            }
            for s in self._states
        ]

    @property
    def current_airflows(self) -> np.ndarray:
        return np.array([s.airflow_cfm for s in self._states])

    @property
    def current_discharges(self) -> np.ndarray:
        return np.array([s.discharge_temp_c for s in self._states])
