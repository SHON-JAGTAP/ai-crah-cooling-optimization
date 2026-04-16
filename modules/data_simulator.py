"""
modules/data_simulator.py
-------------------------
Generates realistic synthetic sensor data for a data-center cooling system.

Changes from v1:
  + Aisle names added to every record (RACK_TO_AISLE lookup)
  + airflow_dist_factor: 4th ML feature = zone_airflow / mean(all_zone_airflows)
    This implements: T_rack = f(F_CRAH, T_discharge, L_IT, A_airflow)
"""

import numpy as np
import pandas as pd
from config import (
    NUM_RACKS, NUM_CRAH_UNITS,
    IT_LOAD_MIN_KW, IT_LOAD_MAX_KW, IT_LOAD_NOISE,
    MIN_AIRFLOW_CFM, MAX_AIRFLOW_CFM,
    DISCHARGE_TEMP_SETPOINT,
    TRAINING_SAMPLES,
    RACK_TO_AISLE,
)

# HVAC: ΔT_C = Q_kW * 1.76 / (airflow_CFM / 1000)
NOISE_STD = 0.3


class DataSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self.rack_crah_map = {r: r % NUM_CRAH_UNITS for r in range(NUM_RACKS)}

        self._it_loads   = self.rng.uniform(IT_LOAD_MIN_KW, IT_LOAD_MAX_KW, size=NUM_RACKS)
        self._airflows   = self.rng.uniform(MIN_AIRFLOW_CFM, MAX_AIRFLOW_CFM, size=NUM_CRAH_UNITS)
        self._discharges = np.full(NUM_CRAH_UNITS, DISCHARGE_TEMP_SETPOINT)

    # -- Physics ---------------------------------------------------------------

    def _compute_rack_temp(self, it_load: float, airflow: float, discharge_temp: float) -> float:
        delta_t = it_load * 1.76 / (max(airflow, 50.0) / 1000.0)
        noise   = self.rng.normal(0, NOISE_STD)
        return round(discharge_temp + delta_t + noise, 2)

    def _drift_it_load(self) -> None:
        delta = self.rng.normal(0, IT_LOAD_NOISE, size=NUM_RACKS)
        self._it_loads = np.clip(self._it_loads + delta, IT_LOAD_MIN_KW, IT_LOAD_MAX_KW)

    @staticmethod
    def _airflow_dist_factor(zone_airflow: float, all_airflows: list[float]) -> float:
        """
        A_airflow = zone_airflow / mean(all_zone_airflows).
        1.0 = perfectly balanced; >1 = over-served; <1 = under-served.
        """
        mean_af = float(np.mean(all_airflows)) if all_airflows else zone_airflow
        return round(zone_airflow / max(mean_af, 1.0), 4)

    # -- Training data generation ---------------------------------------------

    def generate_training_data(self, n_samples: int = TRAINING_SAMPLES) -> pd.DataFrame:
        """
        Generate n_samples rows. Each row simulates a full data-hall snapshot
        (all 4 CRAH airflows drawn together) so A_airflow is physically consistent.
        """
        rows = []
        for _ in range(n_samples):
            # Draw all 4 zone airflows simultaneously
            zone_airflows = self.rng.uniform(MIN_AIRFLOW_CFM, MAX_AIRFLOW_CFM,
                                             size=NUM_CRAH_UNITS).tolist()
            rack_id    = int(self.rng.integers(0, NUM_RACKS))
            crah_id    = self.rack_crah_map[rack_id]
            it_load    = float(self.rng.uniform(IT_LOAD_MIN_KW, IT_LOAD_MAX_KW))
            airflow    = zone_airflows[crah_id]
            discharge  = float(self.rng.uniform(
                DISCHARGE_TEMP_SETPOINT - 3, DISCHARGE_TEMP_SETPOINT + 3
            ))
            rack_temp  = self._compute_rack_temp(it_load, airflow, discharge)
            dist_fac   = self._airflow_dist_factor(airflow, zone_airflows)
            aisle      = RACK_TO_AISLE.get(rack_id, "Aisle-?")

            rows.append({
                "rack_id":            rack_id,
                "crah_id":            crah_id,
                "aisle":              aisle,
                "it_load_kw":         round(it_load,   3),
                "airflow_cfm":        round(airflow,   1),
                "discharge_temp_c":   round(discharge, 2),
                "airflow_dist_factor":dist_fac,
                "rack_temp_c":        rack_temp,
            })
        return pd.DataFrame(rows)

    # -- Real-time step -------------------------------------------------------

    def step(
        self,
        airflows:   np.ndarray | None = None,
        discharges: np.ndarray | None = None,
    ) -> list[dict]:
        self._step_count += 1
        self._drift_it_load()

        if airflows is not None:
            self._airflows = np.clip(airflows, MIN_AIRFLOW_CFM, MAX_AIRFLOW_CFM)
        if discharges is not None:
            self._discharges = np.clip(discharges,
                                        DISCHARGE_TEMP_SETPOINT - 5,
                                        DISCHARGE_TEMP_SETPOINT + 5)

        all_airflows = self._airflows.tolist()
        records = []
        for rack_id in range(NUM_RACKS):
            crah_id   = self.rack_crah_map[rack_id]
            airflow   = float(self._airflows[crah_id])
            discharge = float(self._discharges[crah_id])
            it_load   = float(self._it_loads[rack_id])
            rack_temp = self._compute_rack_temp(it_load, airflow, discharge)
            dist_fac  = self._airflow_dist_factor(airflow, all_airflows)
            aisle     = RACK_TO_AISLE.get(rack_id, "Aisle-?")

            records.append({
                "step":               self._step_count,
                "rack_id":            rack_id,
                "crah_id":            crah_id,
                "aisle":              aisle,
                "it_load_kw":         round(it_load,   3),
                "airflow_cfm":        round(airflow,   1),
                "discharge_temp_c":   round(discharge, 2),
                "airflow_dist_factor":dist_fac,
                "rack_temp_c":        rack_temp,
            })
        return records

    @property
    def current_airflows(self) -> np.ndarray:
        return self._airflows.copy()

    @property
    def current_discharges(self) -> np.ndarray:
        return self._discharges.copy()
