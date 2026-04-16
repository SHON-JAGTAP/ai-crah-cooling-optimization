"""
modules/optimizer.py
---------------------
Rule-based cooling optimizer with aisle-aware action descriptions
and airflow distribution balancing.

Changes from v1:
  + Action descriptions include aisle names
  + Zone spread flagged in actions for compound hotspot zones
  + A_airflow rebalancing hint added to actions
"""

import numpy as np

from config import (
    AIRFLOW_DECREASE_PCT,
    AIRFLOW_INCREASE_PCT,
    ALLOWABLE_ZONE_TEMP_VARIATION,
    CRAH_TO_AISLE,
    DISCHARGE_TEMP_SETPOINT,
    DISCHARGE_TEMP_SETPOINT_MAX,
    DISCHARGE_TEMP_SETPOINT_MIN,
    MAX_AIRFLOW_CFM,
    MIN_AIRFLOW_CFM,
    NUM_CRAH_UNITS,
    SETPOINT_DECREASE_C,
    SETPOINT_INCREASE_C,
)
from modules.hotspot_detector import Severity, ZoneSummary


class CoolingOptimizer:
    """
    Stateless optimizer. Computes target airflow and discharge-temperature
    setpoints per CRAH unit based on zone thermal state.
    Output actions include aisle names and compound hotspot flags.
    """

    def __init__(self):
        self._nominal_airflow   = (MIN_AIRFLOW_CFM + MAX_AIRFLOW_CFM) / 2
        self._nominal_discharge = DISCHARGE_TEMP_SETPOINT

    def optimize(
        self,
        current_airflows:   np.ndarray,
        current_discharges: np.ndarray,
        zones:              dict[int, ZoneSummary],
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:

        new_airflows   = current_airflows.copy().astype(float)
        new_discharges = current_discharges.copy().astype(float)
        actions: list[dict] = []

        # Global A_airflow distribution factor for awareness
        mean_airflow = float(np.mean(current_airflows)) if len(current_airflows) > 0 else self._nominal_airflow

        for crah_id in range(NUM_CRAH_UNITS):
            zone     = zones.get(crah_id)
            if zone is None:
                continue

            sev      = zone.worst_severity
            airflow  = current_airflows[crah_id]
            discharge= current_discharges[crah_id]
            aisle    = CRAH_TO_AISLE.get(crah_id, f"Zone-{crah_id}")
            dist_fac = airflow / max(mean_airflow, 1.0)  # A_airflow

            action = {
                "crah_id":    crah_id,
                "aisle":      aisle,
                "severity":   sev.value,
                "compound":   zone.is_compound_hotspot,
                "zone_spread":round(zone.zone_spread_c, 2),
                "dist_factor":round(dist_fac, 3),
                "changes":    [],
            }

            if sev == Severity.CRITICAL:
                mult  = 1.5 if not zone.is_compound_hotspot else 1.8
                new_af = airflow * (1 + AIRFLOW_INCREASE_PCT * mult)
                new_dt = discharge - SETPOINT_DECREASE_C
                action["changes"].append(
                    f"{aisle}: AIRFLOW +{AIRFLOW_INCREASE_PCT*mult*100:.0f}%"
                    f" ({airflow:.0f}->{new_af:.0f} CFM)"
                )
                action["changes"].append(f"DISCHARGE {discharge:.1f}->{new_dt:.1f}C")
                if zone.is_compound_hotspot:
                    action["changes"].append(
                        f"[Compound hotspot: spread {zone.zone_spread_c:.1f}C > "
                        f"{ALLOWABLE_ZONE_TEMP_VARIATION}C]"
                    )

            elif sev == Severity.HIGH:
                new_af = airflow * (1 + AIRFLOW_INCREASE_PCT)
                new_dt = discharge - SETPOINT_DECREASE_C * 0.5
                action["changes"].append(
                    f"{aisle}: AIRFLOW +{AIRFLOW_INCREASE_PCT*100:.0f}%"
                    f" ({airflow:.0f}->{new_af:.0f} CFM)"
                )

            elif sev == Severity.MEDIUM:
                new_af = airflow * (1 + AIRFLOW_INCREASE_PCT * 0.5)
                new_dt = discharge
                action["changes"].append(
                    f"{aisle}: AIRFLOW +{AIRFLOW_INCREASE_PCT*50:.0f}%"
                    f" ({airflow:.0f}->{new_af:.0f} CFM)"
                )

            elif sev == Severity.OVERCOOL:
                new_af = airflow * (1 - AIRFLOW_DECREASE_PCT)
                new_dt = discharge + SETPOINT_INCREASE_C
                action["changes"].append(
                    f"{aisle}: AIRFLOW -{AIRFLOW_DECREASE_PCT*100:.0f}%"
                    f" ({airflow:.0f}->{new_af:.0f} CFM) [energy save]"
                )
                action["changes"].append(f"DISCHARGE {discharge:.1f}->{new_dt:.1f}C (raised)")

            else:  # NORMAL
                nudge_af = (self._nominal_airflow - airflow) * 0.05
                nudge_dt = (self._nominal_discharge - discharge) * 0.05
                new_af   = airflow   + nudge_af
                new_dt   = discharge + nudge_dt
                action["changes"].append(f"{aisle}: Nominal -- gentle correction applied")

            # A_airflow rebalancing hint
            if dist_fac < 0.7:
                action["changes"].append(
                    f"[A_airflow={dist_fac:.2f}: under-served zone -- consider increasing]"
                )
            elif dist_fac > 1.5:
                action["changes"].append(
                    f"[A_airflow={dist_fac:.2f}: over-served zone -- consider reducing]"
                )

            new_airflows[crah_id]   = np.clip(new_af, MIN_AIRFLOW_CFM, MAX_AIRFLOW_CFM)
            new_discharges[crah_id] = np.clip(new_dt,
                                               DISCHARGE_TEMP_SETPOINT_MIN,
                                               DISCHARGE_TEMP_SETPOINT_MAX)
            action["airflow_after"]   = round(float(new_airflows[crah_id]),   1)
            action["discharge_after"] = round(float(new_discharges[crah_id]), 2)
            action["avg_rack_temp"]   = zone.avg_temp_c
            action["max_rack_temp"]   = zone.max_temp_c
            actions.append(action)

        return new_airflows, new_discharges, actions

    # -- Energy helpers -------------------------------------------------------

    @staticmethod
    def compute_pue(it_load_kw: float, cooling_power_kw: float) -> float:
        total = it_load_kw + cooling_power_kw
        return round(total / max(it_load_kw, 0.001), 3)

    @staticmethod
    def estimate_cooling_power(airflows: np.ndarray) -> float:
        max_total = MAX_AIRFLOW_CFM * NUM_CRAH_UNITS
        ratio     = float(airflows.sum()) / max(max_total, 1)
        return round(ratio ** 3 * 20.0, 2)
