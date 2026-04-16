"""
modules/feedback_loop.py
-------------------------
Orchestrates the complete AI control loop with all Phase 1-5 features.

New in v2:
  + Operation phase tracking (1-5)
  + Control mode: AI / SUPERVISED / LOCAL_AUTO with switching
  + Fail-safe: consecutive error counter -> auto-revert to LOCAL_AUTO
  + Supervised mode integration (action approval queue)
  + BMS connector integration (reads/writes when BMS_ENABLED)
  + Aisle data in snapshots
  + A_airflow (airflow_dist_factor) as 4th ML feature
"""

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_CONTROL_MODE,
    DEFAULT_PHASE,
    FAILSAFE_MAX_CONSECUTIVE_ERRORS,
    FAILSAFE_REVERT_ON_ERROR,
    NUM_CRAH_UNITS,
    NUM_RACKS,
    OPERATION_PHASES,
    SIMULATION_INTERVAL_SEC,
    TRAINING_SAMPLES,
    CRAH_TO_AISLE,
    AISLE_MAP,
)
from modules.bms_connector import NiagaraBMSConnector
from modules.claude_agent import CRAHClaudeAgent
from modules.crah_controller import CRAHController, ControlMode
from modules.data_simulator import DataSimulator
from modules.hotspot_detector import HotspotDetector, Severity
from modules.ml_model import TemperaturePredictor
from modules.optimizer import CoolingOptimizer
from modules.preprocessor import Preprocessor
from modules.supervised_mode import SupervisedModeManager, PendingAction

AI_CALL_EVERY_N_STEPS = 10
HISTORY_LENGTH        = 200


class FeedbackLoop:
    """
    Wires all subsystems together and supports 5 operation phases.

    Modes
    -----
    AI         : Autonomous -- optimizer commands applied immediately.
    SUPERVISED : AI proposes; human approves via dashboard before execution.
    LOCAL_AUTO : Fail-safe -- fixed setpoints, no AI involvement.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.simulator    = DataSimulator(seed=0)
        self.preprocessor = Preprocessor()
        self.model        = TemperaturePredictor()
        self.detector     = HotspotDetector()
        self.optimizer    = CoolingOptimizer()
        self.controller   = CRAHController()
        self.ai_agent     = CRAHClaudeAgent(api_key=api_key)
        self.bms          = NiagaraBMSConnector(simulator=self.simulator)

        # Supervised mode manager
        self.supervised   = SupervisedModeManager(on_execute=self._execute_approved_action)

        # Thread safety
        self._lock    = threading.Lock()
        self._step    = 0
        self._running = False
        self._trained = False

        # Operation phase & mode
        self._phase          = DEFAULT_PHASE
        self._control_mode   = DEFAULT_CONTROL_MODE

        # Fail-safe tracking
        self._consecutive_errors = 0

        # Ring-buffers
        self._rack_history:   deque[list[dict]] = deque(maxlen=HISTORY_LENGTH)
        self._crah_history:   deque[list[dict]] = deque(maxlen=HISTORY_LENGTH)
        self._metric_history: deque[dict]       = deque(maxlen=HISTORY_LENGTH)

        # Latest snapshot fields
        self.current_racks:   list[dict] = []
        self.current_crah:    list[dict] = []
        self.current_alerts:  list[dict] = []
        self.current_actions: list[dict] = []
        self.current_metrics: dict       = {}
        self.current_zones:   dict       = {}
        self.ai_recommendation: str      = "Initialising..."
        self.model_metrics:   dict       = {}

    # -- Training --------------------------------------------------------------

    def train(self, n_samples: int = TRAINING_SAMPLES) -> dict:
        print("[FeedbackLoop] Generating training data ...")
        raw_df    = self.simulator.generate_training_data(n_samples)
        scaled_df = self.preprocessor.fit_transform(raw_df)

        print("[FeedbackLoop] Training temperature predictor ...")
        metrics = self.model.train(scaled_df, self.preprocessor)
        self.model_metrics = metrics
        self._trained = True
        print("[FeedbackLoop] Training complete.")
        return metrics

    # -- Single step -----------------------------------------------------------

    def step(self) -> dict:
        if not self._trained:
            raise RuntimeError("Call train() before step().")

        try:
            snap = self._execute_step()
            self._consecutive_errors = 0
            return snap
        except Exception as exc:
            self._consecutive_errors += 1
            print(f"[FeedbackLoop] Step error ({self._consecutive_errors}): {exc}")
            if (FAILSAFE_REVERT_ON_ERROR
                    and self._consecutive_errors >= FAILSAFE_MAX_CONSECUTIVE_ERRORS):
                self._trigger_failsafe()
            raise

    def _execute_step(self) -> dict:
        mode = self._control_mode

        # --- LOCAL_AUTO: bypass AI entirely ---
        if mode == "LOCAL_AUTO":
            return self._local_auto_step()

        # --- 1. Sensors (BMS or simulator) ---
        raw_records = self.simulator.step(
            airflows=self.controller.current_airflows,
            discharges=self.controller.current_discharges,
        )

        # --- 2. Preprocess for ML ---
        raw_df        = pd.DataFrame(raw_records)
        inference_cols= ["airflow_cfm", "discharge_temp_c", "it_load_kw",
                         "airflow_dist_factor", "rack_temp_c"]
        inf_df = self.preprocessor.transform(
            raw_df[[c for c in inference_cols if c in raw_df.columns]].copy()
        )

        # --- 3. Predict rack temperatures ---
        pred_temps = self.model.predict(inf_df)
        for i, rec in enumerate(raw_records):
            rec["predicted_temp_c"] = round(float(pred_temps[i]), 2)

        # --- 4. Detect hotspots (compound condition) ---
        alerts, zones = self.detector.detect(raw_records)

        # --- 5. Optimize ---
        new_airflows, new_discharges, actions = self.optimizer.optimize(
            self.controller.current_airflows,
            self.controller.current_discharges,
            zones,
        )

        # --- 6. Apply or queue (depends on mode) ---
        if mode == "SUPERVISED":
            self._queue_supervised_actions(
                actions, new_airflows, new_discharges, zones
            )
            # Controller stays at current setpoints; supervised actions will update
            crah_states = self.controller.apply(
                self.controller.current_airflows,
                self.controller.current_discharges,
                dt_seconds=SIMULATION_INTERVAL_SEC,
            )
        else:  # AI mode
            crah_states = self.controller.apply(
                new_airflows, new_discharges,
                dt_seconds=SIMULATION_INTERVAL_SEC,
            )
            # Write to BMS if available
            bms_cmds = {
                crah_id: {
                    "airflow_sp_cfm":     float(new_airflows[crah_id]),
                    "discharge_temp_sp_c":float(new_discharges[crah_id]),
                }
                for crah_id in range(NUM_CRAH_UNITS)
            }
            self.bms.write_crah_commands(bms_cmds)

        # Sync simulator to actual controller outputs
        self.simulator._airflows   = self.controller.current_airflows
        self.simulator._discharges = self.controller.current_discharges

        # --- 7. Metrics ---
        rack_temps    = [r["rack_temp_c"] for r in raw_records]
        it_load_total = round(sum(r["it_load_kw"] for r in raw_records), 2)
        cool_power    = self.optimizer.estimate_cooling_power(self.controller.current_airflows)
        pue           = self.optimizer.compute_pue(it_load_total, cool_power)

        # Aisle summary
        aisle_data = self._build_aisle_data(zones)

        metrics = {
            "step":             self._step,
            "phase":            self._phase,
            "phase_label":      OPERATION_PHASES.get(self._phase, ""),
            "control_mode":     self._control_mode,
            "avg_rack_temp":    round(float(np.mean(rack_temps)), 2),
            "max_rack_temp":    round(float(np.max(rack_temps)),  2),
            "min_rack_temp":    round(float(np.min(rack_temps)),  2),
            "it_load_kw_total": it_load_total,
            "cooling_power_kw": cool_power,
            "pue":              pue,
            "n_hotspots":       len([a for a in alerts if a.severity.numeric > 0]),
            "n_overcooled":     len([a for a in alerts if a.severity.numeric < 0]),
            "n_compound":       len([a for a in alerts if a.is_compound_hotspot]),
            "alert_summary":    self.detector.summary_str(alerts),
        }

        # --- 8. AI recommendation (async) ---
        if self._step % AI_CALL_EVERY_N_STEPS == 0:
            hotspot_racks  = [a.rack_id for a in alerts if a.severity.numeric > 0]
            overcool_racks = [a.rack_id for a in alerts if a.severity.numeric < 0]
            telemetry = {
                **metrics,
                "crah_states":      self.controller.states_as_dicts(),
                "hotspot_racks":    hotspot_racks,
                "overcooled_racks": overcool_racks,
                "aisle_data":       aisle_data,
            }
            threading.Thread(target=self._call_ai, args=(telemetry,), daemon=True).start()

        # Build alert/action dicts
        alert_dicts = [
            {
                "rack_id":            a.rack_id,
                "crah_id":            a.crah_id,
                "aisle":              a.aisle,
                "temp_c":             a.temp_c,
                "severity":           a.severity.value,
                "color":              a.severity.color,
                "message":            a.message,
                "delta_c":            a.delta_c,
                "is_compound":        a.is_compound_hotspot,
            }
            for a in alerts
        ]

        zone_dicts = {
            crah_id: {
                "crah_id":      z.crah_id,
                "aisle":        z.aisle,
                "avg_temp_c":   z.avg_temp_c,
                "max_temp_c":   z.max_temp_c,
                "min_temp_c":   z.min_temp_c,
                "zone_spread_c":z.zone_spread_c,
                "severity":     z.worst_severity.value,
                "compound":     z.is_compound_hotspot,
            }
            for crah_id, z in zones.items()
        }

        crah_dicts = self.controller.states_as_dicts()

        with self._lock:
            self._step           += 1
            self.current_racks    = raw_records
            self.current_crah     = crah_dicts
            self.current_alerts   = alert_dicts
            self.current_actions  = actions
            self.current_metrics  = metrics
            self.current_zones    = zone_dicts
            self._rack_history.append(raw_records)
            self._crah_history.append(crah_dicts)
            self._metric_history.append(metrics)

        return self.get_snapshot()

    # -- Local auto step -------------------------------------------------------

    def _local_auto_step(self) -> dict:
        raw_records = self.simulator.step()
        zone_avg_temps = {}
        for rec in raw_records:
            cid = rec["crah_id"]
            zone_avg_temps.setdefault(cid, []).append(rec["rack_temp_c"])
        zone_avgs = {cid: float(np.mean(temps)) for cid, temps in zone_avg_temps.items()}

        crah_states = self.controller.apply_local_auto(
            rack_avg_temps=zone_avgs, dt_seconds=SIMULATION_INTERVAL_SEC
        )
        rack_temps    = [r["rack_temp_c"] for r in raw_records]
        it_load_total = round(sum(r["it_load_kw"] for r in raw_records), 2)
        cool_power    = self.optimizer.estimate_cooling_power(self.controller.current_airflows)

        metrics = {
            "step":          self._step,
            "phase":         self._phase,
            "phase_label":   OPERATION_PHASES.get(self._phase, ""),
            "control_mode":  "LOCAL_AUTO",
            "avg_rack_temp": round(float(np.mean(rack_temps)), 2),
            "max_rack_temp": round(float(np.max(rack_temps)),  2),
            "min_rack_temp": round(float(np.min(rack_temps)),  2),
            "it_load_kw_total": it_load_total,
            "cooling_power_kw":cool_power,
            "pue":           self.optimizer.compute_pue(it_load_total, cool_power),
            "n_hotspots":    0,
            "n_overcooled":  0,
            "n_compound":    0,
            "alert_summary": "LOCAL_AUTO mode -- AI offline",
        }
        with self._lock:
            self._step          += 1
            self.current_racks   = raw_records
            self.current_crah    = self.controller.states_as_dicts()
            self.current_metrics = metrics
            self._metric_history.append(metrics)

        return self.get_snapshot()

    # -- Supervised mode helpers -----------------------------------------------

    def _queue_supervised_actions(
        self,
        actions:        list[dict],
        new_airflows:   np.ndarray,
        new_discharges: np.ndarray,
        zones:          dict,
    ) -> None:
        for act in actions:
            cid = act["crah_id"]
            z   = zones.get(cid)
            if not z:
                continue
            # Only queue if there's a meaningful change
            af_change  = abs(new_airflows[cid]   - self.controller.current_airflows[cid])
            dt_change  = abs(new_discharges[cid]  - self.controller.current_discharges[cid])
            if af_change < 10 and dt_change < 0.1:
                continue

            reason = "; ".join(act.get("changes", ["Routine optimization"]))
            self.supervised.propose(
                crah_id            = cid,
                severity           = act["severity"],
                proposed_airflow   = float(new_airflows[cid]),
                proposed_discharge = float(new_discharges[cid]),
                current_airflow    = float(self.controller.current_airflows[cid]),
                current_discharge  = float(self.controller.current_discharges[cid]),
                avg_rack_temp      = act.get("avg_rack_temp", 0.0),
                max_rack_temp      = act.get("max_rack_temp", 0.0),
                reason             = reason,
            )

    def _execute_approved_action(self, action: PendingAction) -> None:
        """Called by SupervisedModeManager when an action is approved."""
        target_airflows   = self.controller.current_airflows.copy()
        target_discharges = self.controller.current_discharges.copy()
        target_airflows[action.crah_id]   = action.proposed_airflow
        target_discharges[action.crah_id] = action.proposed_discharge
        self.controller.apply(target_airflows, target_discharges,
                              dt_seconds=SIMULATION_INTERVAL_SEC)
        print(f"[FeedbackLoop] Supervised action {action.action_id} executed for "
              f"{action.aisle} (CRAH-{action.crah_id})")

    # -- Fail-safe -------------------------------------------------------------

    def _trigger_failsafe(self) -> None:
        print(f"[FeedbackLoop] FAIL-SAFE triggered after "
              f"{self._consecutive_errors} consecutive errors.")
        self.controller.revert_to_local_auto()
        with self._lock:
            self._control_mode = "LOCAL_AUTO"
        self.ai_recommendation = (
            "[FAIL-SAFE] AI encountered repeated errors. "
            "System reverted to LOCAL_AUTO mode. "
            "Investigate errors then manually switch back to AI mode."
        )

    # -- Mode / Phase management -----------------------------------------------

    def set_control_mode(self, mode: str) -> bool:
        """Switch control mode. mode: 'AI' | 'SUPERVISED' | 'LOCAL_AUTO'"""
        if mode not in ["AI", "SUPERVISED", "LOCAL_AUTO"]:
            return False
        with self._lock:
            self._control_mode = mode
        self.controller.set_mode(mode)
        print(f"[FeedbackLoop] Control mode -> {mode}")
        return True

    def set_phase(self, phase: int) -> bool:
        if phase not in OPERATION_PHASES:
            return False
        with self._lock:
            self._phase = phase
        print(f"[FeedbackLoop] Phase -> {OPERATION_PHASES[phase]}")
        return True

    # -- AI recommendation -----------------------------------------------------

    def _call_ai(self, telemetry: dict) -> None:
        rec = self.ai_agent.get_recommendations(telemetry)
        with self._lock:
            self.ai_recommendation = rec

    # -- Aisle data helper -----------------------------------------------------

    def _build_aisle_data(self, zones: dict) -> list[dict]:
        aisle_list = []
        for aisle, info in AISLE_MAP.items():
            crah_id = info["crah_id"]
            z = zones.get(crah_id)
            if z:
                aisle_list.append({
                    "aisle":        aisle,
                    "crah_id":      crah_id,
                    "avg_temp":     z.avg_temp_c,
                    "max_temp":     z.max_temp_c,
                    "min_temp":     z.min_temp_c,
                    "spread":       z.zone_spread_c,
                    "severity":     z.worst_severity.value,
                    "compound":     z.is_compound_hotspot,
                    "rack_count":   len(info["rack_ids"]),
                })
        return aisle_list

    # -- Run modes -------------------------------------------------------------

    def run(self, max_steps: int = 100) -> None:
        self._running = True
        for _ in range(max_steps):
            if not self._running:
                break
            try:
                self.step()
            except Exception:
                pass
            time.sleep(SIMULATION_INTERVAL_SEC)
        self._running = False

    def start_background(self) -> None:
        self._running = True
        t = threading.Thread(target=self._background_run, daemon=True)
        t.start()
        print("[FeedbackLoop] Background loop started.")

    def _background_run(self) -> None:
        while self._running:
            try:
                self.step()
            except Exception as exc:
                print(f"[FeedbackLoop] Step error: {exc}")
            time.sleep(SIMULATION_INTERVAL_SEC)

    def stop(self) -> None:
        self._running = False
        self.supervised.stop()

    # -- Snapshot / Dashboard helpers ------------------------------------------

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "step":              self._step,
                "phase":             self._phase,
                "phase_label":       OPERATION_PHASES.get(self._phase, ""),
                "control_mode":      self._control_mode,
                "racks":             list(self.current_racks),
                "crah":              list(self.current_crah),
                "alerts":            list(self.current_alerts),
                "actions":           list(self.current_actions),
                "metrics":           dict(self.current_metrics),
                "zones":             dict(self.current_zones),
                "ai_recommendation": self.ai_recommendation,
                "model_metrics":     dict(self.model_metrics),
                "bms_status":        self.bms.get_status(),
            }

    def get_metric_history(self, field: str) -> list:
        with self._lock:
            return [m.get(field) for m in self._metric_history]

    def get_rack_temp_series(self) -> dict:
        with self._lock:
            series: dict[int, list[float]] = {i: [] for i in range(NUM_RACKS)}
            for step_records in list(self._rack_history)[-50:]:
                for rec in step_records:
                    series[rec["rack_id"]].append(rec["rack_temp_c"])
            return series
