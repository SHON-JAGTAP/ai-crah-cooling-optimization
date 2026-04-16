"""
modules/supervised_mode.py
--------------------------
Phase 4: Human-Supervised Control with Approval Workflow.

When the system is in SUPERVISED mode, the AI optimizer proposes control
actions but does NOT execute them immediately. Each action enters a pending
queue and waits for human approval via the dashboard.

Action lifecycle
----------------
  PENDING   -- proposed, awaiting decision
  APPROVED  -- human approved; executing
  REJECTED  -- human rejected with reason
  EXPIRED   -- timeout elapsed; auto-executed (if configured) or discarded
  EXECUTED  -- command sent to CRAH controller

This module is intentionally BMS-agnostic -- the FeedbackLoop decides
whether to route approved actions to the simulator or the real BMS.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from config import (
    SUPERVISED_ACTION_TIMEOUT_SEC,
    SUPERVISED_AUTO_EXECUTE_ON_TIMEOUT,
    NUM_CRAH_UNITS,
    CRAH_TO_AISLE,
)


class ActionStatus(str, Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED  = "EXPIRED"
    EXECUTED = "EXECUTED"


@dataclass
class PendingAction:
    """A single control action awaiting human approval."""
    action_id:          str
    crah_id:            int
    aisle:              str
    severity:           str
    proposed_airflow:   float       # CFM
    proposed_discharge: float       # degC
    current_airflow:    float
    current_discharge:  float
    avg_rack_temp:      float
    max_rack_temp:      float
    reason:             str         # why AI proposed this
    proposed_at:        float = field(default_factory=time.time)
    status:             ActionStatus = ActionStatus.PENDING
    decided_at:         Optional[float] = None
    decided_by:         str = ""   # "human" | "timeout"
    rejection_reason:   str = ""
    executed_at:        Optional[float] = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.proposed_at

    @property
    def expired(self) -> bool:
        return self.age_seconds > SUPERVISED_ACTION_TIMEOUT_SEC

    def to_dict(self) -> dict:
        return {
            "action_id":          self.action_id,
            "crah_id":            self.crah_id,
            "aisle":              self.aisle,
            "severity":           self.severity,
            "proposed_airflow":   round(self.proposed_airflow,   1),
            "proposed_discharge": round(self.proposed_discharge, 2),
            "current_airflow":    round(self.current_airflow,    1),
            "current_discharge":  round(self.current_discharge,  2),
            "avg_rack_temp":      round(self.avg_rack_temp,      2),
            "max_rack_temp":      round(self.max_rack_temp,      2),
            "reason":             self.reason,
            "status":             self.status.value,
            "age_seconds":        round(self.age_seconds, 1),
            "timeout_seconds":    SUPERVISED_ACTION_TIMEOUT_SEC,
            "decided_by":         self.decided_by,
            "rejection_reason":   self.rejection_reason,
        }


class SupervisedModeManager:
    """
    Manages the queue of pending AI control actions awaiting human approval.

    Usage
    -----
    mgr = SupervisedModeManager(on_execute=my_execute_callback)
    mgr.propose(crah_id=0, ...)
    mgr.approve("action-uuid")
    mgr.reject("action-uuid", "temperature is already stable")
    actions = mgr.get_pending()     # dashboard reads this
    """

    def __init__(self, on_execute: Optional[Callable[[PendingAction], None]] = None):
        """
        Parameters
        ----------
        on_execute : callback invoked when an action is approved / auto-executed.
                     Signature: fn(action: PendingAction) -> None
        """
        self._lock     = threading.Lock()
        self._queue:   dict[str, PendingAction] = {}
        self._history: list[PendingAction] = []
        self._on_execute = on_execute
        self._stats = {"proposed": 0, "approved": 0, "rejected": 0, "expired": 0}

        # Background thread checks for timeouts
        self._running = True
        t = threading.Thread(target=self._timeout_watcher, daemon=True)
        t.start()

    # -------------------------------------------------------------------------
    # Propose
    # -------------------------------------------------------------------------

    def propose(
        self,
        crah_id:            int,
        severity:           str,
        proposed_airflow:   float,
        proposed_discharge: float,
        current_airflow:    float,
        current_discharge:  float,
        avg_rack_temp:      float,
        max_rack_temp:      float,
        reason:             str,
    ) -> PendingAction:
        """
        Add a new action to the pending queue.
        Returns the created PendingAction.
        """
        action = PendingAction(
            action_id          = str(uuid.uuid4())[:8],
            crah_id            = crah_id,
            aisle              = CRAH_TO_AISLE.get(crah_id, f"Zone-{crah_id}"),
            severity           = severity,
            proposed_airflow   = proposed_airflow,
            proposed_discharge = proposed_discharge,
            current_airflow    = current_airflow,
            current_discharge  = current_discharge,
            avg_rack_temp      = avg_rack_temp,
            max_rack_temp      = max_rack_temp,
            reason             = reason,
        )
        with self._lock:
            # Only one pending action per CRAH at a time (replace old one)
            existing = [aid for aid, a in self._queue.items()
                        if a.crah_id == crah_id and a.status == ActionStatus.PENDING]
            for eid in existing:
                old = self._queue.pop(eid)
                old.status = ActionStatus.EXPIRED
                old.decided_by = "replaced"
                self._history.append(old)

            self._queue[action.action_id] = action
            self._stats["proposed"] += 1
        return action

    # -------------------------------------------------------------------------
    # Approve / Reject
    # -------------------------------------------------------------------------

    def approve(self, action_id: str) -> bool:
        """Human approves an action. Returns True if found and approved."""
        with self._lock:
            action = self._queue.get(action_id)
            if action is None or action.status != ActionStatus.PENDING:
                return False
            action.status     = ActionStatus.APPROVED
            action.decided_at = time.time()
            action.decided_by = "human"
            self._stats["approved"] += 1

        self._execute(action)
        return True

    def reject(self, action_id: str, reason: str = "") -> bool:
        """Human rejects an action. Returns True if found and rejected."""
        with self._lock:
            action = self._queue.get(action_id)
            if action is None or action.status != ActionStatus.PENDING:
                return False
            action.status           = ActionStatus.REJECTED
            action.decided_at       = time.time()
            action.decided_by       = "human"
            action.rejection_reason = reason
            self._stats["rejected"] += 1
            self._history.append(self._queue.pop(action_id))
        return True

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_pending(self) -> list[dict]:
        with self._lock:
            return [a.to_dict() for a in self._queue.values()
                    if a.status == ActionStatus.PENDING]

    def get_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [a.to_dict() for a in self._history[-limit:]]

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _execute(self, action: PendingAction) -> None:
        """Invoke the on_execute callback and mark action EXECUTED."""
        if self._on_execute:
            try:
                self._on_execute(action)
            except Exception as exc:
                print(f"[SupervisedMode] Execute callback error: {exc}")
        with self._lock:
            action.status      = ActionStatus.EXECUTED
            action.executed_at = time.time()
            self._history.append(self._queue.pop(action.action_id, action))

    def _timeout_watcher(self) -> None:
        """Background thread: auto-handle expired actions."""
        while self._running:
            time.sleep(5)
            with self._lock:
                expired_ids = [
                    aid for aid, a in self._queue.items()
                    if a.status == ActionStatus.PENDING and a.expired
                ]
            for aid in expired_ids:
                with self._lock:
                    action = self._queue.get(aid)
                if action and action.status == ActionStatus.PENDING:
                    action.status     = ActionStatus.EXPIRED
                    action.decided_at = time.time()
                    action.decided_by = "timeout"
                    self._stats["expired"] += 1
                    if SUPERVISED_AUTO_EXECUTE_ON_TIMEOUT:
                        print(f"[SupervisedMode] Action {aid} timed out -> auto-executing.")
                        action.status = ActionStatus.APPROVED
                        self._execute(action)
                    else:
                        print(f"[SupervisedMode] Action {aid} timed out -> discarded.")
                        with self._lock:
                            self._history.append(self._queue.pop(aid, action))

    def stop(self) -> None:
        self._running = False
