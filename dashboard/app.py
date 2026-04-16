"""
dashboard/app.py
----------------
Flask REST API -- serves the live dashboard and exposes control endpoints.

New endpoints (v2):
  GET  /api/mode                  -- current control mode + phase
  POST /api/mode                  -- switch mode {"mode": "AI|SUPERVISED|LOCAL_AUTO"}
  POST /api/phase                 -- switch phase {"phase": 1-5}
  GET  /api/aisles                -- aisle-level aggregated temperatures
  GET  /api/supervised/pending    -- actions awaiting human approval
  POST /api/supervised/approve/<id>  -- approve an action
  POST /api/supervised/reject/<id>   -- reject an action
  GET  /api/bms/status            -- Niagara BMS connection state + tags
  GET  /api/bms/tags              -- first 50 registered BMS tags
"""

import os
from threading import Lock

from flask import Flask, jsonify, render_template, request

from config import OPERATION_PHASES, POLL_INTERVAL_MS

app = Flask(__name__, template_folder="templates")

_loop = None
_loop_lock = Lock()


def get_loop():
    global _loop
    with _loop_lock:
        if _loop is None:
            from modules.feedback_loop import FeedbackLoop
            _loop = FeedbackLoop(api_key=os.getenv("ANTHROPIC_API_KEY"))
            _loop.train()
            _loop.start_background()
    return _loop


# ---------------------------------------------------------------------------
# Core dashboard endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", poll_interval_ms=POLL_INTERVAL_MS)


@app.route("/api/status")
def api_status():
    loop = get_loop()
    snap = loop.get_snapshot()
    return jsonify(snap)


@app.route("/api/metrics/history")
def api_metrics_history():
    loop  = get_loop()
    field = request.args.get("field", "avg_rack_temp")
    return jsonify({"field": field, "values": loop.get_metric_history(field)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _loop
    with _loop_lock:
        if _loop:
            _loop.stop()
        from modules.feedback_loop import FeedbackLoop
        _loop = FeedbackLoop(api_key=os.getenv("ANTHROPIC_API_KEY"))
        _loop.train()
        _loop.start_background()
    return jsonify({"status": "reset", "ok": True})


# ---------------------------------------------------------------------------
# Control mode & phase endpoints
# ---------------------------------------------------------------------------

@app.route("/api/mode", methods=["GET"])
def api_get_mode():
    loop = get_loop()
    snap = loop.get_snapshot()
    return jsonify({
        "control_mode": snap["control_mode"],
        "phase":        snap["phase"],
        "phase_label":  snap["phase_label"],
        "phases":       OPERATION_PHASES,
    })


@app.route("/api/mode", methods=["POST"])
def api_set_mode():
    data = request.get_json(force=True)
    mode = data.get("mode", "").upper()
    loop = get_loop()
    ok   = loop.set_control_mode(mode)
    return jsonify({"ok": ok, "control_mode": mode if ok else loop.get_snapshot()["control_mode"]})


@app.route("/api/phase", methods=["POST"])
def api_set_phase():
    data  = request.get_json(force=True)
    phase = int(data.get("phase", 3))
    loop  = get_loop()
    ok    = loop.set_phase(phase)
    return jsonify({"ok": ok, "phase": phase, "label": OPERATION_PHASES.get(phase, "")})


# ---------------------------------------------------------------------------
# Aisle data endpoint
# ---------------------------------------------------------------------------

@app.route("/api/aisles")
def api_aisles():
    loop = get_loop()
    snap = loop.get_snapshot()
    zones = snap.get("zones", {})
    from config import AISLE_MAP, CRAH_TO_AISLE
    aisle_list = []
    for aisle, info in AISLE_MAP.items():
        cid = info["crah_id"]
        z   = zones.get(cid, {})
        aisle_list.append({
            "aisle":      aisle,
            "crah_id":    cid,
            "rack_ids":   info["rack_ids"],
            "avg_temp":   z.get("avg_temp_c", 0),
            "max_temp":   z.get("max_temp_c", 0),
            "min_temp":   z.get("min_temp_c", 0),
            "spread":     z.get("zone_spread_c", 0),
            "severity":   z.get("severity", "NORMAL"),
            "compound":   z.get("compound", False),
        })
    return jsonify({"aisles": aisle_list})


# ---------------------------------------------------------------------------
# Supervised mode endpoints
# ---------------------------------------------------------------------------

@app.route("/api/supervised/pending")
def api_supervised_pending():
    loop = get_loop()
    return jsonify({
        "pending": loop.supervised.get_pending(),
        "stats":   loop.supervised.get_stats(),
        "history": loop.supervised.get_history(10),
    })


@app.route("/api/supervised/approve/<action_id>", methods=["POST"])
def api_supervised_approve(action_id):
    loop = get_loop()
    ok   = loop.supervised.approve(action_id)
    return jsonify({"ok": ok, "action_id": action_id})


@app.route("/api/supervised/reject/<action_id>", methods=["POST"])
def api_supervised_reject(action_id):
    loop   = get_loop()
    data   = request.get_json(force=True) or {}
    reason = data.get("reason", "")
    ok     = loop.supervised.reject(action_id, reason)
    return jsonify({"ok": ok, "action_id": action_id})


# ---------------------------------------------------------------------------
# BMS connector endpoints
# ---------------------------------------------------------------------------

@app.route("/api/bms/status")
def api_bms_status():
    loop   = get_loop()
    status = loop.bms.get_status()
    return jsonify(status)


@app.route("/api/bms/tags")
def api_bms_tags():
    loop = get_loop()
    limit = int(request.args.get("limit", 50))
    tags = loop.bms.get_tag_registry(limit=limit)
    return jsonify({"tags": tags, "shown": len(tags),
                    "total_registered": loop.bms._status.registered_tags})
