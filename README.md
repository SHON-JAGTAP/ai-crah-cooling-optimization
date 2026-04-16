# AI-Based Group CRAH Control and Cooling Optimization

> An AI-driven prototype for autonomous data center cooling using Claude Managed Agents, machine learning, and real-time optimization.

---

## Overview

This system simulates a data center with **16 racks** and **4 CRAH units**, uses a **RandomForest + GradientBoosting ensemble** to predict rack temperatures, detects hotspots, and continuously optimizes airflow and temperature setpoints through a feedback loop — with optional **Claude AI recommendations** via the Managed Agents API.

---

## Architecture

```
DataSimulator → Preprocessor → ML Model (Temperature Prediction)
                                       ↓
                               HotspotDetector (Severity Classification)
                                       ↓
                               CoolingOptimizer (Airflow / Setpoint Targets)
                                       ↓
                               CRAHController (Fan Speed Ramp Control)
                                       ↓
                    ┌──────────────────┴──────────────────┐
               FeedbackLoop                        Claude Managed Agent
           (continuous loop)                   (AI recommendations every 10 steps)
                    ↓
              Flask Dashboard (live charts, heatmap, alerts)
```

---

## Module Guide

| File | Responsibility |
|---|---|
| `config.py` | All tuneable parameters (temps, airflow, ML, dashboard) |
| `modules/data_simulator.py` | Physics-based synthetic CRAH sensor data generation |
| `modules/preprocessor.py` | Dedup → clip → outlier removal → smoothing → MinMax scaling |
| `modules/ml_model.py` | Ensemble RF + GB regressor: `T_rack = f(airflow, discharge_temp, IT_load)` |
| `modules/hotspot_detector.py` | Per-rack severity (OVERCOOL / NORMAL / MEDIUM / HIGH / CRITICAL) |
| `modules/optimizer.py` | Rule-based airflow & setpoint optimizer + PUE calculation |
| `modules/crah_controller.py` | Fan-speed ramp control, affinity-law CFM conversion, energy tally |
| `modules/feedback_loop.py` | Orchestrates all modules in a background thread |
| `modules/claude_agent.py` | Claude Managed Agents API integration with graceful fallback |
| `dashboard/app.py` | Flask REST API + serves the live dashboard |
| `dashboard/templates/index.html` | Dark-themed live dashboard with Chart.js visualisations |
| `main.py` | CLI entry point (dashboard or headless mode) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Anthropic API key for Claude AI

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Or create a .env file
echo ANTHROPIC_API_KEY=sk-ant-... > .env
```

### 3. Launch the dashboard

```bash
python main.py
```

Then open **http://localhost:5050** in your browser.

### 4. Run headless (no browser)

```bash
python main.py --headless --steps 100
```

---

## Temperature Thresholds

| Severity | Range | Action |
|---|---|---|
| OVERCOOL | < 22°C | Reduce airflow, raise discharge temp |
| NORMAL | 22–27°C | Gradual return to nominal |
| MEDIUM | 27–28°C | Gentle airflow increase (+7.5%) |
| HIGH | 28–32°C | Airflow increase (+15%) |
| CRITICAL | ≥ 32°C | Max cooling response (+22.5%, lower setpoint) |

---

## ML Model

- **Algorithm**: RandomForest (60%) + GradientBoosting (40%) ensemble
- **Features**: `airflow_cfm`, `discharge_temp_c`, `it_load_kw`
- **Target**: `rack_temp_c` (hot-aisle return temperature)
- **Training data**: 2000 synthetic samples with physical noise
- **Typical MAE**: < 0.5°C on normalised-then-inverted predictions

---

## Claude Managed Agents Integration

The system uses the `managed-agents-2026-04-01` beta API to:
1. Create a persistent CRAH-expert agent at startup
2. Start a session per analysis window
3. Send telemetry every 10 steps and stream recommendations
4. Automatically falls back to rule-based summaries if no API key is set

```python
from modules.claude_agent import CRAHClaudeAgent
agent = CRAHClaudeAgent(api_key="sk-ant-...")
rec   = agent.get_recommendations(telemetry_dict)
```

---

## Dashboard Features

- **Live rack heatmap** — 4×4 grid colour-coded by temperature
- **CRAH unit cards** — fan speed gauge, airflow, discharge temp, mode
- **Temperature trend chart** — rolling avg/max/min over last 60 steps
- **PUE & hotspot history** — dual-axis energy efficiency chart
- **Active alerts panel** — severity-sorted with colour indicators
- **Optimizer actions log** — per-CRAH decisions each cycle
- **Claude AI panel** — live recommendations from Managed Agent
- **ML model metrics** — MAE, R², feature importances

---

## Configuration

Key parameters in `config.py`:

```python
NUM_RACKS              = 16      # Racks in the data hall
NUM_CRAH_UNITS         = 4       # CRAH units (one per zone)
SIMULATION_INTERVAL_SEC= 3       # Seconds between control cycles
TRAINING_SAMPLES       = 2000    # ML training set size
TEMP_HOTSPOT_CRITICAL  = 32.0    # °C — critical alert threshold
AIRFLOW_INCREASE_PCT   = 0.15    # +15% airflow on hotspot
DASHBOARD_PORT         = 5050    # Web dashboard port
```
