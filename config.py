"""
config.py -- Central configuration for the AI-Based CRAH Cooling Optimization System.
All tuneable parameters live here for easy experimentation.
"""

# -- Infrastructure layout ----------------------------------------------------
NUM_RACKS       = 16   # Total racks in the data hall (arranged 4x4)
NUM_CRAH_UNITS  = 4    # Number of CRAH units (one per quadrant)
RACK_ROWS       = 4    # Layout rows
RACK_COLS       = 4    # Layout columns

# -- Temperature thresholds (degC) --------------------------------------------
TEMP_OPTIMAL_LOW       = 22.0   # Below this -> overcooled (waste energy)
TEMP_OPTIMAL_HIGH      = 27.0   # Ideal upper bound
TEMP_HOTSPOT_MEDIUM    = 28.0   # Severity: MEDIUM
TEMP_HOTSPOT_HIGH      = 32.0   # Severity: HIGH
TEMP_HOTSPOT_CRITICAL  = 36.0   # Severity: CRITICAL
TEMP_AMBIENT           = 20.0   # Ambient / outside air temperature

# -- CRAH unit parameters -----------------------------------------------------
MIN_FAN_SPEED     = 20    # % -- Min to keep unit from stalling
MAX_FAN_SPEED     = 100   # %
FAN_SPEED_STEP    = 5     # % increment/decrement per control cycle

MIN_AIRFLOW_CFM   = 400   # CFM per unit minimum
MAX_AIRFLOW_CFM   = 2000  # CFM per unit maximum

DISCHARGE_TEMP_SETPOINT     = 18.0  # degC -- default cold-air discharge
DISCHARGE_TEMP_SETPOINT_MIN = 15.0  # degC -- coldest allowed
DISCHARGE_TEMP_SETPOINT_MAX = 21.0  # degC -- warmest allowed

# -- IT load simulation -------------------------------------------------------
IT_LOAD_MIN_KW  = 1.0    # kW per rack (idle)
IT_LOAD_MAX_KW  = 10.0   # kW per rack (peak)
IT_LOAD_NOISE   = 0.3    # Random fluctuation amplitude

# -- Data simulation ----------------------------------------------------------
SIMULATION_STEPS        = 200   # Steps in one synthetic training run
SIMULATION_INTERVAL_SEC = 3     # Real-time loop interval (seconds)
TRAINING_SAMPLES        = 2000  # Rows used to train the ML model

# -- Pre-processing -----------------------------------------------------------
SMOOTHING_WINDOW   = 5     # Rolling mean window size
Z_SCORE_THRESHOLD  = 3.0   # Outlier removal cut-off

# -- ML model -----------------------------------------------------------------
ML_N_ESTIMATORS  = 200
ML_MAX_DEPTH     = 10
ML_RANDOM_STATE  = 42
ML_TEST_SIZE     = 0.2
MODEL_SAVE_PATH  = "crah_model.joblib"

# -- Optimization -------------------------------------------------------------
AIRFLOW_INCREASE_PCT  = 0.15   # +15% when hotspot detected
AIRFLOW_DECREASE_PCT  = 0.10   # -10% when overcooled
SETPOINT_DECREASE_C   = 0.5    # lower discharge temp on critical hotspot
SETPOINT_INCREASE_C   = 0.5    # raise discharge temp when overcooled

# -- Claude Managed Agents ----------------------------------------------------
ANTHROPIC_MODEL     = "claude-opus-4-5"
AGENT_NAME          = "CRAH-Optimizer"
AGENT_SYSTEM_PROMPT = (
    "You are an expert AI system specialized in data center thermal management "
    "and CRAH (Computer Room Air Handler) unit optimization. "
    "You receive real-time sensor data (rack temperatures, airflow rates, "
    "discharge temperatures, IT loads) and provide precise, actionable "
    "cooling optimization recommendations. Always respond with structured, "
    "concise guidance and explain your reasoning briefly."
)
BETA_HEADER = "managed-agents-2026-04-01"

# -- Dashboard ----------------------------------------------------------------
DASHBOARD_HOST    = "0.0.0.0"
DASHBOARD_PORT    = 5050
DASHBOARD_DEBUG   = False
POLL_INTERVAL_MS  = 3000   # Frontend polling interval (ms)

# =============================================================================
# NEWLY IMPLEMENTED SECTIONS
# =============================================================================

# -- Aisle Layout (named aisles, each served by one CRAH unit) ----------------
AISLE_MAP = {
    "Aisle-A": {"crah_id": 0, "rack_ids": [0,  1,  2,  3]},
    "Aisle-B": {"crah_id": 1, "rack_ids": [4,  5,  6,  7]},
    "Aisle-C": {"crah_id": 2, "rack_ids": [8,  9,  10, 11]},
    "Aisle-D": {"crah_id": 3, "rack_ids": [12, 13, 14, 15]},
}
RACK_TO_AISLE = {
    rack_id: aisle
    for aisle, info in AISLE_MAP.items()
    for rack_id in info["rack_ids"]
}
CRAH_TO_AISLE = {info["crah_id"]: aisle for aisle, info in AISLE_MAP.items()}

# -- Compound Hotspot Condition (2B from doc) ----------------------------------
# Hotspot = (T > threshold) AND (zone_spread > ALLOWABLE_ZONE_TEMP_VARIATION)
ALLOWABLE_ZONE_TEMP_VARIATION    = 4.0   # degC -- max acceptable spread inside one zone
COMPOUND_HOTSPOT_REQUIRES_SPREAD = True  # False -> simple threshold only

# -- A_airflow: 4th ML Feature (airflow distribution factor) ------------------
# T_rack = f(F_CRAH, T_discharge, L_IT, A_airflow)
# A_airflow = zone_airflow / mean(all_zone_airflows)   [1.0 = balanced]
AIRFLOW_DIST_FEATURE_ENABLED = True

# -- Fail-Safe Mechanism -------------------------------------------------------
FAILSAFE_MAX_CONSECUTIVE_ERRORS = 5    # step failures before auto-revert
FAILSAFE_REVERT_ON_ERROR        = True
LOCAL_AUTO_AIRFLOW_CFM          = 1200.0   # fixed airflow in local auto
LOCAL_AUTO_DISCHARGE_C          = 18.0     # fixed discharge temp in local auto
LOCAL_AUTO_FAN_SPEED_PCT        = 60.0     # fixed fan speed in local auto

# -- Control Mode & Operation Phase -------------------------------------------
CONTROL_MODES        = ["AI", "SUPERVISED", "LOCAL_AUTO"]
DEFAULT_CONTROL_MODE = "AI"

OPERATION_PHASES = {
    1: "Phase 1 - Data Collection",
    2: "Phase 2 - Model Training",
    3: "Phase 3 - Simulation Testing",
    4: "Phase 4 - Supervised Deployment",
    5: "Phase 5 - Full Autonomous Control",
}
DEFAULT_PHASE = 3

# -- Supervised Mode ----------------------------------------------------------
SUPERVISED_ACTION_TIMEOUT_SEC        = 90     # auto-execute if not acted on
SUPERVISED_AUTO_EXECUTE_ON_TIMEOUT   = True

# -- Niagara Framework BMS Connector ------------------------------------------
BMS_ENABLED           = False           # True when real Niagara is available
BMS_HOST              = "192.168.1.100"
BMS_PORT              = 80
BMS_USE_HTTPS         = False
BMS_USERNAME          = "admin"
BMS_PASSWORD          = ""              # set via env: NIAGARA_PASSWORD
BMS_STATION_PATH      = "/station"
BMS_POLL_INTERVAL_SEC = 10
BMS_TAG_RACK_TEMP     = "Racks/Rack{rack_id}/InletTemp"
BMS_TAG_CRAH_AIRFLOW  = "CRAH/Unit{crah_id}/AirflowCFM"
BMS_TAG_CRAH_FAN      = "CRAH/Unit{crah_id}/FanSpeedPct"
BMS_TAG_CRAH_DISCH    = "CRAH/Unit{crah_id}/DischargeTemp"
BMS_TAG_CRAH_SETPT    = "CRAH/Unit{crah_id}/DischargeTempSP"
BMS_TAG_IT_LOAD       = "PDU/Unit{rack_id}/PowerKw"
BMS_TOTAL_TAGS        = 1500
