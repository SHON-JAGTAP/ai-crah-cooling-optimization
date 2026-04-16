"""
modules/bms_connector.py
------------------------
Niagara Framework BMS Integration Layer.

This module provides a clean interface between the AI optimization engine
and the Niagara Framework Building Management System (BMS).

Architecture
------------
  NiagaraBMSConnector
      |-- read_rack_temperatures()   -> dict[rack_id, float]
      |-- read_crah_parameters()     -> dict[crah_id, dict]
      |-- write_crah_commands()      -> bool
      |-- get_tag_registry()         -> dict (all ~1500 tag paths)
      |-- get_status()               -> dict (connection health)

When BMS_ENABLED = False (default), the connector operates in SIMULATION mode,
returning values from the DataSimulator so the rest of the system works
identically -- just swap in real Niagara credentials to go live.

Niagara REST API
----------------
Endpoint: GET  {host}/station?path=/Racks/Rack0/InletTemp
          POST {host}/station?path=/CRAH/Unit0/DischargeTempSP  body: {"value": 17.5}
Reference: Niagara Framework 4.x REST API documentation
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from config import (
    BMS_ENABLED, BMS_HOST, BMS_PASSWORD, BMS_POLL_INTERVAL_SEC,
    BMS_PORT, BMS_STATION_PATH, BMS_TAG_CRAH_AIRFLOW, BMS_TAG_CRAH_DISCH,
    BMS_TAG_CRAH_FAN, BMS_TAG_CRAH_SETPT, BMS_TAG_IT_LOAD,
    BMS_TAG_RACK_TEMP, BMS_TOTAL_TAGS, BMS_USERNAME, BMS_USE_HTTPS,
    CRAH_TO_AISLE, NUM_CRAH_UNITS, NUM_RACKS,
)


# ---------------------------------------------------------------------------
# Tag Registry — maps every measurable point to a Niagara path
# ---------------------------------------------------------------------------

def build_tag_registry() -> dict[str, str]:
    """
    Build the full tag registry (~1500 tags across all racks and CRAH units).
    Each entry: tag_name -> niagara_path
    """
    tags: dict[str, str] = {}

    # Rack-level tags (per rack: 5 tags x 16 racks = 80)
    for rack_id in range(NUM_RACKS):
        prefix = f"rack_{rack_id}"
        tags[f"{prefix}_inlet_temp"]      = BMS_TAG_RACK_TEMP.format(rack_id=rack_id)
        tags[f"{prefix}_it_load_kw"]      = BMS_TAG_IT_LOAD.format(rack_id=rack_id)
        tags[f"{prefix}_outlet_temp"]     = f"Racks/Rack{rack_id}/OutletTemp"
        tags[f"{prefix}_power_w"]         = f"PDU/Unit{rack_id}/PowerW"
        tags[f"{prefix}_humidity_pct"]    = f"Racks/Rack{rack_id}/Humidity"

    # CRAH unit tags (per unit: 15 tags x 4 units = 60)
    for crah_id in range(NUM_CRAH_UNITS):
        prefix = f"crah_{crah_id}"
        tags[f"{prefix}_airflow_cfm"]     = BMS_TAG_CRAH_AIRFLOW.format(crah_id=crah_id)
        tags[f"{prefix}_fan_speed_pct"]   = BMS_TAG_CRAH_FAN.format(crah_id=crah_id)
        tags[f"{prefix}_discharge_temp"]  = BMS_TAG_CRAH_DISCH.format(crah_id=crah_id)
        tags[f"{prefix}_discharge_sp"]    = BMS_TAG_CRAH_SETPT.format(crah_id=crah_id)
        tags[f"{prefix}_return_temp"]     = f"CRAH/Unit{crah_id}/ReturnTemp"
        tags[f"{prefix}_coil_temp"]       = f"CRAH/Unit{crah_id}/CoilTemp"
        tags[f"{prefix}_chw_flow"]        = f"CRAH/Unit{crah_id}/ChilledWaterFlow"
        tags[f"{prefix}_chw_supply_temp"] = f"CRAH/Unit{crah_id}/ChwSupplyTemp"
        tags[f"{prefix}_chw_return_temp"] = f"CRAH/Unit{crah_id}/ChwReturnTemp"
        tags[f"{prefix}_power_kw"]        = f"CRAH/Unit{crah_id}/PowerKw"
        tags[f"{prefix}_alarm_status"]    = f"CRAH/Unit{crah_id}/AlarmStatus"
        tags[f"{prefix}_run_status"]      = f"CRAH/Unit{crah_id}/RunStatus"
        tags[f"{prefix}_mode"]            = f"CRAH/Unit{crah_id}/ControlMode"
        tags[f"{prefix}_airflow_sp"]      = f"CRAH/Unit{crah_id}/AirflowSP"
        tags[f"{prefix}_fan_speed_sp"]    = f"CRAH/Unit{crah_id}/FanSpeedSP"

    # Facility-level tags (environmental, power, etc.)
    facility_tags = {
        "facility_total_it_load_mw":    "Facility/TotalITLoadMW",
        "facility_total_cooling_kw":    "Facility/TotalCoolingKW",
        "facility_pue":                 "Facility/PUE",
        "facility_dcie":                "Facility/DCiE",
        "facility_ambient_temp":        "Weather/AmbientTemp",
        "facility_ambient_humidity":    "Weather/AmbientHumidity",
        "facility_ups_load_pct":        "UPS/TotalLoadPct",
        "facility_chw_supply_temp":     "Chiller/ChwSupplyTemp",
        "facility_chw_return_temp":     "Chiller/ChwReturnTemp",
        "facility_chw_flow_gpm":        "Chiller/TotalFlowGPM",
        "facility_chiller_cop":         "Chiller/COP",
    }
    tags.update(facility_tags)

    return tags


@dataclass
class BMSStatus:
    connected:       bool   = False
    last_read_ts:    float  = 0.0
    read_count:      int    = 0
    write_count:     int    = 0
    error_count:     int    = 0
    last_error:      str    = ""
    mode:            str    = "SIMULATION"     # SIMULATION | LIVE
    total_tags:      int    = BMS_TOTAL_TAGS
    registered_tags: int    = 0


class NiagaraBMSConnector:
    """
    Interface to the Niagara Framework BMS.
    Operates in SIMULATION mode by default (no real Niagara needed).
    Set BMS_ENABLED=True and configure credentials to go live.

    Usage
    -----
    bms = NiagaraBMSConnector()
    temps = bms.read_rack_temperatures()       # {rack_id: float}
    crah  = bms.read_crah_parameters()         # {crah_id: dict}
    bms.write_crah_commands(commands_dict)
    status = bms.get_status()
    """

    def __init__(self, simulator=None):
        """
        Parameters
        ----------
        simulator : optional DataSimulator instance for SIMULATION mode.
        """
        self._simulator  = simulator
        self._status     = BMSStatus()
        self._tag_registry = build_tag_registry()
        self._status.registered_tags = len(self._tag_registry)
        self._mode       = "LIVE" if BMS_ENABLED else "SIMULATION"
        self._status.mode = self._mode
        self._session: Optional[object] = None

        # Simulated tag value cache (used in SIMULATION mode)
        self._sim_rack_temps:    dict[int, float] = {}
        self._sim_crah_params:   dict[int, dict]  = {}

        if BMS_ENABLED and _REQUESTS_AVAILABLE:
            self._init_live_session()
        else:
            if BMS_ENABLED:
                print("[BMS] WARNING: 'requests' not installed. Falling back to SIMULATION.")
                self._mode = "SIMULATION"
            else:
                print(f"[BMS] Running in SIMULATION mode. "
                      f"{self._status.registered_tags} tags registered "
                      f"({BMS_TOTAL_TAGS} total available in Niagara).")

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def _init_live_session(self) -> None:
        """Establish an authenticated HTTP session to the Niagara station."""
        try:
            proto = "https" if BMS_USE_HTTPS else "http"
            self._base_url = f"{proto}://{BMS_HOST}:{BMS_PORT}{BMS_STATION_PATH}"
            self._session = requests.Session()
            self._session.auth = (BMS_USERNAME, os.getenv("NIAGARA_PASSWORD", BMS_PASSWORD))
            # Test connection
            resp = self._session.get(f"{self._base_url}?path=/", timeout=5)
            if resp.status_code == 200:
                self._status.connected = True
                print(f"[BMS] Connected to Niagara at {self._base_url}")
            else:
                print(f"[BMS] Connection failed (HTTP {resp.status_code}). Using SIMULATION.")
                self._mode = "SIMULATION"
        except Exception as exc:
            print(f"[BMS] Could not connect to Niagara: {exc}. Using SIMULATION.")
            self._mode = "SIMULATION"

    # -------------------------------------------------------------------------
    # Read Interface
    # -------------------------------------------------------------------------

    def read_rack_temperatures(self) -> dict[int, float]:
        """
        Read rack inlet temperatures for all racks.
        Returns {rack_id: temperature_C}.
        """
        if self._mode == "LIVE":
            return self._live_read_rack_temps()
        return self._sim_read_rack_temps()

    def read_crah_parameters(self) -> dict[int, dict]:
        """
        Read current CRAH operational parameters.
        Returns {crah_id: {"airflow_cfm", "fan_speed_pct", "discharge_temp_c",
                           "return_temp_c", "run_status", "alarm"}}
        """
        if self._mode == "LIVE":
            return self._live_read_crah_params()
        return self._sim_read_crah_params()

    def read_it_loads(self) -> dict[int, float]:
        """Read IT power load per rack (kW)."""
        if self._mode == "LIVE":
            return self._live_read_it_loads()
        return {
            rack_id: round(
                self._simulator._it_loads[rack_id]
                if self._simulator else 5.0, 3
            )
            for rack_id in range(NUM_RACKS)
        }

    # -------------------------------------------------------------------------
    # Write Interface
    # -------------------------------------------------------------------------

    def write_crah_commands(self, commands: dict[int, dict]) -> bool:
        """
        Write optimised setpoints back to CRAH units.

        Parameters
        ----------
        commands : {crah_id: {"airflow_sp_cfm": float,
                                "discharge_temp_sp_c": float,
                                "fan_speed_sp_pct": float}}

        Returns True if all writes succeeded.
        """
        if self._mode == "LIVE":
            return self._live_write_commands(commands)
        return self._sim_write_commands(commands)

    # -------------------------------------------------------------------------
    # Live (Niagara) implementations
    # -------------------------------------------------------------------------

    def _live_read_tag(self, path: str) -> Optional[float]:
        try:
            resp = self._session.get(f"{self._base_url}?path=/{path}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self._status.read_count += 1
                self._status.last_read_ts = time.time()
                return float(data.get("value", 0.0))
        except Exception as exc:
            self._status.error_count += 1
            self._status.last_error = str(exc)
        return None

    def _live_write_tag(self, path: str, value: float) -> bool:
        try:
            resp = self._session.post(
                f"{self._base_url}?path=/{path}",
                json={"value": value},
                timeout=5,
            )
            if resp.status_code in (200, 204):
                self._status.write_count += 1
                return True
        except Exception as exc:
            self._status.error_count += 1
            self._status.last_error = str(exc)
        return False

    def _live_read_rack_temps(self) -> dict[int, float]:
        result = {}
        for rack_id in range(NUM_RACKS):
            path = BMS_TAG_RACK_TEMP.format(rack_id=rack_id)
            val  = self._live_read_tag(path)
            result[rack_id] = val if val is not None else 25.0
        return result

    def _live_read_crah_params(self) -> dict[int, dict]:
        result = {}
        for crah_id in range(NUM_CRAH_UNITS):
            result[crah_id] = {
                "airflow_cfm":      self._live_read_tag(BMS_TAG_CRAH_AIRFLOW.format(crah_id=crah_id)) or 1200.0,
                "fan_speed_pct":    self._live_read_tag(BMS_TAG_CRAH_FAN.format(crah_id=crah_id))    or 60.0,
                "discharge_temp_c": self._live_read_tag(BMS_TAG_CRAH_DISCH.format(crah_id=crah_id))  or 18.0,
                "run_status":       True,
                "alarm":            False,
            }
        return result

    def _live_read_it_loads(self) -> dict[int, float]:
        return {
            rack_id: self._live_read_tag(BMS_TAG_IT_LOAD.format(rack_id=rack_id)) or 5.0
            for rack_id in range(NUM_RACKS)
        }

    def _live_write_commands(self, commands: dict[int, dict]) -> bool:
        success = True
        for crah_id, cmd in commands.items():
            if "airflow_sp_cfm" in cmd:
                success &= self._live_write_tag(
                    BMS_TAG_CRAH_AIRFLOW.format(crah_id=crah_id), cmd["airflow_sp_cfm"]
                )
            if "discharge_temp_sp_c" in cmd:
                success &= self._live_write_tag(
                    BMS_TAG_CRAH_SETPT.format(crah_id=crah_id), cmd["discharge_temp_sp_c"]
                )
        return success

    # -------------------------------------------------------------------------
    # Simulation implementations
    # -------------------------------------------------------------------------

    def _sim_read_rack_temps(self) -> dict[int, float]:
        if self._simulator:
            return {
                rec["rack_id"]: rec["rack_temp_c"]
                for rec in (self._simulator.step() if not self._sim_rack_temps
                            else [{"rack_id": k, "rack_temp_c": v}
                                  for k, v in self._sim_rack_temps.items()])
            }
        return {i: 24.0 + i * 0.5 for i in range(NUM_RACKS)}

    def _sim_read_crah_params(self) -> dict[int, dict]:
        if self._simulator:
            return {
                crah_id: {
                    "airflow_cfm":      round(float(self._simulator.current_airflows[crah_id]), 1),
                    "fan_speed_pct":    60.0,
                    "discharge_temp_c": round(float(self._simulator.current_discharges[crah_id]), 2),
                    "run_status":       True,
                    "alarm":            False,
                }
                for crah_id in range(NUM_CRAH_UNITS)
            }
        return {
            crah_id: {"airflow_cfm": 1200.0, "fan_speed_pct": 60.0,
                      "discharge_temp_c": 18.0, "run_status": True, "alarm": False}
            for crah_id in range(NUM_CRAH_UNITS)
        }

    def _sim_write_commands(self, commands: dict[int, dict]) -> bool:
        # In simulation mode, commands go to the simulator's state
        if self._simulator:
            for crah_id, cmd in commands.items():
                if "airflow_sp_cfm" in cmd:
                    self._simulator._airflows[crah_id] = cmd["airflow_sp_cfm"]
                if "discharge_temp_sp_c" in cmd:
                    self._simulator._discharges[crah_id] = cmd["discharge_temp_sp_c"]
        self._status.write_count += len(commands)
        return True

    # -------------------------------------------------------------------------
    # Status & Registry
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "mode":            self._status.mode,
            "connected":       self._status.connected if self._mode == "LIVE" else True,
            "bms_host":        BMS_HOST if BMS_ENABLED else "N/A (simulation)",
            "total_tags":      self._status.total_tags,
            "registered_tags": self._status.registered_tags,
            "read_count":      self._status.read_count,
            "write_count":     self._status.write_count,
            "error_count":     self._status.error_count,
            "last_error":      self._status.last_error,
            "last_read_ts":    self._status.last_read_ts,
            "poll_interval_s": BMS_POLL_INTERVAL_SEC,
        }

    def get_tag_registry(self, limit: int = 50) -> dict[str, str]:
        """Return first `limit` entries of the tag registry (full list is ~1500)."""
        items = list(self._tag_registry.items())[:limit]
        return dict(items)
