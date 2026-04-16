"""
modules/claude_agent.py
------------------------
Integration with Anthropic's Claude Managed Agents API (beta).

The module:
  1. Creates a persistent Claude agent specialised in CRAH optimization.
  2. Starts a session per analysis request (or reuses an active one).
  3. Sends current system telemetry and receives structured recommendations.
  4. Falls back to a rule-based summary if the API is unavailable.

Beta header required: managed-agents-2026-04-01
"""

import json
import os
import time
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from config import (
    AGENT_NAME,
    AGENT_SYSTEM_PROMPT,
    ANTHROPIC_MODEL,
    BETA_HEADER,
    NUM_CRAH_UNITS,
)


class CRAHClaudeAgent:
    """
    Wrapper around the Claude Managed Agents API for CRAH optimization advice.

    Usage
    -----
    agent = CRAHClaudeAgent(api_key="sk-ant-...")
    rec   = agent.get_recommendations(telemetry_dict)
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key  = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client   = None
        self._agent_id : Optional[str] = None
        self._env_id   : Optional[str] = None
        self._session_id: Optional[str] = None
        self._available = False
        self._last_recommendation: str = "No AI recommendation yet."

        self._init_client()

    # -- Initialisation --------------------------------------------------------

    def _init_client(self) -> None:
        if not _ANTHROPIC_AVAILABLE:
            print("[ClaudeAgent] anthropic package not installed -- using fallback.")
            return
        if not self._api_key:
            print("[ClaudeAgent] No API key -- using fallback rule-based summaries.")
            return
        try:
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                default_headers={"anthropic-beta": BETA_HEADER},
            )
            self._available = True
            print("[ClaudeAgent] Client initialised.")
            self._create_agent()
        except Exception as exc:
            print(f"[ClaudeAgent] Init error: {exc} -- using fallback.")

    def _create_agent(self) -> None:
        """Create a persistent managed agent (called once at startup)."""
        try:
            agent = self._client.beta.agents.create(
                name=AGENT_NAME,
                model=ANTHROPIC_MODEL,
                system_prompt=AGENT_SYSTEM_PROMPT,
            )
            self._agent_id = agent.id
            print(f"[ClaudeAgent] Agent created: {self._agent_id}")
        except Exception as exc:
            print(f"[ClaudeAgent] Could not create agent: {exc}")
            self._available = False

    def _ensure_session(self) -> bool:
        """Create a new session if none is active."""
        if not self._available or not self._agent_id:
            return False
        try:
            # Create environment
            env = self._client.beta.environments.create(
                name="crah-cooling-env",
            )
            self._env_id = env.id

            # Start session
            session = self._client.beta.sessions.create(
                agent_id=self._agent_id,
                environment_id=self._env_id,
            )
            self._session_id = session.id
            return True
        except Exception as exc:
            print(f"[ClaudeAgent] Session creation error: {exc}")
            self._session_id = None
            return False

    # -- Core API call ---------------------------------------------------------

    def get_recommendations(self, telemetry: dict) -> str:
        """
        Send current telemetry to Claude and retrieve cooling recommendations.

        Parameters
        ----------
        telemetry : dict with keys:
            step, avg_rack_temp, max_rack_temp, min_rack_temp,
            crah_states (list), hotspot_racks (list), overcooled_racks (list),
            it_load_kw_total, pue

        Returns
        -------
        Recommendation string (plain text).
        """
        if not self._available:
            return self._fallback_recommendation(telemetry)

        if not self._session_id:
            if not self._ensure_session():
                return self._fallback_recommendation(telemetry)

        prompt = self._build_prompt(telemetry)

        try:
            event = self._client.beta.sessions.events.create(
                session_id=self._session_id,
                event={
                    "type": "user",
                    "content": prompt,
                },
            )
            # Stream & collect response
            response_text = self._collect_sse(event)
            self._last_recommendation = response_text
            return response_text

        except Exception as exc:
            print(f"[ClaudeAgent] API call failed: {exc} -- using fallback.")
            return self._fallback_recommendation(telemetry)

    def _collect_sse(self, event_response) -> str:
        """Extract text from a streaming SSE response."""
        try:
            # SDK may return a streaming object
            if hasattr(event_response, "__iter__"):
                parts = []
                for chunk in event_response:
                    if hasattr(chunk, "content"):
                        parts.append(str(chunk.content))
                return " ".join(parts).strip() or "No response received."
            elif hasattr(event_response, "content"):
                return str(event_response.content)
            return str(event_response)
        except Exception:
            return "Response parsing error."

    # -- Prompt builder --------------------------------------------------------

    @staticmethod
    def _build_prompt(t: dict) -> str:
        crah_lines = "\n".join(
            f"  CRAH-{s['unit_id']}: fan={s['fan_speed_pct']}% | "
            f"airflow={s['airflow_cfm']} CFM | "
            f"discharge={s['discharge_temp_c']}°C | mode={s['mode']}"
            for s in t.get("crah_states", [])
        )
        hotspot_str   = ", ".join(map(str, t.get("hotspot_racks", []))) or "None"
        overcool_str  = ", ".join(map(str, t.get("overcooled_racks", []))) or "None"

        return f"""
=== DATA CENTER COOLING TELEMETRY (Step {t.get("step", "?")}) ===

Rack Temperature Summary:
  Average : {t.get("avg_rack_temp", "?")}°C
  Maximum : {t.get("max_rack_temp", "?")}°C
  Minimum : {t.get("min_rack_temp", "?")}°C

CRAH Unit Status:
{crah_lines}

Active Hotspot Racks  : {hotspot_str}
Overcooled Racks      : {overcool_str}

System Metrics:
  Total IT Load : {t.get("it_load_kw_total", "?")} kW
  Current PUE   : {t.get("pue", "?")}

Based on this data, provide:
1. A brief assessment of the current thermal state (1-2 sentences).
2. The top 2-3 specific, actionable optimization recommendations.
3. An estimated PUE improvement if recommendations are followed.
Keep the response under 200 words and structured.
""".strip()

    # -- Fallback -------------------------------------------------------------

    @staticmethod
    def _fallback_recommendation(t: dict) -> str:
        """Rule-based summary used when API is unavailable."""
        avg  = t.get("avg_rack_temp", 0)
        max_ = t.get("max_rack_temp", 0)
        hots = t.get("hotspot_racks", [])
        cold = t.get("overcooled_racks", [])
        pue  = t.get("pue", 1.5)

        lines = ["[AI Fallback -- Connect API key for Claude analysis]"]
        if max_ > 32:
            lines.append(f"⚠ CRITICAL: Rack(s) {hots} at {max_:.1f}°C. "
                         f"Increase CRAH airflow immediately.")
        elif max_ > 28:
            lines.append(f"⚡ HIGH: Rack(s) {hots} at {max_:.1f}°C. "
                         f"Boost airflow in affected zones.")
        elif avg < 22 and cold:
            lines.append(f"❄ OVERCOOLING: Rack(s) {cold} below optimal. "
                         f"Reduce airflow to save energy.")
        else:
            lines.append(f"✅ System nominal. Average rack temp {avg:.1f}°C. "
                         f"Maintain current settings.")
        lines.append(f"PUE: {pue:.3f} -- {'Optimize cooling to improve.' if pue > 1.5 else 'Good efficiency.'}")
        return "\n".join(lines)

    @property
    def last_recommendation(self) -> str:
        return self._last_recommendation

    @property
    def is_available(self) -> bool:
        return self._available
