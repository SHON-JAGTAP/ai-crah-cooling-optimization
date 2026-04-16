"""
modules/hotspot_detector.py
----------------------------
Classifies rack temperatures and detects hotspots.

Changes from v1:
  + Compound hotspot condition (2B from document):
      Hotspot = (T > threshold) AND (zone_temp_spread > ALLOWABLE_ZONE_TEMP_VARIATION)
  + aisle field added to RackAlert and ZoneSummary
  + zone_spread_c and is_compound_hotspot added to ZoneSummary
  + is_compound_hotspot flag on RackAlert
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import (
    ALLOWABLE_ZONE_TEMP_VARIATION,
    COMPOUND_HOTSPOT_REQUIRES_SPREAD,
    CRAH_TO_AISLE,
    RACK_TO_AISLE,
    TEMP_HOTSPOT_CRITICAL,
    TEMP_HOTSPOT_HIGH,
    TEMP_HOTSPOT_MEDIUM,
    TEMP_OPTIMAL_HIGH,
    TEMP_OPTIMAL_LOW,
)


class Severity(str, Enum):
    OVERCOOL = "OVERCOOL"
    NORMAL   = "NORMAL"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def numeric(self) -> int:
        return {"OVERCOOL": -1, "NORMAL": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[self.value]

    @property
    def color(self) -> str:
        return {
            "OVERCOOL": "#38bdf8",
            "NORMAL":   "#22c55e",
            "MEDIUM":   "#facc15",
            "HIGH":     "#f97316",
            "CRITICAL": "#ef4444",
        }[self.value]


@dataclass
class RackAlert:
    rack_id:             int
    crah_id:             int
    aisle:               str          # e.g., "Aisle-A"
    temp_c:              float
    severity:            Severity
    delta_c:             float        # deviation from TEMP_OPTIMAL_HIGH
    message:             str
    is_compound_hotspot: bool = False  # True when compound condition triggered


@dataclass
class ZoneSummary:
    crah_id:             int
    aisle:               str
    rack_ids:            list[int]
    avg_temp_c:          float
    max_temp_c:          float
    min_temp_c:          float
    zone_spread_c:       float        # max - min within the zone
    worst_severity:      Severity
    is_compound_hotspot: bool         # zone spread exceeds allowable variation
    alerts:              list[RackAlert] = field(default_factory=list)
    overcooled_racks:    list[int]    = field(default_factory=list)
    hotspot_racks:       list[int]    = field(default_factory=list)


def classify_rack(
    rack_id: int,
    crah_id: int,
    temp_c: float,
    zone_spread_c: float = 0.0,
) -> RackAlert:
    """
    Classify a single rack reading.

    Compound condition (2B):
      is_compound_hotspot = (T > TEMP_HOTSPOT_MEDIUM) AND (spread > ALLOWABLE_ZONE_TEMP_VARIATION)
    """
    aisle = RACK_TO_AISLE.get(rack_id, f"Zone-{crah_id}")

    if temp_c < TEMP_OPTIMAL_LOW:
        sev = Severity.OVERCOOL
        msg = f"{aisle}/R{rack_id} overcooled ({temp_c:.1f}°C < {TEMP_OPTIMAL_LOW}°C)"
    elif temp_c < TEMP_OPTIMAL_HIGH:
        sev = Severity.NORMAL
        msg = f"{aisle}/R{rack_id} nominal ({temp_c:.1f}°C)"
    elif temp_c < TEMP_HOTSPOT_MEDIUM:
        sev = Severity.MEDIUM
        msg = f"{aisle}/R{rack_id} slightly warm ({temp_c:.1f}°C)"
    elif temp_c < TEMP_HOTSPOT_HIGH:
        sev = Severity.HIGH
        msg = f"{aisle}/R{rack_id} HOT ({temp_c:.1f}°C)"
    else:
        sev = Severity.CRITICAL
        msg = f"{aisle}/R{rack_id} CRITICAL ({temp_c:.1f}°C) -- immediate action needed"

    # Compound hotspot condition
    compound = (
        COMPOUND_HOTSPOT_REQUIRES_SPREAD
        and temp_c >= TEMP_HOTSPOT_MEDIUM
        and zone_spread_c > ALLOWABLE_ZONE_TEMP_VARIATION
    )
    if compound and sev == Severity.MEDIUM:
        # Upgrade severity one level due to uneven distribution
        sev = Severity.HIGH
        msg += f" [compound: spread {zone_spread_c:.1f}°C > {ALLOWABLE_ZONE_TEMP_VARIATION}°C]"

    return RackAlert(
        rack_id=rack_id,
        crah_id=crah_id,
        aisle=aisle,
        temp_c=temp_c,
        severity=sev,
        delta_c=round(temp_c - TEMP_OPTIMAL_HIGH, 2),
        message=msg,
        is_compound_hotspot=compound,
    )


class HotspotDetector:
    """
    Detects hotspots using compound condition:
      (T_rack > threshold) AND (zone_spread > ALLOWABLE_ZONE_TEMP_VARIATION)
    """

    def detect(
        self, records: list[dict]
    ) -> tuple[list[RackAlert], dict[int, ZoneSummary]]:
        # --- First pass: collect zone temperatures ---------------------------
        zone_temps: dict[int, list[float]]    = {}
        zone_rack_map: dict[int, list[dict]]  = {}
        for rec in records:
            cid = rec["crah_id"]
            zone_temps.setdefault(cid, []).append(rec["rack_temp_c"])
            zone_rack_map.setdefault(cid, []).append(rec)

        # --- Second pass: classify with spread context ----------------------
        alerts: list[RackAlert] = []
        zones:  dict[int, ZoneSummary] = {}

        for crah_id, temp_list in zone_temps.items():
            zone_max    = max(temp_list)
            zone_min    = min(temp_list)
            zone_spread = round(zone_max - zone_min, 2)
            zone_avg    = round(sum(temp_list) / len(temp_list), 2)
            aisle       = CRAH_TO_AISLE.get(crah_id, f"Zone-{crah_id}")

            zone_alerts: list[RackAlert] = []
            for rec in zone_rack_map[crah_id]:
                alert = classify_rack(
                    rec["rack_id"], rec["crah_id"], rec["rack_temp_c"], zone_spread
                )
                alerts.append(alert)
                zone_alerts.append(alert)

            worst           = max(zone_alerts, key=lambda a: a.severity.numeric).severity
            hotspot_racks   = [a.rack_id for a in zone_alerts if a.severity.numeric > 0]
            overcooled_racks= [a.rack_id for a in zone_alerts if a.severity == Severity.OVERCOOL]
            compound_zone   = zone_spread > ALLOWABLE_ZONE_TEMP_VARIATION

            zones[crah_id] = ZoneSummary(
                crah_id=crah_id,
                aisle=aisle,
                rack_ids=[a.rack_id for a in zone_alerts],
                avg_temp_c=zone_avg,
                max_temp_c=round(zone_max, 2),
                min_temp_c=round(zone_min, 2),
                zone_spread_c=zone_spread,
                worst_severity=worst,
                is_compound_hotspot=compound_zone,
                alerts=zone_alerts,
                hotspot_racks=hotspot_racks,
                overcooled_racks=overcooled_racks,
            )

        alerts.sort(key=lambda a: a.severity.numeric, reverse=True)
        return alerts, zones

    @staticmethod
    def summary_str(alerts: list[RackAlert]) -> str:
        counts = {s: 0 for s in Severity}
        for a in alerts:
            counts[a.severity] += 1
        parts = [f"{s.value}: {n}" for s, n in counts.items() if n > 0]
        return " | ".join(parts) if parts else "All nominal"
