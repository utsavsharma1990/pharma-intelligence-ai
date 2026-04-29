"""
Parser: raw ClinicalTrials.gov API JSON → ParsedTrial domain objects.

Why a dedicated parser file?
  - Single source of truth for "how to read the API"
  - Easy to unit test (input: dict, output: ParsedTrial)
  - When the API changes, this is the only file we touch
  - Defensive parsing — every .get() has a default; missing fields don't crash

Design rule: this module NEVER raises on missing fields. Real-world clinical
trial JSON is messy — old trials lack adverse event modules, some have no
sponsors, etc. We accept what's there and default the rest. Validation
(if needed) happens at a higher layer.
"""

from typing import Any, Optional

from src.ingestion.models import (
    AdverseEvent,
    Intervention,
    Location,
    Outcome,
    ParsedTrial,
)


def parse_trial(raw: dict[str, Any]) -> ParsedTrial:
    """
    Convert one raw study dict (from the API) into a ParsedTrial.

    The raw shape is:
        {"protocolSection": {"identificationModule": {...}, "statusModule": {...}, ...}}
    """
    protocol = raw.get("protocolSection", {})

    return ParsedTrial(
        # --- Identification ---
        nct_id         = _safe(protocol, "identificationModule", "nctId", default=""),
        brief_title    = _safe(protocol, "identificationModule", "briefTitle", default=""),
        official_title = _safe(protocol, "identificationModule", "officialTitle"),

        # --- Status ---
        overall_status          = _safe(protocol, "statusModule", "overallStatus"),
        start_date              = _safe(protocol, "statusModule", "startDateStruct", "date"),
        primary_completion_date = _safe(protocol, "statusModule", "primaryCompletionDateStruct", "date"),

        # --- Design ---
        phases           = _safe(protocol, "designModule", "phases", default=[]) or [],
        study_type       = _safe(protocol, "designModule", "studyType"),
        enrollment_count = _safe(protocol, "designModule", "enrollmentInfo", "count"),

        # --- Eligibility ---
        eligibility_criteria = _safe(protocol, "eligibilityModule", "eligibilityCriteria", default=""),
        minimum_age          = _safe(protocol, "eligibilityModule", "minimumAge"),
        sex                  = _safe(protocol, "eligibilityModule", "sex"),

        # --- Interventions ---
        interventions = _parse_interventions(protocol),

        # --- Outcomes ---
        primary_outcomes   = _parse_outcomes(protocol, "primaryOutcomes"),
        secondary_outcomes = _parse_outcomes(protocol, "secondaryOutcomes"),

        # --- Sponsor ---
        lead_sponsor  = _safe(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        sponsor_class = _safe(protocol, "sponsorCollaboratorsModule", "leadSponsor", "class"),

        # --- Locations ---
        locations = _parse_locations(protocol),

        # --- Adverse events ---
        adverse_events = _parse_adverse_events(protocol),
    )


# ---------------------------------------------------------------------------
# Section-level parsers — kept private (underscore prefix)
# ---------------------------------------------------------------------------

def _parse_interventions(protocol: dict) -> list[Intervention]:
    raw_list = _safe(protocol, "armsInterventionsModule", "interventions", default=[]) or []
    return [
        Intervention(
            type=item.get("type", "UNKNOWN"),
            name=item.get("name", ""),
            description=item.get("description"),
        )
        for item in raw_list
    ]


def _parse_outcomes(protocol: dict, key: str) -> list[Outcome]:
    """key is 'primaryOutcomes' or 'secondaryOutcomes'."""
    raw_list = _safe(protocol, "outcomesModule", key, default=[]) or []
    return [
        Outcome(
            measure=item.get("measure", ""),
            description=item.get("description"),
            time_frame=item.get("timeFrame"),
        )
        for item in raw_list
    ]


def _parse_locations(protocol: dict) -> list[Location]:
    raw_list = _safe(protocol, "contactsLocationsModule", "locations", default=[]) or []
    return [
        Location(
            facility=item.get("facility"),
            city=item.get("city"),
            state=item.get("state"),
            country=item.get("country"),
        )
        for item in raw_list
    ]


def _parse_adverse_events(protocol: dict) -> list[AdverseEvent]:
    """
    AEs come in two buckets in the API: 'seriousEvents' and 'otherEvents'.
    We flatten them into a single list with a `serious` boolean flag.
    Each event has a 'stats' array — for our purposes the first entry
    (overall trial-wide counts) is what we care about.
    """
    ae_module = _safe(protocol, "adverseEventsModule", default={}) or {}
    events: list[AdverseEvent] = []

    for raw_event in ae_module.get("seriousEvents", []):
        events.append(_one_ae(raw_event, serious=True))

    for raw_event in ae_module.get("otherEvents", []):
        events.append(_one_ae(raw_event, serious=False))

    return events


def _one_ae(raw_event: dict, *, serious: bool) -> AdverseEvent:
    stats = raw_event.get("stats", [])
    first = stats[0] if stats else {}
    return AdverseEvent(
        term=raw_event.get("term", ""),
        organ_system=raw_event.get("organSystem"),
        num_affected=first.get("numAffected", 0),
        num_at_risk=first.get("numAtRisk", 0),
        serious=serious,
    )


# ---------------------------------------------------------------------------
# _safe: defensive nested-dict accessor.
# Replaces this fragile pattern:
#     protocol["identificationModule"]["nctId"]   # KeyError if missing
# With:
#     _safe(protocol, "identificationModule", "nctId", default="")
# Walks the nested dict, returning `default` if any key is missing.
# ---------------------------------------------------------------------------

def _safe(d: dict, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur