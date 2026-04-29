"""Unit tests for the trial parser."""

import json
from pathlib import Path

import pytest

from src.ingestion.models import ParsedTrial, AdverseEvent
from src.ingestion.parser import parse_trial


@pytest.fixture
def mock_studies() -> list[dict]:
    """Load all mock studies from the JSON file."""
    raw = json.loads(Path("data/raw/mock_trials.json").read_text())
    return raw["studies"]


# ---------- Top-level parsing ----------

def test_parse_returns_parsed_trial(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert isinstance(trial, ParsedTrial)


def test_basic_fields(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert trial.nct_id == "NCT05123456"
    assert "Pembrolizumab" in trial.brief_title
    assert trial.overall_status == "RECRUITING"
    assert trial.phases == ["PHASE3"]
    assert trial.lead_sponsor == "Merck Sharp & Dohme LLC"
    assert trial.enrollment_count == 800


def test_eligibility_text_extracted(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert "Inclusion Criteria" in trial.eligibility_criteria
    assert "Exclusion Criteria" in trial.eligibility_criteria


def test_interventions(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert len(trial.interventions) == 2
    names = trial.intervention_names
    assert "Pembrolizumab" in names
    assert "Carboplatin" in names


def test_outcomes(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert len(trial.primary_outcomes) == 1
    assert "Overall Survival" in trial.primary_outcomes[0].measure
    assert len(trial.secondary_outcomes) == 2


def test_locations(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert len(trial.locations) == 2
    assert any("New York" in (loc.city or "") for loc in trial.locations)


# ---------- Adverse events ----------

def test_serious_vs_non_serious_aes(mock_studies):
    trial = parse_trial(mock_studies[0])
    assert len(trial.serious_adverse_events) == 2
    serious_terms = [ae.term for ae in trial.serious_adverse_events]
    assert "Pneumonitis" in serious_terms
    assert "Colitis" in serious_terms


def test_ae_incidence_calculation(mock_studies):
    trial = parse_trial(mock_studies[0])
    pneumonitis = next(ae for ae in trial.adverse_events if ae.term == "Pneumonitis")
    assert pneumonitis.num_affected == 24
    assert pneumonitis.num_at_risk == 800
    assert pneumonitis.incidence_pct == 3.0


def test_ae_incidence_handles_zero_at_risk():
    """Defensive: division-by-zero shouldn't crash."""
    ae = AdverseEvent(term="Test", num_affected=5, num_at_risk=0)
    assert ae.incidence_pct == 0.0


# ---------- Defensive parsing ----------

def test_parser_handles_empty_input():
    """Missing fields shouldn't crash — they should produce empty defaults."""
    trial = parse_trial({})
    assert trial.nct_id == ""
    assert trial.phases == []
    assert trial.interventions == []
    assert trial.adverse_events == []


def test_parser_handles_partial_input():
    """Trial with only identification — no design, no AEs."""
    trial = parse_trial({
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999999", "briefTitle": "Tiny"}
        }
    })
    assert trial.nct_id == "NCT99999999"
    assert trial.brief_title == "Tiny"
    assert trial.lead_sponsor is None
    assert trial.adverse_events == []


def test_parsed_trial_serializes_to_json(mock_studies):
    """Pydantic gives us free JSON serialization — we'll use this in the cache."""
    trial = parse_trial(mock_studies[0])
    j = trial.model_dump_json()
    assert "NCT05123456" in j
    # Round-trip
    rebuilt = ParsedTrial.model_validate_json(j)
    assert rebuilt.nct_id == trial.nct_id


# ---------- All three mock trials parse cleanly ----------

def test_all_mock_trials_parse(mock_studies):
    for raw in mock_studies:
        trial = parse_trial(raw)
        assert trial.nct_id.startswith("NCT")
        assert trial.brief_title  # non-empty