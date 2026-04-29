"""Unit tests for the domain-aware chunker."""

import json
from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_trial, chunk_trials, _split_text
from src.ingestion.models import ParsedTrial
from src.ingestion.parser import parse_trial


@pytest.fixture
def trial() -> ParsedTrial:
    raw = json.loads(Path("data/raw/mock_trials.json").read_text())
    return parse_trial(raw["studies"][0])  # NCT05123456 (Pembrolizumab)


def test_chunker_produces_chunks(trial):
    chunks = chunk_trial(trial)
    assert len(chunks) >= 5  # at least overview/elig/endpoints/intervs/AEs


def test_each_section_present(trial):
    chunks = chunk_trial(trial)
    sections = {c.section_type for c in chunks}
    assert "overview" in sections
    assert "eligibility" in sections
    assert "endpoints" in sections
    assert "interventions" in sections
    assert "adverse_events" in sections


def test_chunk_ids_are_unique(trial):
    chunks = chunk_trial(trial)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_every_chunk_has_nct_id(trial):
    chunks = chunk_trial(trial)
    for c in chunks:
        assert c.nct_id == "NCT05123456"


def test_metadata_is_propagated(trial):
    chunks = chunk_trial(trial)
    for c in chunks:
        assert c.phase == "PHASE3"
        assert c.sponsor == "Merck Sharp & Dohme LLC"
        assert c.overall_status == "RECRUITING"


def test_overview_chunk_includes_title_and_phase(trial):
    chunks = chunk_trial(trial)
    overview = next(c for c in chunks if c.section_type == "overview")
    assert "NCT05123456" in overview.content
    assert "PHASE3" in overview.content


def test_eligibility_chunk_includes_inclusion_criteria(trial):
    chunks = chunk_trial(trial)
    elig = next(c for c in chunks if c.section_type == "eligibility")
    assert "Inclusion Criteria" in elig.content
    assert "Exclusion Criteria" in elig.content


def test_ae_chunk_distinguishes_serious_from_common(trial):
    chunks = chunk_trial(trial)
    ae = next(c for c in chunks if c.section_type == "adverse_events")
    assert "Serious adverse events" in ae.content
    assert "Common adverse events" in ae.content
    assert "Pneumonitis" in ae.content


def test_to_metadata_dict_has_no_none_values(trial):
    chunks = chunk_trial(trial)
    for c in chunks:
        meta = c.to_metadata_dict()
        for v in meta.values():
            assert v is not None


def test_to_metadata_dict_excludes_content(trial):
    chunks = chunk_trial(trial)
    meta = chunks[0].to_metadata_dict()
    assert "content" not in meta
    assert "nct_id" in meta
    assert "section_type" in meta


# ---------- Bulk + edge cases ----------

def test_chunk_trials_handles_multiple(trial):
    raw = json.loads(Path("data/raw/mock_trials.json").read_text())
    trials = [parse_trial(s) for s in raw["studies"]]
    all_chunks = chunk_trials(trials)
    nct_ids = {c.nct_id for c in all_chunks}
    assert nct_ids == {"NCT05123456", "NCT04567890", "NCT06789012"}


def test_chunker_handles_minimal_trial():
    """Trial with just identification — no eligibility, no AEs."""
    minimal = ParsedTrial(nct_id="NCT99999999", brief_title="Minimal")
    chunks = chunk_trial(minimal)
    # Should at least have an overview chunk
    sections = {c.section_type for c in chunks}
    assert "overview" in sections
    # And NOT have sections we don't have data for
    assert "adverse_events" not in sections
    assert "eligibility" not in sections


# ---------- Splitter edge cases ----------

def test_split_short_text_returns_single_chunk():
    assert _split_text("short text", max_chars=100, overlap=10) == ["short text"]


def test_split_long_text_produces_multiple_chunks():
    long_text = "Sentence one. " * 200  # ~2800 chars
    parts = _split_text(long_text, max_chars=1000, overlap=100)
    assert len(parts) >= 2
    # Every part fits within max_chars (with some tolerance for boundary search)
    for p in parts:
        assert len(p) <= 1100


def test_split_preserves_no_data_loss():
    """The concatenated chunks should contain everything from the original."""
    original = "Para one.\n\nPara two has more text in it.\n\nPara three is here."
    parts = _split_text(original, max_chars=30, overlap=5)
    # We can't easily assert exact equality due to overlap, but key
    # phrases should appear at least once
    joined = " ".join(parts)
    for keyword in ["Para one", "Para two", "Para three"]:
        assert keyword in joined