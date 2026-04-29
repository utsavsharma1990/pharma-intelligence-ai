"""
Domain-aware chunker for clinical trials.

Generic chunking (split every N chars) destroys the semantic structure
of clinical trials. A query about safety should retrieve the AE section,
not a chunk that bridges eligibility + interventions.

This chunker:
  1. Splits trials into LOGICAL SECTIONS (overview, eligibility, endpoints,
     interventions, adverse_events) — preserving meaning
  2. If a section is too large (e.g. long eligibility criteria), splits it
     by character count with overlap
  3. Attaches RICH METADATA to every chunk for hybrid filtering

Why these specific sections?
  - overview: title + sponsor + phase + status (good for "find all trials...")
  - eligibility: who can enroll (good for "patient X qualifies for what?")
  - endpoints: what's being measured (good for "what was OS in trial Y?")
  - interventions: what drugs/doses (good for "compare regimens")
  - adverse_events: safety profile (good for "compare AEs")

These map 1:1 to our four agent specializations.
"""

from typing import Iterable

from src.ingestion.models import ParsedTrial, TrialChunk


# Tunable chunking parameters.
# 1500 chars ≈ 300-400 tokens for English biomedical text — comfortably
# below the embedding model's 512-token limit, with headroom.
MAX_CHUNK_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_trial(trial: ParsedTrial) -> list[TrialChunk]:
    """
    Convert one parsed trial into a list of retrievable chunks.

    Returns 5 chunks if all sections present (overview/eligibility/endpoints/
    interventions/adverse_events), or fewer if some sections are empty.
    Long sections may produce multiple chunks of the same section_type.
    """
    base_metadata = _base_metadata(trial)
    chunks: list[TrialChunk] = []

    # --- Section 1: Overview ---
    overview_text = _build_overview(trial)
    chunks.extend(_make_chunks(
        nct_id=trial.nct_id,
        section_type="overview",
        text=overview_text,
        base_metadata=base_metadata,
    ))

    # --- Section 2: Eligibility ---
    if trial.eligibility_criteria.strip():
        chunks.extend(_make_chunks(
            nct_id=trial.nct_id,
            section_type="eligibility",
            text=_build_eligibility(trial),
            base_metadata=base_metadata,
        ))

    # --- Section 3: Endpoints ---
    endpoints_text = _build_endpoints(trial)
    if endpoints_text.strip():
        chunks.extend(_make_chunks(
            nct_id=trial.nct_id,
            section_type="endpoints",
            text=endpoints_text,
            base_metadata=base_metadata,
        ))

    # --- Section 4: Interventions ---
    if trial.interventions:
        chunks.extend(_make_chunks(
            nct_id=trial.nct_id,
            section_type="interventions",
            text=_build_interventions(trial),
            base_metadata=base_metadata,
        ))

    # --- Section 5: Adverse Events ---
    if trial.adverse_events:
        chunks.extend(_make_chunks(
            nct_id=trial.nct_id,
            section_type="adverse_events",
            text=_build_adverse_events(trial),
            base_metadata=base_metadata,
        ))

    return chunks


def chunk_trials(trials: Iterable[ParsedTrial]) -> list[TrialChunk]:
    """Bulk version: chunk many trials into a flat list."""
    out: list[TrialChunk] = []
    for trial in trials:
        out.extend(chunk_trial(trial))
    return out


# ---------------------------------------------------------------------------
# Section text builders
# Each builder returns a single human-readable string for its section.
# We intentionally include redundant context (NCT, title, phase) at the top
# of each chunk because it dramatically improves retrieval quality —
# chunks that mention "lung cancer" and "Phase III" upfront match better.
# ---------------------------------------------------------------------------

def _header(trial: ParsedTrial) -> str:
    """Common header prepended to every section so chunks are self-contained."""
    parts = [f"Trial {trial.nct_id}: {trial.brief_title}"]
    if trial.primary_phase:
        parts.append(f"Phase: {trial.primary_phase}")
    if trial.overall_status:
        parts.append(f"Status: {trial.overall_status}")
    if trial.lead_sponsor:
        parts.append(f"Sponsor: {trial.lead_sponsor}")
    return " | ".join(parts)


def _build_overview(trial: ParsedTrial) -> str:
    lines = [_header(trial), ""]
    if trial.official_title and trial.official_title != trial.brief_title:
        lines.append(f"Official title: {trial.official_title}")
    if trial.study_type:
        lines.append(f"Study type: {trial.study_type}")
    if trial.enrollment_count:
        lines.append(f"Planned enrollment: {trial.enrollment_count} participants")
    if trial.start_date:
        lines.append(f"Start date: {trial.start_date}")
    if trial.primary_completion_date:
        lines.append(f"Primary completion: {trial.primary_completion_date}")
    return "\n".join(lines)


def _build_eligibility(trial: ParsedTrial) -> str:
    lines = [_header(trial), "", "ELIGIBILITY CRITERIA"]
    if trial.minimum_age:
        lines.append(f"Minimum age: {trial.minimum_age}")
    if trial.sex:
        lines.append(f"Sex: {trial.sex}")
    lines.append("")
    lines.append(trial.eligibility_criteria)
    return "\n".join(lines)


def _build_endpoints(trial: ParsedTrial) -> str:
    if not trial.primary_outcomes and not trial.secondary_outcomes:
        return ""
    lines = [_header(trial), "", "ENDPOINTS"]
    if trial.primary_outcomes:
        lines.append("\nPrimary outcomes:")
        for o in trial.primary_outcomes:
            lines.append(f"- {o.measure}")
            if o.description:
                lines.append(f"  Description: {o.description}")
            if o.time_frame:
                lines.append(f"  Time frame: {o.time_frame}")
    if trial.secondary_outcomes:
        lines.append("\nSecondary outcomes:")
        for o in trial.secondary_outcomes:
            lines.append(f"- {o.measure}")
            if o.time_frame:
                lines.append(f"  Time frame: {o.time_frame}")
    return "\n".join(lines)


def _build_interventions(trial: ParsedTrial) -> str:
    lines = [_header(trial), "", "INTERVENTIONS"]
    for i in trial.interventions:
        lines.append(f"- {i.name} ({i.type})")
        if i.description:
            lines.append(f"  {i.description}")
    return "\n".join(lines)


def _build_adverse_events(trial: ParsedTrial) -> str:
    lines = [_header(trial), "", "ADVERSE EVENTS (SAFETY PROFILE)"]
    serious = [ae for ae in trial.adverse_events if ae.serious]
    other   = [ae for ae in trial.adverse_events if not ae.serious]

    if serious:
        lines.append("\nSerious adverse events:")
        for ae in serious:
            lines.append(
                f"- {ae.term} ({ae.organ_system or 'unspecified system'}): "
                f"{ae.num_affected}/{ae.num_at_risk} ({ae.incidence_pct}%)"
            )
    if other:
        lines.append("\nCommon adverse events:")
        for ae in other:
            lines.append(
                f"- {ae.term}: {ae.num_affected}/{ae.num_at_risk} ({ae.incidence_pct}%)"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunk builder + size-based splitter
# ---------------------------------------------------------------------------

def _base_metadata(trial: ParsedTrial) -> dict:
    """Metadata fields shared by every chunk of a given trial."""
    return {
        "phase":            trial.primary_phase,
        "overall_status":   trial.overall_status,
        "sponsor":          trial.lead_sponsor,
        "sponsor_class":    trial.sponsor_class,
        "enrollment_count": trial.enrollment_count,
        "brief_title":      trial.brief_title,
    }


def _make_chunks(
    nct_id: str,
    section_type: str,
    text: str,
    base_metadata: dict,
) -> list[TrialChunk]:
    """
    Produce one or more TrialChunk objects for a given section.
    Splits if the text exceeds MAX_CHUNK_CHARS, with overlap.
    """
    text = text.strip()
    if not text:
        return []

    parts = _split_text(text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
    chunks: list[TrialChunk] = []
    for idx, part in enumerate(parts):
        chunks.append(TrialChunk(
            chunk_id=f"{nct_id}::{section_type}::{idx}",
            nct_id=nct_id,
            section_type=section_type,
            content=part,
            **base_metadata,
        ))
    return chunks


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Sliding-window splitter with overlap. Tries to break on paragraph
    boundaries first, then on sentence boundaries, then hard-breaks.

    Why overlap? Without it, a chunk boundary in the middle of a sentence
    can split semantic meaning. Overlap = retrieval can find the answer
    even if it spans a chunk boundary.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        # Try to find a clean break point (paragraph > sentence > whitespace)
        if end < n:
            for separator in ["\n\n", "\n", ". ", " "]:
                # Search backwards from `end` for the separator
                cut = text.rfind(separator, start + max_chars // 2, end)
                if cut != -1:
                    end = cut + len(separator)
                    break

        chunks.append(text[start:end].strip())
        if end >= n:
            break
        # Slide forward, keeping overlap to preserve context
        start = max(end - overlap, start + 1)

    return chunks