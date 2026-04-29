"""
Domain models for parsed clinical trials.

Design decision: we define our OWN representation of a trial, separate from
the raw API shape. This is the anti-corruption layer pattern — if CT.gov
changes their API, we fix one file (parser.py) instead of 30.

All models inherit from pydantic.BaseModel for:
  - Type validation at construction time
  - Free JSON serialization (.model_dump() / .model_dump_json())
  - IDE autocomplete on every field
  - Easy comparison and equality
"""

from typing import Optional
from pydantic import BaseModel, Field


class Intervention(BaseModel):
    """A single drug, device, or procedure being tested."""
    type: str = Field(..., description="DRUG | DEVICE | PROCEDURE | BEHAVIORAL | etc")
    name: str
    description: Optional[str] = None


class Outcome(BaseModel):
    """A primary or secondary endpoint."""
    measure: str
    description: Optional[str] = None
    time_frame: Optional[str] = None


class Location(BaseModel):
    """A study site."""
    facility: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class AdverseEvent(BaseModel):
    """
    A single adverse event with its incidence numbers.
    `serious=True` means SAE (serious adverse event — death, hospitalization,
    life-threatening, etc.). `serious=False` means common AE.
    """
    term: str
    organ_system: Optional[str] = None
    num_affected: int = 0
    num_at_risk: int = 0
    serious: bool = False

    @property
    def incidence_pct(self) -> float:
        """Percentage of participants who experienced this AE."""
        if self.num_at_risk == 0:
            return 0.0
        return round(self.num_affected / self.num_at_risk * 100, 2)


class ParsedTrial(BaseModel):
    """
    Clean, flat representation of a clinical trial.

    Contrast with the raw API: instead of
        study["protocolSection"]["identificationModule"]["nctId"]
    we have
        trial.nct_id

    Use ParsedTrial.from_api(raw_dict) (defined in parser.py) to construct.
    """
    # --- Identification ---
    nct_id: str
    brief_title: str
    official_title: Optional[str] = None

    # --- Status & timing ---
    overall_status: Optional[str] = None
    start_date: Optional[str] = None
    primary_completion_date: Optional[str] = None

    # --- Design ---
    phases: list[str] = Field(default_factory=list)
    study_type: Optional[str] = None
    enrollment_count: Optional[int] = None

    # --- Eligibility ---
    eligibility_criteria: str = ""        # the big text blob
    minimum_age: Optional[str] = None
    sex: Optional[str] = None

    # --- Interventions ---
    interventions: list[Intervention] = Field(default_factory=list)

    # --- Outcomes ---
    primary_outcomes:   list[Outcome] = Field(default_factory=list)
    secondary_outcomes: list[Outcome] = Field(default_factory=list)

    # --- Sponsor ---
    lead_sponsor: Optional[str] = None
    sponsor_class: Optional[str] = None    # INDUSTRY | NIH | OTHER

    # --- Locations ---
    locations: list[Location] = Field(default_factory=list)

    # --- Adverse events ---
    adverse_events: list[AdverseEvent] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors that downstream code (chunker, agents) wants
    # ------------------------------------------------------------------

    @property
    def primary_phase(self) -> Optional[str]:
        """Most studies have one phase; this is the canonical one."""
        return self.phases[0] if self.phases else None

    @property
    def serious_adverse_events(self) -> list[AdverseEvent]:
        return [ae for ae in self.adverse_events if ae.serious]

    @property
    def intervention_names(self) -> list[str]:
        return [i.name for i in self.interventions]
        
class TrialChunk(BaseModel):
    """
    A single retrievable chunk produced by the chunker.

    Each chunk has:
      - content: the actual text that will be embedded + searched
      - metadata: filterable fields for hybrid search (nct_id, phase, etc.)
      - chunk_id: stable unique identifier (so we can update without dupes)

    Design: separate `content` from `metadata` because vector stores treat
    them differently. ChromaDB indexes content for similarity search;
    metadata is used for exact-match filtering.
    """
    chunk_id: str         # e.g. "NCT05123456::eligibility::0"
    nct_id: str
    section_type: str     # "overview" | "eligibility" | "endpoints" | "interventions" | "adverse_events"
    content: str

    # Filterable metadata — flat strings/numbers only (vector store constraint)
    phase: Optional[str] = None
    overall_status: Optional[str] = None
    sponsor: Optional[str] = None
    sponsor_class: Optional[str] = None
    enrollment_count: Optional[int] = None
    brief_title: Optional[str] = None

    def to_metadata_dict(self) -> dict:
        """
        Flatten to a dict suitable for vector store metadata.
        ChromaDB only accepts str/int/float/bool — no None values.
        """
        d = self.model_dump(exclude={"content"})
        return {k: v for k, v in d.items() if v is not None}