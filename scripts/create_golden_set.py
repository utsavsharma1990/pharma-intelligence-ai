"""Generate the golden evaluation test set."""
import json
from pathlib import Path

golden_set = [
    {
        "id": "gs_001",
        "query": "What adverse events were reported in NCT05123456?",
        "expected_agent": "safety",
        "expected_nct_ids": ["NCT05123456"],
        "expected_keywords": ["pneumonitis", "colitis", "fatigue"],
        "notes": "Safety agent, specific trial AE lookup"
    },
    {
        "id": "gs_002",
        "query": "What are the eligibility criteria for NCT05123456?",
        "expected_agent": "search",
        "expected_nct_ids": ["NCT05123456"],
        "expected_keywords": ["inclusion", "exclusion", "ECOG", "18"],
        "notes": "Search agent, eligibility section retrieval"
    },
    {
        "id": "gs_003",
        "query": "Compare the safety profiles of NCT05123456 and NCT04567890",
        "expected_agent": "comparative",
        "expected_nct_ids": ["NCT05123456", "NCT04567890"],
        "expected_keywords": ["pneumonitis", "compare"],
        "notes": "Comparative agent, multi-trial AE comparison"
    },
    {
        "id": "gs_004",
        "query": "What is the primary endpoint of NCT04567890?",
        "expected_agent": "search",
        "expected_nct_ids": ["NCT04567890"],
        "expected_keywords": ["overall survival", "OS"],
        "notes": "Search agent, endpoint retrieval"
    },
    {
        "id": "gs_005",
        "query": "What serious adverse events occurred in the nivolumab trial?",
        "expected_agent": "safety",
        "expected_nct_ids": ["NCT04567890"],
        "expected_keywords": ["pneumonitis", "hepatitis", "serious"],
        "notes": "Safety agent, drug name lookup"
    },
    {
        "id": "gs_006",
        "query": "Who sponsors the pembrolizumab NSCLC trial?",
        "expected_agent": "search",
        "expected_nct_ids": ["NCT05123456"],
        "expected_keywords": ["Merck"],
        "notes": "Search agent, sponsor lookup"
    },
    {
        "id": "gs_007",
        "query": "Compare the enrollment size of NCT05123456 vs NCT06789012",
        "expected_agent": "comparative",
        "expected_nct_ids": ["NCT05123456", "NCT06789012"],
        "expected_keywords": ["800", "400"],
        "notes": "Comparative agent, enrollment comparison"
    },
    {
        "id": "gs_008",
        "query": "What drugs are being tested in the CLL trial NCT06789012?",
        "expected_agent": "search",
        "expected_nct_ids": ["NCT06789012"],
        "expected_keywords": ["ibrutinib", "chlorambucil"],
        "notes": "Search agent, intervention lookup"
    },
    {
        "id": "gs_009",
        "query": "What are the cardiac adverse events in the ibrutinib trial?",
        "expected_agent": "safety",
        "expected_nct_ids": ["NCT06789012"],
        "expected_keywords": ["atrial fibrillation", "cardiac"],
        "notes": "Safety agent, organ-system AE filter"
    },
    {
        "id": "gs_010",
        "query": "Compare pembrolizumab and nivolumab trials in terms of study design",
        "expected_agent": "comparative",
        "expected_nct_ids": ["NCT05123456", "NCT04567890"],
        "expected_keywords": ["phase", "NSCLC"],
        "notes": "Comparative agent, design comparison"
    },
]

Path("data/eval").mkdir(parents=True, exist_ok=True)
out = Path("data/eval/golden_set.json")
out.write_text(json.dumps(golden_set, indent=2))
print(f"✅ Written {len(golden_set)} golden test cases to {out}")