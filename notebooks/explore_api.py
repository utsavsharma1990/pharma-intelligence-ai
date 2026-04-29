"""
API exploration using mock data (real API blocked on corporate network).
The mock data matches the exact shape of ClinicalTrials.gov v2 API responses.
"""

import json
from pathlib import Path

MOCK_FILE = Path("data/raw/mock_trials.json")

def pretty(data) -> None:
    print(json.dumps(data, indent=2))

# Load mock data
data = json.loads(MOCK_FILE.read_text())

# -----------------------------------------------------------------------
# 1. Top-level shape
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("1. TOP-LEVEL RESPONSE SHAPE")
print("="*60)
print("Top-level keys  :", list(data.keys()))
print("Total count     :", data.get("totalCount"))
print("Next page token :", data.get("nextPageToken"))
print("Studies returned:", len(data.get("studies", [])))

# -----------------------------------------------------------------------
# 2. Single study structure
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("2. SINGLE STUDY STRUCTURE")
print("="*60)
study    = data["studies"][0]
protocol = study["protocolSection"]
print("protocolSection keys:", list(protocol.keys()))

# -----------------------------------------------------------------------
# 3. Key fields
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("3. KEY FIELDS EXTRACTION")
print("="*60)
id_mod   = protocol["identificationModule"]
status   = protocol["statusModule"]
design   = protocol["designModule"]
elig     = protocol["eligibilityModule"]
outcomes = protocol["outcomesModule"]
sponsor  = protocol["sponsorCollaboratorsModule"]
ae       = protocol["adverseEventsModule"]

print(f"NCT ID   : {id_mod['nctId']}")
print(f"Title    : {id_mod['briefTitle']}")
print(f"Phase    : {design['phases']}")
print(f"Status   : {status['overallStatus']}")
print(f"Sponsor  : {sponsor['leadSponsor']['name']}")
print(f"Enrolled : {design['enrollmentInfo']['count']}")

print(f"\nEligibility (first 300 chars):")
print(elig["eligibilityCriteria"][:300] + "...")

print(f"\nPrimary outcomes: {len(outcomes['primaryOutcomes'])}")
print(f"First outcome   : {outcomes['primaryOutcomes'][0]['measure']}")

print(f"\nSerious AEs:")
for ae_event in ae["seriousEvents"]:
    n = ae_event["stats"][0]["numAffected"]
    total = ae_event["stats"][0]["numAtRisk"]
    pct = round(n/total*100, 1)
    print(f"  {ae_event['term']}: {n}/{total} ({pct}%)")

# -----------------------------------------------------------------------
# 4. Pagination pattern (simulated)
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("4. PAGINATION PATTERN")
print("="*60)
print("Real API uses nextPageToken cursor (not page numbers)")
print("Pass nextPageToken from response back as pageToken param")
print("None token = last page")
print(f"This mock has {data['totalCount']} studies, nextPageToken={data['nextPageToken']}")

# -----------------------------------------------------------------------
# 5. All NCT IDs
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("5. ALL STUDIES IN MOCK")
print("="*60)
for s in data["studies"]:
    p     = s["protocolSection"]
    nct   = p["identificationModule"]["nctId"]
    title = p["identificationModule"]["briefTitle"]
    phase = p["designModule"]["phases"]
    cond  = p["statusModule"]["overallStatus"]
    print(f"  {nct} | {phase} | {cond} | {title[:50]}")

# -----------------------------------------------------------------------
# 6. AE comparison across trials
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("6. ADVERSE EVENTS ACROSS ALL TRIALS")
print("="*60)
for s in data["studies"]:
    p     = s["protocolSection"]
    nct   = p["identificationModule"]["nctId"]
    ae_mod = p.get("adverseEventsModule", {})
    serious = ae_mod.get("seriousEvents", [])
    print(f"\n{nct}:")
    for event in serious:
        n     = event["stats"][0]["numAffected"]
        total = event["stats"][0]["numAtRisk"]
        print(f"  {event['term']}: {n}/{total} ({round(n/total*100,1)}%)")

print("\n✅ Mock exploration complete — ready for ingestion pipeline")