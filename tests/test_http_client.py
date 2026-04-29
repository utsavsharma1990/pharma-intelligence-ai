"""Unit tests for the HTTP client. Uses mock mode — never hits the network."""

import pytest
from pathlib import Path

from src.ingestion.http_client import (
    ClinicalTrialsClient,
    CTPermanentError,
    RateLimiter,
)


def test_rate_limiter_blocks_when_too_fast():
    """Rate limiter should sleep when we exceed the rate."""
    import time
    rl = RateLimiter(requests_per_sec=2)  # min 0.5s between calls

    rl.wait()
    start = time.monotonic()
    rl.wait()  # should sleep ~0.5s
    elapsed = time.monotonic() - start

    assert elapsed >= 0.4, f"Rate limiter didn't sleep: {elapsed}s"


def test_search_studies_mock_returns_studies():
    """Mock mode should return the local file contents."""
    client = ClinicalTrialsClient(use_mock=True)
    result = client.search_studies(condition="lung cancer")

    assert "studies" in result
    assert result["totalCount"] == 3
    assert len(result["studies"]) == 3


def test_get_study_mock_finds_known_nct():
    """Mock mode get_study should find a known NCT ID."""
    client = ClinicalTrialsClient(use_mock=True)
    study = client.get_study("NCT05123456")

    nct = study["protocolSection"]["identificationModule"]["nctId"]
    assert nct == "NCT05123456"


def test_get_study_mock_raises_on_unknown_nct():
    """Unknown NCT ID should raise CTPermanentError, not retry."""
    client = ClinicalTrialsClient(use_mock=True)
    with pytest.raises(CTPermanentError):
        client.get_study("NCT99999999")


def test_mock_file_missing_raises_clear_error(tmp_path):
    """Helpful error message when mock file is missing."""
    client = ClinicalTrialsClient(
        use_mock=True,
        mock_file=tmp_path / "does_not_exist.json"
    )
    with pytest.raises(CTPermanentError, match="Mock file not found"):
        client.search_studies()