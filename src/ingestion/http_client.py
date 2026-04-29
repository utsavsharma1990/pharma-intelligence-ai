"""
HTTP client with retry logic and rate limiting for ClinicalTrials.gov API.

Design decisions:
- tenacity for retries: declarative, well-tested, supports async + sync
- Exponential backoff with jitter: prevents thundering-herd retry storms
- Retry only on transient errors (5xx, network errors), never on 4xx
- Token bucket rate limiter: simple, accurate, respects API politeness limits
- Mock mode toggle: USE_MOCK=true reads from local JSON instead of HTTP

The client supports both:
  1. Real HTTP calls (production / unrestricted networks)
  2. Mock file-based responses (corporate networks / CI / offline dev)

Switching between them is one config flag — no code changes needed.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)
import logging

from src.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions — gives callers clear semantics on what went wrong.
# Caller can catch CTAPIError without caring whether it was a 500 or timeout.
# ---------------------------------------------------------------------------

class CTAPIError(Exception):
    """Base error for any ClinicalTrials.gov API failure."""


class CTRetryableError(CTAPIError):
    """Transient error worth retrying (5xx, network timeout, connection reset)."""


class CTPermanentError(CTAPIError):
    """Non-retryable error (4xx — bad request, not found, forbidden)."""


# ---------------------------------------------------------------------------
# Token bucket rate limiter.
# Why not use `time.sleep(1)` between calls? Because if a request takes 2s,
# we'd unnecessarily wait another 1s. Token bucket only sleeps when needed.
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token bucket: max N requests per second, blocks when exhausted."""

    def __init__(self, requests_per_sec: float):
        self.min_interval = 1.0 / requests_per_sec
        self.last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.monotonic()


# ---------------------------------------------------------------------------
# The main client.
# ---------------------------------------------------------------------------

class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov v2 API.

    Two modes:
      - Real HTTP: hits the live API with retry + rate limiting
      - Mock mode: reads from data/raw/mock_trials.json (use_mock=True)

    Usage:
        client = ClinicalTrialsClient()
        page  = client.search_studies(condition="lung cancer", phase="PHASE3")
        trial = client.get_study("NCT05123456")
    """

    def __init__(self, use_mock: bool = False, mock_file: Optional[Path] = None):
        settings = get_settings().ct
        self.base_url = settings.base_url.rstrip("/")
        self.use_mock = use_mock
        self.mock_file = mock_file or Path("data/raw/mock_trials.json")
        self.rate_limiter = RateLimiter(settings.rate_limit_per_sec)

        # We persist the httpx Client across calls — connection pooling
        # gives a meaningful speedup over creating a new client every request.
        self._http = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "pharma-intelligence-ai/0.1",
                "Accept": "application/json",
            },
            verify=False,  # tolerate corporate SSL inspection; safe for public read API
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._http.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_studies(
        self,
        condition: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        sponsor: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """
        Search studies with optional filters.
        Returns the raw API response: {totalCount, nextPageToken, studies: [...]}
        """
        if self.use_mock:
            return self._load_mock()

        params: dict[str, Any] = {"pageSize": page_size, "format": "json"}
        if condition: params["query.cond"]        = condition
        if phase:     params["filter.phase"]      = phase
        if status:    params["filter.overallStatus"] = status
        if sponsor:   params["query.spons"]       = sponsor
        if page_token: params["pageToken"]        = page_token

        return self._get("/studies", params)

    def get_study(self, nct_id: str) -> dict[str, Any]:
        """Fetch a single study by NCT ID."""
        if self.use_mock:
            mock = self._load_mock()
            for s in mock["studies"]:
                if s["protocolSection"]["identificationModule"]["nctId"] == nct_id:
                    return s
            raise CTPermanentError(f"NCT {nct_id} not found in mock data")

        return self._get(f"/studies/{nct_id}", {"format": "json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mock(self) -> dict[str, Any]:
        """Read mock file from disk. Used in mock mode and tests."""
        if not self.mock_file.exists():
            raise CTPermanentError(
                f"Mock file not found: {self.mock_file}. "
                "Run: python scripts/create_mock_data.py"
            )
        return json.loads(self.mock_file.read_text())

    @retry(
        # tenacity decorator config:
        retry=retry_if_exception_type(CTRetryableError),  # only retry transient errors
        stop=stop_after_attempt(4),                       # try 4 times max
        wait=wait_exponential_jitter(initial=1, max=10),  # 1s, 2s, 4s, 8s — jittered
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,                                     # raise last exception, not RetryError
    )
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a GET with retry + rate limiting.
        Wrapped in @retry so transient failures retry automatically with backoff.
        """
        self.rate_limiter.wait()  # respect API politeness limit

        try:
            resp = self._http.get(f"{self.base_url}{path}", params=params)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            # Network-level errors are always retryable
            raise CTRetryableError(f"Network error: {e}") from e

        # Classify HTTP status codes into retryable vs permanent
        if resp.status_code >= 500:
            raise CTRetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code == 429:                    # rate limited
            raise CTRetryableError("HTTP 429: rate limited")
        if 400 <= resp.status_code < 500:
            raise CTPermanentError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            return resp.json()
        except ValueError as e:
            raise CTPermanentError(f"Invalid JSON: {e}") from e