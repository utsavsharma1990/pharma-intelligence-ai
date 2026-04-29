"""
Pagination-aware fetcher for ClinicalTrials.gov.

Orchestrates the boring details of "fetch ALL pages of a query":
  - Loops with nextPageToken until exhausted
  - Caches each page to disk (so a crash mid-fetch doesn't lose progress)
  - Yields studies one at a time (memory-efficient for large result sets)
  - Optional max_studies cap (for testing or budget limits)

Why a generator (yield) instead of returning a list?
  Some queries can return 50,000+ studies — building a 500MB list in memory
  is wasteful. Generators let downstream code process-as-you-fetch.
"""

import logging
from pathlib import Path
from typing import Iterator, Optional

from src.core.config import get_settings
from src.ingestion.cache import FileCache
from src.ingestion.http_client import ClinicalTrialsClient

logger = logging.getLogger(__name__)


class TrialFetcher:
    """
    Fetches studies from ClinicalTrials.gov with pagination + caching.

    Usage:
        fetcher = TrialFetcher(use_mock=True)
        for study in fetcher.fetch_all(condition="lung cancer", phase="PHASE3"):
            process(study)
    """

    def __init__(
        self,
        use_mock: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        settings = get_settings().ct
        self.client = ClinicalTrialsClient(use_mock=use_mock)
        self.cache = FileCache(
            cache_dir=cache_dir or Path(settings.cache_dir) / "pages",
            ttl_seconds=None,  # never expire — clinical trials change slowly
        )
        self.page_size = settings.page_size

    def fetch_all(
        self,
        condition: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        sponsor: Optional[str] = None,
        max_studies: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Yield every study matching the filters, paginating through all pages.

        Args:
            max_studies: stop after yielding this many studies (None = all)

        Yields:
            Individual study dicts (the items inside response['studies']).
        """
        page_token: Optional[str] = None
        total_yielded = 0
        page_num = 0

        while True:
            page_num += 1
            cache_key = self._cache_key(
                condition, phase, status, sponsor, page_token
            )

            # Try cache first — skip the API call entirely if we already have it
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit: page {page_num} ({cache_key[:60]})")
                page = cached
            else:
                logger.info(f"Cache miss: fetching page {page_num}")
                page = self.client.search_studies(
                    condition=condition,
                    phase=phase,
                    status=status,
                    sponsor=sponsor,
                    page_token=page_token,
                    page_size=self.page_size,
                )
                self.cache.set(cache_key, page)

            studies = page.get("studies", [])
            for study in studies:
                yield study
                total_yielded += 1
                if max_studies is not None and total_yielded >= max_studies:
                    logger.info(f"Hit max_studies cap: {max_studies}")
                    return

            # Decide whether to continue paginating
            page_token = page.get("nextPageToken")
            if not page_token:
                logger.info(
                    f"Pagination complete: {total_yielded} studies across "
                    f"{page_num} page(s)"
                )
                return

    @staticmethod
    def _cache_key(
        condition: Optional[str],
        phase: Optional[str],
        status: Optional[str],
        sponsor: Optional[str],
        page_token: Optional[str],
    ) -> str:
        """
        Build a deterministic cache key from the query parameters.
        Same query params → same key → cache hit on re-runs.
        """
        parts = [
            f"cond={condition or ''}",
            f"phase={phase or ''}",
            f"status={status or ''}",
            f"sponsor={sponsor or ''}",
            f"page={page_token or '0'}",
        ]
        return "studies|" + "|".join(parts)