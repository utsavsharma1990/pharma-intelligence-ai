"""Unit tests for cache + fetcher. Uses mock mode, no network."""

import json
from pathlib import Path

import pytest

from src.ingestion.cache import FileCache
from src.ingestion.fetcher import TrialFetcher


# ---------- FileCache tests ----------

def test_cache_set_and_get(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("my_key", {"hello": "world"})
    assert cache.get("my_key") == {"hello": "world"}


def test_cache_miss_returns_none(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.get("nonexistent") is None


def test_cache_has(tmp_path):
    cache = FileCache(tmp_path)
    assert not cache.has("k")
    cache.set("k", {"v": 1})
    assert cache.has("k")


def test_cache_corrupt_file_returns_none(tmp_path):
    cache = FileCache(tmp_path)
    bad = cache._key_to_path("corrupt_key")
    bad.write_text("not valid json {{{")
    assert cache.get("corrupt_key") is None
    # Corrupt entry should have been auto-evicted
    assert not bad.exists()


def test_cache_ttl_expires(tmp_path):
    import time
    cache = FileCache(tmp_path, ttl_seconds=1)
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}
    time.sleep(1.1)
    assert cache.get("k") is None  # expired


def test_cache_clear(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("a", {"x": 1})
    cache.set("b", {"x": 2})
    deleted = cache.clear()
    assert deleted == 2
    assert cache.get("a") is None


# ---------- TrialFetcher tests ----------

def test_fetcher_yields_all_studies(tmp_path):
    fetcher = TrialFetcher(use_mock=True, cache_dir=tmp_path)
    studies = list(fetcher.fetch_all(condition="lung cancer"))
    assert len(studies) == 3
    nct_ids = [
        s["protocolSection"]["identificationModule"]["nctId"]
        for s in studies
    ]
    assert "NCT05123456" in nct_ids


def test_fetcher_respects_max_studies(tmp_path):
    fetcher = TrialFetcher(use_mock=True, cache_dir=tmp_path)
    studies = list(fetcher.fetch_all(condition="lung cancer", max_studies=2))
    assert len(studies) == 2


def test_fetcher_caches_responses(tmp_path):
    fetcher = TrialFetcher(use_mock=True, cache_dir=tmp_path)

    # First fetch — populates cache
    list(fetcher.fetch_all(condition="lung cancer"))

    # Cache directory should now have at least one entry
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) >= 1

    # Second fetch — should hit cache (we verify by checking file mtime)
    import time
    cache_file = cache_files[0]
    mtime_before = cache_file.stat().st_mtime
    time.sleep(0.05)
    list(fetcher.fetch_all(condition="lung cancer"))
    mtime_after = cache_file.stat().st_mtime
    assert mtime_before == mtime_after, "Cache file was rewritten on hit"