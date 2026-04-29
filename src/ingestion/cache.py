"""
Disk-based cache for raw ClinicalTrials.gov API responses.

Design decisions:
- File-per-key: simple, debuggable (can `cat` any cached response)
- Hash-based filenames: deterministic from query params, safe for filesystems
- JSON storage: human-readable, no binary serialization needed
- TTL support: optional expiry (default: never expire — clinical trial data
  changes slowly enough that "fetch once per ingest run" is fine)

Why not Redis / SQLite? File cache is zero-infrastructure, plays nicely with
git-ignore, and is trivial to inspect. Upgrade to Redis if/when scale demands.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class FileCache:
    """
    Filesystem cache: each entry is a JSON file named by hash of the cache key.

    Usage:
        cache = FileCache(Path("data/raw"))
        cache.set("studies:lung_cancer:page1", {"totalCount": 100, ...})
        data = cache.get("studies:lung_cancer:page1")
    """

    def __init__(self, cache_dir: Path, ttl_seconds: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds  # None = never expire

    def _key_to_path(self, key: str) -> Path:
        """
        Hash the cache key to produce a filesystem-safe filename.
        We use sha1 (not md5) because newer Python versions deprecate md5.
        Truncated to 16 chars — collision probability is negligible at our scale.
        """
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return cached value, or None if missing/expired."""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        # Check TTL if configured
        if self.ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > self.ttl_seconds:
                path.unlink()  # evict expired entry
                return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Strip our internal metadata wrapper before returning
            return payload.get("data")
        except (json.JSONDecodeError, KeyError):
            # Corrupt cache entry — delete and miss
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Write value to cache. Wraps with metadata for debugging."""
        path = self._key_to_path(key)
        # We write to a temp file first, then atomic rename — prevents partial
        # writes if the process crashes mid-write.
        tmp = path.with_suffix(".tmp")
        payload = {
            "_cache_key": key,            # so you can grep cache files by query
            "_cached_at": int(time.time()),
            "data": value,
        }
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def has(self, key: str) -> bool:
        """Check if a cache entry exists (and is not expired)."""
        return self.get(key) is not None

    def clear(self) -> int:
        """Remove all cache entries. Returns count deleted."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count