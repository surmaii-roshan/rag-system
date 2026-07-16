"""
ingest/dedup.py — SHA-256 file deduplication using a JSON manifest.

The manifest maps file_hash → filename. On each ingest run:
  - Hash the file
  - If hash in manifest → skip (duplicate)
  - If hash not in manifest → process, then add to manifest
"""

import hashlib
import json
from pathlib import Path
from typing import Dict

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file. Reads in 8KB chunks for memory efficiency."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_manifest() -> Dict[str, str]:
    """Load the manifest from disk. Returns empty dict if it doesn't exist yet."""
    if Config.MANIFEST_PATH.exists():
        with open(Config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict[str, str]) -> None:
    """Persist the manifest to disk."""
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def is_duplicate(file_hash: str, manifest: Dict[str, str]) -> bool:
    """Return True if this hash has already been ingested."""
    return file_hash in manifest


def update_manifest(file_hash: str, filename: str, manifest: Dict[str, str]) -> None:
    """Add a new hash→filename entry. Does NOT save to disk — call save_manifest() after."""
    manifest[file_hash] = filename
    log.debug(f"Manifest updated: {filename} → {file_hash[:8]}...")