"""
Pinned runtime artifact resolution for live inference.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config

logger = logging.getLogger(__name__)


def get_runtime_artifact_version() -> str:
    """Return the configured pinned runtime-artifact version."""
    return str(Config.DEFAULT_RUNTIME_ARTIFACT_VERSION).strip()


def get_runtime_artifact_dir(version: Optional[str] = None) -> Path:
    """Return the directory that contains the pinned runtime artifacts."""
    artifact_version = str(version or get_runtime_artifact_version()).strip()
    return Config.RUNTIME_ARTIFACTS_DIR / artifact_version


def get_runtime_manifest_path(version: Optional[str] = None) -> Path:
    """Return the manifest path for the pinned runtime artifacts."""
    return get_runtime_artifact_dir(version) / "manifest.json"


def load_runtime_manifest(version: Optional[str] = None) -> Dict[str, Any]:
    """Load the pinned runtime-artifact manifest, or an empty dict if missing."""
    manifest_path = get_runtime_manifest_path(version)
    try:
        with open(manifest_path) as handle:
            return json.load(handle)
    except Exception:
        return {}


def resolve_runtime_artifact_path(
    artifact_name: str,
    version: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a named runtime artifact from the pinned manifest."""
    manifest = load_runtime_manifest(version)
    artifact_entry = (manifest.get("artifacts") or {}).get(str(artifact_name))
    if not isinstance(artifact_entry, dict):
        return None

    relative_path = artifact_entry.get("path")
    if not relative_path:
        return None

    artifact_path = (get_runtime_artifact_dir(version) / relative_path).resolve()
    if artifact_path.exists():
        return artifact_path
    return None


def load_runtime_artifact_json(
    artifact_name: str,
    version: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load a JSON runtime artifact from the pinned manifest."""
    artifact_path = resolve_runtime_artifact_path(artifact_name, version=version)
    if artifact_path is None:
        return None
    try:
        with open(artifact_path) as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _verify_model_artifact_hash_cached(
    local_path_str: str,
    artifact_name: str,
    version: Optional[str],
) -> Optional[bool]:
    return _verify_model_artifact_hash_uncached(Path(local_path_str), artifact_name, version)


def verify_model_artifact_hash(
    local_path: Path,
    artifact_name: str,
    version: Optional[str] = None,
) -> Optional[bool]:
    """
    Cached wrapper around `_verify_model_artifact_hash_uncached` — engines are
    reconstructed per-request, so without caching every request would re-hash
    a multi-MB pickle file. Safe to cache since model files don't change while
    a worker process is running.
    """
    return _verify_model_artifact_hash_cached(str(local_path), artifact_name, version)


def _verify_model_artifact_hash_uncached(
    local_path: Path,
    artifact_name: str,
    version: Optional[str] = None,
) -> Optional[bool]:
    """
    Verify a locally-loaded model file's sha256 against the pinned manifest.

    The manifest's "path" field for model artifacts is relative to the
    runtime-artifact directory, where the actual model files (backend/models/...)
    do not live — only the JSON research artifacts (temperature_scaling,
    pso_ensemble_weights, etc.) resolve correctly via `resolve_runtime_artifact_path`.
    This function instead verifies against `source_path`/`sha256`, computed on
    the model file actually being loaded at `local_path`, so drift between the
    pinned manifest and the model/vectorizer files being served is detectable
    instead of the manifest being purely decorative provenance.

    Returns
    -------
    Optional[bool]
        True if the hash matches, False if it doesn't, None if the artifact
        isn't in the manifest or the local file couldn't be hashed (e.g.
        missing) — callers should treat None as "unable to verify", not as a
        failure.
    """
    manifest = load_runtime_manifest(version)
    entry = (manifest.get("artifacts") or {}).get(str(artifact_name))
    if not isinstance(entry, dict):
        return None
    expected = entry.get("sha256")
    if not expected:
        return None

    try:
        actual = _sha256_file(Path(local_path))
    except OSError:
        return None

    matches = actual == expected
    if not matches:
        logger.warning(
            "Runtime artifact hash mismatch for %r: pinned manifest expects "
            "sha256=%s but %s has sha256=%s. The served model may differ from "
            "the one the pinned research results (calibration, ensemble "
            "weights, etc.) were computed against.",
            artifact_name,
            expected,
            local_path,
            actual,
        )
    return matches


def verify_artifact_or_raise(
    local_path: Path,
    artifact_name: str,
    version: Optional[str] = None,
) -> Optional[bool]:
    """
    Verify a model artifact's hash and raise on a confirmed mismatch.

    Must be called *before* the artifact is deserialized (e.g. before
    `pickle.load`) — checking the hash only after loading can't stop a
    tampered pickle from executing arbitrary code during deserialization,
    it can only notice afterward. Returns `None` rather than raising when
    the hash is merely unverifiable (no pinned manifest entry, or the file
    can't be read), since the caller's own file-not-found/load-error
    handling already covers that case with a clearer message.
    """
    verified = verify_model_artifact_hash(local_path, artifact_name, version=version)
    if verified is False:
        raise RuntimeError(
            f"Refusing to load {local_path} ({artifact_name}): sha256 does not "
            "match the pinned runtime-artifact manifest. The file may have been "
            "tampered with or corrupted."
        )
    return verified


def get_runtime_artifact_metadata(version: Optional[str] = None) -> Dict[str, Any]:
    """Return compact metadata describing the pinned runtime artifacts."""
    manifest = load_runtime_manifest(version)
    artifacts = manifest.get("artifacts") or {}
    return {
        "version": manifest.get("version") or get_runtime_artifact_version(),
        "manifest_path": str(get_runtime_manifest_path(version)),
        "artifacts": {
            name: {
                "path": entry.get("path"),
                "sha256": entry.get("sha256"),
            }
            for name, entry in artifacts.items()
            if isinstance(entry, dict)
        },
    }

