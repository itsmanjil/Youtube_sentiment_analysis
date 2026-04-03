"""
Pinned runtime artifact resolution for live inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config


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

