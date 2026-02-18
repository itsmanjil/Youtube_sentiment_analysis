#!/usr/bin/env python3
"""
Create a reproducibility bundle for thesis experiments.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
GLOB_CHARS = set("*?[]")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_path(raw: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expanduser(raw))
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_candidate = (Path.cwd() / expanded).resolve()
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return (base_dir / expanded).resolve()


def to_display_path(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(base_dir))
    except ValueError:
        return str(resolved)


def run_capture(command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
    )
    return process.returncode, process.stdout, process.stderr


def find_git_root(start_dir: Path) -> Optional[Path]:
    candidates = [start_dir.resolve(), *start_dir.resolve().parents]
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def get_git_info(base_dir: Path) -> Dict[str, object]:
    root = find_git_root(base_dir)
    if root is None:
        return {
            "available": False,
            "reason": "no_git_repo_detected",
        }

    def read_git(*args: str) -> Optional[str]:
        code, stdout, _ = run_capture(["git", "-C", str(root), *args])
        if code != 0:
            return None
        value = stdout.strip()
        return value if value else None

    status = read_git("status", "--porcelain") or ""
    return {
        "available": True,
        "repo_root": str(root),
        "branch": read_git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": read_git("rev-parse", "HEAD"),
        "remote_origin": read_git("remote", "get-url", "origin"),
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
    }


def has_glob_pattern(spec: str) -> bool:
    return any(char in spec for char in GLOB_CHARS)


def expand_artifact_spec(spec: str, base_dir: Path) -> Tuple[List[Path], Optional[str]]:
    if has_glob_pattern(spec):
        resolved_pattern = normalize_path(spec, base_dir)
        matches = [
            Path(match).resolve()
            for match in glob.glob(str(resolved_pattern), recursive=True)
        ]
        files = sorted(path for path in matches if path.is_file())
        if files:
            return files, None
        return [], spec

    candidate = normalize_path(spec, base_dir)
    if not candidate.exists():
        return [], spec
    if candidate.is_file():
        return [candidate], None
    files = sorted(path.resolve() for path in candidate.rglob("*") if path.is_file())
    return files, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_commands(command_args: List[str], command_file: Optional[str], base_dir: Path) -> List[str]:
    commands: List[str] = []

    for command in command_args:
        value = command.strip()
        if value:
            commands.append(value)

    if command_file:
        file_path = normalize_path(command_file, base_dir)
        if not file_path.exists():
            raise FileNotFoundError(f"Command file not found: {file_path}")
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                line = parts[2].strip()
            if line:
                commands.append(line)

    deduped: List[str] = []
    seen = set()
    for command in commands:
        if command in seen:
            continue
        seen.add(command)
        deduped.append(command)
    return deduped


def write_commands(bundle_dir: Path, commands: List[str]) -> Dict[str, str]:
    commands_txt = bundle_dir / "commands.txt"
    commands_sh = bundle_dir / "commands.sh"

    if commands:
        commands_txt.write_text("\n".join(commands) + "\n", encoding="utf-8")
        shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        shell_lines.extend(commands)
        commands_sh.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    else:
        commands_txt.write_text("", encoding="utf-8")
        commands_sh.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n",
            encoding="utf-8",
        )

    try:
        mode = commands_sh.stat().st_mode
        commands_sh.chmod(mode | 0o111)
    except OSError:
        pass

    return {
        "commands_txt": str(commands_txt),
        "commands_sh": str(commands_sh),
    }


def write_environment(bundle_dir: Path) -> Dict[str, object]:
    env_path = bundle_dir / "python_environment.txt"
    freeze_path = bundle_dir / "pip_freeze.txt"

    pip_version_code, pip_version_out, pip_version_err = run_capture(
        [sys.executable, "-m", "pip", "--version"]
    )
    pip_freeze_code, pip_freeze_out, pip_freeze_err = run_capture(
        [sys.executable, "-m", "pip", "freeze"]
    )

    env_lines = [
        f"generated_at_utc: {utc_now()}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"python_implementation: {platform.python_implementation()}",
        f"pip_version_exit_code: {pip_version_code}",
    ]
    if pip_version_out.strip():
        env_lines.append(f"pip_version: {pip_version_out.strip()}")
    if pip_version_err.strip():
        env_lines.append(f"pip_version_stderr: {pip_version_err.strip()}")
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    if pip_freeze_code == 0:
        freeze_path.write_text(pip_freeze_out, encoding="utf-8")
    else:
        error_text = (
            f"# pip freeze failed (exit_code={pip_freeze_code})\n{pip_freeze_err}"
        )
        freeze_path.write_text(error_text, encoding="utf-8")

    return {
        "python_environment_file": str(env_path),
        "pip_freeze_file": str(freeze_path),
        "pip_freeze_exit_code": pip_freeze_code,
    }


def write_checksums(bundle_dir: Path, artifacts: List[Dict[str, object]]) -> str:
    checksum_path = bundle_dir / "artifacts.sha256"
    lines = [f"{item['sha256']}  {item['path']}" for item in artifacts]
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return str(checksum_path)


def default_artifact_specs(split_metadata: str) -> List[str]:
    candidates = [
        split_metadata,
        "requirements.txt",
        "requirements-dl.txt",
        "Pipfile",
        "Pipfile.lock",
    ]
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducibility bundle with manifest, checksums, and environment locks.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/repro_bundles",
        help="Output directory for bundles (relative to backend root).",
    )
    parser.add_argument(
        "--bundle_name",
        default=None,
        help="Optional bundle directory name. Default: repro_bundle_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Command used to generate thesis artifacts. Repeat for multiple commands.",
    )
    parser.add_argument(
        "--command_file",
        default=None,
        help="File containing commands. Supports plain commands or log rows (timestamp<TAB>cwd<TAB>command).",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact file/dir/glob to checksum and register in manifest.",
    )
    parser.add_argument(
        "--split_metadata",
        default="data/split_metadata.json",
        help="Path to split metadata file for dataset provenance.",
    )
    parser.add_argument(
        "--skip_default_artifacts",
        action="store_true",
        help="Skip automatic inclusion of split metadata and environment lock files.",
    )
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Fail if any artifact specification does not resolve to existing files.",
    )
    parser.add_argument("--notes", default=None, help="Optional free-text notes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc)
    bundle_name = args.bundle_name or f"repro_bundle_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    output_root = normalize_path(args.output_dir, BASE_DIR)
    bundle_dir = output_root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    commands = parse_commands(args.command, args.command_file, BASE_DIR)

    artifact_specs: List[str] = list(args.artifact)
    if not args.skip_default_artifacts:
        artifact_specs.extend(default_artifact_specs(args.split_metadata))

    all_files: List[Path] = []
    missing_specs: List[str] = []
    seen_files = set()
    for spec in artifact_specs:
        files, missing = expand_artifact_spec(spec, BASE_DIR)
        for path in files:
            key = str(path)
            if key in seen_files:
                continue
            seen_files.add(key)
            all_files.append(path)
        if missing:
            missing_specs.append(missing)

    if args.fail_on_missing and missing_specs:
        missing_text = ", ".join(missing_specs)
        raise FileNotFoundError(f"Missing artifact specification(s): {missing_text}")

    artifact_entries: List[Dict[str, object]] = []
    for path in sorted(all_files):
        stats = path.stat()
        artifact_entries.append(
            {
                "path": to_display_path(path, BASE_DIR),
                "absolute_path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": stats.st_size,
                "modified_at_utc": datetime.fromtimestamp(
                    stats.st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    command_files = write_commands(bundle_dir, commands)
    environment_files = write_environment(bundle_dir)
    checksum_file = write_checksums(bundle_dir, artifact_entries)

    invocation = " ".join(shlex.quote(arg) for arg in [sys.executable, *sys.argv])
    split_metadata_path = normalize_path(args.split_metadata, BASE_DIR)
    split_metadata_display = to_display_path(split_metadata_path, BASE_DIR)
    split_metadata_record = {
        "path": split_metadata_display,
        "exists": split_metadata_path.exists(),
        "included_in_artifacts": any(
            item["absolute_path"] == str(split_metadata_path) for item in artifact_entries
        ),
    }
    if split_metadata_path.exists() and split_metadata_path.is_file():
        split_metadata_record["sha256"] = sha256_file(split_metadata_path)

    manifest = {
        "bundle_created_at_utc": utc_now(),
        "bundle_name": bundle_name,
        "bundle_dir": str(bundle_dir),
        "backend_root": str(BASE_DIR),
        "invocation": invocation,
        "notes": args.notes,
        "commands": commands,
        "command_count": len(commands),
        "artifact_count": len(artifact_entries),
        "missing_artifact_specs": sorted(set(missing_specs)),
        "split_metadata": split_metadata_record,
        "git": get_git_info(BASE_DIR),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "files": {
            **command_files,
            **environment_files,
            "checksums_file": checksum_file,
        },
        "artifacts": artifact_entries,
    }

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Reproducibility bundle created: {bundle_dir}")
    print(f"- Manifest: {manifest_path}")
    print(f"- Artifacts tracked: {len(artifact_entries)}")
    if missing_specs:
        print(f"- Missing artifact specs: {len(set(missing_specs))}")


if __name__ == "__main__":
    main()
