from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess


def write_run_manifest(output: str | Path, *, command: str, inputs: dict, seed: int = 7) -> Path:
    """Write a compact machine-readable provenance record for every analysis."""
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "core-tbi"]
    versions = {name: _version(name) for name in packages}
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": command, "seed": seed, "inputs": inputs, "package_versions": versions, "git_commit": _git_commit()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
