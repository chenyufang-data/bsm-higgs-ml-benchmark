"""Version standalone code and libraries determining compact data."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from . import __version__


def bundle_info(root: Path) -> dict | None:
    """Version and git commit of a deployed server bundle (its MANIFEST.json), if any."""
    manifest = root / "MANIFEST.json"
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return {key: record.get(key) for key in ("version", "git_commit", "git_dirty")}


def extraction_provenance() -> dict:
    """Extractor file hashes and library versions enter the export signature;
    the compactor version, Python version and bundle record are informational."""
    package = Path(__file__).resolve().parent
    return {
        "extractor": {
            path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sorted(package.glob("*.py"))
        },
        "libraries": {name: version(name) for name in ("numpy", "pandas", "uproot", "awkward", "pyarrow")},
        "version": __version__,
        "python": platform.python_version(),
        "bundle": bundle_info(package.parent),
    }
