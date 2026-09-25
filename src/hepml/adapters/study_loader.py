"""Load a trusted, editable study directory through a small hook contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from hepml_compact.config import load_profile

from hepml.ports import Study


def load_study(path: str | Path | None = None) -> Study:
    if path is None:
        import studies.cg_bbc

        directory = Path(studies.cg_bbc.__file__).parent
    else:
        directory = Path(path).resolve()
    profile = load_profile(directory / "extraction.yaml")
    config = profile.config
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for source in sorted(directory.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes().replace(b"\r\n", b"\n"))
    fingerprint = digest.hexdigest()
    # Give each study its own package namespace, so relative imports work and
    # two studies with plugin.py/features.py cannot collide.
    namespace = f"_hepml_study_{fingerprint}"
    spec = importlib.util.spec_from_file_location(
        namespace,
        directory / "plugin.py",
        submodule_search_locations=[str(directory)],
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load study plugin in {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    for name in ("FEATURES", "RETAINED_COLUMNS"):
        value = getattr(module, name, None)
        if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"Study plugin must declare {name} as a sequence of strings")
    columns = [*module.FEATURES, *module.RETAINED_COLUMNS]
    if len(columns) != len(set(columns)):
        raise ValueError("Study feature/retained columns must be unique and disjoint")
    reserved = {"label", "weight", "xs", "event_id", "source_id", "source_entry", "sample_key"}
    if reserved.intersection(columns):
        raise ValueError("Study columns conflict with reserved metadata columns")
    if not callable(getattr(module, "derive_features", None)):
        raise ValueError("Study plugin must implement derive_features")
    return Study(directory, config, module, fingerprint)
