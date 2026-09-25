"""The analysis package's compactor requirement accepts the compactor in this checkout."""

from pathlib import Path

import pytest
from hepml_compact import __version__
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]


def test_compactor_requirement_accepts_the_checked_out_compactor():
    # CI installs both packages from the checkout; a requirement that excludes the
    # compactor's own version makes that install fail.
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [Requirement(d) for d in project["dependencies"] if Requirement(d).name == "hepml-compactor"]
    assert len(requirements) == 1
    assert requirements[0].specifier.contains(__version__), (
        f"pyproject.toml requires {requirements[0]}, but ./compactor is version {__version__}")
