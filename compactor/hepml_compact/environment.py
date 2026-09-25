"""Compare the running environment with the pinned server constraints.

Exports record their library versions, and a resume refuses a changed
environment. Checking the pins before a batch keeps one dataset on one set.
"""

from __future__ import annotations

import argparse
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def read_constraints(path: Path) -> tuple[str | None, dict[str, str]]:
    """Return (python major.minor or None, {package: exact version})."""
    python, pins = None, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        marker = re.fullmatch(r"#\s*python:\s*(\d+\.\d+)\s*", text)
        if marker:
            python = marker.group(1)
            continue
        text = text.split("#", 1)[0].strip()
        if not text:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*==\s*([A-Za-z0-9_.+-]+)", text)
        if not match:
            raise ValueError(f"{path}: only exact 'name==version' pins are supported: {line!r}")
        pins[match.group(1)] = match.group(2)
    return python, pins


def mismatches(path: Path) -> list[str]:
    python, pins = read_constraints(path)
    problems = []
    running = ".".join(platform.python_version_tuple()[:2])
    if python is not None and running != python:
        problems.append(f"python {running} (pinned {python})")
    for name, wanted in pins.items():
        try:
            found = version(name)
        except PackageNotFoundError:
            found = "missing"
        if found != wanted:
            problems.append(f"{name} {found} (pinned {wanted})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("constraints", type=Path)
    args = parser.parse_args(argv)
    problems = mismatches(args.constraints)
    if problems:
        print("Environment differs from " + str(args.constraints) + ":")
        for item in problems:
            print("  " + item)
        return 1
    print(f"Environment matches {args.constraints}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
