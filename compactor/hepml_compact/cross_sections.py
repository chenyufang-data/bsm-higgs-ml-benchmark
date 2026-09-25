"""Fill sample cross sections from Pythia merged-cross-section tables on the server.

Run as `python -m hepml_compact.cross_sections`. Uses Python + PyYAML only.
Requires a (choice) marker by default; --row first explicitly selects the first
data row when production files have no marker. For a multi-file sample, xs_file
explicitly identifies its sample-wide rate file. For single-file samples,
--xs-filename optionally selects an adjacent filename. Validates every sample
before writing a NEW manifest; never edits the input. Each filled sample records
xs_source (file, SHA-256, chosen row, merging scale and MC uncertainty), which
the export carries into its metadata.

With MLM matching and merging, only this Pythia merged cross section is a valid
rate. Never derive one from the Event.Weight or Event.CrossSection branches.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import yaml


def _chosen_row(path: Path, *, row: str = "choice") -> tuple[float, float, float]:
    """(merging scale, cross section in pb, MC uncertainty in pb) of the nominal row."""
    if row not in {"choice", "first"}:
        raise ValueError(f"Unknown row selection: {row}")
    text = path.read_text(encoding="utf-8")
    if not re.search(r"Cross-section\s*\[pb\]", text, re.IGNORECASE):
        raise ValueError(f"{path}: expected Cross-section [pb] header")
    if row == "choice":
        chosen = [line for line in text.splitlines() if "(choice)" in line]
        if len(chosen) != 1:
            raise ValueError(
                f"{path}: expected exactly one (choice) row, found {len(chosen)}. "
                "If your nominal result is the first data row, use --row first."
            )
        fields = chosen[0].split()
        if len(fields) != 4 or fields[3] != "(choice)":
            raise ValueError(f"{path}: invalid chosen row: {chosen[0]!r}")
    else:
        fields = None
        for line in text.splitlines():
            candidate = line.split()
            if not candidate:
                continue
            try:
                float(candidate[0].replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if len(candidate) != 3 and not (len(candidate) == 4 and candidate[3] == "(choice)"):
                raise ValueError(f"{path}: invalid first data row: {line!r}")
            fields = candidate
            break
        if fields is None:
            raise ValueError(f"{path}: no numeric data rows found")
    scale, xs, uncertainty = (float(value.replace("D", "E").replace("d", "e")) for value in fields[:3])
    if not all(math.isfinite(value) for value in (scale, xs, uncertainty)) or scale <= 0 or xs <= 0 or uncertainty < 0:
        raise ValueError(f"{path}: invalid scale, cross section or MC uncertainty")
    return scale, xs, uncertainty


def chosen_cross_section(path: Path, *, row: str = "choice") -> float:
    return _chosen_row(path, row=row)[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fill_manifest(
    manifest: Path, output: Path, data_root: Path | None = None, *, row: str = "choice", keep_existing: bool = False,
    xs_filename: str | None = None,
) -> int:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new filename")
    if xs_filename is not None and (
        not xs_filename.strip() or xs_filename in {".", ".."} or any(c in xs_filename for c in "/\\:")
    ):
        raise ValueError("--xs-filename must be a filename, not a path; use xs_file for explicit paths")
    config = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("samples"), dict) or not config["samples"]:
        raise ValueError("Manifest must contain a nonempty samples mapping")
    root = data_root if data_root is not None else Path(config.get("data_root", "."))
    if not root.is_absolute():
        root = manifest.resolve().parent / root
    root = root.resolve()
    for key, sample in config["samples"].items():
        files = sample.get("files")
        if not isinstance(files, list) or not files or not all(isinstance(name, str) and name for name in files):
            raise ValueError(f"{key}: expected a nonempty ROOT files list")
        for name in files:
            if not (root / name).is_file():
                raise FileNotFoundError(f"{key}: ROOT file not found: {root / name}")
        if keep_existing and sample.get("xs_pb") is not None:
            xs = float(sample["xs_pb"])
            if not math.isfinite(xs) or xs <= 0:
                raise ValueError(f"{key}: supplied xs_pb must be finite and positive")
            sample["xs_pb"] = xs
            sample.setdefault("xs_source", {"method": "supplied_in_manifest"})
            continue
        xs_file = sample.get("xs_file")
        if xs_file is not None:
            if not isinstance(xs_file, str) or not xs_file.strip():
                raise ValueError(f"{key}: xs_file must be a nonempty path relative to data_root, or absolute")
            xs_path = root / xs_file
        elif len(files) == 1 and xs_filename is not None:
            xs_path = (root / files[0]).parent / xs_filename
        elif len(files) == 1:
            raise ValueError(
                f"{key}: set xs_file, supply --xs-filename for an adjacent rate table, "
                "or supply xs_pb and use --keep-existing. No rate filename is assumed."
            )
        else:
            raise ValueError(
                f"{key}: cannot infer a combined cross section for multiple ROOT files. "
                "Supply xs_pb explicitly and use --keep-existing, or set xs_file to a rate file "
                "that represents the entire sample."
            )
        scale, xs, uncertainty = _chosen_row(xs_path, row=row)
        if sample.get("xs_pb") is not None and float(sample["xs_pb"]) != xs:
            raise ValueError(f"{key}: existing xs_pb differs from the chosen row; resolve before continuing")
        sample["xs_pb"] = xs
        sample["xs_source"] = {
            "method": "pythia_merged_table",
            "file": str(xs_path.resolve()),
            "sha256": _sha256(xs_path),
            "row": row,
            "merging_scale": scale,
            "mc_uncertainty_pb": uncertainty,
        }
    # Resolve the root so writing beside a different output directory cannot
    # silently reinterpret relative input paths. Counts are deliberately untouched.
    config["data_root"] = str(root)
    rendered = f"# Cross sections in pb; rate-table row selection: {row}.\n"
    if keep_existing:
        rendered += "# Explicitly supplied cross sections preserved without reading rate tables for those samples.\n"
    rendered += yaml.safe_dump(config, sort_keys=False)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
    return len(config["samples"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output YAML; existing files are refused")
    parser.add_argument("--data-root", type=Path, help="Override input data_root explicitly")
    parser.add_argument(
        "--xs-filename",
        help="Adjacent rate-table filename for single-file samples; per-sample xs_file takes precedence. No default.",
    )
    parser.add_argument(
        "--row",
        choices=["choice", "first"],
        default="choice",
        help="choice: require (choice) marker; first: use first numeric data row explicitly",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Preserve supplied positive xs_pb, including multi-file backgrounds; fill only missing rates",
    )
    args = parser.parse_args()
    count = fill_manifest(
        args.manifest, args.output, args.data_root, row=args.row, keep_existing=args.keep_existing,
        xs_filename=args.xs_filename,
    )
    print(f"Prepared {count} sample cross sections (row={args.row}): {args.output}")


if __name__ == "__main__":
    main()
