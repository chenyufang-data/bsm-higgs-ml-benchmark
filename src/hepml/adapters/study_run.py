"""Lifecycle shared by report-producing studies.

A study writes one new report directory (figures/, tables/, provenance/) and new
model or dataset directories. It never reuses an existing output, records its
settings, sources and input hashes before computing, re-checks them before
finishing, and ends with provenance/status.json listing the SHA-256 of every
artifact it wrote. A report created by a failed call is marked failed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from hepml_compact.parquet_writer import sha256_file

REPOSITORY = Path(__file__).resolve().parents[3]
PRESERVED = "Use new report and run IDs; completed or partial artifacts are preserved"


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_provenance(destination: Path, config: dict, records: list[dict], *, archive_sources=False) -> None:
    """Preserve exact research settings, source files, dependencies and input hashes."""
    repo = REPOSITORY
    paths = [repo / "pyproject.toml", repo / "compactor/pyproject.toml"]
    if archive_sources:
        paths.extend((repo / "notebooks").glob("*.ipynb"))
    for base in (repo / "src/hepml", repo / "compactor/hepml_compact", Path(config["study"])):
        paths.extend(path for path in base.rglob("*") if path.suffix in {".py", ".yaml"})
    hashes = {}
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(destination / "sources.zip", "w", zipfile.ZIP_DEFLATED)) if archive_sources else None
        for index, path in enumerate(sorted(set(paths))):
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if archive is not None:
                member = path.relative_to(repo).as_posix() if path.is_relative_to(repo) else f"study/{path.name}"
                archive.writestr(member, data)
                hashes[str(path)] = {"sha256": digest, "archive": "sources.zip", "member": member}
            else:
                saved = destination / "sources" / f"{index:03d}_{path.name}"
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(data)
                hashes[str(path)] = {"sha256": digest, "snapshot": str(saved.relative_to(destination))}
        if archive is not None:
            archive.writestr("manifest.json", json.dumps(hashes, indent=2))
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    write_json(destination / "provenance.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), settings=config,
        git_commit=result.stdout.strip() if result.returncode == 0 else None,
        git_status=status.stdout, python=sys.version, executable=sys.executable,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        sources=hashes,
        inputs=[dict(directory=r["directory"], metadata=r["meta"], export=r["export"],
                     export_sha256=r["export_sha256"]) for r in records],
    ))


def run_study(outdir, run, args):
    """Call run(args); if it raises, a report created by this call is marked failed."""
    status = Path(outdir) / "provenance/status.json"
    existed = Path(outdir).exists()
    try:
        return run(args)
    except Exception as error:
        if not existed and status.is_file():
            write_json(status, dict(status="failed", error=str(error)))
        raise


def completed_report(directory: Path, phases) -> dict:
    """Summary of a completed report whose recorded artifacts are unchanged."""
    summary = json.loads((directory / "provenance/summary.json").read_text())
    status = json.loads((directory / "provenance/status.json").read_text())
    if status["status"] != "complete" or summary["phases"] != phases:
        raise ValueError(f"{directory}: require a completed Phase {phases[0]} report")
    for name, digest in status["artifacts"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Completed artifact changed: {name}")
    return summary


def comparable_config(config: dict) -> dict:
    """A recorded evaluation configuration in a form comparable with the current one."""
    return dict(config, background_systematics=tuple(config["background_systematics"]))


def refuse_existing(*paths, message=PRESERVED) -> None:
    if any(Path(path).exists() for path in paths):
        raise FileExistsError(message)


def load_registry(directory: Path) -> pd.DataFrame:
    """The common event registry, after checking the hash recorded when it was frozen."""
    directory = Path(directory)
    metadata = json.loads((directory / "registry.json").read_text())
    if sha256_file(directory / "assignments.parquet") != metadata["assignments_sha256"]:
        raise ValueError("Assignment registry hash changed")
    return pd.read_parquet(directory / "assignments.parquet")


def registry_files(directory: Path) -> list[Path]:
    return [Path(directory) / "assignments.parquet", Path(directory) / "registry.json"]


def hash_files(paths) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if Path(path).is_file()}


def open_report(out: Path, settings: dict, inputs: dict[str, str], records=(), *, preregister=False) -> None:
    """Create the report layout and freeze settings, sources and inputs before any fit."""
    for folder in ("figures", "tables", "provenance"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    save_provenance(out / "provenance", settings, list(records), archive_sources=True)
    if preregister:
        write_json(out / "provenance/preregistration.json", settings)
    write_json(out / "provenance/inputs.json", inputs)
    write_json(out / "provenance/status.json", {"status": "running"})


def reopen_report(out: Path, settings: dict, inputs: dict[str, str], record: dict) -> Path:
    """Continue an interrupted report with the same frozen settings and inputs.

    The interrupted run's provenance and source archive move to provenance/original_run_<n>/;
    this session records its own, and resume.json names every source file that differs.
    Returns the resume record's path.
    """
    provenance = out / "provenance"
    status = json.loads((provenance / "status.json").read_text())["status"]
    if status == "complete":
        raise ValueError(f"{out} is complete; nothing to resume")
    frozen = json.loads((provenance / "preregistration.json").read_text())
    if frozen != json.loads(json.dumps(settings, default=str)):
        raise ValueError("Settings differ from the interrupted run")
    if json.loads((provenance / "inputs.json").read_text()) != inputs:
        raise ValueError("Inputs changed since the interrupted run")
    index = 1 + sum(1 for _ in provenance.glob("original_run_*"))
    original = provenance / f"original_run_{index}"
    original.mkdir()
    for name in ("provenance.json", "sources.zip"):
        (provenance / name).rename(original / name)
    save_provenance(provenance, settings, [], archive_sources=True)
    before = json.loads((original / "provenance.json").read_text())
    after = json.loads((provenance / "provenance.json").read_text())
    changed = sorted(path for path in set(before["sources"]) | set(after["sources"])
                     if before["sources"].get(path, {}).get("sha256") != after["sources"].get(path, {}).get("sha256"))
    path = provenance / "resume.json"
    write_json(path, dict(record, interrupted_status=status, interrupted_git_commit=before.get("git_commit"),
                          resumed_git_commit=after.get("git_commit"), changed_sources=changed,
                          interrupted_provenance=str(original), resumed_utc=after["created_utc"]))
    write_json(provenance / "status.json", {"status": "running"})
    return path


def write_tables(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(out / "tables" / f"{name}.csv", index=False)


def write_validation(directory: Path, record) -> None:
    """Per-model files of application.evaluation.validate_model, under their established names."""
    record.curve.to_parquet(directory / "threshold_curves.parquet", index=False)
    record.bands.to_parquet(directory / "threshold_bands.parquet", index=False)
    record.processes.to_parquet(directory / "process_curves.parquet", index=False)
    record.stability.to_csv(directory / "threshold_stability.csv", index=False)
    for name, frame in record.frozen.items():
        frame.to_csv(directory / f"frozen_{name}.csv", index=False)


def verify_unchanged(out: Path, inputs: dict[str, str]) -> None:
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Input changed during the study: {path}")
    provenance = json.loads((out / "provenance/provenance.json").read_text())
    for path, record in provenance["sources"].items():
        if sha256_file(Path(path)) != record["sha256"]:
            raise ValueError(f"Source changed during the study: {path}")


def finish_report(out: Path, notebook: str, inputs: dict[str, str], artifacts=()) -> None:
    """Re-check inputs and sources, publish the notebook (if any), then hash every artifact written."""
    verify_unchanged(out, inputs)
    if notebook is not None:
        export_notebook_report(out, REPOSITORY / "notebooks" / notebook)
    paths = [path for path in out.rglob("*") if path.is_file() and path.name != "status.json"]
    paths += [Path(path) for path in artifacts]
    write_json(out / "provenance/status.json",
               dict(status="complete", artifacts={str(path): sha256_file(path) for path in paths}))
    print(f"Report: {out / 'report.html' if notebook is not None else out}", flush=True)


def export_notebook_report(directory, template):
    """Execute a read-only template in a fresh kernel and embed plots in HTML."""
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    directory, template = Path(directory).resolve(), Path(template).resolve()
    notebook = nbformat.read(template, as_version=4)
    # Inject only the artifact directory; scientific settings come from its record.
    for cell in notebook.cells:
        if "report-settings" in cell.metadata.get("tags", []):
            cell.source = "from pathlib import Path\nREPORT = Path(" + repr(str(directory)) + ")"
    with tempfile.TemporaryDirectory(prefix="hepml-report-kernel-") as temporary:
        kernel = Path(temporary) / "hepml-report"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps(dict(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="hepml-report", language="python")))
        manager = KernelManager(kernel_name="hepml-report", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        NotebookClient(notebook, timeout=300, km=manager,
                       resources={"metadata": {"path": str(template.parent.parent)}}).execute()
    nbformat.write(notebook, directory / "report.ipynb")
    exporter = HTMLExporter()
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(notebook)
    (directory / "report.html").write_text(html, encoding="utf-8")
