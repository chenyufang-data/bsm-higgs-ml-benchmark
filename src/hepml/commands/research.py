"""Detached notebook stage worker; science remains in the existing CLI commands."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.research import benchmark_directory, verify_stage, write_json
from hepml.cli import main as run_cli


def main():
    directory = Path(sys.argv[1]).resolve()
    started = time.monotonic()
    status = {"status": "running"}
    write_json(directory / "status.json", status)
    try:
        argv = json.loads((directory / "command.json").read_text())["argv"]
        provenance = json.loads((directory / "provenance.json").read_text())

        def verify_inputs():
            for path, record in provenance["sources"].items():
                if sha256_file(Path(path)) != record["sha256"]:
                    raise ValueError(f"Source changed while the stage was queued/running: {path}")
            settings = provenance["settings"]
            if "benchmark_artifacts" in settings:
                benchmark = verify_stage(benchmark_directory(settings))
                if benchmark["artifacts"] != settings["benchmark_artifacts"]:
                    raise ValueError("Benchmark changed while training was queued/running")

        verify_inputs()
        run_cli(argv)
        verify_inputs()
        status["status"] = "complete"
        # Record generated file hashes, including splits and predictions.
        status["artifacts"] = {
            str(p.relative_to(directory)): sha256_file(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in {"status.json", "stage.log"} and "sources" not in p.parts
        }
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "status.json", status)


if __name__ == "__main__":
    main()
