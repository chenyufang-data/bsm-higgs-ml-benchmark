#!/usr/bin/env bash
# Run from the activated conda environment. Copy beside hepml_compact/ on server.
# Example: nohup bash run_extraction.sh all > extraction.log 2>&1 < /dev/null &
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash run_extraction.sh [all|signals|backgrounds] [options]
  --config FILE       Default: <script directory>/../manifests/samples.ready.yaml
  --extraction FILE   Default: <script directory>/study/extraction.yaml
  --outdir DIR        Default: <script directory>/../exports/<study name>-v<compactor version>
  --step-size VALUE   Default: "50 MB"; keep the pilot's value to resume it
  --dry-run           Print planned commands without reading ROOT or writing exports
  --allow-env-mismatch  Start even if libraries differ from constraints-py311.txt
  --help              Show this help

Samples run sequentially; first failure stops the batch with a nonzero exit code.
Signals go to <outdir>/rho01 ... rho05 by rho_tc, or to <outdir>/signals without one;
backgrounds go once to <outdir>/backgrounds.
Logs and a status TSV go to <outdir>/logs/<timestamp>_<mode>_<PID>/.
Rerun the same command to resume; existing exports are checked by the compactor.
New compactor code or libraries cannot resume or revalidate an older export: use a
new --outdir (the default already includes the compactor version).
PYTHON_BIN can select a Python executable; otherwise use the active conda's python.
HELP
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=all
if [[ $# -gt 0 && $1 != -* ]]; then mode=$1; shift; fi
case "$mode" in all|signals|backgrounds) ;; *) usage >&2; exit 2 ;; esac
# Shared manifest beside the versioned bundles, independent of the working directory.
config="$script_dir/../manifests/samples.ready.yaml"
extraction="$script_dir/study/extraction.yaml"
output_root=""
step_size="50 MB"
dry_run=0
allow_env_mismatch=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --allow-env-mismatch) allow_env_mismatch=1; shift ;;
        --config|--extraction|--outdir|--step-size)
            if [[ $# -lt 2 || -z $2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --config) config=$2 ;;
                --extraction) extraction=$2 ;;
                --outdir) output_root=$2 ;;
                --step-size) step_size=$2 ;;
            esac
            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
python_bin=${PYTHON_BIN:-python}
plan_file=$(mktemp)
trap 'rm -f -- "$plan_file"' EXIT

# Validate the entire selected batch before starting any ROOT extraction. Plan
# paths are tab-separated and quoted in Bash; YAML values never become shell code.
"$python_bin" - "$config" "$extraction" "$output_root" "$script_dir" "$mode" "$dry_run" > "$plan_file" <<'PY'
import math
import os
import re
import sys
from decimal import Decimal
from pathlib import Path

import yaml

sys.stdout.reconfigure(newline="\n")
config_path, extraction_path = (Path(p).resolve() for p in sys.argv[1:3])
script_dir, mode, dry = Path(sys.argv[4]), sys.argv[5], sys.argv[6] == "1"
config = yaml.safe_load(config_path.read_text())
recipe = yaml.safe_load(extraction_path.read_text())
name = recipe.get("name", "")
if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
    raise ValueError("Extraction recipe must declare a safe study name")
sys.path.insert(0, str(script_dir))
import hepml_compact  # the bundled source beside this script; imports no dependencies

default_out = script_dir.parent / "exports" / f"{name}-v{hepml_compact.__version__}"
out = Path(sys.argv[3]).resolve() if sys.argv[3] else default_out.resolve()
root = Path(os.environ.get("HEPML_DATA_ROOT") or config.get("data_root", "."))
if not root.is_absolute():
    root = config_path.parent / root
samples = config.get("samples")
if not isinstance(samples, dict) or not samples:
    raise ValueError("No samples configured")
jobs = []
for key, sample in samples.items():
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError(f"Unsafe sample key: {key!r}")
    kind = sample.get("kind")
    if kind not in {"signal", "background"}:
        raise ValueError(f"{key}: kind must be signal/background")
    if mode != "all" and kind != {"signals": "signal", "backgrounds": "background"}[mode]:
        continue
    xs = sample.get("xs_pb")
    if not isinstance(xs, (int, float)) or not math.isfinite(xs) or xs <= 0:
        raise ValueError(f"{key}: xs_pb must be filled, finite and positive")
    files = sample.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(f, str) and f for f in files):
        raise ValueError(f"{key}: files must be a nonempty list")
    if not dry:
        for file in files:
            if not (root / file).is_file():
                raise FileNotFoundError(f"{key}: missing ROOT file {root / file}")
    if kind == "background":
        if sample.get("mass") is not None or sample.get("rho_tc") is not None:
            raise ValueError(f"{key}: background must not have mass/rho_tc")
        folder = "backgrounds"
    else:
        mass = sample.get("mass")
        if not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"{key}: invalid signal mass")
        if sample.get("rho_tc") is None:
            # A study with mass as its only signal parameter keeps every signal in one folder.
            folder = "signals"
        else:
            rho = Decimal(str(sample.get("rho_tc")))
            if not rho.is_finite() or rho < 0:
                raise ValueError(f"{key}: invalid signal rho_tc")
            folder = f"rho{int(rho * 10):02d}" if rho * 10 == int(rho * 10) else "rho" + format(rho.normalize(), "f").replace(".", "p")
    jobs.append((key, (out / folder).as_posix()))
if not jobs:
    raise ValueError(f"No samples selected for mode {mode}")
values = [config_path.as_posix(), extraction_path.as_posix(), out.as_posix(), *[v for job in jobs for v in job]]
if any(any(c in value for c in "\t\n\r") for value in values):
    raise ValueError("Tabs and newlines are not supported in paths or sample keys")
print(config_path.as_posix())
print(extraction_path.as_posix())
print(out.as_posix())
for key, destination in jobs:
    print(f"{key}\t{destination}")
print(f"Planned {len(jobs)} samples ({mode}) with hepml-compactor {hepml_compact.__version__}; "
      f"data root: {root}", file=sys.stderr)
PY

# The local source package works without pip installation from this directory.
cd -- "$script_dir"
exec 3< "$plan_file"
IFS= read -r config <&3
IFS= read -r extraction <&3
IFS= read -r output_root <&3
if [[ $dry_run -eq 0 ]]; then
    "$python_bin" -c 'import hepml_compact, uproot, awkward, pyarrow, pandas, yaml'
    constraints="$script_dir/constraints-py311.txt"
    if [[ -f $constraints ]] && ! "$python_bin" -m hepml_compact.environment "$constraints"; then
        if [[ $allow_env_mismatch -eq 1 ]]; then
            echo "Continuing despite the environment mismatch (--allow-env-mismatch)." >&2
        else
            echo "Refusing to start: the environment differs from $constraints." >&2
            echo "Fix the environment, or rerun with --allow-env-mismatch and a new --outdir." >&2
            exit 2
        fi
    fi
    log_dir="$output_root/logs/$(date +%Y%m%d_%H%M%S)_${mode}_$$"
    mkdir -p -- "$log_dir"
    status_file="$log_dir/status.tsv"
    printf 'time\tsample\tstatus\texit_code\n' > "$status_file"
    echo "Logs: $log_dir"
fi
while IFS=$'\t' read -r sample destination <&3; do
    command=("$python_bin" -u -m hepml_compact --extraction "$extraction" --config "$config"
             --sample "$sample" --outdir "$destination" --step-size "$step_size")
    if [[ $dry_run -eq 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    echo "[$(date -Is)] START $sample -> $destination"
    printf '%s\t%s\tRUNNING\t-\n' "$(date -Is)" "$sample" >> "$status_file"
    if "${command[@]}" > "$log_dir/$sample.log" 2>&1; then
        printf '%s\t%s\tCOMPLETE\t0\n' "$(date -Is)" "$sample" >> "$status_file"
        echo "[$(date -Is)] COMPLETE $sample"
    else
        code=$?
        printf '%s\t%s\tFAILED\t%s\n' "$(date -Is)" "$sample" "$code" >> "$status_file"
        echo "[$(date -Is)] FAILED $sample (exit $code). See $log_dir/$sample.log" >&2
        tail -n 20 -- "$log_dir/$sample.log" >&2
        exit "$code"
    fi
done
exec 3<&-
if [[ $dry_run -eq 0 ]]; then echo "[$(date -Is)] All selected samples completed."; fi
