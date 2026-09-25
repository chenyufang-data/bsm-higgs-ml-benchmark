import json

import awkward as ak
import numpy as np
import pytest
import uproot
import yaml
from hepml_compact.merging_counts import count_file, main
from hepml_compact.root_reader import source_id_for


def _weights(n_events, *, nominal, n_generated, central_slot, seed, share=True):
    """Delphes-like weight rows: 3 raw LHE weights, 4 scale variations, the nominal,
    three merging weights (central in `central_slot`) and one scale-info column."""
    rng = np.random.default_rng(seed)
    accepted = rng.random((n_events, 3)) < [0.4, 0.6, 0.7]
    rows = []
    for event in range(n_events):
        merging = [nominal / n_generated if accepted[event, k] else 0.0 for k in range(3)]
        central = merging[central_slot] != 0
        variations = [nominal / n_generated * r if (central or not share) else 0.0
                      for r in rng.uniform(0.8, 1.2, 4)]
        raw = list(nominal * rng.uniform(0.7, 1.3, 3))
        rows.append(raw + variations + [nominal] + merging + [rng.uniform(10, 30)])
    return rows, accepted[:, central_slot]


def _write(path, rows):
    with uproot.recreate(path) as root:
        root.mktree("Delphes", {"Weight/Weight.Weight": ak.Array(np.asarray(rows, dtype=np.float32).tolist())})
        uuid = str(root.file.uuid)
    return uuid


def test_central_merging_weight_is_found_by_its_zero_pattern(tmp_path):
    rows, accepted = _weights(40, nominal=12.5, n_generated=100, central_slot=2, seed=1)
    uuid = _write(tmp_path / "a.root", rows)
    record = count_file(tmp_path / "a.root", step_size=7)
    assert record["nominal_column"] == 7 and record["central_column"] == 10
    assert record["scale_variation_columns"] == [3, 4, 5, 6]
    assert record["n_generated"] == 100 and record["accepted"] == accepted.sum()
    assert record["entries"] == 40 and record["source_id"] == source_id_for(uuid, "Delphes")
    assert set(record["merging_columns"]) == {"8", "9", "10"}


def test_unidentifiable_central_scale_fails(tmp_path):
    rows, _ = _weights(30, nominal=12.5, n_generated=100, central_slot=0, seed=2, share=False)
    _write(tmp_path / "b.root", rows)
    with pytest.raises(ValueError, match="scale variations share"):
        count_file(tmp_path / "b.root")


def test_sample_counts_pool_files_and_check_the_manifest_rate(tmp_path):
    first, accepted_first = _weights(30, nominal=10.0, n_generated=100, central_slot=1, seed=3)
    second, accepted_second = _weights(20, nominal=10.2, n_generated=80, central_slot=1, seed=4)
    _write(tmp_path / "one.root", first)
    _write(tmp_path / "two.root", second)
    implied = (10.0 * accepted_first.sum() + 10.2 * accepted_second.sum()) / 180
    manifest = tmp_path / "samples.yaml"
    manifest.write_text(yaml.safe_dump({"data_root": str(tmp_path), "samples": {"bkg_x": {
        "kind": "background", "xs_pb": float(2 * implied), "files": ["one.root", "two.root"]}}}))
    output = tmp_path / "counts.json"
    assert main(["--config", str(manifest), "--output", str(output), "--step-size", "9"]) == 0
    sample = json.loads(output.read_text())["samples"]["bkg_x"]
    assert sample["n_stored"] == 50 and sample["n_generated"] == 180
    assert sample["n_accepted"] == accepted_first.sum() + accepted_second.sum()
    assert sample["implied_merged_xs"] == pytest.approx(implied, rel=1e-6)
    assert sample["xs_pb_over_implied"] == pytest.approx(2.0, rel=1e-6)
    with pytest.raises(SystemExit):
        main(["--config", str(manifest), "--output", str(output)])
