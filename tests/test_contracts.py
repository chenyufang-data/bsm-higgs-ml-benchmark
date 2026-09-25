"""File-naming contracts and the features.txt round-trip."""

from pathlib import Path

from hepml_compact.contracts import meta_name, part_glob_pattern, part_name, sample_stem

from hepml.domain.artifacts import (
    BenchmarkFiles,
    assignments_filename,
    dataset_filename,
    model_dirname,
    split_filename,
    split_meta_filename,
    split_meta_path,
)
from studies.cg_bbc.features import DEFAULT_FEATURES


def test_stems():
    assert sample_stem("signal", "sig_200") == "signal_sig_200"
    assert sample_stem("background", "bbj") == "background_bbj"


def test_part_names_match_glob():
    stem = sample_stem("signal", "sig_200")
    name = part_name(stem, 3)
    assert name == "signal_sig_200.part00003.parquet"
    # what compact writes must be found by what prepare globs
    assert Path(name).match(part_glob_pattern(stem))


def test_meta_and_dataset_names():
    assert meta_name("background_ccj") == "background_ccj.meta.json"
    assert dataset_filename(200) == "dataset_sig200_vs_bkg.parquet"
    assert split_filename("test", 200) == "test_sig200.parquet"
    assert split_meta_filename(200) == "split_sig200.meta.json"
    assert assignments_filename(200) == "assignments_sig200.parquet"
    assert model_dirname(200) == "sig200"
    assert split_meta_path(Path("datasets/splits"), 200) == Path("datasets/splits/split_sig200.meta.json")


def test_benchmark_layout_matches_completed_outputs():
    files = BenchmarkFiles(Path("data"))
    assert files.dataset(400) == Path("data/dataset_sig400_vs_bkg.parquet")
    assert files.split("val", 400) == Path("data/splits/val_sig400.parquet")
    assert files.assignments(400) == Path("data/splits/assignments_sig400.parquet")
    assert files.meta(400) == Path("data/splits/split_sig400.meta.json")


def test_features_txt_round_trip(tmp_path):
    # freeze_final writes features.txt this way; predict.read_features reads it
    from hepml.adapters.inference import read_features

    path = tmp_path / "features.txt"
    path.write_text("\n".join(DEFAULT_FEATURES) + "\n", encoding="utf-8")
    assert read_features(path) == DEFAULT_FEATURES
