"""Convenience dispatch to the independently installable CPU compactor."""

from hepml_compact.cli import main as compact_main

from hepml.adapters.configuration import default_study_directory


def main(argv=None):
    args = list(argv or [])
    if not any(arg in {"--study", "--extraction"} or arg.startswith(("--study=", "--extraction=")) for arg in args):
        args += ["--extraction", str(default_study_directory() / "extraction.yaml")]
    return compact_main(args)
