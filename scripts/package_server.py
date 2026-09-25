"""Build the CPU extraction upload, using a whitelist (never includes data/.env).

The ZIP unpacks into a versioned folder, hepml-compactor-<version>/, so a new
deployment never overwrites an older one. MANIFEST.json records the version,
the git commit, whether the bundled files had uncommitted changes, and the
SHA-256 of every bundled file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def compactor_version(repository: Path) -> str:
    text = (repository / "compactor/hepml_compact/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', text, re.MULTILINE)
    if not match:
        raise ValueError("compactor/hepml_compact/__init__.py must define __version__")
    return match.group(1)


def git_state(repository: Path, paths: list[Path]) -> tuple[str | None, bool | None]:
    """(commit, dirty) for the bundled paths, or (None, None) outside a git checkout."""
    try:
        commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=20).stdout.strip()
        inside = [str(p) for p in paths if p.resolve().is_relative_to(repository)]
        status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--", *inside],
                                capture_output=True, text=True, check=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())


def package_server(repository: Path, destination: Path, study: Path | None = None) -> Path:
    repository = repository.resolve()
    from hepml_compact.config import load_profile

    study = (study or repository / "studies" / "cg_bbc").resolve()
    profile = load_profile(study)
    standalone = repository / "compactor"
    version = compactor_version(repository)
    top = f"hepml-compactor-{version}"
    files = {
        "pyproject.toml": standalone / "pyproject.toml",
        "README.md": standalone / "README.md",
        "run_extraction.sh": standalone / "run_extraction.sh",
        "constraints-py311.txt": standalone / "constraints-py311.txt",
        "LICENSE": repository / "LICENSE",
    }
    files.update(
        {path.relative_to(standalone).as_posix(): path for path in sorted((standalone / "hepml_compact").glob("*.py"))}
    )
    # Every recipe of the study (extraction.yaml plus variants such as the truth-tagging
    # one); each must validate before it is shipped.
    for recipe in sorted(study.glob("extraction*.yaml")):
        load_profile(recipe)
        files[f"study/{recipe.name}"] = recipe
    files["study/samples.yaml"] = study / "samples.yaml"
    if profile.hook:
        files["study/extraction.py"] = study / "extraction.py"
    contents = {name: path.read_bytes().replace(b"\r\n", b"\n") for name, path in sorted(files.items())}
    commit, dirty = git_state(repository, [standalone, study, repository / "LICENSE"])
    manifest = {
        "version": version,
        "git_commit": commit,
        "git_dirty": dirty,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    contents["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "hepml-server-extract.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(contents.items()):
            entry = ZipInfo(f"{top}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            bundle.writestr(entry, data)
    temporary.replace(archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    # GNU sha256sum treats a Windows CR as part of the filename. Always emit LF.
    archive.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {archive.name}\n".encode("ascii"))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, help="Default: <HEPML_OUTPUT_ROOT>/packages")
    parser.add_argument("--study", type=Path, help="Study directory to copy (default: studies/cg_bbc)")
    args = parser.parse_args()
    from hepml.adapters.configuration import output_paths

    destination = args.outdir if args.outdir is not None else output_paths().base / "packages"
    archive = package_server(Path(__file__).resolve().parents[1], destination, args.study)
    manifest = json.loads(ZipFile(archive).read(next(n for n in ZipFile(archive).namelist()
                                                    if n.endswith("/MANIFEST.json"))))
    print(archive.resolve())
    print(f"hepml-compactor {manifest['version']} at commit {manifest['git_commit']}"
          + (" (with uncommitted changes)" if manifest["git_dirty"] else ""))


if __name__ == "__main__":
    main()
