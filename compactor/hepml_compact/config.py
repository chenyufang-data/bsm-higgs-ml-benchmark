"""Validated, portable extraction recipes. No channel imports or default study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

TYPES = {"float32", "float64", "int32", "int64"}
IDENTITY_TYPES = {"source_entry": "int64", "source_id": "string", "event_id": "string", "sample_key": "string"}
PREDICATES = {"eq", "gt", "ge", "lt", "le", "abs_lt", "bit_any"}
REDUCTIONS = {"scalar", "first", "list"}
ACCUMULATORS = {"ratio_to_first"}
RECIPE_VERSIONS = (1, 2)
TRUTH = "truth"


def _keys(value, allowed, where):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{where}: expected a mapping with keys {sorted(allowed)}")


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid field/collection/role name: {value!r}")


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _field(value, where, *, event):
    allowed = {"branch", "dtype", "required", "default", "truth"}
    allowed |= {"reduce", "accumulate"} if event else {"map_abs", "map_default"}
    _keys(value, allowed, where)
    if not isinstance(value.get("branch"), str) or not value["branch"]:
        raise ValueError(f"{where}: nonempty branch required")
    dtype = value.get("dtype", "float32")
    if dtype not in TYPES:
        raise ValueError(f"{where}: unsupported dtype")
    for flag in ("required", "truth"):
        if not isinstance(value.get(flag, flag == "required"), bool):
            raise ValueError(f"{where}: {flag} must be boolean")
    if not value.get("required", True) and "default" not in value:
        raise ValueError(f"{where}: optional fields require an explicit default")
    if "default" in value:
        default = value["default"]
        if not isinstance(default, (int, float)) or not math.isfinite(default):
            raise ValueError(f"{where}: default must be finite numeric")
        if dtype.startswith("int") and int(default) != default:
            raise ValueError(f"{where}: integer default required")
    if "map_abs" in value:
        mapping = value["map_abs"]
        if (not dtype.startswith("int") or not isinstance(mapping, dict) or not mapping
                or not all(_integer(k) and k >= 0 and _integer(v) for k, v in mapping.items())
                or not _integer(value.get("map_default"))):
            raise ValueError(f"{where}: map_abs needs an integer dtype, {{|value|: code}} integers and a map_default")
    elif "map_default" in value:
        raise ValueError(f"{where}: map_default requires map_abs")
    if event:
        if value.get("reduce", "scalar") not in REDUCTIONS:
            raise ValueError(f"{where}: reduce must be scalar, first or list")
        if value.get("reduce") == "list" and not value.get("required", True):
            raise ValueError(f"{where}: list-valued event fields must be required")
        if "accumulate" in value and (value["accumulate"] not in ACCUMULATORS or value.get("reduce") != "list"
                                      or not dtype.startswith("float")):
            raise ValueError(f"{where}: accumulate: ratio_to_first needs a float list field (reduce: list)")


def _filters(filters, fields, where):
    if not isinstance(filters, dict):
        raise ValueError(f"{where}: filters must be a mapping")
    for field, predicates in filters.items():
        if field not in fields:
            raise ValueError(f"{where}: unknown filter field {field}")
        if "map_abs" in fields[field]:
            raise ValueError(f"{where}: cannot filter on mapped field {field}")
        _keys(predicates, PREDICATES, where)
        for op, value in predicates.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{where}: filter values must be finite numbers")
            if op == "bit_any" and (
                not fields[field].get("dtype", "float32").startswith("int") or not _integer(value) or value < 0
            ):
                raise ValueError(f"{where}: bit_any requires an integer field and nonnegative integer mask")


def _is_truth(name, spec, collection_truth=False):
    return collection_truth or spec.get("truth", False)


@dataclass
class ExtractionProfile:
    directory: Path
    config: dict
    fingerprint: str
    hook: object = None

    @property
    def name(self):
        return self.config["name"]

    @property
    def fields(self):
        return [f for c in self.config["collections"].values() for f in c["fields"].values()] + list(
            self.config["event_fields"].values()
        )

    @property
    def reference_branches(self):
        return [b for c in self.config["collections"].values() if "owner" in c
                for b in (c["owner"]["refs"], c["owner"]["uid"])]

    @property
    def required_branches(self):
        return list(dict.fromkeys(
            [f["branch"] for f in self.fields if f.get("required", True)] + self.reference_branches
        ))

    @property
    def optional_branches(self):
        return list(
            dict.fromkeys(
                f["branch"]
                for f in self.fields
                if not f.get("required", True) and f["branch"] not in self.required_branches
            )
        )

    @property
    def roles(self):
        return {
            role: selection["collection"]
            for selection in self.config["selection"].values()
            for role in selection.get("roles", [])
        }

    @property
    def output_dtypes(self):
        result = {"label": "int32"}
        for name, spec in self.config["event_fields"].items():
            kind = spec.get("dtype", "float32")
            result[name] = "list:" + kind if spec.get("reduce") == "list" else kind
        for collection, spec in self.config["collections"].items():
            result.update({f"{collection}_{k}": "list:" + f.get("dtype", "float32") for k, f in spec["fields"].items()})
            result[f"{collection}_source_index"] = "list:int64"
            if "owner" in spec:
                result[f"{collection}_owner"] = "list:int32"
        result.update({f"role_{role}": "int64" for role in self.roles})
        result.update(IDENTITY_TYPES)
        return result

    @property
    def truth_columns(self):
        columns = [name for name, spec in self.config["event_fields"].items() if spec.get("truth", False)]
        for collection, spec in self.config["collections"].items():
            columns += [f"{collection}_{k}" for k, f in spec["fields"].items() if _is_truth(k, f, spec.get("truth"))]
        return columns

    @property
    def accumulators(self):
        return {name: spec["accumulate"] for name, spec in self.config["event_fields"].items() if "accumulate" in spec}

    def process_chunk(self, arrays, label):
        from .extraction import extract_chunk

        return extract_chunk(arrays, label, self)


def selection_fingerprint(config: dict) -> str:
    """Hash of everything that decides event membership and role indices.

    Two recipes with the same selection fingerprint select the same events from the
    same files, whatever extra fields or collections they store. A recipe hook can
    change selection arbitrarily, so a hooked recipe is compared by its full
    fingerprint instead (see recipes_compatible).
    """
    selection = config.get("selection", {}) or {}
    used = {}
    for group in selection.values():
        fields = config["collections"][group["collection"]]["fields"]
        names = set(group.get("filters", {})) | ({group["sort_by"]} if "sort_by" in group else set())
        # Multiplicity comes from the first required field of the collection.
        names.add(next(k for k, f in fields.items() if f.get("required", True)))
        for name in names:
            spec = fields[name]
            used[f"{group['collection']}.{name}"] = {k: spec[k] for k in ("branch", "dtype", "required", "default")
                                                     if k in spec}
    payload = dict(tree=config.get("tree"), units=config.get("units"), selection=selection, fields=used,
                   hook=config.get("hook"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def recipes_compatible(export_config: dict, export_fingerprint: str, profile: ExtractionProfile) -> bool:
    """Whether data exported with export_config can be read with profile's study.

    Identical recipes are always compatible. Otherwise both must be hook-free and
    select events identically; extra or missing stored fields are then allowed, and
    a missing field fails later with its column name if the study needs it.
    """
    if export_fingerprint == profile.fingerprint:
        return True
    if profile.hook is not None or export_config.get("hook"):
        return False
    return selection_fingerprint(export_config) == selection_fingerprint(profile.config)


def load_profile(path: str | Path) -> ExtractionProfile:
    path = Path(path).resolve()
    if path.is_dir():
        path /= "extraction.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _keys(
        raw, {"schema_version", "name", "tree", "units", "collections", "event_fields", "selection", "hook"}, "profile"
    )
    version = raw.get("schema_version")
    if version not in RECIPE_VERSIONS:
        raise ValueError(f"Unsupported extraction schema_version; expected one of {RECIPE_VERSIONS}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(raw.get("name", ""))):
        raise ValueError("Extraction name must be a safe nonempty name")
    if not isinstance(raw.get("tree"), str) or not raw["tree"]:
        raise ValueError("ROOT tree name required")
    if raw.get("units") != {"momentum": "GeV", "angle": "radians", "cross_section": "pb"}:
        raise ValueError("Supported units are momentum=GeV, angle=radians, cross_section=pb; no implicit conversion")
    collections = raw.get("collections")
    if not isinstance(collections, dict) or not collections:
        raise ValueError("At least one collection is required")
    columns = ["label", *IDENTITY_TYPES]
    uses_v2 = False
    for name, spec in collections.items():
        _name(name)
        _keys(spec, {"fields", "keep", "owner", "truth"}, name)
        uses_v2 |= bool(set(spec) - {"fields"})
        fields = spec.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError(f"{name}: nonempty fields required")
        if not isinstance(spec.get("truth", False), bool):
            raise ValueError(f"{name}: truth must be boolean")
        if spec.get("truth") and not name.startswith(TRUTH + "_"):
            raise ValueError(f"{name}: a truth collection's name must start with '{TRUTH}_'")
        for field, value in fields.items():
            _name(field)
            if field in {"source_index", "owner"}:
                raise ValueError(f"{field} is a reserved field name")
            _field(value, f"{name}.{field}", event=False)
            uses_v2 |= bool(set(value) & {"truth", "map_abs", "map_default"})
            if value.get("truth") and not spec.get("truth") and not field.startswith(TRUTH + "_"):
                raise ValueError(f"{name}.{field}: a truth field's name must start with '{TRUTH}_'")
            columns.append(f"{name}_{field}")
        if not any(f.get("required", True) for f in fields.values()):
            raise ValueError(f"{name}: needs at least one required field to define object multiplicity")
        columns.append(f"{name}_source_index")
        if "keep" in spec and spec["keep"] != "owned":
            _filters(spec["keep"], fields, f"{name}.keep")
        if "owner" in spec:
            columns.append(f"{name}_owner")
    for name, spec in collections.items():
        if "owner" not in spec:
            if spec.get("keep") == "owned":
                raise ValueError(f"{name}: keep: owned requires an owner")
            continue
        owner = spec["owner"]
        _keys(owner, {"collection", "refs", "uid"}, f"{name}.owner")
        if not all(isinstance(owner.get(k), str) and owner[k] for k in ("collection", "refs", "uid")):
            raise ValueError(f"{name}.owner: collection, refs and uid are required")
        parent = collections.get(owner["collection"])
        if parent is None or owner["collection"] == name or "keep" in parent or "owner" in parent:
            raise ValueError(f"{name}.owner: must name another collection that has no keep or owner of its own")
    events = raw.get("event_fields")
    if not isinstance(events, dict) or not {"weight", "xs"}.issubset(events):
        raise ValueError("event_fields must declare weight and xs (with explicit defaults if optional)")
    for name, spec in events.items():
        _name(name)
        _field(spec, name, event=True)
        uses_v2 |= bool(set(spec) & {"truth", "accumulate"}) or spec.get("reduce") == "list"
        if spec.get("truth") and not name.startswith(TRUTH + "_"):
            raise ValueError(f"{name}: a truth field's name must start with '{TRUTH}_'")
        columns.append(name)
    selection = raw.setdefault("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("selection must be a mapping")
    for group, spec in selection.items():
        _name(group)
        if group in {"input", "selected"}:
            raise ValueError(f"{group}: reserved cutflow name")
        _keys(spec, {"collection", "filters", "count", "sort_by", "roles"}, group)
        if spec.get("collection") not in collections:
            raise ValueError(f"{group}: unknown collection")
        if "keep" in collections[spec["collection"]]:
            raise ValueError(f"{group}: selection cannot use a collection with a storage filter (keep)")
        fields = collections[spec["collection"]]["fields"]
        if "sort_by" in spec and spec["sort_by"] not in fields:
            raise ValueError(f"{group}: unknown sort field")
        _filters(spec.get("filters", {}), fields, group)
        count = spec.get("count", {})
        _keys(count, {"exact", "min", "max"}, group)
        if "exact" in count and len(count) > 1:
            raise ValueError(f"{group}: exact cannot be combined with min/max")
        if any(not _integer(v) or v < 0 for v in count.values()):
            raise ValueError(f"{group}: counts must be nonnegative integers")
        if count.get("min", 0) > count.get("max", float("inf")):
            raise ValueError(f"{group}: min exceeds max")
        roles = spec.get("roles", [])
        if not isinstance(roles, list):
            raise ValueError(f"{group}: roles must be a list")
        for role in roles:
            _name(role)
            columns.append(f"role_{role}")
        if roles and (spec.get("sort_by") not in fields or count.get("exact", count.get("min", 0)) < len(roles)):
            raise ValueError(f"{group}: roles need sort_by and a count guaranteeing enough objects")
    if uses_v2 and version < 2:
        raise ValueError("keep, owner, truth, map_abs, list fields and accumulators require schema_version 2")
    if len(columns) != len(set(columns)):
        raise ValueError("Output column names must be unique (including role and metadata columns)")
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode())
    hook = None
    if "hook" in raw:
        if raw["hook"] != "extraction.py":
            raise ValueError("The optional trusted hook must be extraction.py beside extraction.yaml")
        hook_path = path.parent / "extraction.py"
        digest.update(hook_path.read_bytes().replace(b"\r\n", b"\n"))
        spec = importlib.util.spec_from_file_location(f"_compact_hook_{digest.hexdigest()}", hook_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hook = getattr(module, "select", None)
        if not callable(hook):
            raise ValueError("extraction.py must define select(collections, keep, roles, config)")
    return ExtractionProfile(path.parent, raw, digest.hexdigest(), hook)


def load_compact_sample(manifest: Path, key: str, data_root: str | None = None) -> tuple[list[Path], dict]:
    """Resolve files: CLI root > HEPML_DATA_ROOT > manifest-relative data_root."""
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    samples = raw.get("samples") or {}
    if key not in samples:
        raise ValueError(f"Sample {key!r} is not configured in {manifest}; add the new production bookkeeping first")
    sample = samples[key]
    files = sample.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(f, str) and f for f in files):
        raise ValueError(f"Sample {key!r} needs a nonempty files list")
    if sample.get("kind") not in {"signal", "background"}:
        raise ValueError(f"Sample {key!r} kind must be signal or background")
    root = Path(data_root or os.environ.get("HEPML_DATA_ROOT") or raw.get("data_root", "."))
    if not root.is_absolute():
        root = manifest.resolve().parent / root
    return [root / name for name in files], sample
