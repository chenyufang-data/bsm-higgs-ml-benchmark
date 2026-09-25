"""Extract aligned ragged objects, owner links and roles; never compute training features."""

from __future__ import annotations

from dataclasses import dataclass, field

import awkward as ak
import numpy as np
import pyarrow as pa

ARROW_TYPES = {
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "string": pa.string(),
}


def arrow_type(kind: str) -> pa.DataType:
    return pa.list_(ARROW_TYPES[kind[5:]]) if kind.startswith("list:") else ARROW_TYPES[kind]


@dataclass
class SelectedChunk:
    table: pa.Table
    offsets: np.ndarray
    cutflow: dict[str, int]
    accumulators: dict = field(default_factory=dict)

    @property
    def frame(self):
        """Pandas view for inspection and tests; the writer uses `table`."""
        return self.table.to_pandas()


def _cast(values, dtype, name):
    flat = ak.to_numpy(ak.flatten(values, axis=None))
    if ak.any(ak.is_none(values, axis=-1)) or not np.isfinite(flat).all():
        raise ValueError(f"{name}: missing or nonfinite values")
    if dtype.startswith("int"):
        bounds = np.iinfo(dtype)
        if np.any(flat != np.floor(flat)) or np.any(flat < bounds.min) or np.any(flat > bounds.max):
            raise ValueError(f"{name}: invalid or overflowing integer")
    result = ak.values_astype(values, dtype)
    if not np.isfinite(ak.to_numpy(ak.flatten(result, axis=None))).all():
        raise ValueError(f"{name}: overflow converting to {dtype}")
    return result


def _predicates(fields, filters, reference):
    """Object-level mask for {field: {op: value}} filters, as in the selection groups."""
    mask = ak.ones_like(reference, dtype=bool)
    for name, predicates in filters.items():
        values = fields[name]
        for op, value in predicates.items():
            if op == "eq":
                passed = values == value
            elif op == "gt":
                passed = values > value
            elif op == "ge":
                passed = values >= value
            elif op == "lt":
                passed = values < value
            elif op == "le":
                passed = values <= value
            elif op == "abs_lt":
                passed = abs(values) < value
            else:  # bit_any validated by the config loader
                passed = (values & value) != 0
            mask = mask & passed
    return mask


def resolve_owners(refs, uid, owner_counts, what):
    """Owner index of every target object, matching reference IDs within each event.

    `refs` holds, per event and per owner object, the unique IDs it references
    (a Delphes TRefArray reads as records with a `refs` field; a plain nested
    integer list is accepted too). `uid` holds each target object's unique ID.
    Every reference must match exactly one target in the same event, and no target
    may have two owners. Unreferenced targets get -1.
    """
    if "refs" in ak.fields(refs):
        refs = refs["refs"]
    if not bool(ak.all(ak.num(refs, axis=1) == owner_counts)):
        raise ValueError(f"{what}: reference lists do not align with their owner collection")
    n = len(uid)
    uid_counts = ak.to_numpy(ak.num(uid, axis=1))
    uid_flat = ak.to_numpy(ak.flatten(uid, axis=None)).astype(np.int64)
    if np.any(uid_flat < 0) or np.any(uid_flat >= 2**32):
        raise ValueError(f"{what}: unique IDs must fit in 32 unsigned bits")
    target_key = (np.repeat(np.arange(n, dtype=np.int64), uid_counts) << 32) | uid_flat
    owners_per_event = ak.to_numpy(ak.num(refs, axis=1))
    refs_per_owner = ak.to_numpy(ak.flatten(ak.num(refs, axis=2), axis=None))
    owner_local = ak.to_numpy(ak.flatten(ak.local_index(refs, axis=1), axis=None))
    ref_flat = ak.to_numpy(ak.flatten(refs, axis=None)).astype(np.int64)
    event_of_owner = np.repeat(np.arange(n, dtype=np.int64), owners_per_event)
    ref_key = (np.repeat(event_of_owner, refs_per_owner) << 32) | ref_flat
    order = np.argsort(target_key, kind="stable")
    ordered = target_key[order]
    low = np.searchsorted(ordered, ref_key, side="left")
    high = np.searchsorted(ordered, ref_key, side="right")
    if np.any(high - low != 1):
        raise ValueError(f"{what}: every reference must match exactly one object in the same event")
    target = order[low]
    if len(np.unique(target)) != len(target):
        raise ValueError(f"{what}: an object is referenced by more than one owner")
    owner = np.full(len(target_key), -1, dtype=np.int64)
    owner[target] = np.repeat(owner_local, refs_per_owner)
    return ak.unflatten(owner, uid_counts)


def _map_abs(values, mapping, default):
    counts = ak.to_numpy(ak.num(values, axis=1))
    flat = np.abs(ak.to_numpy(ak.flatten(values, axis=None)).astype(np.int64))
    codes = np.full(len(flat), default, dtype=np.int64)
    for key, code in mapping.items():
        codes[flat == key] = code
    return ak.unflatten(codes, counts)


def _list_column(values, dtype):
    array = ak.to_arrow(ak.values_astype(values, dtype), list_to32=True, extensionarray=False)
    return array.cast(pa.list_(ARROW_TYPES[dtype]))


def _accumulate(values, name, n):
    """Per-index sums of w_i / w_0 over every event in the chunk (before selection)."""
    lengths = ak.to_numpy(ak.num(values, axis=1))
    if n == 0:
        return dict(kind="ratio_to_first", events=0, sum=None)
    if len(set(lengths.tolist())) != 1 or lengths[0] == 0:
        raise ValueError(f"{name}: every event needs the same nonzero number of weights")
    weights = ak.to_numpy(ak.flatten(values, axis=None)).astype(np.float64).reshape(n, lengths[0])
    if not np.isfinite(weights).all() or np.any(weights[:, 0] == 0):
        raise ValueError(f"{name}: weights must be finite with a nonzero first entry")
    sums = (weights / weights[:, :1]).sum(axis=0)
    return dict(kind="ratio_to_first", events=int(n), sum=[float(x) for x in sums])


def extract_chunk(arrays, label, profile):
    n = len(arrays)
    config = profile.config
    collections = {}
    for name, spec in config["collections"].items():
        fields = spec["fields"]
        reference = arrays[next(f["branch"] for f in fields.values() if f.get("required", True))]
        if reference.ndim != 2 or bool(ak.any(ak.is_none(reference, axis=0))):
            raise ValueError(f"{name}: collection branches must be event-by-object lists")
        counts = ak.num(reference, axis=1)
        collection = {}
        for field_name, field_config in fields.items():
            branch = field_config["branch"]
            if branch not in arrays.fields:
                if field_config.get("required", True):
                    raise ValueError(f"Missing required branch {branch}")
                values = ak.full_like(reference, field_config["default"], dtype=field_config.get("dtype", "float32"))
            else:
                values = arrays[branch]
            if (
                values.ndim != 2
                or bool(ak.any(ak.is_none(values, axis=0)))
                or not bool(ak.all(ak.num(values) == counts))
            ):
                raise ValueError(f"{name}.{field_name}: collection field lengths do not align")
            # Selection uses source precision, as in the original pipeline.
            _cast(values, field_config.get("dtype", "float32"), f"{name}.{field_name}")
            # Delphes can store tiny signed masses from numerical roundoff.
            # Preserve them; local four-vectors use mass squared, as before.
            if field_name == "pt" and bool(ak.any(values < 0)):
                raise ValueError(f"{name}.{field_name}: negative kinematics")
            collection[field_name] = values
        collections[name] = collection
    keep = np.ones(n, dtype=bool)
    roles = {}
    cutflow = {"input": n}
    for group, spec in config["selection"].items():
        fields = collections[spec["collection"]]
        reference = next(iter(fields.values()))
        mask = _predicates(fields, spec.get("filters", {}), reference)
        counts = ak.to_numpy(ak.sum(mask, axis=1))
        count = spec.get("count", {})
        accepted = np.ones(n, dtype=bool)
        for op, value in count.items():
            accepted &= counts == value if op == "exact" else counts >= value if op == "min" else counts <= value
        keep &= accepted
        cutflow[group] = int(keep.sum())
        indices = ak.local_index(reference, axis=1)[mask]
        if spec.get("roles"):
            order = ak.argsort(fields[spec["sort_by"]][mask], axis=1, ascending=False, stable=True)
            indices = ak.pad_none(indices[order], len(spec["roles"]))
            for position, role in enumerate(spec["roles"]):
                roles[role] = ak.to_numpy(ak.fill_none(indices[:, position], -1)).astype(np.int64)
    if profile.hook:
        keep, roles = profile.hook(collections, keep.copy(), {k: v.copy() for k, v in roles.items()}, config)
    keep = np.asarray(keep)
    if keep.shape != (n,) or keep.dtype != np.bool_ or set(roles) != set(profile.roles):
        raise ValueError("Selection hook must return a boolean event mask and exactly the declared role arrays")

    accumulators = {}
    for name in profile.accumulators:
        accumulators[name] = _accumulate(arrays[config["event_fields"][name]["branch"]], name, n)

    dtypes = profile.output_dtypes
    selected = int(keep.sum())
    columns = {"label": pa.array(np.full(selected, label, dtype=np.int32), type=pa.int32())}
    for name, spec in config["event_fields"].items():
        branch, dtype, reduce = spec["branch"], spec.get("dtype", "float32"), spec.get("reduce", "scalar")
        if branch not in arrays.fields:
            if spec.get("required", True):
                raise ValueError(f"Missing required branch {branch}")
            values = ak.Array(np.full(n, spec["default"]))
        else:
            values = arrays[branch]
            if reduce == "first":
                if values.ndim != 2:
                    raise ValueError(f"{name}: reduce=first needs event-by-value lists")
                values = ak.firsts(values)
                if "default" in spec:
                    values = ak.fill_none(values, spec["default"])
            elif reduce == "list" and values.ndim != 2:
                raise ValueError(f"{name}: reduce=list needs event-by-value lists")
            if reduce != "list" and values.ndim != 1:
                raise ValueError(f"{name}: expected one scalar per event")
        chosen = _cast(values[keep], dtype, name)
        if reduce == "list":
            columns[name] = _list_column(chosen, dtype)
            continue
        chosen = ak.to_numpy(chosen)
        if name in {"weight", "xs"} and (np.any(chosen < 0) or (len(chosen) and chosen.sum() == 0)):
            raise ValueError(f"{name}: current normalization policy requires nonnegative values with positive total")
        columns[name] = pa.array(chosen, type=ARROW_TYPES[dtype])
    for name, spec in config["collections"].items():
        fields = collections[name]
        reference = next(iter(fields.values()))[keep]
        stored = ak.ones_like(reference, dtype=bool)
        owner = None
        if "owner" in spec:
            link = spec["owner"]
            parent = collections[link["collection"]]
            owner = resolve_owners(arrays[link["refs"]][keep], arrays[link["uid"]][keep],
                                   ak.num(next(iter(parent.values()))[keep], axis=1), f"{name}.owner")
            if not bool(ak.all(ak.num(owner, axis=1) == ak.num(reference, axis=1))):
                raise ValueError(f"{name}: unique IDs do not align with the collection")
        keep_rule = spec.get("keep")
        if keep_rule == "owned":
            stored = owner >= 0
        elif keep_rule:
            stored = _predicates({k: v[keep] for k, v in fields.items()}, keep_rule, reference)
        for field_name, values in fields.items():
            field_config = spec["fields"][field_name]
            dtype = field_config.get("dtype", "float32")
            values = values[keep][stored]
            if "map_abs" in field_config:
                values = _map_abs(values, field_config["map_abs"], field_config["map_default"])
            columns[f"{name}_{field_name}"] = _list_column(values, dtype)
        columns[f"{name}_source_index"] = _list_column(ak.local_index(reference, axis=1)[stored], "int64")
        if owner is not None:
            columns[f"{name}_owner"] = _list_column(owner[stored], "int32")
    for role, collection in profile.roles.items():
        indices = np.asarray(roles[role])
        counts = ak.to_numpy(ak.num(next(iter(collections[collection].values())), axis=1))
        if (
            indices.shape != (n,)
            or indices.dtype.kind not in "iu"
            or np.any(indices[keep] < 0)
            or np.any(indices[keep] >= counts[keep])
        ):
            raise ValueError(f"{role}: invalid object indices for selected events")
        columns[f"role_{role}"] = pa.array(indices[keep].astype(np.int64), type=pa.int64())
    names = [name for name in dtypes if name in columns]
    if set(names) != set(dtypes) - set(("source_entry", "source_id", "event_id", "sample_key")):
        raise ValueError("Extracted columns do not match the recipe's output schema")
    table = pa.table([columns[name] for name in names],
                     schema=pa.schema([(name, arrow_type(dtypes[name])) for name in names]))
    cutflow["selected"] = selected
    return SelectedChunk(table, np.flatnonzero(keep), cutflow, accumulators)
