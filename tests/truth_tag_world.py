"""A synthetic truth-tag world: Delphes-like events whose tag bits follow the registered efficiencies,
exported with the truth-tag recipe and with the direct selection, plus a registry of the direct events.

Generator weights follow the MLM layout with JetMatching:doVeto=off: raw LHE weights, scale variations
that vanish when the central merging scale rejects the event, the nominal weight, one merging weight per
scale (central in a sample-dependent slot) and a scale-info value. `merging-counts.json` beside the
exports holds what hepml_compact.merging_counts would count, with each sample's rate set to its implied
merged rate. Backgrounds carry hard-process partons that follow their process cards, extra jets
included (some of which overlap another sample), exported under `partons/` with extraction_partons.yaml."""

import json
from contextlib import contextmanager
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import yaml
from hepml_compact.config import load_profile
from hepml_compact.export import export_sample
from hepml_compact.merging_counts import BRANCH, _Columns, identify
from hepml_compact.parquet_writer import sha256_file

from hepml.domain.splits import extend_registry
from hepml.domain.truth_tagging import TaggingModel

STUDY = Path(__file__).resolve().parents[1] / "studies/cg_bbc"
SETTINGS = yaml.safe_load((STUDY / "validation.yaml").read_text())["truth_tagging"]
MODEL = TaggingModel.from_settings(SETTINGS)


def simulate(model, n_events, seed, efficiency_model=None):
    """Events of 3-6 jets whose tag bits are drawn like Delphes, from efficiency_model (default: model)."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(3, 7, n_events)
    offsets = np.r_[0, np.cumsum(counts)]
    n = offsets[-1]
    pt, eta = rng.uniform(20, 250, n), rng.uniform(-3, 3, n)
    flavour = rng.choice([5, 5, 4, 21, 1], n)
    eps_b, eps_c = (efficiency_model or model).efficiencies(flavour, pt)
    btag = model.b_bit * (rng.random(n) < eps_b) + model.c_bit * (rng.random(n) < eps_c)
    return offsets, pt, eta, flavour, btag.astype(int)


class MemorySource:
    """In-memory Delphes events with a per-sample ROOT UUID, so event IDs differ between samples."""

    def __init__(self, arrays, uuid):
        self.arrays, self.uuid = arrays, uuid

    def inspect(self, path, study, step_size):
        return dict(path=str(path), source_id=f"memory-{self.uuid}", root_uuid=self.uuid, size=1, mtime_ns=1,
                    entries=len(self.arrays), chunk_entries=int(step_size), branches=list(study.required_branches),
                    missing_optional=[])

    @contextmanager
    def open(self, source, tree):
        yield self

    def read(self, branches, start, stop):
        return self.arrays[start:stop]

    def assert_unchanged(self, source):
        pass


def mlm_weights(n_events, seed, central_slot, *, nominal=1.0):
    """Stored events of a doVeto=off production: each is accepted at one merging scale at least."""
    rng = np.random.default_rng(seed)
    accepted = rng.random((n_events, 3)) < [0.55, 0.7, 0.8]
    accepted[~accepted.any(axis=1), 2] = True
    n_generated = int(n_events * 1.1)
    rows = []
    for event in range(n_events):
        merging = [nominal / n_generated if accepted[event, k] else 0.0 for k in range(3)]
        central = accepted[event, central_slot]
        variations = [nominal / n_generated * r if central else 0.0 for r in rng.uniform(0.8, 1.2, 4)]
        rows.append([*(nominal * rng.uniform(0.7, 1.3, 3)), *variations, nominal, *merging, rng.uniform(10, 30)])
    return rows


def merging_record(weights, source_id, xs_pb):
    """What merging_counts records for a one-file sample."""
    columns = _Columns(len(weights[0]))
    columns.add(np.asarray(weights, dtype=np.float32).astype(np.float64))
    record = dict(source_id=source_id, entries=len(weights), n_weights=len(weights[0]), **identify(columns))
    implied = record["nominal_weight"] * record["accepted"] / record["n_generated"]
    return dict(kind=None, xs_pb=xs_pb, n_stored=len(weights), n_accepted=record["accepted"],
                n_generated=record["n_generated"], files=[record], implied_merged_xs=implied,
                xs_pb_over_implied=xs_pb / implied)


# Outgoing core partons and the flavours extra jets draw from, per background card.
HARD_PROCESSES = {"bkg_bbc": ([5, -5, 4], [21, 1, -4]), "bkg_bbj": ([5, -5, 21], [21, 2, 4]),
                  "bkg_cjj": ([4, 1, 2], [21, -4, 4]), "bkg_ccj": ([4, -4, 21], [21, 3, 4]),
                  "bkg_bjj": ([5, 1, 2], [21, 1, 4])}


def hard_partons(key, n_events, seed):
    """Delphes Particle PID/Status lists: two incoming partons, the core and 0-2 extra jets
    outgoing (status 23), then a few final-state particles (status 1)."""
    rng = np.random.default_rng(seed)
    core, extras = HARD_PROCESSES[key]
    pids, statuses = [], []
    for _ in range(n_events):
        outgoing = core + rng.choice(extras, rng.integers(0, 3)).tolist()
        pids.append([21, 21, *outgoing, 211, -211, 22])
        statuses.append([21, 21, *[23] * len(outgoing), 1, 1, 1])
    return pids, statuses


def delphes_like(n_events, seed, flavours, mass_scale=1.0, weights=None, partons=None):
    offsets, pt, eta, _, _ = simulate(MODEL, n_events, seed)
    pt = pt * mass_scale
    rng = np.random.default_rng(seed + 1)
    flavour = rng.choice(flavours, len(pt))
    eps_b, eps_c = MODEL.efficiencies(flavour, pt)
    btag = MODEL.b_bit * (rng.random(len(pt)) < eps_b) + MODEL.c_bit * (rng.random(len(pt)) < eps_c)
    split = lambda values: [values[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]  # noqa: E731
    uniform = lambda low, high: split(rng.uniform(low, high, len(pt)))  # noqa: E731
    return ak.Array({
        "Jet/Jet.PT": split(pt), "Jet/Jet.Eta": split(eta), "Jet/Jet.Phi": uniform(-np.pi, np.pi),
        "Jet/Jet.Mass": uniform(2, 15), "Jet/Jet.BTag": split(btag.astype(int)), "Jet/Jet.DeltaEta": uniform(0, 0.4),
        "Jet/Jet.DeltaPhi": uniform(0, 0.4), "Jet/Jet.EhadOverEem": uniform(0.5, 3), "Jet/Jet.Flavor": split(flavour),
        "Event/Event.Weight": [[1.0]] * n_events, "Event/Event.CrossSection": [[1.0]] * n_events,
        "Event/Event.Number": [[i] for i in range(n_events)], "MissingET/MissingET.MET": [[10.0]] * n_events,
        "MissingET/MissingET.Phi": [[0.0]] * n_events, "ScalarHT/ScalarHT.HT": [[200.0]] * n_events,
        "Electron_size": [0] * n_events, "Muon_size": [0] * n_events, "Photon_size": [0] * n_events,
        "Weight/Weight.Weight": weights if weights is not None else [[1.0, 1.1, 0.9]] * n_events,
        **({"Particle/Particle.PID": partons[0], "Particle/Particle.Status": partons[1]} if partons else {})})


def build_world(root, samples, study=STUDY):
    """samples: (folder, key, kind, flavours, events, mass, rho_tc, xs_pb, pt_scale). Returns the three roots."""
    root = Path(root)
    truth_recipe = load_profile(study / "extraction_truth_tag.yaml")
    direct_config = json.loads(json.dumps(truth_recipe.config))
    direct_config["name"] = "cg_bbc"
    direct_config["selection"] = yaml.safe_load((study / "extraction.yaml").read_text())["selection"]
    (root / "recipes").mkdir(parents=True)
    (root / "recipes/direct.yaml").write_text(yaml.safe_dump(direct_config))
    direct_recipe = load_profile(root / "recipes/direct.yaml")
    parton_recipe = load_profile(study / "extraction_partons.yaml")
    direct_events, counts = [], {}
    for index, (folder, key, kind, flavours, n, mass, rho, xs, scale) in enumerate(samples):
        # The merging weights' nominal is chosen so that the implied merged rate equals xs.
        weights = mlm_weights(n, 200 + index, index % 3)
        record = merging_record(weights, f"memory-uuid-{key}", xs)
        nominal = xs / record["implied_merged_xs"]
        weights = mlm_weights(n, 200 + index, index % 3, nominal=nominal)
        counts[key] = dict(merging_record(weights, f"memory-uuid-{key}", xs), kind=kind)
        partons = hard_partons(key, n, 300 + index) if key in HARD_PROCESSES else None
        arrays = delphes_like(n, 100 + index, flavours, scale, weights, partons)
        options = dict(sample=key, kind=kind, xs_pb=xs, step_size=1000)
        if kind == "signal":
            options.update(mass=mass, rho_tc=rho)
        recipes = [("truth", truth_recipe), ("direct", direct_recipe)] + ([("partons", parton_recipe)] if partons else [])
        for side, recipe in recipes:
            manifest = export_sample([Path(f"{key}.root")], root / side / folder, recipe,
                                     source_reader=MemorySource(arrays, f"uuid-{key}"), **options)
            assert manifest["status"] == "complete"
        direct_events.append(pd.concat([pd.read_parquet(root / "direct" / folder / c["file"],
                                                        columns=["event_id", "sample_key"])
                                        for c in manifest["chunks"] if c["file"]]))
    (root / "merging-counts.json").write_text(json.dumps(dict(schema=1, branch=BRANCH, samples=counts)))
    registry = root / "registry"
    registry.mkdir()
    extend_registry(pd.DataFrame(), pd.concat(direct_events)).to_parquet(registry / "assignments.parquet", index=False)
    (registry / "registry.json").write_text(json.dumps(dict(seed=42, assignments_sha256=sha256_file(
        registry / "assignments.parquet"))))
    return root / "truth", root / "direct", registry
