"""Truth tagging under parameterised Delphes flavour tagging. Filesystem-free.

Delphes tags every jet with two independent random draws, a b tagger and a c
tagger, whose efficiencies depend only on the jet's flavour label and pT. The
cg_bbc selection keeps an event when, among the jets in acceptance, exactly two
have a tag word equal to the b bit and exactly one equal to the c bit; a jet with
both bits or none is neutral. Every passing outcome is therefore a choice
(b1, b2, c1) of jets, and its probability is the product of the per-jet state
probabilities. Truth tagging turns each passing outcome into one weighted entry;
in expectation its yields and distributions equal those of direct tagging.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from functools import cache
from itertools import combinations

import numpy as np

STATES = ("none", "b_only", "c_only", "both")  # index = has_b + 2 * has_c
FLAVOUR_CLASSES = ("b", "c", "other")
_FUNCTIONS = {"tanh": np.tanh, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "log": np.log}
_OPERATORS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def compile_efficiency(text):
    """A Delphes efficiency expression of pt (GeV), evaluated without eval() and vectorised."""
    tree = ast.parse(str(text), mode="eval")

    def evaluate(node, pt):
        if isinstance(node, ast.Expression):
            return evaluate(node.body, pt)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pt":
            return pt
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left, pt), evaluate(node.right, pt))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand, pt)
            return -value if isinstance(node.op, ast.USub) else value
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
                and len(node.args) == 1 and not node.keywords):
            return _FUNCTIONS[node.func.id](evaluate(node.args[0], pt))
        raise ValueError(f"Unsupported efficiency expression: {text!r}")

    def efficiency(pt):
        pt = np.asarray(pt, dtype=float)
        return np.broadcast_to(np.asarray(evaluate(tree, pt), dtype=float), pt.shape).copy()

    probe = efficiency(np.geomspace(1.0, 1.0e4, 200))
    if not (np.isfinite(probe).all() and (probe >= 0).all() and (probe <= 1).all()):
        raise ValueError(f"Efficiency {text!r} leaves [0, 1] between 1 GeV and 10 TeV")
    return efficiency


@dataclass(frozen=True)
class TaggingModel:
    """The pre-registered tagging parameterisation and the direct selection it must reproduce."""

    b_bit: int
    c_bit: int
    b_label: int
    c_label: int
    pt_min: float
    abs_eta_max: float
    b_tag: tuple  # efficiency callables for (b, c, other) jets
    c_tag: tuple

    @classmethod
    def from_settings(cls, settings):
        eff, flavour, acceptance = settings["efficiencies"], settings["flavour"], settings["acceptance"]
        tagger = {name: tuple(compile_efficiency(eff[name][key]) for key in ("b", "c", "default"))
                  for name in ("b_tag", "c_tag")}
        b_bit, c_bit = int(eff["b_tag"]["bit_value"]), int(eff["c_tag"]["bit_value"])
        if b_bit <= 0 or c_bit <= 0 or b_bit & c_bit:
            raise ValueError("Tag bits must be distinct positive bit values")
        return cls(b_bit=b_bit, c_bit=c_bit, b_label=int(flavour["b"]), c_label=int(flavour["c"]),
                   pt_min=float(acceptance["pt_gt"]), abs_eta_max=float(acceptance["abs_eta_lt"]), **tagger)

    def flavour_class(self, flavour):
        """0 = b, 1 = c, 2 = any other label (light quarks, gluons, unmatched)."""
        label = np.abs(np.asarray(flavour))
        return np.where(label == self.b_label, 0, np.where(label == self.c_label, 1, 2))

    def efficiencies(self, flavour, pt):
        kind = self.flavour_class(flavour)

        def pick(functions):
            values = np.stack([function(pt) for function in functions])
            return np.take_along_axis(values, kind[None, :], axis=0)[0]

        return pick(self.b_tag), pick(self.c_tag)

    def in_acceptance(self, pt, eta):
        return (np.asarray(pt) > self.pt_min) & (np.abs(np.asarray(eta)) < self.abs_eta_max)

    def observed_states(self, btag):
        btag = np.asarray(btag)
        return ((btag & self.b_bit) != 0).astype(np.int64) + 2 * ((btag & self.c_bit) != 0)


def state_probabilities(eps_b, eps_c):
    """(jets, 4) probabilities of the states in STATES for independent b and c draws."""
    eps_b, eps_c = np.asarray(eps_b, dtype=float), np.asarray(eps_c, dtype=float)
    return np.stack([(1 - eps_b) * (1 - eps_c), eps_b * (1 - eps_c), eps_c * (1 - eps_b), eps_b * eps_c], axis=1)


@cache
def outcomes(n):
    """(outcomes, 3) positions (b1, b2, c1) among n jets in descending pT; b1 before b2."""
    rows = [(i, j, k) for i, j in combinations(range(n), 2) for k in range(n) if k not in (i, j)]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


@cache
def _outcome_lookup(n):
    lookup = np.full((n, n, n), -1, dtype=np.int64)
    table = outcomes(n)
    lookup[table[:, 0], table[:, 1], table[:, 2]] = np.arange(len(table))
    return lookup


def _outcome_groups(offsets, pt, eta, flavour, btag, model):
    """Events grouped by their number n of acceptance jets, with every passing outcome's probability.

    Yields (events, jets (events, n) flat indices in descending pT, outcomes(n),
    probability (events, outcomes), passes, direct_index (-1 where direct tagging failed)).
    Acceptance jets are ordered by descending pT with ties in original order, as the
    compactor orders roles.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    pt, eta = np.asarray(pt, dtype=float), np.asarray(eta, dtype=float)
    btag = np.asarray(btag)
    n_events = len(offsets) - 1
    event_of_jet = np.repeat(np.arange(n_events), np.diff(offsets))
    accepted = np.flatnonzero(model.in_acceptance(pt, eta))
    ordered = accepted[np.lexsort((-pt[accepted], event_of_jet[accepted]))]  # stable: ties keep ROOT order
    n_accepted = np.bincount(event_of_jet[ordered], minlength=n_events)
    starts = np.concatenate([[0], np.cumsum(n_accepted)[:-1]])
    eps_b, eps_c = model.efficiencies(flavour, pt)
    states = state_probabilities(eps_b, eps_c)
    p_b, p_c, p_other = states[:, 1], states[:, 2], states[:, 0] + states[:, 3]
    for n in np.unique(n_accepted):
        if n < 3:
            continue
        events = np.flatnonzero(n_accepted == n)
        jets = ordered[starts[events][:, None] + np.arange(n)]
        table = outcomes(int(n))
        others = np.prod(p_other[jets], axis=1, keepdims=True)
        chosen = p_other[jets][:, table].prod(axis=2)  # the three role jets' neutral probabilities
        probability = p_b[jets][:, table[:, 0]] * p_b[jets][:, table[:, 1]] * p_c[jets][:, table[:, 2]] * others / chosen
        words = btag[jets]
        is_b, is_c = words == model.b_bit, words == model.c_bit
        passes = (is_b.sum(axis=1) == 2) & (is_c.sum(axis=1) == 1)
        first = np.argmax(is_b, axis=1)
        second = np.argmax(is_b & (np.arange(n)[None, :] != first[:, None]), axis=1)
        direct_index = np.where(passes, _outcome_lookup(int(n))[first, second, np.argmax(is_c, axis=1)], -1)
        yield events, jets, table, probability, passes, direct_index


def enumerate_outcomes(offsets, pt, eta, flavour, btag, model):
    """Every passing tag outcome of every event, and its direct-tag outcome.

    Jet arrays are flat, in original order, with offsets of length events + 1.
    Returns per-entry arrays (event; b1, b2, c1 as flat jet indices; probability;
    direct) and per-event arrays (pass_probability; direct_pass).
    """
    n_events = len(offsets) - 1
    parts = {key: [] for key in ("event", "b1", "b2", "c1", "probability", "direct")}
    pass_probability = np.zeros(n_events)
    direct_pass = np.zeros(n_events, dtype=bool)
    for events, jets, table, probability, passes, direct_index in _outcome_groups(offsets, pt, eta, flavour, btag,
                                                                                   model):
        pass_probability[events] = probability.sum(axis=1)
        direct_pass[events] = passes
        parts["event"].append(np.repeat(events, len(table)))
        for position, role in enumerate(("b1", "b2", "c1")):
            parts[role].append(jets[:, table[:, position]].ravel())
        parts["probability"].append(probability.ravel())
        parts["direct"].append((np.arange(len(table))[None, :] == direct_index[:, None]).ravel())
    entries = {key: (np.concatenate(values) if values else np.zeros(0, dtype=float if key == "probability" else
                                                                     bool if key == "direct" else np.int64))
               for key, values in parts.items()}
    return entries, dict(pass_probability=pass_probability, direct_pass=direct_pass)


def event_uniforms(event_ids, seed):
    """A reproducible uniform number in [0, 1) per event, from sha256(seed, event ID)."""
    return np.array([int(hashlib.sha256(f"{seed}\0{event}".encode()).hexdigest()[:13], 16) / 16**13
                     for event in event_ids])


def draw_outcomes(offsets, pt, eta, flavour, btag, model, uniform):
    """One passing outcome per event, drawn in proportion to its probability.

    uniform holds one number in [0, 1) per event. Returns role flat jet indices
    (-1 when an event has fewer than three acceptance jets) and each event's pass
    probability, the weight that keeps the draw unbiased.
    """
    n_events = len(offsets) - 1
    uniform = np.asarray(uniform, dtype=float)
    if uniform.shape != (n_events,) or ((uniform < 0) | (uniform >= 1)).any():
        raise ValueError("Need one uniform number in [0, 1) per event")
    roles = {role: np.full(n_events, -1, dtype=np.int64) for role in ("b1", "b2", "c1")}
    pass_probability = np.zeros(n_events)
    for events, jets, table, probability, _, _ in _outcome_groups(offsets, pt, eta, flavour, btag, model):
        cumulative = np.cumsum(probability, axis=1)
        total = cumulative[:, -1]
        pick = np.minimum((cumulative <= (uniform[events] * total)[:, None]).sum(axis=1), len(table) - 1)
        rows = np.arange(len(events))
        for position, role in enumerate(("b1", "b2", "c1")):
            roles[role][events] = jets[rows, table[pick, position]]
        pass_probability[events] = total
    return roles, pass_probability


def sampling_chi2(expected, pass_probability, drawn_bin):
    """Drawn rows against all outcomes in fixed bins: chi2, degrees of freedom.

    expected[e, k] is the probability mass of event e's passing outcomes in bin k; a
    drawn row carries the event's pass probability in its bin. The covariance is the
    per-event multinomial of the draw: diag(sum_e P_e E_ek) - E^T E.
    """
    expected, weight = np.asarray(expected, dtype=float), np.asarray(pass_probability, dtype=float)
    bins = expected.shape[1]
    drawn = np.bincount(np.asarray(drawn_bin), weights=weight, minlength=bins)
    difference = drawn - expected.sum(axis=0)
    covariance = np.diag(weight @ expected) - expected.T @ expected
    rank = np.linalg.matrix_rank(covariance)
    return float(difference @ np.linalg.pinv(covariance) @ difference), int(rank)


def direct_roles(offsets, pt, eta, btag, model):
    """The compactor's direct selection: pass flag and role positions within each event's jet list (-1 if failed)."""
    offsets = np.asarray(offsets, dtype=np.int64)
    pt, btag = np.asarray(pt, dtype=float), np.asarray(btag)
    n_events = len(offsets) - 1
    event_of_jet = np.repeat(np.arange(n_events), np.diff(offsets))
    accepted = model.in_acceptance(pt, eta)
    roles = {}
    counts = {}
    for name, bit, size in (("b", model.b_bit, 2), ("c", model.c_bit, 1)):
        selected = np.flatnonzero(accepted & (btag == bit))
        selected = selected[np.lexsort((-pt[selected], event_of_jet[selected]))]
        counts[name] = np.bincount(event_of_jet[selected], minlength=n_events)
        starts = np.concatenate([[0], np.cumsum(counts[name])[:-1]])
        for position in range(size):
            index = np.full(n_events, -1, dtype=np.int64)
            has = counts[name] > position
            flat = selected[starts[has] + position]
            index[has] = flat - offsets[:-1][has]
            roles[f"{name}{position + 1}"] = index
    passes = (counts["b"] == 2) & (counts["c"] == 1)
    return passes, {role: np.where(passes, index, -1) for role, index in roles.items()}
