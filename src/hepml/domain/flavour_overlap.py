"""Heavy-flavour overlap between MLM samples, decided by their outgoing hard-process partons.

Samples whose extra jets may contain a heavy quark can generate the same final state as a
sample with that heavy quark in its core process. The overlapping events are removed from
the extra-jet sample; the rules and each sample's card are recorded in validation.yaml:
flavour_overlap.
"""

from __future__ import annotations

import numpy as np

LIGHT = (1, 2, 3, 21)
CLASSES = ("b", "bbar", "c", "cbar", "charm", "b_flavour", "light", "other", "total")
REMOVALS = {
    "any_charm": lambda n: n["charm"] > 0,
    "charm_pair": lambda n: (n["c"] > 0) & (n["cbar"] > 0),
}


def outgoing_counts(offsets, pid, status, outgoing_status):
    """Per event, counts of each flavour class among partons with |status| == outgoing_status."""
    offsets, pid, status = (np.asarray(a) for a in (offsets, pid, status))
    n_events = len(offsets) - 1
    event = np.repeat(np.arange(n_events), np.diff(offsets))
    out = np.abs(status) == outgoing_status

    def count(mask):
        return np.bincount(event[out & mask], minlength=n_events)

    b, bbar, c, cbar = (count(pid == code) for code in (5, -5, 4, -4))
    light, total = count(np.isin(np.abs(pid), LIGHT)), count(np.ones_like(out))
    return dict(b=b, bbar=bbar, c=c, cbar=cbar, charm=c + cbar, b_flavour=b + bbar, light=light,
                other=total - (b + bbar + c + cbar + light), total=total)


def matches_card(counts, card, outgoing_range):
    """Per event: the outgoing partons are quarks or gluons, their number is in range, and each
    carded class count is exact (integer) or within [min, max] (max None: unbounded)."""
    low, high = outgoing_range
    matched = (counts["other"] == 0) & (counts["total"] >= low) & (counts["total"] <= high)
    for name, rule in card.items():
        if name not in CLASSES:
            raise ValueError(f"Unknown flavour class in card: {name}")
        values = counts[name]
        if isinstance(rule, int):
            matched &= values == rule
        else:
            minimum, maximum = rule
            matched &= values >= minimum
            if maximum is not None:
                matched &= values <= maximum
    return matched


def removed(counts, rule):
    if rule not in REMOVALS:
        raise ValueError(f"Unknown overlap rule: {rule}")
    return REMOVALS[rule](counts)
