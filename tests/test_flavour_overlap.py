import numpy as np
import pytest
import yaml

from hepml.domain.flavour_overlap import matches_card, outgoing_counts, removed
from tests.truth_tag_world import STUDY

DESIGN = yaml.safe_load((STUDY / "validation.yaml").read_text())["flavour_overlap"]


def _counts(events):
    """events: lists of (pid, status) per event, incoming partons included."""
    offsets = np.r_[0, np.cumsum([len(e) for e in events])]
    pid = [p for e in events for p, _ in e]
    status = [s for e in events for _, s in e]
    return outgoing_counts(offsets, pid, status, DESIGN["outgoing_status"])


def test_rules_follow_the_cards():
    incoming = [(21, -21), (21, -21)]
    bbj = _counts([incoming + [(5, 23), (-5, 23), (21, 23)],            # core: kept
                   incoming + [(5, 23), (-5, 23), (1, 23), (4, 23)],    # extra charm: bbc has it
                   incoming + [(5, 23), (-5, 23), (21, 23), (211, 1)]])  # final-state hadrons ignored
    assert matches_card(bbj, DESIGN["cards"]["bkg_bbj"], DESIGN["outgoing_partons"]).all()
    assert removed(bbj, DESIGN["remove"]["bkg_bbj"]).tolist() == [False, True, False]
    cjj = _counts([incoming + [(4, 23), (1, 23), (2, 23)],
                   incoming + [(4, 23), (1, 23), (2, 23), (-4, 23)],    # c c~ pair: ccj has it
                   [(4, 21), (4, 21), (4, 23), (4, 23), (1, 23), (2, 23)]])  # same-sign: only cjj
    assert matches_card(cjj, DESIGN["cards"]["bkg_cjj"], DESIGN["outgoing_partons"]).all()
    assert removed(cjj, DESIGN["remove"]["bkg_cjj"]).tolist() == [False, True, False]


def test_cards_reject_other_content():
    incoming = [(21, -21), (21, -21)]
    events = _counts([incoming + [(5, 23), (-5, 23)],                    # too few partons
                      incoming + [(5, 23), (-5, 23), (21, 23), (11, 23)],  # a lepton
                      incoming + [(5, 23), (5, 23), (21, 23)]])           # b b, not b b~
    assert not matches_card(events, DESIGN["cards"]["bkg_bbj"], DESIGN["outgoing_partons"]).any()
    with pytest.raises(ValueError, match="Unknown"):
        matches_card(events, {"top": 1}, DESIGN["outgoing_partons"])
    with pytest.raises(ValueError, match="Unknown"):
        removed(events, "any_bottom")
