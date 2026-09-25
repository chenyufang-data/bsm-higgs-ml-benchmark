"""Reference selections use physical events and exactly the declared search budget."""

import numpy as np
import pandas as pd
import pytest

from hepml.application.baselines import apply_cut, cut_candidates, cut_reference
from hepml.application.datasets import registered_splits
from hepml.domain.config import EvaluationConfig
from hepml.domain.metrics import OBJECTIVES, asimov_z


def test_cut_reference_matches_direct_counts_and_records_unsupported_selection():
    settings = dict(mass_features=['m1', 'm2'], activity_feature='ht',
                    relative_half_widths=[.1,.3], minimum_activity_over_mass=[0.,1.])
    frame = pd.DataFrame(dict(event_id=list('abcdef'), sample=['s','s','a','a','b','b'],
        sample_key=['s','s','a','a','b','b'], target=[1,1,0,0,0,0], sample_weight=[2.,2.,1.,1.,10.,10.],
        m1=[200,260,190,150,240,200], m2=[150,250,150,290,300,280], ht=[250,250,220,190,190,190]))
    cfg = EvaluationConfig(lumi_pb_inv=100, signal_k_factor=1.95, background_k_factor=1.26, min_background_neff=1)
    points, grid, processes, _ = cut_reference(frame,frame,200,settings,cfg,stored_lumi=10)
    assert len(grid) == 5 and len(processes) == 15
    for candidate in cut_candidates(settings):
        keep = apply_cut(frame,200,settings,candidate)
        row = grid.loc[grid.candidate_id == candidate['candidate_id']].iloc[0]
        S = frame.loc[keep & (frame.target == 1), 'sample_weight'].sum()*10*1.95
        B = frame.loc[keep & (frame.target == 0), 'sample_weight'].sum()*10*1.26
        assert row.S == pytest.approx(S)
        assert row.B == pytest.approx(B)
        assert row.Z_Asimov == pytest.approx(asimov_z(S,B))
    for objective in OBJECTIVES:
        point = points[objective]
        assert point['metrics'][objective] == grid.loc[grid.eligible,objective].max()
        assert point['selection_source'] == 'validation'
    impossible = EvaluationConfig(min_background_neff=10000)
    points,_,_,_ = cut_reference(frame,frame,200,settings,impossible,stored_lumi=10)
    assert all(p['status'] == 'no_valid_selection' and p['candidate'] is None for p in points.values())
    tied = frame.assign(m1=200, m2=200, ht=300)
    points,_,_,_ = cut_reference(tied,tied,200,settings,cfg,stored_lumi=10)
    assert all(p['candidate']['candidate_id'] == 0 for p in points.values())


def test_registry_projection_is_order_independent_and_rejects_missing_events():
    frame = pd.DataFrame(dict(event_id=list('abcdef'), sample_key=['s']*3+['b']*3, sample_weight=np.arange(6.)))
    registry = frame[['event_id','sample_key']].assign(split=['train','val','test']*2)
    splits = registered_splits(frame.sample(frac=1,random_state=3),registry)
    for name, part in splits.items():
        assert set(part.event_id) == set(registry.loc[registry.split==name,'event_id'])
        assert part.set_index('event_id').sample_weight.to_dict() == frame.set_index('event_id').loc[part.event_id,'sample_weight'].to_dict()
    with pytest.raises(ValueError,match='membership'):
        registered_splits(frame.iloc[:-1],registry)
    with pytest.raises(ValueError,match='Duplicate'):
        registered_splits(pd.concat([frame,frame]),registry)
