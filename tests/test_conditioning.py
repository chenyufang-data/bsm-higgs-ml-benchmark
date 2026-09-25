"""Physical identity, hypothesis priors and physics-weight isolation."""

import numpy as np
import pandas as pd
import pytest

from hepml.application.conditioning import assert_split_isolation, condition_frame, expand_hypotheses
from hepml.application.training import make_xyw
from hepml.domain.metrics import safe_auc


@pytest.fixture
def inputs():
    def frame(prefix, size, target):
        return pd.DataFrame(dict(event_id=[f'{prefix}{i}' for i in range(size)], sample_key=prefix,
                                 sample=prefix, target=target, sample_weight=3. if target else 17.,
                                 x=np.arange(size, dtype=float)))
    bg = frame('bg', 6, 0)
    frames = {m: pd.concat([frame(f's{m}', n, 1), bg], ignore_index=True) for m, n in [(200, 4), (400, 10)]}
    conditioning = dict(feature='hypothesis_mass', support=[200, 400], prior={'200': .5, '400': .5})
    return frames, conditioning


def expand(frames, conditioning):
    return expand_hypotheses(frames, physics_features=['x'], conditioning=conditioning,
                             reference_mass=200, split='train')[0]


def test_priors_replica_conservation_and_unchanged_physical_weights(inputs):
    frames, conditioning = inputs
    pooled = expand(frames, conditioning)
    assert len(pooled) == 26 and pooled.event_id.nunique() == 20
    assert not pooled.duplicated(['event_id', 'hypothesis_id']).any()
    np.testing.assert_allclose(pooled.groupby(['target', 'hypothesis_mass']).fit_weight.sum(), [3, 3, 2, 2])
    bg = pooled[pooled.target == 0]
    np.testing.assert_allclose(bg.groupby('event_id').fit_weight.sum(), 1.)
    assert pooled.fit_weight.sum() == pytest.approx(len(frames[200]))
    np.testing.assert_array_equal(pooled.sample_weight, np.where(pooled.target == 1, 3., 17.))
    assert bg.generated_mass.isna().all()
    assert safe_auc(pooled.target, pooled.hypothesis_mass, pooled.fit_weight) == pytest.approx(.5)
    assert safe_auc(pooled.target, pooled.hypothesis_mass) != pytest.approx(.5)


def test_changed_background_and_reused_identity_are_rejected(inputs):
    frames, conditioning = inputs
    frames[400].loc[frames[400].target == 0, 'x'] += 1
    with pytest.raises(AssertionError):
        expand(frames, conditioning)
    frames[400].loc[frames[400].target == 0, 'x'] -= 1
    frames[400].loc[0, 'event_id'] = frames[200].event_id.iloc[0]
    with pytest.raises(ValueError, match='identity reused'):
        expand(frames, conditioning)
    frames[400].loc[0, 'event_id'] = frames[400].event_id.iloc[1]
    with pytest.raises(ValueError, match='unique physical'):
        expand(frames, conditioning)


def test_explicit_condition_and_metadata_guards(inputs):
    frames, conditioning = inputs
    conditioned = condition_frame(frames[200], 400, conditioning)
    assert (conditioned.hypothesis_mass == 400).all()
    assert 'hypothesis_mass' not in frames[200]
    for mass in [None, 300, np.nan, np.inf]:
        with pytest.raises(ValueError, match='supported'):
            condition_frame(frames[200], mass, conditioning)
    with pytest.raises(ValueError, match='conflicts'):
        condition_frame(conditioned, 200, conditioning)
    pooled = expand(frames, conditioning)
    for feature in ['generated_mass', 'hypothesis_id', 'split', 'fit_weight']:
        with pytest.raises(ValueError):
            make_xyw(pooled, ['x', feature])


def test_split_isolation_uses_physical_ids(inputs):
    frames, conditioning = inputs
    pooled = expand(frames, conditioning)
    other = pooled.assign(event_id='val_' + pooled.event_id)
    assert_split_isolation(pooled, other, {'test_event'})
    with pytest.raises(ValueError, match='boundaries'):
        assert_split_isolation(pooled, other, {pooled.event_id.iloc[0]})
    with pytest.raises(ValueError, match='boundaries'):
        assert_split_isolation(pooled, pooled.assign(hypothesis_id='different'), set())
