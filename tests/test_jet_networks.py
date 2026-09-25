import importlib.util

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hepml.adapters.jet_networks import (  # noqa: E402
    build_network,
    fit_network,
    jet_nodes,
    load_network,
    minkowski,
    predict,
    save_network,
)

CONFIGS = {"lorentznet": dict(hidden=16, blocks=3, c_weight=1e-3, dropout=0.0),
           "deep_sets": dict(hidden=16, layers=2, dropout=0.0),
           "particle_transformer": dict(embed_dims=[16, 32, 16], pair_embed_dims=[8, 8], num_heads=2, num_layers=2,
                                        num_cls_layers=1, trim=False)}
HAS_WEAVER = importlib.util.find_spec("weaver") is not None


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"{q}{role}": values for role in ("b1", "b2", "c1") for q, values in
                         (("pt", rng.uniform(25, 300, n)), ("eta", rng.uniform(-2.5, 2.5, n)),
                          ("phi", rng.uniform(-np.pi, np.pi, n)), ("mass", rng.uniform(2, 20, n)))})


def _boost_and_rotate(momenta, rapidity, angle):
    """A boost along x followed by a rotation about y: a general Lorentz transformation."""
    boost = np.eye(4)
    boost[0, 0] = boost[1, 1] = np.cosh(rapidity)
    boost[0, 1] = boost[1, 0] = np.sinh(rapidity)
    rotation = np.eye(4)
    rotation[1, 1] = rotation[3, 3] = np.cos(angle)
    rotation[1, 3], rotation[3, 1] = np.sin(angle), -np.sin(angle)
    return momenta @ (rotation @ boost).T


def test_nodes_carry_the_jet_masses_and_beams():
    frame = _frame(50)
    momenta, scalars = jet_nodes(frame, unit_gev=100.0, beams=True)
    assert momenta.shape == (50, 5, 4) and scalars.shape == (50, 5, 4)
    mass2 = minkowski(torch.as_tensor(momenta, dtype=torch.float64), torch.as_tensor(momenta, dtype=torch.float64))
    np.testing.assert_allclose(np.sqrt(np.clip(mass2[:, 0].numpy(), 0, None)) * 100, frame.massb1, rtol=1e-3, atol=0.05)
    np.testing.assert_array_equal(momenta[:, 3], [[1, 0, 0, 1]] * 50)
    assert (scalars.sum(axis=2) == 1).all() and (scalars[:, 3:, 3] == 1).all()


def test_lorentznet_is_lorentz_invariant_and_deep_sets_permutation_invariant():
    momenta, scalars = jet_nodes(_frame(40), unit_gev=100.0, beams=True)
    torch.manual_seed(0)
    network = build_network("lorentznet", CONFIGS["lorentznet"], 4, 5).double().eval()
    transformed = _boost_and_rotate(momenta.astype(np.float64), 0.7, 0.4)
    with torch.no_grad():
        before = network(torch.as_tensor(momenta, dtype=torch.float64), torch.as_tensor(scalars, dtype=torch.float64))
        after = network(torch.as_tensor(transformed), torch.as_tensor(scalars, dtype=torch.float64))
    np.testing.assert_allclose(after.numpy(), before.numpy(), rtol=1e-6, atol=1e-8)
    control = build_network("deep_sets", CONFIGS["deep_sets"], 4, 5).double().eval()
    order = [2, 0, 4, 1, 3]
    with torch.no_grad():
        first = control(torch.as_tensor(momenta, dtype=torch.float64), torch.as_tensor(scalars, dtype=torch.float64))
        permuted = control(torch.as_tensor(momenta[:, order], dtype=torch.float64),
                           torch.as_tensor(scalars[:, order], dtype=torch.float64))
    np.testing.assert_allclose(permuted.numpy(), first.numpy(), rtol=1e-10)


@pytest.mark.skipif(not HAS_WEAVER, reason="the upstream ParT (weaver-core) is not installed")
def test_particle_transformer_ignores_token_order_and_rotations_about_the_beam():
    momenta, scalars = jet_nodes(_frame(40), unit_gev=100.0, beams=False)
    torch.manual_seed(0)
    network = build_network("particle_transformer", CONFIGS["particle_transformer"], 4, 3).double().eval()
    angle = 0.9
    rotated = momenta.astype(np.float64).copy()
    rotated[..., 1] = np.cos(angle) * momenta[..., 1] - np.sin(angle) * momenta[..., 2]
    rotated[..., 2] = np.sin(angle) * momenta[..., 1] + np.cos(angle) * momenta[..., 2]
    order = [2, 0, 1]
    with torch.no_grad():
        first = network(torch.as_tensor(momenta, dtype=torch.float64), torch.as_tensor(scalars, dtype=torch.float64))
        turned = network(torch.as_tensor(rotated), torch.as_tensor(scalars, dtype=torch.float64))
        permuted = network(torch.as_tensor(momenta[:, order], dtype=torch.float64),
                           torch.as_tensor(scalars[:, order], dtype=torch.float64))
    assert first.shape == (40,)
    np.testing.assert_allclose(turned.numpy(), first.numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(permuted.numpy(), first.numpy(), rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="jet tokens only"):
        network(*(torch.as_tensor(a, dtype=torch.float64) for a in jet_nodes(_frame(4), unit_gev=100.0, beams=True)))


@pytest.mark.parametrize("kind", ["lorentznet", "deep_sets",
                                  pytest.param("particle_transformer", marks=pytest.mark.skipif(
                                      not HAS_WEAVER, reason="the upstream ParT (weaver-core) is not installed"))])
def test_fit_keeps_the_best_epoch_and_reloads_exactly(tmp_path, kind):
    rng = np.random.default_rng(1)
    frames = [_frame(300, seed) for seed in (2, 3)]
    labels = [rng.integers(0, 2, 300) for _ in frames]
    for frame, label in zip(frames, labels):  # a learnable difference: signal jets are harder
        frame.loc[label == 1, "ptb1"] *= 2.0
    beams = kind != "particle_transformer"  # as registered: ParT takes the jets alone
    arrays = [(*jet_nodes(frame, unit_gev=100.0, beams=beams), label, rng.uniform(0.1, 1.0, 300))
              for frame, label in zip(frames, labels)]
    training = dict(learning_rate=1e-2, weight_decay=0.0, batch_size=64, max_epochs=4, patience=2)
    model, record = fit_network(kind, CONFIGS[kind], training, arrays[0], arrays[1], seed=7, device="cpu")
    aucs = [row["val_fit_weighted_auc"] for row in record["history"]]
    assert record["best_val_fit_weighted_auc"] == max(aucs) and record["history"][record["best_epoch"]]["val_fit_weighted_auc"] == max(aucs)
    scores = predict(model, arrays[1][0], arrays[1][1], device="cpu")
    save_network(tmp_path / kind, model, record)
    reloaded, saved = load_network(tmp_path / kind, device="cpu")
    np.testing.assert_array_equal(predict(reloaded, arrays[1][0], arrays[1][1], device="cpu"), scores)
    assert saved["best_epoch"] == record["best_epoch"]


def test_stage_c_reports_unformable_categories_instead_of_failing():
    from hepml.commands.jet_network_study import stage_c_z

    frame = _frame(6).assign(target=[1, 0, 0, 0, 0, 0], sample=["sig", "bkg", "bkg", "bkg", "bkg", "bkg"])
    weights = np.array([1.0, 1.0, 50.0, 1.0, 1.0, 46.0])  # one event holds half the background
    stage_c = {"category_background_efficiencies": [0.4, 0.2]}
    settings = {"binning": {"min_background_neff": 1}, "normalization_uncertainty": 0.05,
                "primary_shape_uncertainty": 0.02}
    z, status = stage_c_z(frame, np.linspace(1, 0, 6), weights, 200, settings, stage_c, np.array([0, 1, np.inf]))
    assert np.isnan(z) and status.startswith("categories not formed")
