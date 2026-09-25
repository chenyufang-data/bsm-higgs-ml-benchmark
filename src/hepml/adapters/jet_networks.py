"""Jet-level networks for Phase 10 (optional dependency: PyTorch).

LorentzNet (Gong et al., JHEP 07 (2022) 030) and a Deep Sets control over role-jet nodes
plus two beam spurions, and the upstream Particle Transformer (Qu, Li and Qian, ICML 2022;
weaver-core) over the same jets as tokens. Only the network study imports this module;
extraction and the BDT workflow never import torch.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from hepml.domain.physics import four_vector

ROLES = ("b1", "b2", "c1")
KINEMATICS = ("pt", "eta", "phi", "mass")
ROLE_COLUMNS = [f"{quantity}{role}" for role in ROLES for quantity in KINEMATICS]


def jet_nodes(frame, *, unit_gev, beams):
    """Node four-momenta (E, px, py, pz) / unit_gev with shape [events, nodes, 4], and one-hot
    node scalars [events, nodes, roles + 1]; the extra class marks the beam spurions (1, 0, 0, +-1)."""
    momenta = []
    for role in ROLES:
        px, py, pz, e = four_vector(*(frame[f"{q}{role}"].to_numpy(np.float64) for q in KINEMATICS))
        momenta.append(np.stack([e, px, py, pz], axis=1) / unit_gev)
    kinds = list(range(len(ROLES)))
    if beams:
        for sign in (1.0, -1.0):
            momenta.append(np.tile([1.0, 0.0, 0.0, sign], (len(frame), 1)))
            kinds.append(len(ROLES))
    scalars = np.zeros((len(frame), len(kinds), len(ROLES) + 1), np.float32)
    scalars[:, np.arange(len(kinds)), kinds] = 1.0
    return np.stack(momenta, axis=1).astype(np.float32), scalars


def minkowski(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def psi(value):
    return torch.sign(value) * torch.log(torch.abs(value) + 1)


def _neighbours(n_nodes):
    return torch.tensor([[j for j in range(n_nodes) if j != i] for i in range(n_nodes)])


class LGEB(nn.Module):
    """Lorentz group equivariant block on a fully connected graph without self-edges."""

    def __init__(self, hidden, n_scalar, c_weight, last):
        super().__init__()
        self.c_weight, self.last = c_weight, last
        self.phi_e = nn.Sequential(nn.Linear(2 * hidden + 2, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.phi_m = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.phi_h = nn.Sequential(nn.Linear(2 * hidden + n_scalar, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        if not last:
            output = nn.Linear(hidden, 1, bias=False)
            nn.init.xavier_uniform_(output.weight, gain=0.001)
            self.phi_x = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), output)

    def forward(self, h, x, scalars, neighbours):
        events, nodes, hidden = h.shape
        degree = neighbours.shape[1]
        h_i, h_j = h[:, :, None, :].expand(-1, -1, degree, -1), h[:, neighbours]
        x_i, x_j = x[:, :, None, :].expand(-1, -1, degree, -1), x[:, neighbours]
        difference = x_i - x_j
        invariants = torch.stack([psi(minkowski(difference, difference)), psi(minkowski(x_i, x_j))], dim=-1)
        m = self.phi_e(torch.cat([h_i, h_j, invariants], dim=-1).reshape(-1, 2 * hidden + 2))
        m = (m * self.phi_m(m)).reshape(events, nodes, degree, hidden)
        if not self.last:
            shift = torch.clamp(difference * self.phi_x(m), min=-100, max=100)
            x = x + self.c_weight * shift.mean(dim=2)
        update = self.phi_h(torch.cat([h, m.sum(dim=2), scalars], dim=-1).reshape(-1, 2 * hidden + scalars.shape[-1]))
        return h + update.reshape(events, nodes, hidden), x


class LorentzNet(nn.Module):
    def __init__(self, *, n_scalar, n_nodes, hidden, blocks, c_weight, dropout):
        super().__init__()
        self.embedding = nn.Linear(n_scalar, hidden)
        self.blocks = nn.ModuleList(LGEB(hidden, n_scalar, c_weight, last=index == blocks - 1)
                                    for index in range(blocks))
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("neighbours", _neighbours(n_nodes), persistent=False)

    def forward(self, momenta, scalars):
        h, x = self.embedding(scalars), momenta
        for block in self.blocks:
            h, x = block(h, x, scalars, self.neighbours)
        return self.decoder(h.mean(dim=1)).squeeze(-1)


class DeepSets(nn.Module):
    """Control: a per-node network on role and four-momentum, sum pooling, no pairwise terms."""

    def __init__(self, *, n_scalar, n_nodes, hidden, layers, dropout):
        super().__init__()
        encoder = [nn.Linear(n_scalar + 4, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            encoder += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, momenta, scalars):
        return self.decoder(self.encoder(torch.cat([scalars, momenta], dim=-1)).sum(dim=1)).squeeze(-1)


class ParticleTransformerNet(nn.Module):
    """The upstream ParT on jet tokens: per-token log pT, eta, log E and role, standardized by
    ParT's input batch normalization, and ParT's own pairwise features from the four-vectors."""

    def __init__(self, *, n_scalar, n_nodes, **config):
        super().__init__()
        from weaver.nn.model.ParticleTransformer import ParticleTransformer

        self.part = ParticleTransformer(input_dim=3 + n_scalar, num_classes=1, **config)

    def forward(self, momenta, scalars):
        if bool((scalars[..., len(ROLES):] != 0).any()):
            raise ValueError("ParT takes jet tokens only: beam spurions have no transverse momentum or rapidity")
        energy, px, py, pz = momenta.unbind(-1)
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        tokens = torch.stack([torch.log(pt), torch.asinh(pz / pt), torch.log(energy.clamp(min=1e-8))], dim=-1)
        features = torch.cat([tokens, scalars], dim=-1).transpose(1, 2)  # (events, features, tokens)
        vectors = torch.stack([px, py, pz, energy], dim=1)  # ParT's (px, py, pz, E) order
        mask = torch.ones_like(features[:, :1], dtype=torch.bool)
        return self.part(features, v=vectors, mask=mask).squeeze(-1)


MODELS = {
    "lorentznet": lambda c, s, n: LorentzNet(n_scalar=s, n_nodes=n, hidden=c["hidden"], blocks=c["blocks"],
                                             c_weight=c["c_weight"], dropout=c["dropout"]),
    "deep_sets": lambda c, s, n: DeepSets(n_scalar=s, n_nodes=n, hidden=c["hidden"], layers=c["layers"],
                                          dropout=c["dropout"]),
    "particle_transformer": lambda c, s, n: ParticleTransformerNet(n_scalar=s, n_nodes=n, **c),
}


def build_network(kind, config, n_scalar, n_nodes):
    if kind not in MODELS:
        raise ValueError(f"Unknown network {kind}")
    return MODELS[kind](config, n_scalar, n_nodes)


def predict(model, momenta, scalars, *, device, batch=16384):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(momenta), batch):
            logits = model(torch.as_tensor(momenta[start:start + batch], device=device),
                           torch.as_tensor(scalars[start:start + batch], device=device))
            scores.append(torch.sigmoid(logits.double()).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def fit_network(kind, config, training, train, val, *, seed, device):
    """Weighted cross-entropy fit; the checkpoint is the epoch with the highest fit-weighted
    validation AUC. train/val: (momenta, scalars, labels, fit_weights). Returns the model at
    its checkpoint and a record with the per-epoch history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    momenta, scalars, labels, weights = (torch.as_tensor(np.asarray(a, dtype=np.float32), device=device) for a in train)
    weights = weights / weights.mean()
    model = build_network(kind, config, scalars.shape[-1], scalars.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["max_epochs"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    best, best_auc, best_epoch, history, started = None, -np.inf, -1, [], time.perf_counter()
    for epoch in range(training["max_epochs"]):
        model.train()
        order = torch.randperm(len(labels), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), training["batch_size"]):
            index = order[start:start + training["batch_size"]]
            if len(index) < 2:  # batch normalization needs two events
                continue
            loss = (weights[index] * loss_function(model(momenta[index], scalars[index]), labels[index])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        schedule.step()
        scores = predict(model, val[0], val[1], device=device)
        auc = float(roc_auc_score(val[2], scores, sample_weight=val[3]))
        history.append(dict(epoch=epoch, train_loss=total / len(labels), val_fit_weighted_auc=auc,
                            seconds=time.perf_counter() - started))
        if auc > best_auc:
            best, best_auc, best_epoch = copy.deepcopy(model.state_dict()), auc, epoch
        elif epoch - best_epoch >= training["patience"]:
            break
    model.load_state_dict(best)
    record = dict(kind=kind, config=config, training=training, seed=seed, best_epoch=best_epoch,
                  best_val_fit_weighted_auc=best_auc, epochs_run=len(history), seconds=time.perf_counter() - started,
                  peak_gpu_mb=(torch.cuda.max_memory_allocated(device) / 2**20 if device != "cpu" else None),
                  parameters=sum(p.numel() for p in model.parameters()), history=history,
                  n_scalar=int(scalars.shape[-1]), n_nodes=int(scalars.shape[1]))
    return model, record


def save_network(directory, model, record):
    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "network.json").write_text(json.dumps(record, indent=2) + "\n")


def load_network(directory, *, device):
    record = json.loads((Path(directory) / "network.json").read_text())
    model = build_network(record["kind"], record["config"], record["n_scalar"], record["n_nodes"])
    model.load_state_dict(torch.load(Path(directory) / "model.pt", map_location=device, weights_only=True))
    return model.to(device), record
