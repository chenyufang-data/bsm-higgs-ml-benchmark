"""Bounded cut references and diagnostics using the shared evaluation contract."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from hepml.application.evaluation import evaluation_weights
from hepml.domain.metrics import OBJECTIVES, operating_curve


def cut_candidates(settings):
    widths, activity = settings['relative_half_widths'], settings['minimum_activity_over_mass']
    if (not widths or not activity or not settings['mass_features']
            or len(set(widths)) != len(widths) or len(set(activity)) != len(activity)
            or any(not np.isfinite(x) or x <= 0 for x in widths)
            or any(not np.isfinite(x) or x < 0 for x in activity)):
        raise ValueError('Invalid preregistered cut grid')
    candidates = [dict(candidate_id=0, no_cut=True, relative_half_width=None, minimum_activity_over_mass=None)]
    for width, minimum in product(widths, activity):
        candidates.append(dict(candidate_id=len(candidates), no_cut=False,
                               relative_half_width=width, minimum_activity_over_mass=minimum))
    return candidates


def apply_cut(frame, mass, settings, candidate):
    """A fixed union of mass windows followed by one activity threshold."""
    if candidate['no_cut']:
        return np.ones(len(frame), dtype=bool)
    values = frame[settings['mass_features']].to_numpy(dtype=float)
    activity = frame[settings['activity_feature']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(activity).all() or mass <= 0:
        raise ValueError('Invalid cut-reference inputs')
    window = np.any(np.abs(values - mass) <= mass * candidate['relative_half_width'], axis=1)
    return window & (activity >= mass * candidate['minimum_activity_over_mass'])


def cut_reference(full, validation, mass, settings, config, *, stored_lumi):
    """Evaluate exactly the declared recipes; independently select each objective on validation."""
    weights, normalization = evaluation_weights(full, validation, stored_lumi=stored_lumi,
        target_lumi=config.lumi_pb_inv, signal_k_factor=config.signal_k_factor,
        background_k_factor=config.background_k_factor)
    rows, process_rows = [], []
    for candidate in cut_candidates(settings):
        prediction = validation.assign(bdt_score=apply_cut(validation, mass, settings, candidate).astype(float))
        curve, processes = operating_curve(prediction, weights, config, thresholds=[.5])
        # The threshold .5 encodes a Boolean selection, not a BDT operating point.
        rows.append(dict(curve.iloc[0].drop('threshold'), **candidate))
        process_rows.append(processes.drop(columns='threshold').assign(candidate_id=candidate['candidate_id']))
    grid = pd.DataFrame(rows)
    points = {}
    for objective in OBJECTIVES:
        supported = grid[grid.eligible & (grid[objective] > 0) & np.isfinite(grid[objective])]
        if supported.empty:
            points[objective] = dict(status='no_valid_selection', objective=objective, candidate=None)
        else:
            chosen = supported.sort_values([objective, 'candidate_id'], ascending=[False, True]).iloc[0]
            candidate = cut_candidates(settings)[int(chosen.candidate_id)]
            points[objective] = dict(status='valid', objective=objective, candidate=candidate,
                                    metrics=chosen.to_dict(), selection_source='validation', tie_rule='candidate_order')
    return points, grid, pd.concat(process_rows, ignore_index=True), normalization
