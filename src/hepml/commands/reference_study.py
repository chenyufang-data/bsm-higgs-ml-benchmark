"""Phase 2: fixed-policy, mass-specific BDTs and a bounded cut-based reference."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hepml_compact.contracts import part_glob_pattern
from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.configuration import default_study_directory, fit_parameters, load_analysis
from hepml.adapters.dataset_files import discover_samples, prepare_registered_reference
from hepml.adapters.evaluation_report import save_reference_figures
from hepml.adapters.inference import load_model
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    finish_report,
    hash_files,
    load_registry,
    open_report,
    refuse_existing,
    registry_files,
    run_study,
    write_json,
    write_tables,
    write_validation,
)
from hepml.adapters.xgboost_model import train_xgb
from hepml.application.baselines import cut_candidates, cut_reference
from hepml.application.evaluation import (
    check_yield_closure,
    point_rows,
    process_support,
    roc_tables,
    selected_support,
    validate_model,
)
from hepml.application.training import build_fit_weights, make_xyw
from hepml.domain.artifacts import BenchmarkFiles, model_dirname
from hepml.domain.metrics import OBJECTIVES, safe_auc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study')
    parser.add_argument('--anchor-dataset', required=True)
    parser.add_argument('--anchor-mass', type=int, default=200)
    parser.add_argument('--registry-dir', required=True)
    parser.add_argument('--compact-root', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--runs-dir', required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--training-override', action='append', default=[], metavar='KEY=VALUE',
                        help='Recorded change to one analysis.yaml training setting, e.g. max_depth=5; repeatable')
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def _run(args):
    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / 'analysis.yaml')
    cfg = analysis.evaluation
    settings = yaml.safe_load((study.directory / 'validation.yaml').read_text())['reference_study']
    masses, seeds = settings['masses'], settings['seeds']
    if (not masses or not seeds or len(set(masses)) != len(masses) or len(set(seeds)) != len(seeds)
            or settings['fit_policy'] not in {'none', 'balanced-mixture'} or settings['bootstrap_replicates'] < 2):
        raise ValueError('Invalid reference study settings')
    candidates = cut_candidates(settings['cut_reference'])
    out, datasets, runs = (Path(getattr(args, name)).resolve() for name in ('outdir', 'dataset_dir', 'runs_dir'))
    ids = ([f'{args.run_prefix}-m{m}-seed{s}' for m in masses for s in seeds]
           + [f'{args.run_prefix}-m{m}-cuts' for m in masses])
    refuse_existing(out, datasets, *(runs / name for name in ids))
    if not args.run_prefix or Path(args.run_prefix).name != args.run_prefix or any(x in args.run_prefix for x in '/\\:'):
        raise ValueError('run-prefix must be a directory name')
    anchor, registry_dir = Path(args.anchor_dataset).resolve(), Path(args.registry_dir).resolve()
    registry = load_registry(registry_dir)
    anchor_files = BenchmarkFiles(anchor)
    anchor_meta = json.loads(anchor_files.meta(args.anchor_mass).read_text())
    stored_lumi = float(anchor_meta['lumi'])
    directory = Path(args.compact_root).resolve() / settings['signal_directory']
    signals, _ = discover_samples(directory)
    records = []
    for mass in masses:
        meta = signals[str(mass)]
        if meta.get('rho_tc') != settings['rho_tc']:
            raise ValueError('Reference coupling differs from compact metadata')
        export_path = directory / meta['export_manifest']
        records.append(dict(directory=str(directory), meta=meta,
                            export=json.loads(export_path.read_text()), export_sha256=sha256_file(export_path)))
    inputs = hash_files([*registry_files(registry_dir), anchor_files.dataset(args.anchor_mass),
                         *sorted(anchor_files.splits.glob('*'))])
    for record in records:
        inputs[str(directory / record['meta']['export_manifest'])] = record['export_sha256']
        for path in directory.glob(part_glob_pattern(record['meta']['_stem'])):
            inputs[str(path)] = sha256_file(path)
    params = fit_parameters(analysis, args.training_override)
    settings_record = dict(study=str(study.directory), arguments=vars(args), analysis=asdict(analysis), study_settings=settings,
                           features=list(study.plugin.FEATURES), stored_lumi_pb_inv=stored_lumi,
                           cut_search_budget_per_mass=len(candidates), fit_parameters=params, test_evaluated=False)
    open_report(out, settings_record, inputs, records, preregister=True)
    rows, support_rows, seed_rows, run_records, checks, inventories, cut_rows, cut_grids = [], [], [], [], [], [], [], []
    features = list(study.plugin.FEATURES)
    background_assignments = None
    for mass in masses:
        full, splits = prepare_registered_reference(study, directory, signals[str(mass)], anchor, registry,
            datasets, anchor_mass=args.anchor_mass, lumi=stored_lumi)
        train, val = splits['train'], splits['val']
        background_projection = pd.concat([part.loc[part.target == 0, ['event_id', 'sample_weight']].assign(split=name)
                                          for name, part in splits.items()]).sort_values('event_id').reset_index(drop=True)
        if background_assignments is None:
            background_assignments = background_projection
        else:
            pd.testing.assert_frame_equal(background_assignments, background_projection)
        checks.append(dict(check=f'm{mass}: common_registry_and_background_weights', passed=True))
        for name, part in splits.items():
            for sample, group in part.groupby('sample'):
                inventories.append(dict(mass=mass, split=name, sample=sample, target=int(group.target.iloc[0]),
                                         mc_count=len(group), stored_weight_sum=group.sample_weight.sum()))
        points, grid, processes, normalization = cut_reference(full, val, mass, settings['cut_reference'], cfg,
                                                              stored_lumi=stored_lumi)
        cut_grids.append(grid.assign(mass=mass))
        cut_dir = runs / f'{args.run_prefix}-m{mass}-cuts'
        if cut_dir.exists():
            raise FileExistsError(f'Cut run already exists: {cut_dir}')
        cut_dir.mkdir(parents=True)
        write_json(cut_dir / 'selection.json', dict(mass=mass, settings=settings['cut_reference'],
                   evaluation_config=asdict(cfg), operating_points=points, test_evaluated=False))
        grid.to_csv(cut_dir / 'search.csv', index=False)
        processes.to_parquet(cut_dir / 'processes.parquet', index=False)
        for objective, point in points.items():
            item = dict.fromkeys(['S', 'B', 'S_over_B', *OBJECTIVES, 'background_neff',
                                  'signal_efficiency', 'background_efficiency', 'candidate_id',
                                  'relative_half_width', 'minimum_activity_over_mass'])
            item.update(point.get('metrics', {}))
            cut_rows.append(dict(item, mass=mass, objective=objective, status=point['status']))
            if point['status'] == 'valid':
                selected = processes[processes.candidate_id == point['candidate']['candidate_id']]
                table = process_support(val, selected, normalization, cfg, stored_lumi=stored_lumi)
                support_rows.append(table.assign(mass=mass, run_id=cut_dir.name, method='cuts', objective=objective))
        X_train, y_train, _ = make_xyw(train, features)
        X_val, y_val, _ = make_xyw(val, features)
        for seed in seeds:
            started = time.perf_counter()
            run_id = f'{args.run_prefix}-m{mass}-seed{seed}'
            model_dir = runs / run_id / 'models' / model_dirname(mass)
            model_dir.mkdir(parents=True)
            model = train_xgb(X_train, y_train, X_val, y_val, seed=seed, **params, record_training=True,
                fit_weights_train=build_fit_weights(train, settings['fit_policy'], reference=full),
                fit_weights_val=build_fit_weights(val, settings['fit_policy'], reference=full))
            predictions = {}
            columns = ['event_id', 'sample_key', 'sample', 'target', 'sample_weight']
            for name, frame, X in (('train', train, X_train), ('val', val, X_val)):
                predictions[name] = frame[columns].assign(bdt_score=model.predict_proba(X)[:, 1])
                predictions[name].to_parquet(model_dir / f'preds_{name}.parquet', index=False)
            record = validate_model(full, predictions, cfg, stored_lumi=stored_lumi,
                                    replicates=settings['bootstrap_replicates'], seed=settings['bootstrap_seed'])
            metadata = record.metadata
            check_yield_closure(full, val, record.weights, cfg, stored_lumi=stored_lumi)
            checks.append(dict(check=f'{run_id}: full_yield_and_k_factor_closure', passed=True))
            model.get_booster().save_model(model_dir / 'model.ubj')
            metrics = dict(mass=mass, rho_tc=settings['rho_tc'], features=features, evaluation=metadata,
                lumi_pb_inv=cfg.lumi_pb_inv, fit_policy=settings['fit_policy'], test_evaluated=False,
                xgb=dict(best_iteration=int(model.best_iteration), params=model.get_params(), weight_mode=settings['fit_policy']))
            write_json(model_dir / 'metrics.json', metrics)
            write_json(model_dir / 'operating_points.json', metadata)
            write_json(model_dir / 'learning_curves.json', model.evals_result())
            _, _, reload_predict = load_model(model_dir)
            np.testing.assert_allclose(reload_predict(X_val), predictions['val'].bdt_score, rtol=1e-6, atol=2e-7)
            checks.append(dict(check=f'{run_id}: reload_score_closure', passed=True))
            write_validation(model_dir, record)
            roc_tables(full, predictions, cfg, stored_lumi=stored_lumi).to_parquet(model_dir / 'roc.parquet', index=False)
            for objective, table in selected_support(record, val, cfg, stored_lumi=stored_lumi):
                support_rows.append(table.assign(mass=mass, run_id=run_id, method='bdt', objective=objective))
            rows.extend(point_rows(metadata, mass=mass, run_id=run_id, seed=seed, policy=settings['fit_policy']))
            seed_rows.append(dict(mass=mass, run_id=run_id, seed=seed, auc_weighted=metadata['auc_weighted'],
                auc_unweighted=metadata['auc_unweighted'], train_auc_unweighted=safe_auc(y_train,predictions['train'].bdt_score),
                best_iteration=int(model.best_iteration), seconds=time.perf_counter()-started))
            run_records.append(dict(mass=mass, seed=seed, run_id=run_id, model_directory=str(model_dir)))
            write_json(model_dir / 'provenance.json', dict(report=str(out), dataset=str(datasets), inputs=inputs))
            print(f'Completed m{mass} seed{seed}: weighted validation AUC={metadata["auc_weighted"]:.4f}; test sealed', flush=True)
    tables = dict(operating_points=pd.DataFrame(rows), seed_comparison=pd.DataFrame(seed_rows),
        process_support=pd.concat(support_rows, ignore_index=True) if support_rows else pd.DataFrame(columns=['run_id','objective']),
        split_inventory=pd.DataFrame(inventories), checks=pd.DataFrame(checks),
        cut_operating_points=pd.DataFrame(cut_rows), cut_search=pd.concat(cut_grids, ignore_index=True))
    write_tables(out, tables)
    summary = dict(schema_version=2, phases=[2], runs=run_records, evaluation_config=asdict(cfg), settings=settings,
                   fit_parameters=params, training_overrides=args.training_override,
                   dataset=str(datasets), registry=str(registry_dir), test_evaluated=False, status='checkpoint_ready',
                   note='Unweighted policy retained from the development pilot. These are fixed-rho references, not proof of rho independence. Phase 3 remains a separate study.')
    write_json(out / 'provenance/summary.json', summary)
    save_reference_figures(out)
    artifacts = [p for p in datasets.rglob('*') if p.is_file()]
    artifacts += [p for m in masses for p in (runs / f'{args.run_prefix}-m{m}-cuts').rglob('*') if p.is_file()]
    artifacts += [p for r in run_records for p in Path(r['model_directory']).glob('*') if p.is_file()]
    finish_report(out, '02_bdt_baselines.ipynb', inputs, artifacts)


if __name__ == '__main__':
    main()
