"""Two synthetic masses share backgrounds, publish a report, and keep test scores sealed."""

import importlib.util
import json
import shutil
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml
from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.dataset_files import discover_samples, load_sample_frame
from hepml.adapters.study_loader import load_study
from hepml.cli import main
from hepml.commands.shape_audit import load_anchor
from hepml.domain.splits import extend_registry

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_reference_study_two_mass_report(tmp_path, root_factory, monkeypatch, jets_recipe):
    monkeypatch.setenv('MPLBACKEND','Agg')
    study = tmp_path/'study'
    shutil.copytree(REPO/'studies/cg_bbc',study)
    cfg = yaml.safe_load((study/'analysis.yaml').read_text())
    cfg['training'].update(n_estimators=12,early_stopping_rounds=3)
    cfg['evaluation']['min_background_neff']=10
    (study/'analysis.yaml').write_text(yaml.safe_dump(cfg))
    settings = yaml.safe_load((study/'validation.yaml').read_text())
    settings['reference_study'].update(seeds=[42,7],bootstrap_replicates=4)
    settings['parameterized_study'].update(pilot_estimators=5, probe_estimators=5,
                                           learning_curve_step=5, bootstrap_replicates=4)
    settings['comparison_study'].update(bootstrap_replicates=20, seeds=[42, 7])
    settings['grid_study'].update(masses=[200, 400], couplings=[0.1, 0.4], training_coupling=0.4, seeds=[42, 7],
                                  coupling_directories={'0.1': 'rho01', '0.4': 'rho04'}, bootstrap_replicates=4)
    # Synthetic ROOT files carry no parton truth, so the binned fit uses reconstructed jets here.
    settings['binned_fit_study'].update(jets='reconstructed')
    # 160 synthetic background events with coarse 12-tree scores: the top score can be shared by half the
    # background, so one wide boundary keeps the test stable. Production uses the registered boundaries.
    settings['binned_fit_stage_b'].update(category_background_efficiencies=[0.9])
    # Stage C's sets all take stage B's boundary here, so reproducing stage B is exact by construction.
    settings['binned_fit_stage_c'].update(category_background_efficiencies=[0.9],
                                          descriptive_category_sets={'four_categories': [0.9], 'stage_b': [0.9]})
    # A tiny network pilot on the synthetic overlap-removed grid, on the CPU.
    settings['jet_network_pilot'].update(
        grid_report='tt_hf_grid', seeds=[42, 7],
        models={'lorentznet': dict(hidden=8, blocks=2, c_weight=1e-3, dropout=0.0),
                'deep_sets': dict(hidden=8, layers=2, dropout=0.0)},
        training=dict(optimizer='adamw', learning_rate=1e-3, weight_decay=0.0, batch_size=256, max_epochs=2,
                      schedule='cosine', patience=1))
    settings['jet_network_pilot']['primary'].update(bootstrap_replicates=20)
    settings['particle_transformer_pilot'].update(
        grid_report='tt_hf_grid', lorentznet_report='network_pilot', seeds=[42, 7],
        models={'particle_transformer': dict(embed_dims=[16, 32, 16], pair_embed_dims=[8, 8], num_heads=2,
                                             num_layers=2, num_cls_layers=1, trim=False)},
        training=settings['jet_network_pilot']['training'])
    settings['particle_transformer_pilot']['primary'].update(bootstrap_replicates=20)
    (study/'validation.yaml').write_text(yaml.safe_dump(settings))
    compact=tmp_path/'compact/rho01'
    for mass,seed in [(200,42),(400,43),(None,44)]:
        # Unequal signal sizes give the loss-weight control non-trivial anchored weights.
        source=root_factory(seed=seed,signal=mass is not None,entries=1600 if mass==400 else 2000)
        args=['compact','--extraction',str(jets_recipe),'--input',str(source),'--outdir',str(compact),
              '--kind','signal' if mass else 'background','--sample',f'sig{mass}' if mass else 'bkg',
              # A dominant background (S/B ~ 1e-3), the regime of the real grid and of the binned fit's closure.
              '--xs-pb','2' if mass else '20000']
        if mass:
            args+=['--mass',str(mass),'--rho-tc','0.1']
        main(args)
    anchor=tmp_path/'anchor'
    main(['prepare','--study',str(study),'--indir',str(compact),'--outdir',str(anchor),'--mass','200','--write-splits'])
    registry=load_anchor(anchor,200)
    signals,_=discover_samples(compact)
    plugin=load_study(study)
    frame=load_sample_frame(compact,signals['400'],'sig400',list(plugin.plugin.FEATURES),3000,plugin)
    registry=extend_registry(registry,frame[['event_id','sample_key']])
    registry_dir=tmp_path/'registry'
    registry_dir.mkdir()
    registry.to_parquet(registry_dir/'assignments.parquet',index=False)
    (registry_dir/'registry.json').write_text(json.dumps({'assignments_sha256':sha256_file(registry_dir/'assignments.parquet')}))
    report,datasets,runs=tmp_path/'report',tmp_path/'datasets',tmp_path/'runs'
    argv=['reference-study','--study',str(study),'--anchor-dataset',str(anchor),'--registry-dir',str(registry_dir),
          '--compact-root',str(compact.parent),'--dataset-dir',str(datasets),'--runs-dir',str(runs),
          '--run-prefix','refs','--outdir',str(report)]
    main(argv)
    status=json.loads((report/'provenance/status.json').read_text())
    assert status['status']=='complete'
    assert all(sha256_file(Path(p))==h for p,h in status['artifacts'].items())
    assert not list(runs.rglob('*preds_test*'))
    points=pd.read_csv(report/'tables/operating_points.csv')
    assert set(points.mass)=={200,400} and len(points)==12
    assert len(pd.read_csv(report/'tables/cut_search.csv'))==26
    first=pd.read_parquet(datasets/'splits/assignments_sig200.parquet')
    second=pd.read_parquet(datasets/'splits/assignments_sig400.parquet')
    common=first.merge(second,on='event_id',suffixes=('_200','_400'))
    assert len(common)>0 and (common.split_200==common.split_400).all()
    notebook=nbformat.read(report/'report.ipynb',as_version=4)
    pngs=[o for c in notebook.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    assert len(pngs)==6
    with pytest.raises(FileExistsError):
        main(argv)

    conditional_report, shared = tmp_path/'conditional_report', tmp_path/'shared_model'
    conditional_args = ['parameterized-study', '--study', str(study), '--reference-report', str(report),
                        '--outdir', str(conditional_report), '--run-dir', str(shared)]
    main(conditional_args)
    status = json.loads((conditional_report/'provenance/status.json').read_text())
    assert status['status'] == 'complete'
    assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())
    checks = pd.read_csv(conditional_report/'tables/checks.csv')
    assert len(checks) == 7 and checks.passed.all()
    assert len(list(shared.rglob('model.ubj'))) == 1
    assert not list(shared.rglob('*preds_test*'))
    notebook = nbformat.read(conditional_report/'report.ipynb', as_version=4)
    pngs = [o for c in notebook.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert len(pngs) == 6
    np.testing.assert_allclose(pd.read_csv(conditional_report/'tables/mass_only_probe.csv').weighted_auc, .5)
    with pytest.raises(FileExistsError):
        main(conditional_args)
    seed7_report = tmp_path/'conditional_report_seed7'
    main(['parameterized-study', '--study', str(study), '--reference-report', str(report),
          '--outdir', str(seed7_report), '--run-dir', str(tmp_path/'shared_model_seed7'), '--seed', '7'])
    assert json.loads((seed7_report/'provenance/summary.json').read_text())['settings']['seed'] == 7
    # A conditional fit must use the Phase 2 fits' hyperparameters; overrides are recorded, never silent.
    with pytest.raises(ValueError, match='Hyperparameters differ'):
        main(['parameterized-study', '--study', str(study), '--reference-report', str(report), '--outdir',
              str(tmp_path/'deeper_report'), '--run-dir', str(tmp_path/'deeper_model'), '--training-override', 'max_depth=5'])
    assert not (tmp_path/'deeper_report').exists() and not (tmp_path/'deeper_model').exists()

    comparison_report = tmp_path/'comparison_report'
    comparison_args = ['comparison-study', '--study', str(study), '--reference-report', str(report),
                       '--parameterized-report', str(conditional_report), '--parameterized-report', str(seed7_report),
                       '--runs-dir', str(runs), '--run-prefix', 'cmp', '--outdir', str(comparison_report)]
    main(comparison_args)
    status = json.loads((comparison_report/'provenance/status.json').read_text())
    assert status['status'] == 'complete'
    assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())
    summary = json.loads((comparison_report/'provenance/summary.json').read_text())
    assert set(summary['gate']) == {'200', '400'} and summary['seed_rule'] == 'every_seed_non_inferior'
    assert all(set(by_seed) == {'42', '7'} for by_seed in summary['gate_by_seed'].values())
    assert summary['outcome'] in {'non-inferior', 'inferior', 'inconclusive'} and not summary['test_evaluated']
    # One control per seed at m400; the anchor mass needs none.
    assert [(record['mass'], record['seed']) for record in summary['runs']] == [(400, 42), (400, 7)]
    checks = pd.read_csv(comparison_report/'tables/checks.csv')
    assert checks.passed.all() and 'm400 seed7: control_matches_parameterized_loss_weights' in set(checks.check)
    ratios = pd.read_csv(comparison_report/'tables/loss_weights.csv').set_index(['mass', 'model']).signal_to_background_loss
    assert ratios[400, 'control'] == pytest.approx(ratios[400, 'candidate'])
    assert ratios[200, 'reference'] == pytest.approx(ratios[200, 'candidate'])
    assert ratios[400, 'reference'] != pytest.approx(ratios[400, 'candidate'])
    assert len(pd.read_csv(comparison_report/'tables/bootstrap_replicas.csv')) == 20 * 2 * (2 + 3)
    gate = pd.read_csv(comparison_report/'tables/noninferiority.csv').query("purpose == 'gate' and role == 'primary'")
    assert set(gate.metric) == {'signal_efficiency_at_0.25', 'auc_weighted'} and len(gate) == 2 * 2 * 2
    # A mass passes only if every seed does.
    table = pd.read_csv(comparison_report/'tables/gate.csv', dtype={'seed': str})
    for mass in (200, 400):
        per_seed = table[(table.mass == mass) & (table.seed != 'all')].outcome
        combined = table[(table.mass == mass) & (table.seed == 'all')].outcome.item()
        assert combined == summary['gate'][str(mass)]
        assert (combined == 'non-inferior') == (per_seed == 'non-inferior').all()
        assert (combined == 'inferior') == (per_seed == 'inferior').any()
    assert not list(runs.rglob('*preds_test*'))
    notebook = nbformat.read(comparison_report/'report.ipynb', as_version=4)
    pngs = [o for c in notebook.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert len(pngs) == 3
    assert {path.name for path in (comparison_report/'figures').glob('*.pdf')} == {
        'paired_roc.pdf', 'efficiency_ratio.pdf', 'paired_differences.pdf'}
    with pytest.raises(FileExistsError):
        main(comparison_args)

    # Phase 9: train at rho 0.4, evaluate both couplings on their own genuine samples.
    for mass, seed in [(200, 52), (400, 53)]:
        main(['compact', '--extraction', str(jets_recipe), '--input', str(root_factory(seed=seed, entries=1800)),
              '--outdir', str(compact.parent/'rho04'), '--kind', 'signal', '--sample', f'sig{mass}r04',
              '--xs-pb', '3', '--mass', str(mass), '--rho-tc', '0.4'])
    grid_report, grid_datasets = tmp_path/'grid_report', tmp_path/'grid_datasets'
    grid_args = ['grid-study', '--study', str(study), '--reference-report', str(report),
                 '--compact-root', str(compact.parent), '--registry-dir', str(tmp_path/'grid_registry'),
                 '--dataset-dir', str(grid_datasets), '--runs-dir', str(runs), '--run-prefix', 'grid',
                 '--outdir', str(grid_report)]
    main(grid_args)
    status = json.loads((grid_report/'provenance/status.json').read_text())
    assert status['status'] == 'complete'
    assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())
    summary = json.loads((grid_report/'provenance/summary.json').read_text())
    assert not summary['test_evaluated'] and set(summary['frozen_threshold_directories']) == {'m200', 'm400'}
    assert pd.read_csv(grid_report/'tables/checks.csv').passed.all()
    grid = pd.read_csv(grid_report/'tables/grid.csv')
    assert len(grid[grid.method == 'bdt']) == 2 * 2 * 2 * 3 and len(grid[grid.method == 'cuts']) == 2 * 2 * 3
    assert set(pd.read_csv(grid_report/'tables/registry_extension.csv').sample_key) == {'sig200r04', 'sig400r04'}
    # Each mass is trained once per seed at rho 0.4 and scored at both couplings.
    assert len(list(runs.glob('grid-m*-seed*/models/sig*/model.ubj'))) == 4
    assert len(list(runs.glob('grid-m*-seed*/models/sig*/points/rho0*/operating_points.json'))) == 8
    assert (grid_datasets/'rho04/dataset_sig200_vs_bkg.parquet').is_file()
    anchor_rows = pd.read_parquet(grid_datasets/'rho01/splits/val_sig200.parquet')
    pd.testing.assert_frame_equal(anchor_rows, pd.read_parquet(datasets/'splits/val_sig200.parquet'))
    assert not list(runs.rglob('*preds_test*'))
    notebook = nbformat.read(grid_report/'report.ipynb', as_version=4)
    pngs = [o for c in notebook.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert len(pngs) == 4
    assert {path.name for path in (grid_report/'figures').glob('*.pdf')} == {
        'sensitivity_grid.pdf', 'bdt_vs_cuts.pdf', 'auc_and_seed_spread.pdf', 'sensitivity_vs_mass.pdf'}
    with pytest.raises(FileExistsError):
        main(grid_args)

    binned_report = tmp_path/'binned_report'
    binned_args = ['binned-fit-study', '--study', str(study), '--grid-report', str(grid_report),
                   '--compact-root', str(compact.parent), '--outdir', str(binned_report)]
    main(binned_args)
    status = json.loads((binned_report/'provenance/status.json').read_text())
    assert status['status'] == 'complete'
    assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())
    assert pd.read_csv(binned_report/'tables/checks.csv').passed.all()
    significance = pd.read_csv(binned_report/'tables/significance.csv')
    assert len(significance) == 4 * 2 * 4 * 2  # points x templates x shape terms x MC term
    comparison = pd.read_csv(binned_report/'tables/comparison.csv')
    assert len(comparison) == 4 and (comparison.Z_binned > 0).all()
    assert (comparison.infinite_mc_over_primary >= 1 - 1e-9).all()  # MC statistics can only cost sensitivity
    bins = pd.read_csv(binned_report/'tables/bins.csv')
    assert (bins.background_neff >= 10 - 1e-9).all()  # a sparse tail always joins its neighbour
    notebook = nbformat.read(binned_report/'report.ipynb', as_version=4)
    pngs = [o for c in notebook.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert len(pngs) == 3
    assert {path.name for path in (binned_report/'figures').glob('*.pdf')} == {
        'calibration.pdf', 'cb_mass_templates.pdf', 'binned_significance.pdf', 'uncertainty_dependence.pdf'}
    with pytest.raises(FileExistsError):
        main(binned_args)

    # Stage B: seed-42 BDT-score categories x c-b mass, without the compact exports.
    stage_b = tmp_path/'binned_report_b'
    stage_b_args = ['binned-fit-study', '--stage', 'B', '--study', str(study), '--grid-report', str(grid_report),
                    '--outdir', str(stage_b)]
    main(stage_b_args)
    status = json.loads((stage_b/'provenance/status.json').read_text())
    assert status['status'] == 'complete'
    assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())
    assert pd.read_csv(stage_b/'tables/checks.csv').passed.all()
    categories = pd.read_csv(stage_b/'tables/categories.csv')
    assert len(categories) == 4 * 2  # points x categories
    assert np.allclose(categories.groupby(['mass', 'rho_tc']).background_fraction.sum(), 1)
    significance = pd.read_csv(stage_b/'tables/significance.csv')
    assert len(significance) == 4 * 3 * 4 * 2  # points x fits x shape terms x MC term
    comparison = pd.read_csv(stage_b/'tables/comparison.csv')
    assert len(comparison) == 4 and (comparison.Z_binned > 0).all()
    assert (comparison.infinite_mc_over_primary >= 1 - 1e-9).all()
    # The one-category fit is stage A's observable on reconstructed jets and validation templates.
    stage_a = pd.read_csv(binned_report/'tables/significance.csv')
    stage_a = stage_a[(stage_a.jets == 'reconstructed') & (stage_a.templates == 'validation')]
    single = significance[significance.fit == 'single_category'].drop(columns='fit')
    merged = single.merge(stage_a, on=['mass', 'rho_tc', 'shape_uncertainty', 'mc_statistics'])
    assert len(merged) == len(single) and np.allclose(merged.Z_x, merged.Z_y, rtol=1e-12)
    notebook = nbformat.read(stage_b/'report.ipynb', as_version=4)
    pngs = [o for c in notebook.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert len(pngs) == 4
    assert {path.name for path in (stage_b/'figures').glob('*.pdf')} == {
        'category_templates.pdf', 'binned_significance.pdf', 'fit_comparison.pdf', 'uncertainty_dependence.pdf'}
    with pytest.raises(FileExistsError):
        main(stage_b_args)

    # Truth tagging, separate from the direct results above: a synthetic world whose tag bits follow the
    # registered efficiencies, its closure, then the truth-tagged grid and both binned-fit stages.
    from tests.truth_tag_world import build_world
    truth_root, direct_root, tt_registry = build_world(tmp_path/'tt_world', [
        ('backgrounds', 'bkg_bbc', 'background', [5, 5, 4, 21], 6000, None, None, 20000.0, 1.0),
        ('backgrounds', 'bkg_cjj', 'background', [5, 4, 21, 1], 12000, None, None, 20000.0, 1.0),
        # 0.2 pb signals keep S/B below 1e-2 even after the matching acceptance and overlap removal thin the
        # background: the binned fit's one-bin closure compares a Gaussian bin with Cowan's formula, equal only
        # for S << B, as in production (S/B ~ 1e-3).
        *[(f'rho0{round(rho * 10)}', f'sig_m{mass}_rho0{round(rho * 10)}', 'signal', [5, 5, 4, 2], 2500, mass, rho,
           0.2, mass / 300) for rho in (0.1, 0.4) for mass in (200, 400)]], study=study)
    closure = tmp_path/'truth-tag-closure-v1'
    main(['truth-tag-closure', '--study', str(study), '--truth-tag-root', str(truth_root), '--direct-root',
          str(direct_root), '--registry-dir', str(tt_registry), '--registry-out', str(tmp_path/'tt_registry'),
          '--outdir', str(closure)])
    assert json.loads((closure/'provenance/summary.json').read_text())['adopted']
    tt_grid = tmp_path/'tt_grid_report'
    tt_args = ['truth-tag-grid-study', '--study', str(study), '--grid-report', str(grid_report), '--closure-report',
               str(closure), '--dataset-dir', str(tmp_path/'tt_datasets'), '--runs-dir', str(runs), '--run-prefix',
               'ttgrid', '--outdir', str(tt_grid)]
    main(tt_args)
    status = json.loads((tt_grid/'provenance/status.json').read_text())
    assert status['status'] == 'complete'
    assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())
    tt_summary = json.loads((tt_grid/'provenance/summary.json').read_text())
    assert tt_summary['tagging'] == 'truth' and not tt_summary['test_evaluated']
    assert pd.read_csv(tt_grid/'tables/checks.csv').passed.all()
    assert pd.read_csv(tt_grid/'tables/sampling_check.csv').passed.all()
    tt_table = pd.read_csv(tt_grid/'tables/grid.csv')
    assert set(tt_table.method) == {'bdt', 'cuts'} and len(tt_table[tt_table.method == 'bdt']) == 4 * 2 * 3
    assert not list(runs.rglob('ttgrid*/**/preds_test*'))
    metrics = json.loads((runs/'ttgrid-m400-seed42/models/sig400/metrics.json').read_text())
    assert metrics['fit_policy'] == 'pass_probability'
    with pytest.raises(FileExistsError):
        main(tt_args)
    # An interrupted run, resumed, reproduces the uninterrupted one: each fit calls validate_model once
    # at its training coupling and once per coupling, so the 8th call fails inside the third fit.
    import hepml.commands.grid_study as grid_module
    real_validate, calls = grid_module.validate_model, []
    def interrupted(*arguments, **options):
        calls.append(1)
        if len(calls) == 8:
            raise RuntimeError('simulated interruption')
        return real_validate(*arguments, **options)
    resumed_grid = tmp_path/'tt_grid_resumed'
    resumed_args = ['truth-tag-grid-study', '--study', str(study), '--grid-report', str(grid_report), '--closure-report',
                    str(closure), '--dataset-dir', str(tmp_path/'tt_datasets_resumed'), '--runs-dir', str(runs),
                    '--run-prefix', 'ttresumed', '--outdir', str(resumed_grid)]
    monkeypatch.setattr(grid_module, 'validate_model', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        main(resumed_args)
    monkeypatch.setattr(grid_module, 'validate_model', real_validate)
    assert json.loads((resumed_grid/'provenance/status.json').read_text())['status'] == 'failed'
    main(resumed_args + ['--resume'])
    record = json.loads((resumed_grid/'provenance/resume.json').read_text())
    assert len(record['reused_runs']) == 2 and len(record['refitted_runs']) == 2
    assert len(record['incomplete_runs_set_aside']) == 1
    assert json.loads((resumed_grid/'provenance/status.json').read_text())['status'] == 'complete'
    for name in ('grid', 'auc', 'process_support', 'rates', 'sampling_check'):
        uninterrupted = pd.read_csv(tt_grid/f'tables/{name}.csv')
        resumed = pd.read_csv(resumed_grid/f'tables/{name}.csv')
        if 'run_id' in uninterrupted:
            uninterrupted, resumed = uninterrupted.drop(columns='run_id'), resumed.drop(columns='run_id')
        pd.testing.assert_frame_equal(resumed, uninterrupted)
    assert pd.read_csv(resumed_grid/'tables/checks.csv').passed.all()
    with pytest.raises(ValueError, match='complete'):
        main(resumed_args + ['--resume'])
    tt_a, tt_b = tmp_path/'tt_binned_a', tmp_path/'tt_binned_b'
    main(['binned-fit-study', '--study', str(study), '--grid-report', str(tt_grid), '--outdir', str(tt_a)])
    main(['binned-fit-study', '--stage', 'B', '--study', str(study), '--grid-report', str(tt_grid), '--outdir', str(tt_b)])
    for report_dir in (tt_a, tt_b):
        assert json.loads((report_dir/'provenance/status.json').read_text())['status'] == 'complete'
        assert json.loads((report_dir/'provenance/summary.json').read_text())['tagging'] == 'truth'
        assert pd.read_csv(report_dir/'tables/checks.csv').passed.all()
    assert json.loads((tt_a/'provenance/summary.json').read_text())['settings']['jets'] == 'reconstructed'
    tt_c = tmp_path/'tt_binned_c'
    main(['binned-fit-study', '--stage', 'C', '--study', str(study), '--grid-report', str(tt_grid), '--stage-b-report',
          str(tt_b), '--outdir', str(tt_c)])
    assert json.loads((tt_c/'provenance/status.json').read_text())['status'] == 'complete'
    assert json.loads((tt_c/'provenance/summary.json').read_text())['stage'] == 'binned_fit_C'
    assert pd.read_csv(tt_c/'tables/checks.csv').passed.all()
    stage_c_table = pd.read_csv(tt_c/'tables/comparison.csv')
    np.testing.assert_allclose(stage_c_table.binned_over_stage_b, 1.0, rtol=1e-12)
    assert (stage_c_table.Z_grid_bdt_mc < stage_c_table.Z_grid_bdt).all()
    with pytest.raises(ValueError, match='truth-tagged'):  # stage C needs a truth-tagged grid
        main(['binned-fit-study', '--stage', 'C', '--study', str(study), '--grid-report', str(grid_report),
              '--stage-b-report', str(stage_b), '--outdir', str(tmp_path/'stage_c_direct')])
    assert not (tmp_path/'stage_c_direct').exists()
    # The matching acceptance: only events the central merging scale accepted, weighted by xs / N_accepted.
    counts_path = tmp_path/'tt_world/merging-counts.json'
    counts = json.loads(counts_path.read_text())['samples']
    acc_grid, acc_data = tmp_path/'tt_acc_grid', tmp_path/'tt_acc_datasets'
    acc_args = ['truth-tag-grid-study', '--study', str(study), '--grid-report', str(grid_report), '--closure-report',
                str(closure), '--dataset-dir', str(acc_data), '--runs-dir', str(runs), '--run-prefix', 'ttacc',
                '--outdir', str(acc_grid), '--merging-counts', str(counts_path)]
    main(acc_args)
    assert json.loads((acc_grid/'provenance/status.json').read_text())['status'] == 'complete'
    assert pd.read_csv(acc_grid/'tables/checks.csv').passed.all()
    assert json.loads((acc_grid/'provenance/summary.json').read_text())['matching_acceptance']['counts'] == str(counts_path)
    acceptance_table = pd.read_csv(acc_grid/'tables/matching_acceptance.csv')
    assert (acceptance_table.rows_kept == acceptance_table.exported_accepted).all()
    assert (acceptance_table.exported_accepted < acceptance_table.exported_events).all()
    before = pd.read_parquet(tmp_path/'tt_datasets/background/val.parquet').set_index('event_id')
    after = pd.read_parquet(acc_data/'background/val.parquet').set_index('event_id')
    assert set(after.index) < set(before.index)
    scale = after.sample_key.map({key: counts[key]['n_stored'] / counts[key]['n_accepted'] for key in counts})
    np.testing.assert_allclose(after.sample_weight, before.loc[after.index, 'sample_weight'] * scale, rtol=1e-12)
    pd.testing.assert_series_equal(after.fit_weight, before.loc[after.index, 'fit_weight'])
    main(['binned-fit-study', '--study', str(study), '--grid-report', str(acc_grid), '--outdir',
          str(tmp_path/'tt_acc_binned_a')])
    assert pd.read_csv(tmp_path/'tt_acc_binned_a/tables/checks.csv').passed.all()
    # A rate that the files' own merged rate contradicts stops the study before any row is written.
    wrong = json.loads(counts_path.read_text())
    wrong['samples']['bkg_cjj']['implied_merged_xs'] *= 1.05
    (tmp_path/'wrong-counts.json').write_text(json.dumps(wrong))
    wrong_args = [*acc_args[:-2], '--merging-counts', str(tmp_path/'wrong-counts.json')]
    wrong_args[wrong_args.index(str(acc_data))] = str(tmp_path/'tt_wrong_datasets')
    wrong_args[wrong_args.index(str(acc_grid))] = str(tmp_path/'tt_wrong_grid')
    wrong_args[wrong_args.index('ttacc')] = 'ttwrong'
    with pytest.raises(ValueError, match='matching-acceptance gates'):
        main(wrong_args)
    assert not (tmp_path/'tt_wrong_datasets').exists()
    checks = pd.read_csv(tmp_path/'tt_wrong_grid/tables/checks.csv')
    assert list(checks[~checks.passed].check.str.startswith('bkg_cjj: Pythia merged xs')) == [True]
    # Heavy-flavour overlap removal on top: cjj events with a c c~ pair are ccj's, and leave.
    parton_root = tmp_path/'tt_world/partons'
    hf_grid, hf_data = tmp_path/'tt_hf_grid', tmp_path/'tt_hf_datasets'
    hf_args = ['truth-tag-grid-study', '--study', str(study), '--grid-report', str(grid_report), '--closure-report',
               str(closure), '--dataset-dir', str(hf_data), '--runs-dir', str(runs), '--run-prefix', 'tthf',
               '--outdir', str(hf_grid), '--merging-counts', str(counts_path), '--parton-root', str(parton_root)]
    main(hf_args)
    assert pd.read_csv(hf_grid/'tables/checks.csv').passed.all()
    assert json.loads((hf_grid/'provenance/summary.json').read_text())['flavour_overlap']['remove']['bkg_cjj'] == 'charm_pair'
    overlap = pd.read_csv(hf_grid/'tables/flavour_overlap.csv').set_index('sample')
    assert overlap.loc['bkg_bbc', 'events_removed'] == 0 and overlap.loc['bkg_cjj', 'events_removed'] > 0
    from hepml.adapters.truth_tag_datasets import hard_process_counts
    from hepml.domain.flavour_overlap import removed
    manifest = next((parton_root/'backgrounds').glob('*bkg_cjj.export.json'))
    pairs = removed(hard_process_counts(manifest, 23), 'charm_pair')
    accepted = pd.read_parquet(acc_data/'background/val.parquet').set_index('event_id')
    kept = pd.read_parquet(hf_data/'background/val.parquet').set_index('event_id')
    expected = accepted[~((accepted.sample_key == 'bkg_cjj') & accepted.index.isin(pairs[pairs].index))]
    assert set(kept.index) == set(expected.index) and len(kept) < len(accepted)
    pd.testing.assert_series_equal(kept.sample_weight, expected.loc[kept.index, 'sample_weight'])
    main(['binned-fit-study', '--study', str(study), '--grid-report', str(hf_grid), '--outdir',
          str(tmp_path/'tt_hf_binned_a')])
    assert json.loads((tmp_path/'tt_hf_binned_a/provenance/summary.json').read_text())['flavour_overlap']
    hf_b, hf_c = tmp_path/'tt_hf_binned_b', tmp_path/'tt_hf_binned_c'
    main(['binned-fit-study', '--stage', 'B', '--study', str(study), '--grid-report', str(hf_grid), '--outdir', str(hf_b)])
    main(['binned-fit-study', '--stage', 'C', '--study', str(study), '--grid-report', str(hf_grid), '--stage-b-report',
          str(hf_b), '--outdir', str(hf_c)])
    # Phase 10a: the jet-level networks on the same rows, weights and validation events as the grid's BDTs.
    if importlib.util.find_spec('torch'):
        network = tmp_path/'network_pilot'
        main(['jet-network-study', '--study', str(study), '--grid-report', str(hf_grid), '--stage-c-report', str(hf_c),
              '--runs-dir', str(runs), '--run-prefix', 'net', '--outdir', str(network), '--device', 'cpu'])
        assert json.loads((network/'provenance/status.json').read_text())['status'] == 'complete'
        assert pd.read_csv(network/'tables/checks.csv').passed.all()
        primary = pd.read_csv(network/'tables/primary.csv')
        assert set(primary.model) == {'lorentznet', 'deep_sets'} and len(primary) == 4
        assert len(pd.read_csv(network/'tables/fits.csv')) == 2 * 2 * 2 and not list(runs.rglob('net-*/*test*'))
        with pytest.raises(ValueError, match='registered'):  # only the registered grid
            main(['jet-network-study', '--study', str(study), '--grid-report', str(acc_grid), '--stage-c-report',
                  str(hf_c), '--runs-dir', str(runs), '--run-prefix', 'net2', '--outdir', str(tmp_path/'net2')])
        if importlib.util.find_spec('weaver'):
            # The upstream ParT on the same tokens, against the BDT and against LorentzNet of the same seed.
            part = tmp_path/'part_pilot'
            main(['jet-network-study', '--design', 'particle_transformer_pilot', '--study', str(study), '--grid-report',
                  str(hf_grid), '--stage-c-report', str(hf_c), '--lorentznet-report', str(network), '--runs-dir',
                  str(runs), '--run-prefix', 'part', '--outdir', str(part), '--device', 'cpu'])
            assert pd.read_csv(part/'tables/checks.csv').passed.all()
            compared = pd.read_csv(part/'tables/primary.csv')
            assert set(zip(compared.model, compared.against)) == {('particle_transformer', 'bdt'),
                                                                   ('particle_transformer', 'lorentznet')}
            assert set(pd.read_csv(part/'tables/metrics.csv').model) == {'bdt', 'lorentznet', 'particle_transformer'}
            summary = json.loads((part/'provenance/summary.json').read_text())
            assert summary['stage'] == 'particle_transformer_pilot' and set(summary['secondary']) == {'m200', 'm400'}
            # The presentation benchmark from the saved reports and models: relative significance, AUC, importances.
            benchmark = tmp_path/'benchmark'
            main(['benchmark-report', '--study', str(study), '--grid-report', str(hf_grid), '--lorentznet-report',
                  str(network), '--part-report', str(part), '--outdir', str(benchmark), '--device', 'cpu',
                  '--permutation-repeats', '1', '--shap-events', '50'])
            assert pd.read_csv(benchmark/'tables/checks.csv').passed.all()
            methods = pd.read_csv(benchmark/'tables/methods.csv')
            assert set(methods.method) == {'cut_based', 'bdt', 'lorentznet', 'particle_transformer'}
            assert (methods[methods.method == 'cut_based'].z_asimov_ratio == 1).all()
            assert len(list((benchmark/'figures').glob('*.png'))) == 5 and (benchmark/'benchmark.md').is_file()
    with pytest.raises(ValueError, match='--merging-counts'):  # registered on top of the acceptance only
        main([*hf_args[:-4], '--parton-root', str(parton_root)])
    # The direct-tag reports are untouched.
    for report_dir in (grid_report, binned_report, stage_b):
        status = json.loads((report_dir/'provenance/status.json').read_text())
        assert all(sha256_file(Path(p)) == h for p, h in status['artifacts'].items())

    final = tmp_path/'releases'
    main(['freeze', '--conditional', '--workdir', str(shared), '--finaldir', str(final)])
    release = final/'conditional_v1'
    record = json.loads((release/'threshold.json').read_text())
    assert set(record['hypotheses']) == {'200', '400'}
    for mass in [200, 400]:
        predict = ['predict', '--model-dir', str(release), '--input', str(datasets/f'splits/val_sig{mass}.parquet'),
                   '--split', 'val']
        with pytest.raises(ValueError, match='explicitly'):
            main(predict)
        with pytest.raises(ValueError, match='supported'):
            main([*predict, '--hypothesis-mass', '300'])
        main([*predict, '--hypothesis-mass', str(mass), '--objective', 'Z_syst_10pct'])
        prediction = pd.read_parquet(release/f'preds_infer_val_m{mass}.parquet')
        expected = pd.read_parquet(shared/f'm{mass}/preds_val.parquet')
        np.testing.assert_allclose(prediction.bdt_score, expected.bdt_score, atol=2e-6)
        main(['summarize', '--model-dir', str(release), '--hypothesis-mass', str(mass), '--split', 'val'])
        summary = json.loads((release/f'report_infer_val_m{mass}.json').read_text())
        assert summary['yields_weighted']['k_factors'] == {'signal': 1.95, 'background': 1.26}
        assert summary['objective'] == 'Z_syst_10pct'
        for point in summary['operating_points']:
            reference = record['hypotheses'][str(mass)]['evaluation']['operating_points'][point['objective']]['metrics']
            assert point['S'] == pytest.approx(reference['S'])
            assert point['B'] == pytest.approx(reference['B'])
