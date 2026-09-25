# Notebooks

Report templates. Each study command executes one of them in a fresh kernel on its own saved
artifacts and publishes the executed notebook and an HTML version next to the report's tables and
figures. The notebooks only display results: selections, splits, training and metrics live in
`src/`.

| Notebook | Report |
| --- | --- |
| `01_inspect_data.ipynb` | Exported samples, event counts and weights |
| `02_run_benchmark.ipynb`, `03_compare_results.ipynb` | Dataset preparation, training runs and comparisons on fixed splits |
| `02_bdt_baselines.ipynb` | BDT and cut-based references |
| `03_parameterized_bdt.ipynb` | A BDT conditioned on the signal hypothesis |
| `04_physics_results.ipynb` | Sensitivity grid and binned fits |
| `05_truth_tagging.ipynb` | Closure test of probability-weighted tagging |
| `06_jet_networks.ipynb` | LorentzNet and Particle Transformer against the BDT |

Install the kernel with `python -m ipykernel install --sys-prefix --name hepml`.
