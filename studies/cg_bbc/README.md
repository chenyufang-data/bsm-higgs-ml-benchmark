# Example channel: cg_bbc

A simplified example configuration of the analysis channel used in the benchmark. A hypothetical
charged particle of mass m is produced with a bottom quark and decays into a charm and a bottom
quark, so the final state has three tagged jets: two bottom-like, one charm-like. Two background
processes with the same tagged final state are included. The process and its cut-based selection are
published in [arXiv:2511.19604](https://arxiv.org/abs/2511.19604).

| File | Content |
| --- | --- |
| `extraction*.yaml` | Extraction recipes: which objects and fields to export, and the event selection |
| `features.py`, `plugin.py` | Model inputs: the four-momentum components of the three jets. Also the three columns the cut-based baseline selects on |
| `analysis.yaml` | Luminosity, rate corrections, training parameters and operating-point rules |
| `validation.yaml` | Study designs: masses, seeds, the cut-based baseline, the tagging model and the network pilots, fixed before training |
| `samples.yaml` | Example sample manifest (signal masses 200 and 400 GeV, two backgrounds); paths and rates are placeholders |

The benchmark itself used a fuller configuration.
- The BDT was trained on fifteen engineered observables of the same three jets: invariant masses,
  angular separations and momenta, as shown in the SHAP figure.
- More background processes were included.
- The networks and the permutation importance use the same twelve four-momentum components as this
  example.
