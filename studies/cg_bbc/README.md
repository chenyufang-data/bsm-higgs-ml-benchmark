# Example channel: cg_bbc

Configuration of the analysis channel used in the benchmark: a hypothetical charged particle produced
with a bottom quark and decaying into a charm and a bottom quark, so that the final state has three
tagged jets (two bottom-like, one charm-like). The process and its cut-based selection are published
in [arXiv:2511.19604](https://arxiv.org/abs/2511.19604).

| File | Content |
| --- | --- |
| `extraction*.yaml` | Extraction recipes: which objects and fields to export, and the event selection |
| `features.py`, `plugin.py` | The BDT's engineered features from the three tagged jets |
| `analysis.yaml` | Luminosity, rate corrections, training parameters and operating-point rules |
| `validation.yaml` | Study designs: masses, seeds, decision rules and controls, fixed before each study ran |
| `samples.yaml` | Example sample manifest; paths and rates are placeholders |
