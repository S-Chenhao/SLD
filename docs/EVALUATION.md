# Single-checkpoint SLD evaluation

This release publishes the six original author-supplied checkpoint files, one per PDE/task. The evaluation runs only SLD. It contains no baseline model definitions, multi-checkpoint search, cross-fold statistics, prefilled paper metrics, or panel/latent-response plotting code.

## Run

Install `requirements.txt` and open `task1_unseen_ic/SLD_evaluation.ipynb` or `task2_later_time/SLD_evaluation.ipynb` from Jupyter inside the repository. The setup finds the repository root from a nested task folder. It supports `SLD_DATA_ROOT` and `SLD_RESULTS_ROOT`.

Select the PDEs in `PDES`. The default device is CUDA when available, otherwise CPU. Each PDE has one explicit checkpoint path. `TEST_ICS=None` selects the defaults below; changing that setting is an explicit custom evaluation.

Raw MATLAB files live in `data/gray_scott`, `data/ginzburg_landau`, and `data/burgers`. If the files are not already there, set `DOWNLOAD_DATA=True` to download only the configured test trajectories from [ChenhaoSI/SLD-dataset](https://huggingface.co/datasets/ChenhaoSI/SLD-dataset). The release pins dataset revision `99e5242aaf83fe64071b6bf6d8e6505e75165d92`; all 60 expected HR/LR filenames were verified in that public revision.

Evaluation loads one MAT pair at a time. A Gray–Scott HR trajectory occupies about 3 GiB in RAM before selected float32 windows are copied. Prediction and target tensors are processed one window at a time. Small metric JSON files are written to `results/<task>/<pde>/metrics.json`; full predictions are not exported.

## Protocol

| Task | PDE | Test ICs | LR window starts |
| --- | --- | --- | --- |
| Unseen IC | All three | 1 | 100, 150, 200, 250 |
| Later time | Gray–Scott | 1, 3, 7, 8, 9 | 225, 250 |
| Later time | Ginzburg–Landau | 1, 2, 4, 7, 8 | 225, 250 |
| Later time | Burgers | 6, 7, 8, 9, 10 | 225, 250 |

Each window is 25 LR frames to 100 HR frames, with two channels and spatial ratio 8. For GL/Burgers the HR start is four times the LR start in the stored arrays. Both Burgers tasks read `2DBurgers_IC{i}_ri100_sig040_indIC_...mat`.

Gray–Scott preserves the supplied temporal preprocessing `LR[:,50::2]` and `HR[:,200::2]`; the listed starts apply after this slicing. Raw GS starts are `50+2*s` for LR and `200+8*s` for HR, with stride 2. For later-time starts 225 and 250, raw HR starts are 2000 and 2200. The JSON records the complete raw frame mapping. This follows the supplied loaders; exact equivalence to the manuscript's raw-array time-index description remains unverified.

CaseB `_1` identifies held-out IC 1 according to the supplied evaluation's filename convention. CaseC `_1` identifies one run, not test IC 1. The original CaseC evaluation used seed 2 for all PDEs; the authors explicitly confirmed using the current training notebook defaults instead: GS seed 4, GL seed 48, and Burgers seed 1. Their validation IDs are respectively `[4,5,2,6,10]`, `[9,6,3,5,10]`, and `[3,2,5,1,4]`. Early training windows use starts 100, 150 and 200 from all ten identities.

The supplied checkpoints do not record split IDs, seeds, or data checksums. The configured partitions are therefore explicit release configuration, rather than metadata recovered from weights. The training notebooks remain representative training examples and need not reproduce the exact train/validation split of each published Task 1 checkpoint.

## Model loading and metrics

The shared `sld/model.py` retains the active SLD inference architecture and the original checkpoint parameter names. A minimal registered `ode_wrapper.cell` preserves duplicate state-dictionary paths expected by the supplied checkpoints. The architecture performs exactly the same forward computation as the original SLD model.

Checkpoints are loaded with `torch.load(..., weights_only=True, map_location='cpu')` and `load_state_dict(..., strict=True)`. LR/HR mean and standard-deviation buffers are restored directly. No normalization is recomputed from test data. Optimizer and scheduler state in the original files are unused by inference.

For errors `e = prediction - truth`, the evaluator pools over every tested pixel, both channels, all 100 target frames and all configured windows:

```text
Relative L2 = sqrt(sum(e**2) / (sum(truth**2) + 1e-12))
MAE         = sum(abs(e)) / (number_of_elements + 1e-12)
```

This matches the original notebooks' core testing cells. There is no first-10-frame exclusion, batch-average approximation, or checkpoint-level mean/standard deviation. Float64 sums improve accumulation stability. Per-IC and per-window metrics are included with the same definitions.

JSON results include checkpoint SHA-256, saved normalization, configured test IDs/windows, raw-frame mapping, pooled metric sums, and per-IC/per-window results. Checkpoint `best_error` is labeled recorded training validation metadata; it is not reported as freshly evaluated test performance.

The six original checkpoint bytes and SHA-256 values are listed in `CHECKPOINTS.json`. Release validation is recorded in `EVALUATION_VALIDATION.json`; no full-dataset or manuscript aggregate accuracy claim is implied by smoke checks.
