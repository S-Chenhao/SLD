# Reproducing the supplied SLD experiments

## Scope of this release

This repository organizes the author's six SLD experiment notebooks and three dataset generators. The notebooks retain their model and numerical-solver code, loss definitions, training hyperparameters, and supplied seeds. Preparation repairs execution and path issues and explicitly corrects the Gray–Scott Task 2 temporal split.

Raw datasets are hosted on Hugging Face, and six representative trained SLD checkpoints are included. Two compact SLD-only evaluation notebooks provide one-checkpoint test metrics. Baseline implementations and complete manuscript-results aggregation are not part of this release. The accompanying manuscript was used to identify the tasks and data contracts; the manuscript PDF is not bundled.

## Protocol and supplied splits

Task 1 evaluates unseen initial conditions. Each notebook holds out one trajectory, selects seven of the remaining nine for training, and uses the other two for validation. All splits use four windows with LR starts `100,150,200,250` in their loader arrays. The full manuscript evaluation repeats this over ten held-out trajectories.

| System | Supplied Task 1 held-out IC | Torch / NumPy seed | Python split seed |
| --- | --- | --- | --- |
| Gray–Scott | 6 | 1 | Unseeded in supplied code |
| Ginzburg–Landau | 7 | 42 | Unseeded in supplied code |
| Burgers | 5 | 42 | Unseeded in supplied code |

A rerun of Task 1 can choose different training/validation identities because the supplied Python `random` state is unseeded. Record the printed identity assignments and use an explicit seed when designing a new reproducible run; an added seed cannot recover an unknown original assignment.

Task 2 trains on early windows of all ten identities. Training LR starts are `100,150,200`; validation/test LR starts are `225,250`, divided five identities/five identities. The supplied Python seeds are 4 for Gray–Scott, 48 for Ginzburg–Landau, and 1 for Burgers. The manuscript aggregates five repeated partitions; each supplied notebook executes one.

Task 2 receives coarse observations from the later window it reconstructs. Its objective is later-time reconstruction. Training and evaluation windows occupy disjoint intervals.

## Gray–Scott temporal preprocessing

The raw generated trajectories contain 3001 HR frames and 751 LR frames. The supplied model loaders apply:

```python
lres = lres[:, 50::2, :, :]
hres = hres[:, 200::2, :, :]
```

The resulting arrays contain 351 LR and 1401 HR frames. Segment starts apply to these arrays; the physics-loss configuration uses effective HR spacing `dt=10`. Consequently, a segment starting at loader LR index `s` begins at raw LR index `50+2*s` and raw HR index `200+8*s`.

For example, Task 2 training starts `100,150,200` correspond to raw HR starts `1000,1400,1800`; later starts `225,250` correspond to `2000,2200`. These supplied preprocessing choices are preserved. They differ from the manuscript's description of starts in the raw stored arrays with HR spacing 5. Exact manuscript time-window reproduction is therefore unverified.

The supplied GS generator stores its initial state, then saves after an update when `step % 5 == 0`. This produces stored internal step indices `0,1,6,11,...`, with a shorter first interval. That supplied timing behavior is also preserved and requires reconciliation before claiming exact paper timing.

The original Task 2 `GSDataset` accepted a `segments` argument but ignored it. It used the fixed starts `100,150,200,250` for every split, overlapping the intended early/later protocols. The prepared notebook uses the supplied `segments` argument. Synthetic time-coded data checks verify the requested starts and nonoverlapping HR frame sets.

## Shared Burgers dataset

The authors' later supplied CaseB notebook and both evaluation notebooks use the same independent smooth-IC dataset as CaseC and the generator:

```text
2DBurgers_IC{ic}_ri100_sig040_indIC_2x300x32x32.mat
2DBurgers_IC{ic}_ri100_sig040_indIC_2x1200x256x256.mat
```

Both training tasks and released evaluation now use these names directly. The initial unseen-IC source expected legacy `ri100` filenames; its compatibility fallback was removed after the authors confirmed the shared dataset. The other training model, loss and hyperparameter code is retained from the initial source.

## Files, paths, and device selection

Each notebook's setup cell discovers the root via `README.md` and `data_generation/`, so kernels can start at the repository root or within a notebook folder. Paths use `pathlib`; hardcoded Windows checkpoint directories and GPU visibility masks have been removed.

- `SLD_DATA_ROOT`: defaults to `<repository>/data`, with child directories `gray_scott`, `ginzburg_landau`, and `burgers`.
- `SLD_RESULTS_ROOT`: defaults to `<repository>/results`, with task/PDE child directories.
- Models use CUDA when available and otherwise CPU. GS derivative and residual allocations now honor this selected device.

Burgers and GL training continuation cells had inherited main-block indentation. They are merged into the corresponding main cell. The independently indented GS model/training cell is dedented after the preceding setup cell, retaining its supplied execution order.

Saved notebook outputs, execution counts, and volatile metadata are removed. Optional obsolete or duplicate visualization cells are omitted; core SLD model code is retained. Unused GL helpers that referenced an undefined loss class or allocated only on CUDA are removed. Legacy internal class names, such as `PhySR_ODE_ConvGRU`, remain in the code.

GL clean/noisy filenames and completion counts now use the current 1200 HR / 300 LR / 32 × 32 dimensions. The configured observation-noise level is zero in the primary supplied generators. Optional extra-noise exports are not required by the default training flow.

## Dataset generation and resources

MATLAB files use key `uv` and axes `[channel,time,height,width]`. The GL/Burgers saved HR arrays use float32; GS retains float64. Ten GS HR arrays require about 29.3 GiB of disk, with more than 6 GiB needed for simultaneous A/B records and a stacked HR array before serialization and other working allocations. GL and Burgers each require about 5.9 GiB for ten HR arrays. Allow additional space for LR/noisy variants, plots, model states, histories, and temporary arrays.

Install `pyrtools` to retain binomial-pyramid downsampling for GS/GL. GS stops with a dependency error if it is absent; GL retains its supplied Gaussian fallback, which changes the operator. The listed requirements include `pyrtools`. Burgers performs Fourier cropping with the supplied amplitude correction.

Use the generator's default scientific settings when reproducing the supplied data recipe. Changing grid sizes or step counts for a smoke test changes the experiment and is not a reproduction of the complete datasets.

## Evaluation and validation limits

The training loops create test loaders but do not automatically run final held-out evaluation after training. GS includes a disabled optional cell that loads a selected training checkpoint and invokes the existing `test_gs` helper; its original batch-average metric is not the manuscript's repeated-partition aggregate. The separate released `SLD_evaluation.ipynb` notebooks now provide the single-checkpoint test-reporting pipeline for all three PDEs.

Release validation covers notebook schema and syntax, portable setup paths, six small CPU model forward passes, physical-loss checks, synthetic segment tests, and tiny generator/operator smoke checks. The checks use shortened sequences and small grids to exercise executable paths. They do not measure reconstruction accuracy, establish convergence, or validate the published numerical results. No full dataset generation, 1000-epoch training, GPU experiment, or paper-metric reproduction was performed.

The exact check results and validation environment are saved in [VALIDATION.json](VALIDATION.json). [PREPARATION.json](PREPARATION.json) maps original filenames to prepared notebooks and records the changes.

## License

A software license has not yet been selected by the authors. Public repository visibility alone does not grant an open-source license. A `LICENSE` file can be added after that choice is confirmed.

## Published checkpoint evaluation

See [EVALUATION.md](EVALUATION.md) for the two SLD-only evaluation notebooks, six checkpoint paths, saved normalization, authors' confirmed splits, pooled metrics and pinned Hugging Face dataset revision. Their checks are recorded separately in [EVALUATION_VALIDATION.json](EVALUATION_VALIDATION.json). The original preparation validation above documents the first release before checkpoints were supplied.
