# SparseRadioMap v0.1.0

A reproducible application project by **Yuntao Xue**: per-image reconstruction of simulated 2D radio maps from sparse observed coordinates and signal values. Four methods: IDW, grid-gradient regularization, regularized low-rank matrix factorization, and a Fourier-encoded coordinate MLP. This is neither NeRF2 nor a claim of algorithmic novelty.

## Executed versions

| Version | Scope | Status |
| --- | --- | --- |
| `baseline_v01` | 30 regions / 60 no-car DPM images; 18/6/6 split; 16 development checks, 216 tuning runs, 864 test reconstructions; neural methods 200 steps | Preserved unchanged in local results and `history/baseline_v01/snapshot.zip` |
| `budget_dev_v02` | First two preselected development regions, two Tx each, both mechanisms, all three fractions, seed 11; 48 trajectories | Each trajectory saves 200 / 1,000 / 3,000-step predictions |
| `budget_val_v02` | All six validation regions / 12 images; both mechanisms, 1% / 5% / 10%, seeds 11/22/33; 432 trajectories | 1,296 checkpoint reconstructions, full observed-set diagnostics |
| `test_v02` | Original 12 test images × 3 fractions × 2 mechanisms × 3 seeds × 4 methods | 864 new fits; follow-up exploratory evaluation on the previously viewed holdout |

The test set was already viewed in v01. v02 is **not** an untouched confirmatory test. No test truth or previous test ranking was used to choose checkpoints or hyperparameters. Validation balanced region-averaged hidden RMSE over both missingness mechanisms and all three fractions. Low-rank selected 3,000 steps; MLP retained 200. Architecture, rank, learning rate and regularization remained unchanged. The predeclared small regularization-search gate was not triggered; no extra search was performed.

Validation means (dB):

| Method | Steps | Observed RMSE | Hidden RMSE |
| --- | ---: | ---: | ---: |
| Low-rank | 200 | 3.89 | 10.88 |
| Low-rank | 1,000 | 3.86 | 10.77 |
| Low-rank | 3,000 | 3.86 | 10.74 |
| MLP | 200 | 3.39 | 5.20 |
| MLP | 1,000 | 1.93 | 5.26 |
| MLP | 3,000 | 1.36 | 5.28 |

Longer training improved MLP observation fitting but not its aggregate validation hidden error under this model and protocol. Low-rank showed a small hidden-error improvement and an observed-error plateau. These are limited empirical observations, not universal claims about low-rank models or neural fields. Condition-specific curves and regularization/objective decomposition are in the report.

Start with the [Experiment report](experiments/v02/reports/report.md), [budget candidate evidence](experiments/v02/reports/budget_candidate_scores.csv), [new summary](experiments/v02/results/test_v02_summary.csv), [region-paired version differences](experiments/v02/results/baseline_new_paired.csv), and [changelog](CHANGELOG.md).

![Validation hidden-error budget curves](experiments/v02/figures/budget_hidden.png)

![Exploratory v02 holdout results](experiments/v02/figures/performance_test_v02.png)

## Data and evaluation

The original [RadioMapSeer](https://radiomapseer.github.io/index.html) ZIP was downloaded, CRC checked and SHA-256 verified. Only original **DPM without cars** is used: 701 regions × 80 Tx, native 256×256 at 1 m/pixel. These are urban propagation **simulations, not field measurements**. The original audit decoded 112,861 DPM/building/antenna PNGs; no corruption, six DPM duplicate groups, 104 antenna duplicate groups and 395 antenna pixels overlapping rasterized buildings were recorded. JSON `(x,y)` corresponds to PNG `(row=255-y, col=x)`.

The fixed selection seed is 20261013. All Tx/variants from a map belong to the same split. Buildings only define outdoor sampling/scoring; outdoor zero-gray values remain valid. Observations are noiseless, without replacement: floor(p × valid pixels). Random and 64×64-hole mechanisms have equal observed counts and share masks across methods. Holes remain in bounds and do not imply that all their pixels are outdoors.

`y=gray/255`; saved negative pathgain is `-147+100*y` dB, following the [dataset paper](https://arxiv.org/abs/2212.11777). This is not RSS in dBm or exact pre-truncation truth. Gray zero is truncated/quantized; the author's additional 0.2 threshold is not applied. The target is the saved image.

Main metrics use hidden outdoor points only, with observed, hole-interior and prespecified outdoor high-gradient errors reported separately. Raw predictions are saved; all methods use the same [0,1] clipping policy and disclose unclipped error/out-of-range fraction. Region means and cluster bootstrap resample maps, retaining Tx/seed structure. Fit counts describe computational runs, not independent statistical samples; the statistical unit is the map region. Six holdout regions support exploratory intervals only. Small smooth–IDW differences should not be presented as a large practical advantage. Geographic overlap, real observation noise, pre-truncation recovery and GPU determinism remain unverified.

## Install and reproduce

Recommended: Python 3.12, CPU. **The experiments used local environments; no `.venv` is supplied in either package.** A separate clean environment without system-site-packages was installed and tested; [clean_environment.json](experiments/v02/reports/clean_environment.json) records the resolved package versions. The clean environment passed all 15 tests and re-fit a real ZIP-only example with zero cache reuse. The minimal public staging package itself also passed all 15 tests and executed four fresh fits; those outputs were subsequently compared with the complete local baseline and matched exactly.

`requirements-lock.txt` is a pinned **direct-dependency version list**, not a complete transitive or wheel-hash lock. Resolved transitive versions can vary across time/platforms. The exact validated clean installation used:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt -e '.[test]'
.venv/bin/python -m pytest -q
```

Standard Python tooling can alternatively create a venv, install the version list and the editable project. That alternative is provided for convenience; the recorded clean validation used `uv`.

Reuse the SHA-256-verified original archive (no new download is needed locally):

```bash
.venv/bin/python scripts/pipeline.py all --output experiments/v02 --archive data/raw/RadioMapSeer.zip
```

The public candidate excludes arrays. Running `all` regenerates them from the archive. For an independent full rerun, use a **new** output directory, such as `experiments/independent_v02`, to avoid borrowing records. A successful cache-only invocation is explicitly not an independent reproduction. If the archive is absent, the official downloader remains available:

```bash
.venv/bin/python scripts/download.py
```

The expected raw SHA-256 is `c80672cea3b3693d4a0a57eefb36a1a7e3ef78e04a99e6842d5a96afb6cc072f`. Scientific results are isolated by output directory/version/run manifest. Source, data, parameters, mask and prediction hashes are checked; mismatched, missing, duplicate or stale records are rejected. Frozen parameter selection binds the scientific source and validation plan; source changes require a new freeze/output version.

For CPU execution on a GPU-capable host, prepare a **new** output with `--cpu`, for example `all --output experiments/independent_cpu_v02 --archive data/raw/RadioMapSeer.zip --cpu`. This records explicit CPU parameters and never amends the supplied freeze. Fresh runs also accept `--cpu`; the flag was exercised on a real ZIP-only example, with four new fits matching the baseline exactly. GPU execution remains untested.

Stage commands: `prepare`, `dev-budget`, `val-budget`, `ridge`, `select`, `main`, `verify`, `summarize`. The `ridge` stage applies only the preregistered evidence gate; it does not search automatically when the gate is false. Versioned records live under `experiments/v02/results/<version>/`.

### Genuine first run / independent example

```bash
.venv/bin/python scripts/pipeline.py fresh --output experiments/fresh_new --archive data/raw/RadioMapSeer.zip --fresh-split dev
```

The output must be empty. This reads only the original ZIP, processes a real image, creates masks, fits all four methods, scores and exports. Existing processed arrays, masks, predictions and run records are never used for fitting. Reference outputs are read only **after** fitting to compare results. The executed clean development example exactly matched the corresponding baseline mask/predictions/metrics, with four fits and zero cache hits.

After the final configuration is frozen, `--fresh-split test` independently re-fits one original holdout condition under v02. The full local workspace includes reference arrays; the minimal public package omits them, so comparisons that need those arrays are explicitly marked unavailable rather than claimed as repeated. Baseline score-recomputation evidence from the actual local run is preserved in the report.

Only CPU was actually tested. The original reconstruction implementations retain device detection, but the instrumented budget trajectory currently validates CPU only. No claim is made about cross-device determinism.

## Diagnostic semantics

All logged MAE/RMSE and objectives are evaluated on the **complete observation set after the parameter update**, not on the MLP minibatch. The explicitly named minibatch MSE is before the update. Fixed checkpoints share initialization, masks and batch RNG sequence along one trajectory; callback scoring has no effect on optimization. Low-rank objective is MSE plus ridge on residual factors; MLP's coupled Adam weight decay corresponds to L2 penalty `reg/2 * sum(parameter²)`. Diagnostic time includes scoring/forward overhead; main reconstruction time excludes scoring and I/O. Legacy final-loss metadata in ordinary reconstructions is from before the final optimizer update and is not substituted for the post-update full-observation diagnostics.

## Packaging, license and attribution

Original code: MIT © 2026 Yuntao Xue. Original RadioUNet audit reference code: MIT © 2019 Ron Levie, separate from this implementation.

The earlier report missed the **CC BY 4.0 notice adjacent to the original download in the official website HTML**. [DATA_LICENSE.md](DATA_LICENSE.md) preserves the corrected primary evidence; this does not infer a data license from the paper or code. [FIGURE_ATTRIBUTION.md](FIGURE_ATTRIBUTION.md) credits Çağkan Yapar, Ron Levie, Gitta Kutyniok and Giuseppe Caire, links the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), and describes masking, colour mapping, reconstructions, error maps and plot-layout modifications. The authors do not endorse these results.

The **complete local package** includes the baseline snapshot, processed arrays, masks, predictions, training traces, full records, figures and verification; the 3.26 GB original ZIP is referenced externally by checksum. The **minimal public candidate** includes code, configs, necessary numerical evidence, reports and selected attributed dataset-derived display figures. It excludes the original ZIP, processed arrays, masks, predictions, environments, caches, detailed old history and unrelated local paths. Thus display images are explicitly included; "derived data excluded" refers to arrays, not these licensed presentation images.

Please cite the [RadioMapSeer dataset paper](https://arxiv.org/abs/2212.11777) separately. This repository publishes the minimal public candidate as SparseRadioMap v0.1.0. No GitHub Release, PyPI package or website deployment is included; the data authors have not been contacted. Future directions—censoring, building-aware priors, uncertainty and DeepMIMO—remain unimplemented.
