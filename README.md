# Conformal Bounds Through Sensor Blackouts

Reproducible experiments for **Lei Chen and Cemil Emre Yavas**, *Conformal Bounds Through Sensor Blackouts*.

This package implements counterfactual reset-error calibration for linear systems with intermittent full-state measurements. It contains a six-method primary benchmark, a fresh-split comparison of five interval shapes calibrated to the same event, an observable innovation-censoring experiment, and separate noise-shift and dynamics-mismatch controls.

## Main findings

At a nominal 95% simultaneous coverage target, the all-window method achieved 94.6–95.1% coverage across the nine primary plant/noise combinations. Calibration on sampled masks achieved 48.6–83.9% on the same all-schedule criterion. All-window bounds were 7.9–26.0% narrower than agewise union bounds and 46.3–75.6% narrower than propagated primitive-noise envelopes, using equally weighted ages and measurement-scale-normalized widths.

A second study uses 540,000 fresh test episodes. Recalibrated propagation, training-maximum, and temporal-tail shapes all attain approximately nominal all-schedule coverage. RMS-shaped bounds are 37.2–61.9% narrower than recalibrated propagation-shaped bounds, averaged by noise law, but do not uniformly outperform tail-shaped bounds. Under observable censoring with switching noise, only 4.8% of packets are discarded; observed-mask coverage is 85.1%, while all-window coverage is 99.0%. Despite larger uniform-age width, the recalibrated propagated shape certifies 59.96% of steps versus 43.39% for the RMS shape. Average width and decision usefulness are distinct objectives.

These are synthetic experiments. The guarantee assumes known linear dynamics, known applied controls, full-state reset measurements, exchangeable complete noise episodes, exogenous additive noise, and a fixed horizon and outage cap. A change in the noise law can invalidate coverage. The greedy schedule is a privileged causal stress test that observes the current true error.

## Installation

The reference environment is Python 3.12.1. Create an isolated environment and install the pinned CPU dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The CPU implementation uses float64. The recorded primary run used ten worker processes, one numerical-library thread per worker, and approximately 4.54 GiB peak summed resident memory. The primary experiment completed in 46.3 seconds on a 12-core arm64 system; runtime depends on hardware and numerical libraries. Fewer workers reduce concurrent memory usage. All data are generated locally from deterministic random-number streams.

## Reproduce the results

Run these commands from the package directory:

```sh
python -m unittest discover -s tests -v
python run_experiments.py --mode main --workers 10
python run_experiments.py --mode sensitivity --workers 10
python run_shift_check.py
python summarize.py
python run_extensions.py --workers 8
python run_mismatch_check.py --workers 8
python summarize_extensions.py
python verify_results.py
```

The runners regenerate `results/`. `verify_results.py` compares the numerical CSV files with the recorded SHA-256 hashes. Exact hashes are intended for the pinned reference environment; different numerical-library builds can produce small floating-point differences. Execution times and hardware metadata are recorded separately from the numerical checksums.

A quick execution check is available with:

```sh
python run_experiments.py --mode smoke --workers 1 --output smoke-results
```

The optional GPU cross-check requires a supported Metal or CUDA backend:

```sh
python -m pip install -r requirements-gpu.txt
python run_gpu_check.py
```

It recalculates all 540,000 primary test-episode maxima in float32 and compares their coverage decisions with the float64 results. The recorded cross-check had zero binary differences and a maximum standardized-score difference below 3.90e-6.

## Experiments

- **Primary:** 3 plants × 3 disturbance laws × 30 repetitions, each with 400 training, 999 calibration, and 2,000 test episodes. Each episode contains 100 transitions and 101 states, with maximum outage age 16 and failure target 0.05.
- **Sensitivity:** double integrator, Gaussian and switching noise, 20 repetitions per cell. Outage caps 4, 8, and 32 at 999 calibration episodes; calibration sizes 199, 499, and 1,999 at cap 16. The primary cap-16/999-sample result supplies the reference point.
- **Noise shift:** 30 new test repetitions per noise law for the double integrator. Process and measurement noise amplitudes are both multiplied by 1, 1.25, 1.5, or 2 while retaining the original calibration bounds.
- **Common-event shape comparison:** fresh seeds, 3 plants × 3 laws × 30 fits; 400 training, 999 calibration and 2,000 test episodes per fit. Shapes are RMS, propagated primitive-noise scale, age-constant scale, training maximum, and the 95th percentile across training episodes of temporal maxima. Each uses its own all-window quantile. Original primitive-noise and observed-mask constructions supply reference comparisons.
- **Observable censoring:** a synchronized predictor and the current measurement form an innovation. The packet is discarded when its largest absolute coordinate, normalized by a training-only fresh-reset innovation RMS, exceeds 2.5. Age-cap reception overrides this gate. No true state enters the gate decision.
- **Dynamics mismatch:** nominal double-integrator calibration; true A[0,0] is increased by 0, 0.005, 0.01 or 0.02; periodic reception; 30 fresh paired noise batches per law, 2,000 episodes per batch. The control input matrix and additive-noise law are unchanged. Only realized-schedule coverage is evaluated.
- **Certificates:** a fixed saturated proportional–derivative controller generates feedback trajectories. Bounds assess whether every position coordinate lies in [-1, 1]. The controller is identical across bound methods. Certificate availability and episodes with a false certificate are reported separately.

Methods are `all_window`, `observed_mask`, `age_union`, `primitive`, `constant`, and `gaussian_union`. Policies are `iid`, `bursty`, `severe`, `periodic`, `greedy`, and `all_schedules`. The last policy is the exact maximum over legal reset windows, equivalent to maximizing over all legal schedules for this observer.

The age cap forces reception after an age-L blackout. The Markov parameters describe the underlying dropout chain before this cap. Small-sample age-union thresholds remain infinite when the conformal order statistic requires an appended infinite value.

## Files

| File or directory | Purpose |
|---|---|
| `src/core.py` | Plant matrices, noise generation, reset-window recurrence, scale fitting, conformal quantiles, schedules, and competing bounds |
| `run_experiments.py` | Fixed primary/sensitivity configurations, parallel execution, certificate simulation, and result recording |
| `src/extensions.py` | Training-only shape alternatives, direct feedback rollout, observable censoring, and certificate metrics |
| `run_extensions.py` | Fresh-split common-event shape comparison and observable censoring |
| `run_mismatch_check.py` | Paired dynamics-mismatch negative control |
| `summarize_extensions.py` | Additional independent-fit intervals and paired comparisons |
| `experiment_protocol.json` | Fixed parameters and provenance of the additional experiments |
| `tests/test_extensions.py` | Observable-rule, independent observer identity, scale-invariance, propagated-shape, and mismatch checks |
| `results/extensions_raw/*.npz` | Training shapes, calibration scores, fitted bounds, packed decisions, gate ages, and seeds for the fresh study |
| `run_shift_check.py` | Paired noise-amplitude negative control |
| `run_gpu_check.py` | Optional independent GPU score computation and decision comparison |
| `summarize.py` | Repetition-level confidence intervals and paired comparisons |
| `verify_results.py` | Numerical-file integrity verification |
| `tests/test_core.py` | Explicit-dynamics identity, exhaustive schedule equivalence, quantile boundary, primitive-envelope, and age-cap tests |
| `results/*_metrics.csv` | Individual repetition/method/policy records |
| `results/*_summary.csv` | Means and 95% Student-t intervals across independent repetitions |
| `results/*_paired.csv` | Paired coverage differences and width reductions |
| `results/raw/*.npz` | Calibration scores, test maxima, fitted radii, packed coverage decisions, age histograms, and seeds |
| `results/*_environment.json` | Configuration lists, software versions, timing, and resource measurements |
| `results/gpu_check.json` | Recorded GPU validation outcome |
| `expected_checksums.json` | Reference hashes for deterministic numerical CSV outputs |
| `CITATION.cff` | Machine-readable attribution |
| `LICENSE` | MIT license |

## Record definitions

`coverage` is the fraction of complete test episodes whose required coordinates/times are all covered. `covered` gives the corresponding integer count. `width_uniform_age` averages full coordinate widths over ages 0…L after division by each coordinate's measurement-noise standard deviation. `width` instead weights ages by the evaluated schedule's occupancy; for `all_schedules`, it weights by the number of valid windows at each age.

`false_certificate` is the fraction of episodes with at least one incorrect safe-state certificate. `certified_fraction` is the fraction of time steps certified, and `safe_certified_fraction` is the fraction of truly safe time steps receiving a certificate. These fields apply to the primary realized schedules. `true_safe_fraction` reports the underlying safe-time fraction of the feedback trajectory.

Each compressed score record contains:

- `scale`: the independent-training age/coordinate shape;
- `cal_all`, `cal_observed`, `cal_primitive`: episode-level calibration scores;
- `bound_METHOD`: fitted age-by-coordinate radius arrays;
- `test_score_all_schedules`: each test episode's maximum standardized window error;
- `age_hist_POLICY`: counts of realized ages;
- `covered_POLICY_METHOD`: bit-packed episode containment decisions, decoded with `numpy.unpackbits(array)[:ntest]`;
- `seed`: the complete seed vector; and `seconds`: measured task runtime.

The main seed vector is `[20261119, plant_id, law_id, repetition, T, L, ntrain, ncal, ntest, group_id]`. `SeedSequence` spawns five streams for training noise, calibration noise, test noise, calibration masks, and test masks. Plant IDs follow integrator/oscillator/planar; noise IDs follow Gaussian/correlated-t/switching; group IDs are 0 for the primary experiment and 1 for sensitivity. The negative control uses a separately specified seed family in `run_shift_check.py`.

Confidence intervals describe Monte Carlo uncertainty across independently fitted calibrations. They are distinct from the conformal coverage guarantee, which is marginal over calibration and a new episode. Width reductions in paired files are `1 - all_window / comparator`; coverage differences are `all_window - comparator`.

## Additional-study conventions

The additional study was designed after the primary analysis and uses new training, calibration, and test random streams. Its seed is `[20261120, plant_id, law_id, repetition]`, spawning four streams for training, calibration, testing, and calibration masks. The dynamics-mismatch seed is `[20261121, law_id, repetition]`. These numerical seed labels do not encode execution dates. All fixed configurations and outcomes are retained.

In additional results, `all_window` denotes the RMS shape; `primitive_calibrated` is the propagated primitive shape recalibrated on actual window errors; `max_shape` pools training maxima across all times of equal age; `tail_shape` uses a coordinate/age-specific 95th percentile of training episode temporal maxima. The same coverage event and calibration sample size are used for these shapes. A recalibrated Gaussian-union multiplier cancels exactly and gives the RMS all-window result.

Summary rows with `plant=balanced` average the three plant-specific estimates within each repetition before calculating a 95% Student-t interval across 30 independent repetition means. In paired records, width values are `1 - all_window / comparator`; other values are `all_window - comparator`. Confidence intervals describe repeated-fit Monte Carlo variation, and are not conditional coverage guarantees for an individual calibrated table.
