# Geometry10 FINAL — merged, in-place upgrade

**Target:** `hossein-omidi/mbpo_custom`, `PostProcessing`, audited commit
`92a1b00e5a0cbd5f770514396c0a347f5b9aca6d` (or a descendant with recognized source files).

This single installer supersedes both `PV_geometry10_patch` (V1) and
`PV_geometry10_inplace_v2` (V2). V1 created a sibling directory and never changed
its original Git checkout. V2 applied the source changes in place but required a
clean original revision. FINAL retains the full V2 geometry-aware implementation,
adds a migration mechanism for exact V1/V2 source versions and strengthens model
validation. It never creates another workspace or repository, creates backup
copies of your checkout, modifies the index, commits, pushes, deletes datasets,
or copies old experiments. The internal `_previous` directory contains only
release-recognition templates; it is not a backup of your project and is not
installed as runtime source. Do not run the old installers again.

## 1. Install in the original checkout (not the sibling)

Extract the ZIP into the original repository, normally:

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
# Verify you are in the intended existing checkout:
git rev-parse --show-toplevel
git branch --show-current
git status --short
```

The top-level path must be the actual repository you wish to update. If it is
`~/projects/PVLIB_Paper` instead of `~/projects/PVLIB_Paper/mbpo_custom`, stop:
you are using a parent repository, not the audited checkout. FINAL refuses a
nested directory lacking its own `.git` marker. It also refuses the V1 sibling
if that directory is merely an archived copy without its own Git metadata.

Activate your existing Python environment, then run:

```bash
python PV_geometry10_final/install.py --root . --dry-run
python PV_geometry10_final/install.py --root . --apply

git status --short
git diff --stat
git diff --check
```

The dry run identifies each target as base, v1, v2, final, or multiple equivalent
versions. It generates every postimage independently from the pinned base blobs
rather than applying two patches in succession. It compiles the proposed Python
files and checks the complete Git patch before writing. If all files already
match FINAL, it exits successfully without writing them again.

Supported working-tree states:

- Original audited checkout (no Geometry10 changes yet).
- Exact V1 source changes copied into the original checkout.
- Exact V2 source changes already present in the original checkout.
- A mixture of those recognized file versions, including committed or staged
  earlier changes on a descendant of the audited commit.
- Already installed FINAL (idempotent no-op).

Unrecognized target contents, unknown staged contents, symlinks, ignored new
source files, incompatible index flags, unresolved index conflicts, and changes
in the eight pinned critical callers cause refusal before applying. Other local
changes outside the audited targets/callers are preserved. Files are compared
byte-for-byte, so a checkout with CRLF conversion or custom edits may be refused;
do not force installation or reset your work to bypass this. Inspect the named
file and reconcile it deliberately. Use the audited LF source version if you
know it is safe. A shallow clone must contain the audited base Git object.

No automatic `git add` is performed. New source files appear as `??` in
`git status` until staged. `git diff --stat` shows tracked edits only; inspect and
stage the intended source files after verification. You do not need to commit the
installer archive or extracted installer directory. The previously created
`mbpo_custom_geometry10` sibling is left untouched; FINAL does not migrate its
outputs or checkpoints. Keep it if it contains useful data. Do not delete it
until you have checked its contents.

## 2. Final technical contract

The active managed Torch SAC/MBPO configuration is
`examples/config/pv_tracking/conf_geometry10.py`; the old compact reference
configuration remains for provenance. `mbpo/geometry10.py` defines the shared
10-feature contract:

| Index | Feature | Encoding |
| ---: | --- | --- |
| 0 | DNI | DNI / 2000 |
| 1 | DHI | DHI / 2000 |
| 2 | measured DC power | P / 2000 |
| 3 | panel tilt | beta / 90 |
| 4–5 | panel azimuth | sin, cos |
| 6 | solar zenith | z / 180 |
| 7–8 | solar azimuth | sin, cos |
| 9 | remaining horizon | (H-t)/H |

Current power remains a causal-in-time observation, not future reward leakage.
It provides feedback about omitted conditions and the reference for power-delta
prediction. The schema rejects out-of-range physical power rather than silently
saturating it. This is a compact observation in a partially observable task, not
proof of a fully Markovian or globally optimal state.

SAC actor: 10 inputs, 2 actions. Twin critic: 12 inputs. Real and synthetic replay
store 10D states. MBPO weather model: 6 inputs (DNI, DHI, solar geometry, tau) and
2 outputs (DNI/DHI deltas). Power model: 14 inputs (current state, action,
next irradiance) and one power delta output. Actual next irradiance is used only
as a conditional input when fitting the supervised power model; simulated
rollouts use predicted irradiance. Next solar geometry is calculated from the
scenario ID, episode clock and physical site's ephemeris, not future measured
weather. The original sticky whole-day model holdout and train-only rollout
sampling remain; metadata is aligned to each sampled replay row.

Panel tilt clipping, circular azimuth, terminal time, actuator displacement and
net reward are reconstructed analytically. Predicted weather/power are bounded;
raw non-finite predictions are rejected before clipping. FINAL additionally
validates model-fit observation/action batches, refuses invalid synthetic
validation/counterfactual rows and inconsistent terminal flags, and strengthens
checkpoint replay bounds/indices/terminal validation. The weather and power
models are probabilistic approximations; exact geometry reconstruction does not
guarantee their predictive skill or improved policy performance.

Unchanged: physical power backend, historical data, actuator parameters,
net-energy objective, action dimension, 195-step horizon, 5-minute interval,
reward scale, discount, network hidden widths, learning hyperparameters, real
ratio, one-step rollout length and experiment split/statistical methodology.
Existing managed CLI commands continue using the new default. The native compact
mode remains in PVTrackingEnv, but the managed Torch stack is Geometry10-only.
Old six-input policies and previous dynamics checkpoints cannot be loaded by the
new stack. The stricter metadata/source fingerprints can also reject V1/V2
checkpoints; use a fresh experiment ID and retrain rather than weakening guards.
Existing artifacts stay on disk. To reproduce old compact experiments use the
original Git revision and matching environment/configuration.

## 3. Coverage and verification

The plan writes 24 target files (19 transformed existing sources, one generated
configuration, shared schema, two runtime tests and this documentation). It
checks 20 original Git blob hashes plus eight unchanged managed-pipeline callers.
All 28 hashes were compared with the pinned public GitHub tree. The two uploaded
archives were inspected and their earlier postimages retained for recognition.

Executed in the packaging environment: 11 numerical schema tests and 14 installer
fixture tests (25 total). Fixtures cover clean install, V1/V2 migration, mixed
versions, descendant commits, staged versions, idempotence, unrelated local edits,
conflicts, symlinks, ignored files, wrong root and dependency drift. These are not
a substitute for installing against the actual repository and running the
physical simulator. The packaging environment could not clone GitHub and lacked
Gym/PVLib, so the full source transform, training, checkpoint loading and
representation pipeline have not been executed end-to-end here.

Run after installation, using your project environment and existing prepared
NSRDB catalogue:

```bash
python -m pytest -q geometry10_tests
```

The six managed integration tests exercise shared configuration, actor/critic
optimization, replay/checkpoints, ensemble fitting and mixing, CLI metadata,
paired compact-vs-Geometry10 physical transitions, and invalid synthetic outputs.
The 11 numerical tests cover circular encoding, feature order/bounds, power
normalization, integer terminal time, scenario context, and action clipping.
The installer fixture tests are separately runnable with:

```bash
python -m pytest -q PV_geometry10_final/tests/test_installer.py
```

Do not begin expensive training if installation or any integration test fails.
Several historical tests and standalone workflows still assume six inputs; the
pinned repository lacks `SAC_workspace/src/evaluation/evaluation.py`, required by
some legacy standalone evaluation workflows. This patch targets the current
managed Torch pipeline, not the archived TensorFlow/Ray routes. No claim is made
that every historical test or script is now compatible.

## 4. Train and evaluate using existing commands

Use a **new experiment name**, even if an earlier Geometry10 pilot exists.

```bash
python -m PV_experiments.train --name geom10_final_pilot --seeds 0:1 --epochs 100
python -m PV_experiments.evaluate --name geom10_final_pilot --mode smoke --draws 3
python -m PV_experiments.evaluate --name geom10_final_pilot --mode validation --draws 3
python -m PV_experiments.evaluate --name geom10_final_pilot --mode model
python -m PV_experiments.represent --name geom10_final_pilot --kind all
python -m PV_experiments.advanced_report --name geom10_final_pilot --kind all
```

The model-validation step requires a fitted ensemble. Check the MBPO model skill
and accepted synthetic transition counts; a run with no accepted synthetic
experience is not evidence of useful model-based learning. Pilot results are for
integration and preliminary behavior, not final conclusions. Preserve the held-out
test split until the selected observation/configuration and evaluation protocol
are frozen. No assertion is made that Geometry10 necessarily outperforms the
astronomical controller.
