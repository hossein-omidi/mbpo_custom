# PV tracking failure — root-cause ranking (Stage 0 eval)

**Checkpoint:** `seed:6923_2026-05-24_20-45-32_az5ovrd/best_eval_checkpoint`  
**Eval:** `evaluation/pv_stage0_single_day/` — date `2020-06-21`, clearsky, `movement_penalty=0`, `--deterministic`

This note ranks the verified causes for the current checkpoint after code review
plus the deterministic-vs-stochastic ablation. It separates **primary failure
mechanisms** from **secondary contributors** and from **observed outcomes**.
“Confidence” = probability this cause is **active in this run** (not mutually
exclusive; several apply at once).

---

## Measured outcome (definitions)

| Symbol | Definition | Learned | Sun | Fixed |
|--------|------------|---------|-----|-------|
| \(G\) | \(\sum_t r_t\) ≈ total energy (kWh) | **1.427** | **1.712** | **1.375** |
| \(\rho_G\) | \(G_{\text{learned}}/G_{\text{sun}}\) | **0.834** | 1 | — |
| \(\bar{A}\) | mean \(\|a_0\|+\|a_1\|\) on steps with altitude ≥ 5° | **0.068** | **1.279** | 0 |
| \(\rho_A\) | \(\bar{A}_{\text{learned}}/\bar{A}_{\text{sun}}\) | **0.053** | 1 | — |
| \(\bar{E}\) | mean \(\|\text{tilt}-\text{zenith}\|\) (deg) | **15.03** | **4.60** | — |

All 10 learned rollouts are **identical** because eval is deterministic on the same fixed-day MDP.
This does **not** imply \(\sigma=0\); deterministic eval bypasses the sampled noise term.

---

## Root-cause ranking

1. **RC3 — Wrong-signed tilt mean control**: highest-priority mechanism. The
   learned mean tilt action often moves **away** from the greedy zenith-tracking
   direction.
2. **RC2 — Deploy mean action magnitude too small**: even when the sign is
   correct, `tanh(μ)` is much too small to cover the day’s zenith swing.
3. **RC5 — Training contract mismatch**: the checkpoint was trained on Stage 1
   summer random-day data, then judged against the Stage 0 single-day proof.
4. **RC1 — Deterministic deploy uses `tanh(μ)` rather than sampled actions**:
   real mechanism, but the new ablation shows it is **not** the main limiter for
   this checkpoint.
5. **RC4 — Local optimum / quasi-fixed behavior**: useful description of the
   failure mode, but not a separate upstream bug.

The short conclusion is the one you stated: the dominant problem is the
**learned mean policy itself**, especially its **tilt sign** and **scale**.

---

## RC1 — Deterministic deploy is real, but not the main limiter here

**Confidence: 100% for the mechanism; secondary causal importance for this checkpoint**

### Mechanism (proved from code)

Policy class: `FeedforwardGaussianPolicy` with `squash=True` (`softlearning/policies/gaussian_policy.py`):

- Training sample: \(a = \tanh(\mu(s) + \sigma(s)\odot\varepsilon)\), \(\varepsilon\sim\mathcal{N}(0,I)\)
- Eval (`--deterministic`, `eval_deterministic=True`):  
  \[
  \pi_{\text{eval}}(s) = \tanh(\mu_\theta(s))
  \]

`mbpo` training eval uses the same rule (`softlearning/algorithms/rl_algorithm.py` → `set_deterministic(self._eval_deterministic)`).

### Corollary (not optional)

Off-policy SAC optimizes \(\mathbb{E}_{a\sim\pi}[\cdot]\) under the **stochastic** policy.  
**Deployment quality is a functional of \(\mu\) only.** High training `policy/actions-std` does **not** imply large eval actions.

### Evidence (this trial, epoch 29, `progress.csv`)

| Metric | Value |
|--------|-------|
| `policy/shifts-mean` | **−0.0098** (signed batch-average pre-tanh mean; near zero) |
| `policy/log_scale_diags-mean` | **−0.148** (\(\exp(\cdot)\approx 0.86\): still substantial unsquashed std) |
| `policy/actions-mean` | **−0.033** (signed sampled batch mean; **not** action magnitude) |
| `policy/actions-std` | **0.593** (stochastic spread) |
| Eval rollout \(\bar{A}\) | **0.068** |

### Ablation (same checkpoint)

| Eval mode | Mean reward / energy (kWh) | Interpretation |
|-----------|----------------------------|----------------|
| Deterministic (`tanh(μ)`) | **1.4274** | Canonical deploy policy |
| Stochastic (`tanh(μ + σ\varepsilon)`) | **1.3653 ± 0.0956** | Reintroducing sampling did **not** rescue performance |

If “lost exploration at test time” were the main blocker, stochastic evaluation
should have improved sharply. It did not. For this checkpoint the issue is **not**
“\(\sigma\) collapsed to zero”; it is a **weak mean policy with still substantial
stochasticity**.

**Conclusion:** RC1 is mathematically real, but the ablation demotes it from
“main cause” to **secondary contributor**. The mean policy \(\mu\) is the real
problem to fix.

**Do not say:** “mode of the Gaussian” — the eval action is **not** \(\mathbb{E}[a]\); it is **\(\tanh(\mu)\)** with no noise.

---

## RC2 — Deploy mean action magnitude is far below control authority

**Confidence: 100%** (primary mechanism; direct measurement)

### Mechanism

Env (`mbpo/env/pv_tracking.py`):

\[
\Delta\text{tilt}_t = 5^\circ \cdot a^{(\text{tilt})}_t,\quad a^{(\text{tilt})}_t\in[-1,1]
\]

Sun tracker saturates \(|a|\approx 1\) when \(|\text{zenith}-\text{tilt}|>5^\circ\) (`scripts/eval_utils.py` rate limit).

Learned: \(|a^{(\text{tilt})}|\approx 0.07 \Rightarrow |\Delta\text{tilt}|\approx 0.35^\circ/\text{step}\).

Over \(T=39\) steps, net tilt change **≈ −6.8°** (30° → 23°) vs zenith swing **≈ 57°** (69° → 12° → 55° on this grid).

### Necessary condition for sun tracking (order-of-magnitude)

Let \(D = \max_t |\text{zenith}_t - \text{tilt}_0|\). Greedy tracking needs \(\sum_t |\Delta\text{tilt}_t| \gtrsim D\).  
With rate limit 5°/step, need \(\bar{|a^{(\text{tilt})}|} \gtrsim D/(5T)\). For \(D\sim 57^\circ\), \(T=39\): \(\bar{|a|}\gtrsim 0.29\).  
Observed **0.07** → **impossible** to reach sun trajectory without changing \(\mu\) scale.

### SAC temperature (supporting, not primary)

`target_entropy = 0` pushes SAC toward lower entropy; `alpha = max(exp(log α), min_alpha)` with `min_alpha=0.35` (`mbpo/algorithms/mbpo.py` line 743).

Final `alpha = 0.35` exactly → **temperature floor**, not unbounded collapse.

**Correction:** α at floor **does not** by itself prove “entropy collapse”; it proves **you cannot reduce α further**. The actionable fact is **small \(\mu\)**, not small α alone.

---

## RC3 — Wrong-signed tilt mean control (highest-priority mechanism)

**Confidence: 100%** (primary mechanism)

### Greedy one-step rule (sun baseline semantics)

Target \(\text{tilt}^\star_t = \text{zenith}_t\). Error \(e_t = \text{tilt}_t - \text{zenith}_t\).  
For \(|e_t|>5^\circ\), optimal direction: \(\text{sign}(a^{(\text{tilt})}_t) = -\text{sign}(e_t)\).

### Step 0 (productive sun, same MDP)

| | zenith | tilt | \(e\) | required \(\text{sign}(a)\) | learned \(a\) | sun \(a\) |
|---|--------|------|-------|---------------------------|---------------|-----------|
| t=0 | 69.17° | 29.66° | **−39.5°** | **+** | **−0.068** | **+1.0** |

### Full-episode statistics (`rollout_1.csv`, altitude ≥ 5°)

| Test | Result |
|------|--------|
| Historical sign agreement with greedy rule | **8 / 38 = 21%** |
| Historical \(\text{corr}(e_t, a^{(\text{tilt})}_t)\) | **+0.73** |

Positive correlation means: when tilt is **below** zenith (\(e<0\)), learned tends **negative** \(a\) — **worsening** misalignment. This is **not** “cautious tracking”; it is **incorrect closed-loop direction**.

The diagnostic implementation now computes this from the **pre-action
observation** (the state that actually produced the action), which is the
causally correct definition. Re-run `diagnose_tracking.py` to refresh the exact
numbers under the corrected diagnostic; the ranking does not change.

### Why energy-only RL does not forbid this

Reward \(r_t = \text{energy}_t(\text{tilt}_t,\text{az}_t,\text{sun}_t)\) with no term in \(\|\text{tilt}-\text{zenith}\|\).  
Observations include zenith, but **no loss term** enforces \(a \propto -\text{sign}(e)\). Wrong-sign \(\mu\) can be consistent with local Q improvements.

---

## RC4 — Local optimum description, not a separate upstream bug

**Confidence: 100%** (outcome characterization)

### Statement

There exists a policy with **small, wrong-signed** tilt motion such that:

\[
G_{\text{fixed}} < G_{\text{learned}} < G_{\text{sun}}
\]

Measured: **1.375 < 1.427 < 1.712** kWh on the **same** day, weather, seeds.

### Interpretation

- **Not** a pvlib bug, reset bug, or action scaling bug (CSV consistency **PASS**).
- **Not** “eval unfair to sun” (same `get_eval_environment` kwargs; baselines use same incremental actions).
- A **quasi-fixed** pose near 24–30° can yield moderate POA on 2-axis (azimuth matters; high `cos_aoi` at peak **does not** imply small \(|\text{tilt}-\text{zenith}|\)).

Pure SAC on \(r_t=\text{energy}_t\) has **no guarantee** of converging to the sun-tracking policy without imitation, shaping, or architectural bias.

---

## RC5 — Training MDP / protocol ≠ Stage 0 eval (this checkpoint)

**Confidence: 100%** for this checkpoint; **0%** as cause of eval unfairness

### Training (`params.json`)

| Field | Value |
|-------|--------|
| `config_version` | `pv_tracking_stage1_clearsky_explore_2026-05-24` |
| `n_epochs` | **30** (not 150) |
| `randomize_day` | **true** (2020-06-01 … 2020-08-31) |
| `eval_deterministic` | true |

### Eval (`evaluation_summary.json`)

| Field | Value |
|-------|--------|
| `fixed_eval_dates` | `["2020-06-21"]` |
| `randomize_day` | **false** |

**Conclusion:** Policy optimized for **average summer-day** return + in-training eval on random days; holdout is **one fixed day**. Eval protocol is fair; **training contract** did not match Stage 0 stationary proof.

`real_ratio=1.0` removes model-rollout bias; it **does not** fix RC1–RC4.

`best_eval_checkpoint` maximizes `evaluation/return-average` (~**1.41**), not sun ratio on 2020-06-21 (~**1.43** measured offline).

---

## Ruled out

| Hypothesis | Why |
|------------|-----|
| Wrong action units / scaling | \(\Delta\text{tilt}=5^\circ a_0\) verified in CSV |
| No env exploration | Actions in \([-1,1]\); sun uses full box |
| Multi-day chained episode | 39 steps, single date per episode |
| Cloud / season noise at eval | Clearsky, one date |
| “Entropy collapsed to zero” as primary story | α floored at 0.35, but training `actions-std` ≈ 0.59 and \(\exp(\text{log_scale mean})\approx 0.86\) |

**Diagnostic bug (fixed):** `verify_paired_fairness` used hardcoded `movement_penalty=0.0001`; Stage 0 uses **0**. Energy metrics were always valid.

---

## Causal summary (one paragraph)

The pipeline implements SAC correctly, but this checkpoint learned a **bad mean
policy**: its deploy action magnitude is far too small and its tilt direction is
often wrong-signed relative to the greedy zenith tracker. Deterministic deploy
using \(\tanh(\mu)\) is a real SAC property, but the new ablation shows that
turning stochastic sampling back on does **not** fix the checkpoint. That means
the core problem is the learned \(\mu\) itself, not just missing test-time
noise. Training was also **Stage 1 / 30 epochs / random summer days**, not the
Stage 0 stationary proof, so even a mathematically correct SAC run was being
asked to pass a different gate than the one it was optimized for.

---

## Principled fixes (ordered)

1. Retrain from the **Stage 0 single-day contract** first, then Stage 1; verify `params.json` before long runs.
2. Use **`physical` observation mode** as the canonical training/deployment contract; keep `legacy` only as an explicit ablation.
3. Use principled SAC temperature defaults (`target_entropy='auto'`, no manual `min_alpha` floor) unless an ablation justifies overriding them.
4. Gate on **sun** (`diagnose_tracking.py --gate`, energy ratio ≥ 0.95).
5. **BC / demonstrations** (`collect_sun_demonstrations.py`) — strongest direct fix for RC3.
6. Optional: auxiliary loss on \(\cos(\text{AOI})\) or \(|\text{tilt}-\text{zenith}|\) (changes objective).
7. Do **not** expect stochastic eval or `min_alpha` alone to fix a bad mean policy.
