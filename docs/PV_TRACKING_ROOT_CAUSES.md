# PV tracking failure — verified root causes (Stage 0 eval)

**Checkpoint:** `seed:6923_2026-05-24_20-45-32_az5ovrd/best_eval_checkpoint`  
**Eval:** `evaluation/pv_stage0_single_day/` — date `2020-06-21`, clearsky, `movement_penalty=0`, `--deterministic`

This document states five root causes with **code-backed mechanisms** and **measured evidence**.  
“Confidence” = probability this cause is **active in this run** (not mutually exclusive; several apply at once).

---

## Measured outcome (definitions)

| Symbol | Definition | Learned | Sun | Fixed |
|--------|------------|---------|-----|-------|
| \(G\) | \(\sum_t r_t\) ≈ total energy (kWh) | **1.427** | **1.712** | **1.375** |
| \(\rho_G\) | \(G_{\text{learned}}/G_{\text{sun}}\) | **0.834** | 1 | — |
| \(\bar{A}\) | mean \(\|a_0\|+\|a_1\|\) on steps with altitude ≥ 5° | **0.068** | **1.279** | 0 |
| \(\rho_A\) | \(\bar{A}_{\text{learned}}/\bar{A}_{\text{sun}}\) | **0.053** | 1 | — |
| \(\bar{E}\) | mean \(\|\text{tilt}-\text{zenith}\|\) (deg) | **15.03** | **4.60** | — |

All 10 learned rollouts are **identical** (deterministic policy, \(\sigma=0\) across seeds).

---

## RC1 — Eval deploys squashed mean, not training samples

**Confidence: 100%** (mechanism always true at eval; explains train/eval action gap)

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
| `policy/actions-mean` | **−0.033** (typical \(\|a\|\) scale in training batches) |
| `policy/actions-std` | **0.593** (stochastic spread) |
| Eval rollout \(\bar{A}\) | **0.068** |

**Conclusion:** Exploration during training **does not survive** in \(\pi_{\text{eval}}\). This is by construction, not env limitation.

**Do not say:** “mode of the Gaussian” — the eval action is **not** \(\mathbb{E}[a]\); it is **\(\tanh(\mu)\)** with no noise.

---

## RC2 — Mean action magnitude far below control authority

**Confidence: 100%** (direct measurement)

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

`target_entropy = 0` drives \(\mathcal{H}(\pi)\to 0\); `alpha = max(exp(log α), min_alpha)` with `min_alpha=0.35` (`mbpo/algorithms/mbpo.py` line 743).

Final `alpha = 0.35` exactly → **temperature floor**, not unbounded collapse.

**Correction:** α at floor **does not** by itself prove “entropy collapse”; it proves **you cannot reduce α further**. The actionable fact is **small \(\mu\)**, not small α alone.

---

## RC3 — Wrong-signed tilt control (independent of magnitude)

**Confidence: 95%** (stepwise sign test + correlation)

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
| Sign agreement with greedy rule | **8 / 38 = 21%** |
| \(\text{corr}(e_t, a^{(\text{tilt})}_t)\) | **+0.73** |

Positive correlation means: when tilt is **below** zenith (\(e<0\)), learned tends **negative** \(a\) — **worsening** misalignment. This is **not** “cautious tracking”; it is **incorrect closed-loop direction**.

### Why energy-only RL does not forbid this

Reward \(r_t = \text{energy}_t(\text{tilt}_t,\text{az}_t,\text{sun}_t)\) with no term in \(\|\text{tilt}-\text{zenith}\|\).  
Observations include zenith, but **no loss term** enforces \(a \propto -\text{sign}(e)\). Wrong-sign \(\mu\) can be consistent with local Q improvements.

---

## RC4 — Local optimum of the energy landscape

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
| “Entropy collapsed to zero” as primary story | α floored at 0.35; training `actions-std` ≈ 0.59 |

**Diagnostic bug (fixed):** `verify_paired_fairness` used hardcoded `movement_penalty=0.0001`; Stage 0 uses **0**. Energy metrics were always valid.

---

## Causal summary (one paragraph)

The pipeline implements SAC correctly, but **evaluates \(\tanh(\mu)\)** while **training samples** use \(\tanh(\mu+\sigma\varepsilon)\). For this checkpoint, \(\mu\) is **small** (\(\rho_A\approx 0.05\)) and **misaligned in sign** with the greedy zenith tracker (\(\text{corr}(e,a)>0\)), producing a **local energy maximum** above fixed and far below sun. Training was **Stage 1 / 30 epochs / random summer days**, not Stage 0 stationary — so even perfect SAC on the training MDP would not automatically pass the June-21 gate without retraining or imitation.

---

## Principled fixes (ordered)

1. Retrain `stage0_single_day.py` (150 epochs, `randomize_day=False`); verify `params.json` before long runs.
2. Gate on **sun** (`diagnose_tracking.py --gate`, energy ratio ≥ 0.95).
3. **BC / demonstrations** (`collect_sun_demonstrations.py`) — strongest fix for RC3.
4. Optional: auxiliary loss on \(\cos(\text{AOI})\) or \(|\text{tilt}-\text{zenith}|\) (changes objective).
5. Do **not** expect `min_alpha` alone to fix wrong sign when α is already at floor.
