# PV Policy Quality Diagnostic v1.1 — paired Monte Carlo action comparisons

## Scope / provenance

This is a **read-only diagnostic** replacement for `pv_policy_quality_diagnostic.py` from v1. It does not change SAC or MBPO network parameters, optimizers, replay buffers, physics, evaluation results, or checkpoint contents. Backup the existing script if locally customized, then copy this script into the repository root. The Bash plotting workflow is unaffected.

## Why this correction is necessary

The v1 `critic_vs_conditional_mc_pair_order_agreement` ranks means calculated from only a handful of stochastic continuations. It does **not** account for the variability of *paired action differences*. Candidates are evaluated with identical stochastic seeds, so the correct diagnostic must compute the variance of `MC(candidate_A, seed_k) - MC(candidate_B, seed_k)` over independent `k`, instead of assuming candidate means are independently sampled. This avoids declaring small noisy reversals to be critic failures. Different commands that the real simulator clips into identical physical transitions must not be ranked as different decisions.

## New outputs

- `paired_candidate_differences.csv` — rows for each candidate pair, containing the signed critic-value difference, signed **paired** Monte Carlo difference, paired sample SD/SE, and a nominal 95% paired Student-t interval when `--repeats >= 20`.
- `quality_diagnostic.json` probe fields: `conditional_paired_mc` with resolved comparison count/agreement, and `n_physically_equivalent_candidate_pairs`.
- Legacy `candidate_scores.csv`, `critic_vs_conditional_mc_pair_order_agreement`, and existing CLI remain compatible; the legacy agreement remains *raw descriptive only*.

A `status` of `insufficient_stochastic_repeats` is intentional for `--repeats < 20`; the original raw rank number is **not** a verified comparison. `unresolved_conditional_difference` denotes a paired interval containing zero. `resolved_conditional_difference` denotes a nominal interval excluding zero, subject to unadjusted multiple comparisons, approximate t inference, and independent stochastic continuation seeds. `physically_equivalent_on_tested_day` excludes redundant actions due to physical clipping/identical outcomes from rankings.

These intervals describe **policy-action randomness conditional on the selected NSRDB weather day, policy checkpoint, starting state, and action set**. They do not describe variability across weather days or training seeds; they do not establish population-level critic accuracy or global policy optimality. With 20 repeats the t approximation is more stable than with 5 but still assumption-dependent. The file reports nominal intervals, *not* corrected familywise confidence statements.

## Suggested run

```bash
python pv_policy_quality_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --eval-dir MBPO_workspace/evaluation/scientific_v52_fresh \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --days 2024-12-23 2018-12-27 2021-07-02 \
  --steps 25 50 75 \
  --repeats 20 \
  --model-diagnostic \
  --outdir MBPO_workspace/evaluation/policy_quality_v11
```

The same random continuation seed is used across candidates at each decision point (common random numbers) and an independent seed is used for each stochastic repeat. The paired difference is computed **within each seed**. This is appropriate for the finite comparison set; it does not convert the selected scenarios into an independent test set.

## Focused tests

```bash
python -m unittest test_policy_quality_diagnostic.py -v
```

Eleven synthetic tests verify exact replay, basic SAC soft-return identities, paired noise cancellation, physical-action equivalence, short-sample abstention, and non-resolved intervals.

## Findings from the user's uploaded diagnostic

The provided JSON records **three days, steps 25/50/75, 5 stochastic continuations per candidate**, so current raw ranking percentages should not be treated as decisive. The model one-step forecast was only slightly more accurate than persistence on 2024-12-23 and 2018-12-27 and slightly less accurate on 2021-07-02. No code-level sign or action-boundary bug follows from those results. Independent validation, checkpoint progression, and model-real-only ablations remain needed before claiming a SAC/MBPO training fault or modifying core training algorithms.
