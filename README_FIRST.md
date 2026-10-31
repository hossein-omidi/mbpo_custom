# SAC / MBPO — Updated Stage 5.1 diagnostic package

This update **does not change the training algorithm, physics, replay buffers, source checkpoints, or existing paper figures**. It corrects audit/test weaknesses identified in two external reviews and directly checks them on the provided data. It also bundles the scientific review and the earlier independent assessment.

## What changed

- `scripts/stage5_evidence_audit.py` now imports `scripts/stage5_additional_checks.py`. **Copy both files together**. It parses real UTC timestamps and enforces the pinned 12:05-start / 117-step / 5-minute grid, checks recorded cross-controller weather, validates the gross-only oracle separately, re-derives the design-weighted means and oracle gross result, and flags Stage 4/5 SAC training-history provenance mismatches. It summarizes approximate FIFO season coverage without calling it a bug.
- The audit now explicitly distinguishes Stage 4 **reported** shadow and checkpoint-action results from freshly re-executed tests. It marks `overall_status=AUDITED_WITH_REPRODUCIBILITY_FLAGS` when conflicts remain, and no longer prints a misleading unqualified PASS.
- Expanded `tests/test_stage5_evidence_audit.py` (20 tests), plus 14 original Stage 4 tests. Includes negative tests for invalid timestamps, wrong time intervals, non-UTC, weather mismatch, oracle gross integrity, nonzero movement cost, weighted summaries, mismatched alpha, and tampered saved Stage 4 claims.
- A separate `check_checkpoint_training_provenance.py` verifies an actual **trusted** local checkpoint against its training CSV and prints file SHA-256.

## Safe extraction and tests

From your repository root (the actual folder `~/projects/PVLIB_Paper/mbpo_custom`):

```bash
# Extract into a distinct, non-overwriting folder to preserve existing scripts:
mkdir -p stage5_1_review
unzip SAC_MBPO_STAGE5_1_CRITICAL_REVIEW_AND_AUDIT.zip -d stage5_1_review

# Run tests in the extracted folder, not your existing repository tests directory:
python -m pytest -q stage5_1_review/tests/test_stage5_evidence_audit.py
python -m pytest -q stage5_1_review/stage4_original/tests/test_sac_mbpo_deep_diagnostic.py
```

**Caution:** The first tests only audit static CSV/JSON behavior; the latter test package is not a proof of convergence and may require Torch/NumPy. No checkpoints are loaded by the Stage 5.1 audit.

## Run Stage 5.1 using your real local paths (no placeholders)

```bash
python stage5_1_review/scripts/stage5_evidence_audit.py \
  --deep "$PWD/SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip" \
  --eval-dir "$PWD/MBPO_workspace/evaluation/pvwatts_seed0_heldout1" \
  --sac-training-csv "$PWD/SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv" \
  --mbpo-training-csv "$PWD/MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv" \
  --outdir "$PWD/stage5_1_evidence_output" \
  --bundle-eval --bundle-plots
```

The file paths above reflect the previously reported folder names; **check existence first**. Stage 4 `SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip` must be in your repository root or its path must be adjusted.

Do not use sample shell paths like `/path/to/mbpo_custom` literally. If your ZIP is in Windows Downloads, locate it via WSL and supply its absolute path.

### Verify the exact checkpoint/history combination (highest priority)

```bash
python stage5_1_review/check_checkpoint_training_provenance.py \
  --checkpoint "$PWD/SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt" \
  --training-csv "$PWD/SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv" \
  --output "$PWD/stage5_1_sac_checkpoint_history.json"
```

**Only load checkpoints you created and trust.** The local checker calls `torch.load(weights_only=False)`, which is unsafe for untrusted serialized files.

Expected Stage 4 SAC checkpoint SHA-256: `749e28902aede9d64154bfb27431b3a47522f33f43e6e85015f51db1c28f1688`. If a different checkpoint SHA appears, this is **not** automatically a bug: it indicates a different file, and you must determine which checkpoint produced the evaluation.

For MBPO, use its own run path and compare the SHA to `69da37c2de95ce50e25f71a51181c2a89716e60fbc8d1d03346a3a04e178037a`.

## Inspect the included files

- `REVIEW_DECISIONS_STAGE5_1.md`: detailed accept/qualify/reject decisions on both reviewers and the paper-safe interpretation.
- `verified_example/stage5_audit.json`: full-data **recomputed** Stage 5.1 result using the supplied files. In particular it contains `training.sac.provenance_comparison.status=FAIL_CHECKPOINT_HISTORY_MISMATCH`.
- `verified_example/paired_evaluation_days.csv`: paired-day descriptive differences; official design weights are checked separately.
- `STAGE5_INDEPENDENT_SCIENTIFIC_DIAGNOSIS.md`: previous Stage 5 scientific assessment, included unchanged for provenance and comparison.
- `scripts/repo_preflight.py`: original repo completeness check. Run as `python stage5_1_review/scripts/repo_preflight.py --repo-root "$PWD" --output stage5_repo_preflight.json`.
- `scripts/print_verified_commands.py`: **prints only**, does not launch retraining; demonstrates the correct `--epochs` / `--power-model pvwatts_sapm` CLI. Its printed `--all-scenarios` evaluation example is **not** a promise of held-out testing or a final publication protocol.
- `stage4_original/`: original Stage 4 full checkpoint/physics diagnostic, intended for opt-in reruns on your trusted local data.

## Send back

Send the new `stage5_1_evidence_output/stage5_audit.json` and the SAC checkpoint-provenance JSON if the SAC mismatch persists. The full new evidence ZIP contains trajectories only if you specify `--bundle-eval`; otherwise it is compact.

Do **not** use this package to tune on the already-viewed final days and still call them unseen. Allocate a new untouched test set before any training-model ablation.
