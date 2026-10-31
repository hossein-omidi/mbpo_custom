SAC / MBPO -- ONE-FILE AUTOMATIC PROJECT DIAGNOSIS
================================================

WHY YOU GOT "file or directory not found"
The earlier commands expected a directory named "stage5_1_review" and scripts under it. Those files were not at that location. No SAC or MBPO failure was detected by those errors.

THIS REPLACEMENT DOES NOT REQUIRE THE OLD STAGE 5.1 ZIP.

Step 1. Download sac_mbpo_collect.py from the ChatGPT response.
Step 2. Open your WSL terminal and enter:
        cd ~/projects/PVLIB_Paper/mbpo_custom
        explorer.exe .
     Windows Explorer will open your WSL project folder. Copy the downloaded
     sac_mbpo_collect.py file into that folder (not in any subdirectory).
Step 3. Confirm it is there:
        ls -l sac_mbpo_collect.py
Step 4. Run exactly one Python command:
        python sac_mbpo_collect.py
Step 5. Upload this generated file back to ChatGPT:
        sac_mbpo_collect_output/SEND_ME_SAC_MBPO_PROJECT_DIAGNOSTIC.zip
     If the output directory already exists, the tool will use
     sac_mbpo_collect_output_1, _2, and so on, and print the exact location.

OPTIONAL: Run `python sac_mbpo_collect.py --self-test` to check the collector.
You do NOT need to run pytest or any older Stage 5 commands.

WHAT IT COLLECTS
- Git commit and branch (without remote credentials or shell history)
- Names/sha256 of selected research code files (and copies only these .py/.json files)
- Both SAC run folders, all discovered "latest.pt" files, their hashes and
  checkpoint configuration/optimizer metadata where readable
- Their actual "episodes.csv" histories and a per-run consistency check
- MBPO run checkpoints and training histories
- Evaluation summaries, paired episode CSVs and scenario-selection files
- A comparison against Stage 4's previously reported checkpoint hashes/alpha
- Clear warnings if source files, checkpoints, or training data disagree

PRIVACY / SAFETY
- Read-only for existing project: no training, gradient steps, physics calls,
  checkpoint writes, or changes to existing code.
- It creates a NEW output folder containing a ZIP and metadata.
- It does not bundle checkpoint weights, weather files, .env, secrets, or .git.
- By default it does bundle specifically allowlisted research Python/JSON code.
  Review ZIP contents before sharing. If you do not want to share source code:
       python sac_mbpo_collect.py --no-source
- By default, it loads your own local .pt checkpoints using PyTorch pickle
  so it can compare saved alpha, epoch, and optimizer updates. DO NOT run this
  on checkpoints obtained from untrusted people. For hash-only inspection:
       python sac_mbpo_collect.py --no-load-checkpoints
- If PyTorch is missing or loading fails, the collector still produces an
  evidence ZIP with hashes and CSV summaries, marking that check inconclusive.
- Detailed trajectory CSVs are NOT bundled by default (already sent in Stage 5).
  If you need to send them again:
       python sac_mbpo_collect.py --include-trajectories

DATA INTERPRETATION
- Comparing alpha and steps confirms/dismisses specific provenance concerns;
  it does NOT prove critic convergence, correct SAC optimization, or MBPO advantage.
- If your checkpoint hash differs from Stage 4, that can simply mean `latest.pt`
  was replaced by another trained model. Its own episodes.csv might still match.
- We should NOT alter replay size, alpha, reward normalization or actor/critic
  code until we know which run generated the plotted evaluation.
