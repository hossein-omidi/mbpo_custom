#!/usr/bin/env bash
# Portable, offline modern SAC/MBPO statistical plots. No Ray/TF/Conda activation.
# Usage: ./plot_modern_results.sh <evaluation_directory> [--outdir <path>]
#        [--mbpo-training-csv <path>] [--sac-training-csv <path>]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: ./plot_modern_results.sh <evaluation_directory> [--outdir DIR] [--mbpo-training-csv CSV] [--sac-training-csv CSV]' >&2
  exit 2
fi
cd "$ROOT"
exec "${PYTHON:-python}" -m MBPO_workspace.workflows.06_plot_suite --eval-dir "$1" "${@:2}"
