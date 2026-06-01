# Source before mbpo run_local (do not execute directly).
#   source scripts/pv_cpu_env.sh
#   PV_CPU_PROFILE=dual source scripts/pv_cpu_env.sh
#
# Profiles for a 16-logical-CPU VM (adjust if nproc differs):
#   single — one training process
#   dual   — two Stage 0 runs in parallel (baseline + paper ablation)

PV_CPU_PROFILE="${PV_CPU_PROFILE:-single}"

_detected="$(nproc 2>/dev/null || echo 16)"
if [[ "$_detected" -lt 8 ]]; then
  echo "[pv_cpu_env] Warning: only ${_detected} CPUs detected; dual profile may be unsafe." >&2
fi

case "$PV_CPU_PROFILE" in
  dual)
    export CPUS="${CPUS:-6}"
    export TRIAL_CPUS="${TRIAL_CPUS:-2}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
    export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
    export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
    ;;
  single)
    export CPUS="${CPUS:-10}"
    export TRIAL_CPUS="${TRIAL_CPUS:-4}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
    export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
    export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
    ;;
  *)
    echo "[pv_cpu_env] Unknown PV_CPU_PROFILE=$PV_CPU_PROFILE (use single or dual)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

export GPUS="${GPUS:-0}"
export TRIAL_GPUS="${TRIAL_GPUS:-0}"

echo "[pv_cpu_env] profile=$PV_CPU_PROFILE nproc=${_detected} CPUS=$CPUS TRIAL_CPUS=$TRIAL_CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS"
