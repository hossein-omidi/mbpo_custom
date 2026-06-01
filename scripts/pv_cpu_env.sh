# Source before mbpo run_local (do not execute directly).
#   source scripts/pv_cpu_env.sh
#   PV_CPU_PROFILE=full source scripts/pv_cpu_env.sh   # one job, 16-thread VM
#   PV_CPU_PROFILE=dual source scripts/pv_cpu_env.sh   # two jobs at once (not recommended now)

PV_CPU_PROFILE="${PV_CPU_PROFILE:-full}"

_detected="$(nproc 2>/dev/null || echo 16)"
if [[ "$_detected" -lt 8 ]]; then
  echo "[pv_cpu_env] Warning: only ${_detected} CPUs detected." >&2
fi

case "$PV_CPU_PROFILE" in
  full|single)
    # One training process on a 16-logical-CPU machine: leave headroom for Ray/OS.
    export CPUS="${CPUS:-14}"
    export TRIAL_CPUS="${TRIAL_CPUS:-6}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-6}"
    export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-6}"
    export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
    ;;
  dual)
    export CPUS="${CPUS:-6}"
    export TRIAL_CPUS="${TRIAL_CPUS:-2}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-2}"
    export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
    export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
    ;;
  *)
    echo "[pv_cpu_env] Unknown PV_CPU_PROFILE=$PV_CPU_PROFILE (use full or dual)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

export GPUS="${GPUS:-0}"
export TRIAL_GPUS="${TRIAL_GPUS:-0}"

echo "[pv_cpu_env] profile=$PV_CPU_PROFILE nproc=${_detected} CPUS=$CPUS TRIAL_CPUS=$TRIAL_CPUS OMP=$OMP_NUM_THREADS"
