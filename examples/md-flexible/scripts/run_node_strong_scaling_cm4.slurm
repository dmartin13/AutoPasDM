#!/bin/bash
#SBATCH -J dap-node-scaling
#SBATCH -D ./
#SBATCH -o ./dap-node-scaling.%j.out
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

set -euo pipefail

module purge

module load slurm_setup
module load stack/24.4.0
module load llvm/19.1.3
module load gcc/13.2.0
module load openmpi/4.1.8-gcc13

# Load the same compiler / MPI environment that was used to build both binaries.
# For Intel MPI on CoolMUC-4 this is typically, for example:
# module load intel-toolkit

if [[ $# -ne 2 ]]; then
  echo "Usage: sbatch $0 <AutoPas-source-root> <DistributedAutoPas-source-root>"
  echo
  echo "Both projects are expected to have been built in a 'build/' directory."
  exit 2
fi

AUTOPAS_ROOT=$(realpath "$1")
DAP_ROOT=$(realpath "$2")

REFERENCE="${AUTOPAS_ROOT}/build/examples/md-flexible/md-flexible"
DISTRIBUTED="${DAP_ROOT}/build/examples/md-flexible/md-flexible"
RUNNER="${DAP_ROOT}/examples/md-flexible/scripts/run_strong_scaling.py"
CONFIG="${DAP_ROOT}/examples/md-flexible/input/dap_performance_node_strong_scaling.yaml"

for file in "$REFERENCE" "$DISTRIBUTED" "$RUNNER" "$CONFIG"; do
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file"
    exit 2
  fi
done

RESULT_DIR="${SLURM_SUBMIT_DIR}/node_strong_scaling_cm4_${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

# One AutoPas instance / MPI rank per node. AutoPas uses all physical cores of
# the node through OpenMP. Disable dynamic thread counts and bind threads to
# physical cores for reproducible measurements.
export OMP_NUM_THREADS=112
export OMP_DYNAMIC=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "CoolMUC-4 hybrid node strong-scaling benchmark"
echo "  allocated nodes       : ${SLURM_JOB_NUM_NODES}"
echo "  MPI ranks per node    : 1"
echo "  OpenMP threads / rank : ${OMP_NUM_THREADS}"
echo "  reference             : ${REFERENCE}"
echo "  distributed           : ${DISTRIBUTED}"
echo "  config                : ${CONFIG}"
echo "  results               : ${RESULT_DIR}"
echo

# Here MPI-rank count equals node count. Each srun job step takes the requested
# subset of nodes from the common allocation and places exactly one 112-thread
# rank on each node.
python3 "$RUNNER" \
  --reference "$REFERENCE" \
  --distributed "$DISTRIBUTED" \
  --config "$CONFIG" \
  --ranks 2 \
  --repetitions 5 \
  --warmups 1 \
  --omp-threads 112 \
  --launcher 'srun --exact --nodes={ranks} --ntasks={ranks} --ntasks-per-node=1 --cpus-per-task=112 --cpu-bind=cores' \
  --csv "${RESULT_DIR}/raw.csv" \
  --summary-csv "${RESULT_DIR}/summary.csv" \
  --keep-logs "${RESULT_DIR}/logs"