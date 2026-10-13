#!/bin/bash
#SBATCH -J autopasdm-node-scaling
#SBATCH -D ./
#SBATCH -o ./autopasdm-node-scaling.%j.out
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

set -euo pipefail

module purge

module load slurm_setup
module load stack/24.4.0
module load llvm/19.1.3
module load gcc/13.2.0
module load openmpi/4.1.8-gcc13

if [[ $# -ne 2 ]]; then
  echo "Usage: sbatch $0 <AutoPas-source-root> <AutoPasDM-source-root>"
  echo
  echo "Both projects are expected to have been built in a 'build/' directory."
  exit 2
fi

AUTOPAS_ROOT=$(realpath "$1")
AUTOPASDM_ROOT=$(realpath "$2")

REFERENCE="${AUTOPAS_ROOT}/build/examples/md-flexible/md-flexible"
DISTRIBUTED="${AUTOPASDM_ROOT}/build/examples/md-flexible/md-flexible"
RUNNER="${AUTOPASDM_ROOT}/examples/md-flexible/scripts/run_strong_scaling.py"
CONFIG="${AUTOPASDM_ROOT}/examples/md-flexible/input/autopasdm_performance_node_strong_scaling.yaml"

for file in "$REFERENCE" "$DISTRIBUTED" "$RUNNER" "$CONFIG"; do
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file"
    exit 2
  fi
done

RESULT_DIR="${SLURM_SUBMIT_DIR}/node_strong_scaling_cm4_${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

# One AutoPas instance / MPI rank per socket.
# CoolMUC-4 has two sockets with 56 physical cores each.
# Each MPI rank is bound to one socket and uses all 56 cores via OpenMP.
export OMP_NUM_THREADS=56
export OMP_DYNAMIC=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "CoolMUC-4 hybrid node strong-scaling benchmark"
echo "  allocated nodes       : ${SLURM_JOB_NUM_NODES}"
echo "  MPI ranks per node    : 2"
echo "  OpenMP threads / rank : ${OMP_NUM_THREADS}"
echo "  AutoPas instances/node: 2"
echo "  reference             : ${REFERENCE}"
echo "  distributed           : ${DISTRIBUTED}"
echo "  config                : ${CONFIG}"
echo "  results               : ${RESULT_DIR}"
echo

# Strong scaling over nodes:
#
#   2 MPI ranks -> 1 node
#   4 MPI ranks -> 2 nodes
#   8 MPI ranks -> 4 nodes
#
# Two MPI ranks are placed per node, each bound to one socket.
python3 "$RUNNER" \
  --reference "$REFERENCE" \
  --distributed "$DISTRIBUTED" \
  --config "$CONFIG" \
  --ranks 1,2,4,8 \
  --repetitions 5 \
  --warmups 1 \
  --omp-threads 56 \
  --launcher 'srun --exact --ntasks={ranks} --ntasks-per-node=2 --cpus-per-task=56 --distribution=block:block --cpu-bind=sockets' \
  --csv "${RESULT_DIR}/raw.csv" \
  --summary-csv "${RESULT_DIR}/summary.csv" \
  --keep-logs "${RESULT_DIR}/logs"