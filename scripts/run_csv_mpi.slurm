#!/bin/bash -l
#SBATCH -J ordersmc-dag
#SBATCH -p nodes
#SBATCH -N 2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -o ordersmc_%j.out

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
eval "$(conda shell.bash hook)"
conda activate /users/psedrou/.conda/envs/atlas
module load openmpi/5.0.8-gcc14.2.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl=^ofi,psm2
export FI_PROVIDER=tcp

PROJECT=/users/psedrou/fastscratch/OrderSMC-DAG
DATA=${DATA:?Set DATA to the input CSV path}
METADATA=${METADATA:?Set METADATA to the reviewed JSON/YAML metadata path}
CONFIG=${CONFIG:-$PROJECT/examples/config.standard.yaml}
OUTPUT=${OUTPUT:-$PROJECT/results/run_${SLURM_JOB_ID}}

cd "$PROJECT"
mkdir -p "$OUTPUT" results
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
srun --mpi=pmix -n "$SLURM_NTASKS" \
    python -u -m ordersmc.cli fit "$DATA" \
        --metadata "$METADATA" \
        --config "$CONFIG" \
        --backend mpi \
        --save-table \
        --output "$OUTPUT"
