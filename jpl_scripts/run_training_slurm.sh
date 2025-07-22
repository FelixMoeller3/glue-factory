#!/bin/bash
# Cmd params 'run_training_slurm.sh [exp_name] [path to conf]'
# Run multi gpu on same node. Dont spawn multiple processes as glue-factory does it automatically

#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --output=train.out
#SBATCH --mail-type=END
#SBATCH --mail-user=r.kreft@stud.ethz.ch
#SBATCH --job-name="jpl_training"

module load eth_proxy

# Check if --resume flag is provided
RESUME_FLAG=""
if [[ "$*" == *"--resume"* ]]; then
    RESUME_FLAG="--restore"
fi

# set distributed flag
DISTRIBUTED_FLAG=""
if [ "$SLURM_GPUS_ON_NODE" -gt 1 ] || [ "$SLURM_NNODES" -gt 1 ]; then
    DISTRIBUTED_FLAG="--distributed"
fi

echo "Exp-Name: $1"
echo "Conf-Path: $2"

# Print SLURM variables so you see how your resources are allocated
echo "[sbatch-master] Job Name: $SLURM_JOB_NAME"
echo "[sbatch-master] Job ID: $SLURM_JOB_ID"
echo "[sbatch-master] Num Nodes: $SLURM_NNODES"
echo "[sbatch-master] Allocated Node(s): $SLURM_NODELIST"
echo "[sbatch-master] Number of Tasks: $SLURM_NTASKS"
echo "[sbatch-master] MasterNodeID: $SLURM_NODEID"
echo "[sbatch-master] Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "Current path: $(pwd)"
echo "Current user: $(whoami)"

source ~/jpl_venv/bin/activate
cd ~/glue-factory || exit

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.train "$1" --conf="$2" "$RESUME_FLAG" "$DISTRIBUTED_FLAG"
echo "Finished training!"
