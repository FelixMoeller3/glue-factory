#!/bin/bash
# Cmd params 'run_training.sh [exp_name] [path to conf]'

#SBATCH --time=20000
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=gt_gen.out
#SBATCH --mail-type=END
#SBATCH --mail-user=r.kreft@stud.ethz.ch
#SBATCH --job-name="jpl_gt_generation"

module load eth_proxy

source /cluster/home/rkreft/jpl_venv/bin/activate
cd /cluster/home/rkreft/glue-factory

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python gluefactory/ground_truth_generation/deeplsd_gt_multiple_files.py --output_folder deeplsd_gt --num_H 100 --n_jobs_dataloader 1 --n_gpus 2 --image_name_list minidepth_image_list.txt
echo "Finished training!"
