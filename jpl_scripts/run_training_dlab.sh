#!/bin/bash
# TRAIN ON DLAB MACHINE
#activates venv and runs training. Dont forget to activate tmux before
# Cmd params 'run_training_dlab.sh [exp_name] [path to conf]'

echo "Exp-Name: $1"
echo "Conf-Path: $2"

source /local/home/rkreft/shared_team_folder/jpl_venv/bin/activate
cd ~/glue-factory || exit

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.train "$1" --conf="$2"
echo "Finished training!"
