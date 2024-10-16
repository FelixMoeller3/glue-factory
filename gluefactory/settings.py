from pathlib import Path
import os

my_scratch_space = Path(os.environ.get('SCRATCH'))

cluster_team_folder = Path(
    "/local/home/rkreft/shared_team_folder"
)  # cluster team folder for 3dv
root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = my_scratch_space / "data"  # datasets and pretrained weights
TRAINING_PATH = cluster_team_folder / "outputs/training/"  # training checkpoints
EVAL_PATH = my_scratch_space / "outputs/results/"  # evaluation results
