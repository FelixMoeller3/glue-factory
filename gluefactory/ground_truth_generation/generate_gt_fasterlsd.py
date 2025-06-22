"""
Run the homography adaptation for all images in a given folder
to regress and aggregate line distance function maps.
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm
from pytlsd import lsd
from pathlib import Path
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from afm_op import afm

from gluefactory.datasets import get_dataset
from gluefactory.settings import EVAL_PATH
from gluefactory.datasets.homographies_deeplsd import warp_lines
# TODO: Is it correct to do this?
from gluefactory.ground_truth_generation.superpoint_gt_multiple_files import  sample_homography
from gluefactory.datasets.photometric_transforms import random_contrast

homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}


def ha_df(img, num=100, border_margin=3, min_counts=5):
    """ Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        num: number of homographies used during HA.
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2]
    size = (w, h)
    df_maps, angles, closests, counts = [], [], [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (border_margin * 2, border_margin * 2))
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'),
                       axis=-1)
    raster_lines = np.zeros_like(img)

    # Loop through all the homographies
    for i in range(num):
        # Generate a random homography
        if i == 0:
            H = np.eye(3)
        else:
            H = sample_homography(img.shape, **homography_params)
        H_inv = np.linalg.inv(H)

        # Warp the image
        warped_img = cv2.warpPerspective(img, H, size,
                                         borderMode=cv2.BORDER_REPLICATE)

        # Regress the DF on the warped image
        warped_lines = lsd(warped_img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)

        # Warp the lines back
        lines = warp_lines(warped_lines, H_inv)

        # Get the DF and angles
        num_lines = len(lines)
        cuda_lines = torch.from_numpy(lines[:, :, [1, 0]].astype(np.float32))
        cuda_lines = cuda_lines.reshape(-1, 4)[None].cuda()
        offset = afm(
            cuda_lines,
            torch.IntTensor([[0, num_lines, h, w]]).cuda(), h, w)[0]
        offset = offset[0].permute(1, 2, 0).cpu().numpy()[:, :, [1, 0]]
        closest = pix_loc + offset
        df = np.linalg.norm(offset, axis=-1)
        angle = np.mod(np.arctan2(
            offset[:, :, 0], offset[:, :, 1]) + np.pi / 2, np.pi)

        df_maps.append(df)
        angles.append(angle)
        closests.append(closest)

        # Compute the valid pixels
        count = cv2.warpPerspective(np.ones_like(img), H_inv, size,
                                    flags=cv2.INTER_NEAREST)
        count = cv2.erode(count, kernel)
        counts.append(count)
        raster_lines += (df < 1).astype(np.uint8) * count

        # Compute the median of all DF maps, with counts as weights
    df_maps, angles = np.stack(df_maps), np.stack(angles)
    counts, closests = np.stack(counts), np.stack(closests)

    # Median of the DF
    df_maps[counts == 0] = np.nan
    avg_df = np.nanmedian(df_maps, axis=0)

    # Median of the closest
    closests[counts == 0] = np.nan
    avg_closest = np.nanmedian(closests, axis=0)

    # Median of the angle
    circ_bound = (np.minimum(np.pi - angles, angles)
                  * counts).sum(0) / counts.sum(0) < 0.3
    angles[:, circ_bound] -= np.where(
        angles[:, circ_bound] > np.pi / 2,
        np.ones_like(angles[:, circ_bound]) * np.pi,
        np.zeros_like(angles[:, circ_bound]))
    angles[counts == 0] = np.nan
    avg_angle = np.mod(np.nanmedian(angles, axis=0), np.pi)

    # Generate the background mask and a saliency score
    raster_lines = np.where(raster_lines > min_counts, np.ones_like(img),
                            np.zeros_like(img))
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    bg_mask = (1 - raster_lines).astype(float)

    return avg_df, avg_angle, avg_closest[:, :, [1, 0]], bg_mask


def process_image(img_data, randomize_contrast, num_H, output_folder_path):
    img = img_data["image"]
    if randomize_contrast is not None:
        img = randomize_contrast(img)

    # Run homography adaptation
    df, angle, closest, bg_mask = ha_df(img, num=num_H)

    # Save the DF in a hdf5 file

    complete_out_folder = (output_folder_path / str(img_data["name"][0])).parent
    complete_out_folder.mkdir(parents=True, exist_ok=True)
    output_file_path = (
        complete_out_folder / f"{Path(img_data['name'][0]).name.split('.')[0]}.hdf5"
    )

    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("df", data=df.flatten())
        f.create_dataset("line_level", data=angle.flatten())
        f.create_dataset("closest", data=closest.flatten())
        f.create_dataset("bg_mask", data=bg_mask.flatten())


def export_ha(data_loader, output_folder, num_H=100,
              rdm_contrast=False, n_jobs=1):

    # Random contrast object
    randomize_contrast = random_contrast() if rdm_contrast else None

    # Process each image in parallel
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(process_image)(img_path, randomize_contrast, num_H, output_folder)
        for img_path in tqdm(data_loader, total=len(data_loader))
    )

def get_dataset_and_loader(num_workers):  # folder where dataset images are placed
    config = {
        "name": "oxford_paris_mini",  # name of dataset class in gluefactory > datasets
        "grayscale": True,  # commented out things -> dataset must also have these keys but has not
        "preprocessing": {"resize": [800, 800]},
        "train_batch_size": 1,  # prefix must match split mode
        "num_workers": num_workers,
        "split": "train",  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get("split", "train"))
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', type=str, help='Output folder.')
    parser.add_argument('--num_H', type=int, default=100,
                        help='Number of homographies used during HA.')
    parser.add_argument('--random_contrast', action='store_true',
                        help='Add random contrast to the images (disabled by default).')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs to run in parallel.')
    parser.add_argument(
        "--n_jobs_dataloader",
        type=int,
        default=1,
        help="Number of jobs the dataloader uses to load images",
    )
    args = parser.parse_args()

    out_folder_path = EVAL_PATH / args.output_folder
    out_folder_path.mkdir(exist_ok=True, parents=True)

    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N JOBS: ", args.n_jobs)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader)
    export_ha(dataloader, args.output_folder, args.num_H,
              args.random_contrast, args.n_jobs)
    print("Done with ground-truth generation for faster LSD")

