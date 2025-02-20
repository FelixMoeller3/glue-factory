"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import pickle
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points_torch,
)
from ..models.cache_loader import pad_local_features
from ..settings import DATA_PATH, root
from ..utils.image import read_image
from ..utils.tensor import batch_to_device
from ..utils.tools import fork_rng
from ..visualization.viz2d import (
    plot_image_grid,
    plot_images,
    plot_keypoints,
)
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset
from .utils import warp_points

logger = logging.getLogger(__name__)


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


def plot_predictions(pred, data):

    # read image
    img_0 = data["view0"]["image"][0].permute(1, 2, 0).cpu().numpy()
    
    # read ground truth
    kp_0 = data["view0"]["cache"]["keypoints"][0].cpu().numpy()
    kp_heatmap0 = data["view0"]["cache"]["superpoint_heatmap"][0].cpu().numpy()
    df_0 = data["view0"]["cache"]["deeplsd_distance_field"][0].cpu().numpy()

    # read predictions
    kp_jpl = pred["keypoints0"][0].cpu().numpy()
    df_jpl = pred["line_distancefield0"][0].cpu().numpy()
    kp_heatmap_jpl = pred["keypoint_and_junction_score_map0"][0].cpu().numpy()

    # plot predictions
    plot_images(
        [img_0, df_jpl, kp_heatmap_jpl],
        ["Keypoints", "DF", "Keypoint Heatmap"],
        cmaps=["gray", "viridis", "gray"],
    )
    plot_keypoints([kp_jpl])
    fig_pred = plt.gcf()
    plt.close()

    # plot ground truth
    plot_images(
        [img_0, df_0, kp_heatmap0],
        ["Keypoints", "DF", "Keypoint Heatmap"],
        cmaps=["gray", "viridis", "gray"],
    )
    plot_keypoints([kp_0])
    fig_gt = plt.gcf()
    plt.close()

    return {"Predictions": fig_pred, "Ground Truth": fig_gt}


class OxfordParisTwoViewDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "revisitop1m_POLD2/jpg",  # the top-level directory
        "img_list": "gluefactory/datasets/oxford_paris_images.txt",

        # splits
        "train_size": 11500,
        "val_size": 500,
        "rand_shuffle_seed": 0,  # or None to 
        "reseed": False,
        
        # image loading
        "grayscale": False,
        "triplet": False,
        "right_only": False,  # image0 is orig (rescaled), image1 is right
        
        # homography
        "homography": {
            "difficulty": 0.8,
            "translation": 1.0,
            "max_angle": 60,
            "n_angles": 100,
            "patch_shape": [640, 480],
            "min_convexity": 0.05,
        },

        # photometric augmentations
        "photometric": {
            "name": "dark",
            "p": 0.75,
            # 'difficulty': 1.0,  # currently unused
        },

        # groundtruth loading
        "load_features": {
            "do": False,
            "check_exists": False,
            "thresh": 0.0,
            "max_num_keypoints": 500,
            "force_num_keypoints": True,
            "point_gt": {
                "use_score_heatmap": False,
            },
        },
    }

    def _init(self, conf):
        with open(root / self.conf.img_list, "r") as f:
            self.img_list = [file_name.strip("\n") for file_name in f.readlines()]
        
        # load image names
        images = self.img_list
        if self.conf.rand_shuffle_seed is not None:
            np.random.RandomState(conf.rand_shuffle_seed).shuffle(images)
        
        train_images = images[: conf.train_size]
        val_images = images[conf.train_size : conf.train_size + conf.val_size]
        self.images = {
            "train": train_images,
            "val": val_images,
            "test": images,
            "all": images,
        }
        print(f"DATASET OVERALL(NO-SPLIT) IMAGES: {len(images)}")
        print(f"TRAIN IMAGES: {len(train_images)}")
        print(f"VAL IMAGES: {len(val_images)}")

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_names: list[str], split):
        self.conf = conf
        self.split = split
        self.image_names = np.array(image_names)
        self.image_dir = DATA_PATH / conf.data_dir

        # filter out where missing groundtruth
        if self.conf.load_features.do and self.conf.load_features.check_exists:
            self.image_names = self._check_gt_exists()

        aug_conf = conf.photometric
        aug_name = aug_conf.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'
        self.photo_augment = augmentations[aug_name](aug_conf)
        self.left_augment = (
            IdentityAugmentation() if conf.right_only else self.photo_augment
        )
        self.img_to_tensor = IdentityAugmentation()

    def _check_gt_exists(self):
        new_img_path_list = []

        for img_path in self.image_names:

            full_artificial_img_path = self.image_dir / img_path
            img_folder = (
                full_artificial_img_path.parent / full_artificial_img_path.stem
            )
            keypoint_file = img_folder / "keypoint_scores.npy"
            af_file = img_folder / "angle.jpg"
            df_file = img_folder / "df.jpg"
            img_file = img_folder / "base_image.jpg"


            if keypoint_file.exists() and af_file.exists() and df_file.exists() and img_file.exists():
                new_img_path_list.append(img_path)
            else:
                logging.warning(f"GT files not found for {img_file}")

        logger.info(f"NUMBER OF IMAGES WITH GT: {len(new_img_path_list)}")
        return new_img_path_list


    def _transform_keypoints(self, features, data):
        """Transform keypoints by a homography, threshold them,
        and potentially keep only the best ones."""
        # Warp points
        features["keypoints"] = warp_points_torch(
            features["keypoints"], data["H_"], inverse=False
        )
        h, w = data["image"].shape[1:3]
        valid = (
            (features["keypoints"][:, 0] >= 0)
            & (features["keypoints"][:, 0] <= w - 1)
            & (features["keypoints"][:, 1] >= 0)
            & (features["keypoints"][:, 1] <= h - 1)
        )
        features = {k: v[valid] for k, v in features.items()}

        # Threshold
        if self.conf.load_features.thresh > 0:
            valid = features["keypoint_scores"] >= self.conf.load_features.thresh
            features = {k: v[valid] for k, v in features.items()}

        # Get the top keypoints and pad
        n = self.conf.load_features.max_num_keypoints
        if n > -1:
            inds = torch.argsort(-features["keypoint_scores"])
            features = {k: v[inds[:n]] for k, v in features.items()}

            if self.conf.load_features.force_num_keypoints:
                features = pad_local_features(
                    features, self.conf.load_features.max_num_keypoints
                )

        return features

    def df_and_angle_to_offset(self, df, angle):
        """Convert a DF and angle representation back to an offset map."""
        # Calculate x and y components of the offset using angle and magnitude (df)
        offset_x = df * np.sin(angle + np.pi / 2)
        offset_y = df * np.cos(angle + np.pi / 2)

        # Stack offset_x and offset_y to create the offset map
        offset = np.stack((offset_x, offset_y), axis=-1)

        return offset

    def offset_to_df_and_angle(self, offset):
        """Convert an offset map into a DF and angle representation."""
        df = np.linalg.norm(offset, axis=-1)
        angle = np.arctan2(offset[:, :, 0], offset[:, :, 1])
        return df, angle

    def warp_data(self, img, df, angle, offset, H, ps: list):
        h, w = img.shape[:2]
        ps = tuple(ps)

        valid_mask = cv2.warpPerspective(
            np.ones_like(df), H, ps, flags=cv2.INTER_NEAREST
        ).astype(bool)

        # Warp the closest point on a line
        pix_loc = np.stack(
            np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), axis=-1
        )
        closest = pix_loc + offset
        warped_closest = warp_points(closest.reshape(-1, 2), H).reshape(h, w, 2)
        warped_pix_loc = warp_points(pix_loc.reshape(-1, 2), H).reshape(h, w, 2)
        # angle = np.arctan2(warped_closest[:, :, 0] - warped_pix_loc[:, :, 0],
        #                    warped_closest[:, :, 1] - warped_pix_loc[:, :, 1])
        offset_norm = np.linalg.norm(offset, axis=-1)
        zero_offset = offset_norm < 1e-3
        offset_norm[zero_offset] = 1
        scaling = np.linalg.norm(warped_closest - warped_pix_loc, axis=-1) / offset_norm
        scaling[zero_offset] = 0

        warped_closest[:, :, 0] = cv2.warpPerspective(
            warped_closest[:, :, 0], H, (w, h), flags=cv2.INTER_NEAREST
        )
        warped_closest[:, :, 1] = cv2.warpPerspective(
            warped_closest[:, :, 1], H, (w, h), flags=cv2.INTER_NEAREST
        )
        warped_offset = warped_closest - pix_loc

        # Warp the DF
        warped_df = cv2.warpPerspective(df, H, ps, flags=cv2.INTER_LINEAR)
        warped_scaling = cv2.warpPerspective(scaling, H, ps, flags=cv2.INTER_LINEAR)
        warped_df *= warped_scaling

        # Warp the angle
        closest = pix_loc + np.stack([np.sin(angle), np.cos(angle)], axis=-1)
        warped_closest = warp_points(closest.reshape(-1, 2), H).reshape(h, w, 2)
        warped_angle = np.mod(
            np.arctan2(
                warped_closest[:, :, 0] - warped_pix_loc[:, :, 0],
                warped_closest[:, :, 1] - warped_pix_loc[:, :, 1],
            ),
            np.pi,
        )
        warped_angle = cv2.warpPerspective(warped_angle, H, ps, flags=cv2.INTER_NEAREST)

        return (valid_mask, warped_df, warped_angle, warped_offset)

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def _read_view(self, img, img_folder, name, H_conf, ps, left=False):
        data = sample_homography(img, H_conf, ps)
        data["scene"], data["name"] = name.split("/")
        data["name"] = data["name"][:-4]
        data["scales"] = 1
        if left:
            data["image"] = self.left_augment(data["image"], return_tensor=True)
        else:
            data["image"] = self.photo_augment(data["image"], return_tensor=True)

        gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        if self.conf.grayscale:
            data["image"] = (data["image"] * gs).sum(0, keepdim=True)

        if self.conf.load_features.do:
            H = data["H_"].copy()
            data["H_"] = torch.from_numpy(data["H_"])

            # read keypoints, keypoints_scores
            features = {}
            kp_file = img_folder / "keypoints.npy"
            kps_file = img_folder / "keypoint_scores.npy"

            # Load keypoints and scores
            features["keypoints"] = torch.from_numpy(np.load(kp_file)).to(
                dtype=torch.float32
            )
            features["keypoint_scores"] = torch.from_numpy(np.load(kps_file)).to(
                dtype=torch.float32
            )
            features = batch_to_device(features, data["image"].device)
            features = self._transform_keypoints(features, data)

            # prepare keypoint heatmap
            heatmap = np.zeros_like(data["image"][0], dtype=np.float32)
            integer_kp_coordinates = torch.round(features["keypoints"]).to(dtype=torch.int)
            
            if self.conf.load_features.point_gt.use_score_heatmap:
                heatmap[integer_kp_coordinates[:, 1], integer_kp_coordinates[:, 0]] = features["keypoint_scores"]
            else:
                heatmap[integer_kp_coordinates[:, 1], integer_kp_coordinates[:, 0]] = 1.0
            heatmap = torch.from_numpy(heatmap).to(dtype=torch.float32)

            features["superpoint_heatmap"] = heatmap

            # Load pickle file for DF max and min values
            with open(img_folder / "values.pkl", "rb") as f:
                values = pickle.load(f)

            # Load DF
            df_img = read_image(img_folder / "df.jpg", True)
            df_img = df_img.astype(np.float32) / 255.0
            df_img *= values["max_df"]

            # Load AF
            af_img = read_image(img_folder / "angle.jpg", True)
            af_img = af_img.astype(np.float32) / 255.0
            af_img *= np.pi

            # Get closest point to line for each pixel
            # offset = self.df_and_angle_to_offset(df_img, af_img)
            ofx_img = read_image(img_folder / "offset_x.jpg", True)
            ofx_img = ofx_img.astype(np.float32) / 255.0
            ofy_img = read_image(img_folder / "offset_y.jpg", True)
            ofy_img = ofy_img.astype(np.float32) / 255.0
            offset = np.stack((ofx_img, ofy_img), axis=-1)
            offset = offset * values["max_offset"]
            offset = offset + values["min_offset"]

            """
            # check offset calculation
            df, angle = self.offset_to_df_and_angle(offset)
            assert np.allclose(df, df_img), "DF not equal"

            try:
                assert np.allclose(angle, af_img), "Angle not equal"
            except AssertionError:
                # print values where the angle is not equal
                print(f"Angle not equal at {np.where(angle != af_img)}")
                print(f"Angle: {angle[np.where(angle != af_img)]}")
                print(f"AF: {af_img[np.where(angle != af_img)]}")
                exit()
            """

            # Warp the DF, and AF according to the homography
            ref_valid_mask, warped_df, warped_angle, warped_offset = self.warp_data(
                img, df_img, af_img, offset, H, ps
            )

            # convert angle field to 2 channel direction field (x,y)
            # warped_direction_field = np.stack(
            #     (np.cos(warped_angle), np.sin(warped_angle)), axis=-1
            # )

            # Add the warped features to the dictionary
            features["deeplsd_distance_field"] = torch.from_numpy(warped_df).to(dtype=torch.float32)
            features["deeplsd_angle_field"] = torch.from_numpy(warped_angle).to(dtype=torch.float32)
            # features["line_level"] = (
            #     torch.from_numpy(warped_direction_field)
            #     .to(dtype=torch.float32)
            #     .permute(2, 0, 1)
            # )
            features["ref_valid_mask"] = torch.from_numpy(ref_valid_mask).to(
                dtype=torch.float32
            )

            features = batch_to_device(features, data["image"].device)
            data["cache"] = features

        return data

    def getitem(self, idx):
        name = self.image_names[idx]
        img_path = self.image_dir / name
        img_folder_path = img_path.parent / img_path.stem
        img = read_image(img_folder_path / "base_image.jpg", False)
        if img is None:
            logging.warning("Image %s could not be read.", name)
            img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
        img = img.astype(np.float32) / 255.0
        size = img.shape[:2][::-1]
        ps = self.conf.homography.patch_shape

        left_conf = omegaconf.OmegaConf.to_container(self.conf.homography)
        if self.conf.right_only:
            left_conf["difficulty"] = 0.0

        data0 = self._read_view(img, img_folder_path, name, left_conf, ps, left=True)
        data1 = self._read_view(img, img_folder_path, name, self.conf.homography, ps, left=False)

        H = compute_homography(data0["coords"], data1["coords"], [1, 1])

        data = {
            "name": name,
            "original_image_size": np.array(size),
            "H_0to1": H.astype(np.float32),
            "idx": idx,
            "view0": data0,
            "view1": data1,
        }

        if self.conf.triplet:
            # Generate third image
            data2 = self._read_view(img, self.conf.homography, ps, left=False)
            H02 = compute_homography(data0["coords"], data2["coords"], [1, 1])
            H12 = compute_homography(data1["coords"], data2["coords"], [1, 1])

            data = {
                "H_0to2": H02.astype(np.float32),
                "H_1to2": H12.astype(np.float32),
                "view2": data2,
                **data,
            }

        return data

    def __len__(self):
        return len(self.image_names)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = OxfordParisTwoViewDataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
            # Visualize the DF and line level
            images.append(
                [data[f"view{i}"]["cache"]["df"][0] for i in range(2)]
            )
            images.append(
                [data[f"view{i}"]["cache"]["line_level"][0] for i in range(2)]
            )
            # Visualize the keypoints
            images.append(
                [data[f"view{i}"]["cache"]["keypoint_heatmap"][0] for i in range(2)]
            )

    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.savefig("oxford_paris_2view_jpldd.png", dpi=args.dpi)


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
