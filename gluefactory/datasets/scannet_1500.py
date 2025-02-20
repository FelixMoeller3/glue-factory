"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def get_relative_transform(pose0, pose1):
    R0 = pose0[..., :3, :3]  # Bx3x3
    t0 = pose0[..., :3, [3]]  # Bx3x1

    R1 = pose1[..., :3, :3]  # Bx3x3
    t1 = pose1[..., :3, [3]]  # Bx3x1

    R_0to1 = R1.transpose(-1, -2) @ R0  # Bx3x3
    t_0to1 = R1.transpose(-1, -2) @ (t0 - t1)  # Bx3x1
    T_0to1 = np.concatenate([R_0to1, t_0to1], axis=-1)  # Bx3x4

    return T_0to1

class scannet_1500(BaseDataset, torch.utils.data.Dataset):

    default_conf = {
        "scannet_path": "Scannet-Xfeat/ScanNet1500/",
        "gt_path": "Scannet-Xfeat/ScanNet1500/test.npz",
        "subset": None,
        "cache_images": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "grayscale": False,
    }

    def _init(self, conf):
        assert conf.batch_size == 1
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        self.config = conf

        self.root = DATA_PATH / self.config["scannet_path"]
        self.gt = DATA_PATH / self.config["gt_path"]
        
        if not self.root.exists():
            raise FileNotFoundError("Scannet Data not Found")

        self.pairs = self._read_gt()

        self.image_cache = {}
        if self.config["cache_images"]:
            self.load_images()

    def load_images(self):
        for pair in tqdm(self.pairs, desc="Caching images"):
            if pair["image0"] not in self.image_cache:
                self.image_cache[pair["image0"]] = self._read_image(pair["image0"])
            if pair["image1"] not in self.image_cache:
                self.image_cache[pair["image1"]] = self._read_image(pair["image1"])

    def _read_image(self, path):
        img = load_image(path, self.conf.grayscale)
        return self.preprocessor(img)
        
    def get_dataset(self, split):
        assert split in ["val", "test"]
        return self

    def _read_gt(self):
        pairs = []
        gt_poses = np.load(self.gt)
        names = gt_poses["name"]

        for i in range(len(names)):
            scene_id = names[i, 0]
            scene_idx = names[i, 1]
            scene = f"scene{scene_id:04d}_{scene_idx:02d}"

            image0 = str(int(names[i, 2]))
            image1 = str(int(names[i, 3]))

            K0 = np.loadtxt(
                os.path.join(
                    self.root,
                    "scannet_test_1500",
                    scene,
                    "intrinsic/intrinsic_color.txt",
                )
            )
            K1 = K0

            pose_0 = np.loadtxt(
                os.path.join(
                    self.root,
                    "scannet_test_1500",
                    scene,
                    "pose",
                    image0 + ".txt",
                )
            )
            pose_1 = np.loadtxt(
                os.path.join(
                    self.root,
                    "scannet_test_1500",
                    scene,
                    "pose",
                    image1 + ".txt",
                )
            )
            T_0to1 = get_relative_transform(pose_0, pose_1)

            pairs.append(
                {
                    "image0": os.path.join(
                        self.root,
                        "scannet_test_1500",
                        scene,
                        "color",
                        image0 + ".jpg",
                    ),
                    "image1": os.path.join(
                        self.root,
                        "scannet_test_1500",
                        scene,
                        "color",
                        image1 + ".jpg",
                    ),
                    "K0": K0,
                    "K1": K1,
                    "T_0to1": T_0to1,
                }
            )

        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):

        if self.pairs[idx]['image0'] not in self.image_cache:
            data0 = self._read_image(self.pairs[idx]['image0'])
        else:
            data0 = self.image_cache[self.pairs[idx]['image0']]

        if self.pairs[idx]['image1'] not in self.image_cache:
            data1 = self._read_image(self.pairs[idx]['image1'])
        else:
            data1 = self.image_cache[self.pairs[idx]['image1']]

        # H = data1["transform"] @ self.pairs[idx]["T_0to1"].astype(np.float32) @ np.linalg.inv(data0["transform"])
        H = self.pairs[idx]["T_0to1"].astype(np.float32)

        # print( data0["transform"])
        assert (data0['transform'] == data1['transform']).all()

        # print(data0['transform'].shape)
        return {
            "T_0to1": H,
            "H_0": data0["transform"],
            "H_1": data1["transform"],
            "scene": idx,
            "idx": idx,
            "name": f"{idx}/{idx}.ppm",
            "view0": data0,
            "view1": data1,
            "K0":self.pairs[idx]['K0'],
            "K1":self.pairs[idx]['K1']
        }


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = scannet_1500(conf)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
