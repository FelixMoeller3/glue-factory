import logging
import os
import pickle
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gluefactory.datasets import BaseDataset
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import load_image, read_image, ImagePreprocessor

logger = logging.getLogger(__name__)


class HyperSimOneViewJPLDD(BaseDataset):
    """
    Subset of the Oxford Paris dataset as defined here: https://cmp.felk.cvut.cz/revisitop/
    Supports loading groundtruth and only serves images for that gt exists.
    Dataset only deals with loading one element. Batching is done by Dataloader!

    Adapted to use POLD2 structure of files -> Files and gt in same folder besides each other
    Some facts:
    - Pold2 gt is generated same size as original image and can be resized
    """

    default_conf = {
        "data_dir": "hypersim/ai_001_001/images/scene_cam_00_final_preview",  # as subdirectory of DATA_PATH(defined in settings.py)
        "grayscale": False,
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "val_batch_size": 1,
        "all_batch_size": 1,
        "device": None,  # specify device to move image data to. if None is given, just read, skip move to device
        "split": "train",  # train, val, test
        "seed": 0,
        "num_workers": 0,  # number of workers used by the Dataloader
        "prefetch_factor": None,
        "reshape": None,  # ex 800  # if reshape is activated AND multiscale learning is activated -> reshape has prevalence
        "square_pad": True,  # square padding is needed to batch together images with current scaling technique(keeping aspect ratio). Can and should be deactivated on benchmarks
        "multiscale_learning": {
            "do": False,
            "scales_list": [1000, 800, 600, 400],  # use interger scales to have resize keep aspect ratio -> not squashing img by forcing it to square
            "scale_selection": 'random' # random or round-robin
        },
        "load_features": {
            "do": False,
            "check_exists": True,
            "point_gt": {
                "data_keys": ["superpoint_heatmap", "gt_keypoints", "gt_keypoints_scores"],
                "use_score_heatmap": True,
                "max_num_keypoints": 63, # the number of gt_keypoints used for training. The heatmap is generated using all kp. (IN KP GT KP ARE SORTED BY SCORE) 
                                         # -> Can also be set to None to return all points but this can only be used when batchsize=1. Min num kp in oxparis: 63
                "use_deeplsd_lineendpoints_as_kp_gt": False,  # set true to use deep-lsd line endpoints as keypoint groundtruth
                "use_superpoint_kp_gt": True  # set true to use default HA-Superpoint groundtruth
            },
            "line_gt": {
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
                "enforce_threshold": 5.0,  # Enforce values in distance field to be no greater than this value
            },
        },
        # frame.0000.color.jpg
        "img_glob": f"**/frame.[0-9][0-9][0-9][0-9].color.jpg",  # glob pattern to find images
        # img list path from repo root -> use checked in file list, it is similar to pold2 file
        "rand_shuffle_seed": None,  # seed to randomly shuffle before split in train and val
        "val_size": 0,  # size of validation set given
        "train_size": 98,
    }

    def _init(self, conf):
        
        if not (DATA_PATH / conf.data_dir).exists():
            raise ValueError(f"Dataset not found at {DATA_PATH / conf.data_dir}.")
        # load image names
        self.img_list = list((DATA_PATH / conf.data_dir).glob(conf.img_glob))
        print(self.img_list)


        images = self.img_list
        if self.conf.rand_shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(images)
        train_images = images[: conf.train_size]
        val_images = images[conf.train_size : conf.train_size + conf.val_size]
        self.images = {
            "train": train_images,
            "val": val_images,
            "test": images,
            "all": images,
        }
        print(f"DATASET OVERALL(NO-SPLIT) IMAGES: {len(images)}")

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, self.images[split], split)



class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_sub_paths: list[str], split):
        super().__init__()
        self.split = split
        self.conf = conf
        self.grayscale = bool(conf.grayscale)
        
        # Initialize Image Preprocessors for square padding and resizing
        self.preprocessors = {} # stores preprocessor for each reshape size
        if self.conf.reshape is not None:
            self.register_image_preprocessor_for_size(self.conf.reshape)
        if self.conf.multiscale_learning.do:
            for scale in self.conf.multiscale_learning.scales_list:
                self.register_image_preprocessor_for_size(scale)

        if self.conf.multiscale_learning.do:
            if self.conf.multiscale_learning.scale_selection == 'round-robin':
                self.scale_selection_idx = 0
            # Keep track uf how many selected with current scale for batching (all img in same batch need same size)
            self.num_select_with_current_scale = 0
            self.current_scale = None
        # we need to make sure that the appropriate batch size for the dataset conf is set correctly.
        self.relevant_batch_size = self.conf[f"{split}_batch_size"]

        self.img_dir = DATA_PATH / conf.data_dir
        
        # Extract image paths
        self.image_sub_paths = image_sub_paths

        # making them relative for system independent names in export files (path used as name in export)
        if len(self.image_sub_paths) == 0:
            raise ValueError(f"Could not find any image in folder: {self.img_dir}.")
        logger.info(f"NUMBER OF IMAGES: {len(self.image_sub_paths)}")
        logger.info(f"KNOWN BATCHSIZE FOR MY SPLIT({self.split}) is {self.relevant_batch_size}")
       

    def register_image_preprocessor_for_size(self, size: int) -> None:
        """
        We use image preprocessor to reshape images and square pad them. We resize keeping the aspect ratio of images.
        Thus image sizes can be different even when long side scaled to same length. Thus square padding is needed so that
        all images can be stuck together in a batch.
        """
        self.preprocessors[size] = ImagePreprocessor({
                                                        "resize": size,
                                                        "edge_divisible_by": None,
                                                        "side": "long",
                                                        "interpolation": "bilinear",
                                                        "align_corners": None,
                                                        "antialias": True,
                                                        "square_pad": bool(self.conf.square_pad),
                                                        "add_padding_mask": True,}
                                                     )

    def _read_image(self, img_path, enforce_batch_dim=False):
        """
        Read image as tensor and puts it on device
        """
        img = load_image(img_path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        full_artificial_img_path = self.image_sub_paths[idx]
        
        img = self._read_image(full_artificial_img_path)
        orig_shape = img.shape[-1], img.shape[-2]
        size_to_reshape_to = self.select_resize_shape(orig_shape)
        data = {
            "name": str(full_artificial_img_path),
        }  # keys: 'name', 'scales', 'image_size', 'transform', 'original_image_size', 'image'
        if size_to_reshape_to == orig_shape:
            data['image'] = img
        else:
            data = {**data, **self.preprocessors[size_to_reshape_to](img)}

        return data
    
    def do_change_size_now(self) -> bool:
        """
        Based on current state decides whether to change shape to reshape images to.
        This decision is needed as all images in a batch need same shape. So we only potentially change shape
        when a new batch is starting.

        Returns:
            bool: should shape be potentially changed?
        """
        # check if batch changes
        if self.num_select_with_current_scale % self.relevant_batch_size == 0:
            self.num_select_with_current_scale = 0  # if batch changes set counter to 0
            return True
        else:
            return False
        

    def select_resize_shape(self, original_img_size: tuple):
        """
        Depending on whether resize or multiscale learning is activated the shape to resize the
        image to is returned. If none of it is activated, the original image size will be returned.
        Reshape has prevalence over multiscale learning!
        """
        do_reshape = self.conf.reshape is not None
        do_ms_learning = self.conf.multiscale_learning.do
        if not do_reshape and not do_ms_learning:
            return original_img_size

        if do_reshape:
            return int(self.conf.reshape)

        if do_ms_learning:
            if self.do_change_size_now():
                self.num_select_with_current_scale += 1
                scales_list = self.conf.multiscale_learning.scales_list
                scale_selection = self.conf.multiscale_learning.scale_selection
                assert len(scales_list) > 1 # need more than one scale for multiscale learning to make sense

                if scale_selection == "random":
                    choice = int(random.choice(scales_list))
                    self.current_scale = choice
                    return choice
                elif scale_selection == "round-robin":
                    current_scale = scales_list[self.scale_selection_idx]
                    self.current_scale = current_scale
                    self.scale_selection_idx += 1
                    self.scale_selection_idx = self.scale_selection_idx % len(scales_list)
                    return int(current_scale)
            else:
                self.num_select_with_current_scale += 1
                return self.current_scale

        raise Exception("Shouldn't end up here!")
    
    
    def set_num_selected_with_current_scale(self, value: int) -> None:
        """
        Sets the self.num_selected_with_current_scale variable to a certain value.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets

        Args:
            value (int): new value for variable
        """
        self.num_select_with_current_scale = value


    def get_current_scale(self) -> int:
        """
        Returns the current used scale to reshape images to. Returns None if multiscale learning is deactivated.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets

        Returns:
            int: current scale used to reshape in multi-scale training. None if its deactivated
        """
        return self.current_scale
    
    
    def set_current_scale(self, value):
        """
        Sets the current scale used for multiscale training. Used to set size of reshape of this dataset during batch.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets.

        Returns:
            int: current scale used to reshape in multi-scale training. None if its deactivated
        """
        self.current_scale = value


    def __len__(self):
        return len(self.image_sub_paths)
