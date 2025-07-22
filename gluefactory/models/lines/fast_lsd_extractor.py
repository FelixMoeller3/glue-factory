from typing import Optional

import numpy as np
import torch
from faster_pytlsd import lsd as fast_lsd
from pytlsd import lsd

from gluefactory.models.lines.line_refinement import filter_outlier_lines, merge_lines
from gluefactory.models.lines.line_utils import preprocess_angle
from gluefactory.utils.image import compute_image_grad

from ..base_model import BaseModel


class FastLSDLineExtractor(BaseModel):
    """
    This is meant to be a simple wrapper to use LSD or fast LSD in the JPL pipeline (joint_point_line_extractor.py)
    """

    default_conf = {
        "name": "lines.fast_lsd_extractor",
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "use_img_grad_angle": True,  # Dont use the angle-field but use the image gradient as surrogate
        "merge": False,
        "grad_nfa": True,
        "filtering": "normal",
        "grad_thresh": 3,
        "faster_lsd": True,
        "return_line_descriptors": False,
        "trainable": False,
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"
        # currently line descriptors arte not implemented
        if self.conf.return_line_descriptors:
            raise NotImplementedError(
                "Line descriptors are not implemented yet for FasterLSD"
            )

    def detect_lines(
        self, img: np.array, df: np.array, line_level: Optional[np.array] = None
    ) -> np.array:
        """
        detect lines in one image.
        Args:
            img: image as numpy array
            df: denormalized distance field as numpy array
            line_level: line anglefield / line level as numpy array. Not needed if conf.use_img_grad_angle is True.
        Returns: numpy array containing lines as (x1, y1 \\ x2, y2) tuples so of shape (n_lines, 2, 2)
        """
        # Run LSD
        img_grad_angle = None
        gradnorm = np.maximum(5 - df, 0).astype(np.float64)

        if self.conf.use_img_grad_angle:
            img_grad_angle = compute_image_grad(img)[3]
            angle = np.mod(img_grad_angle - np.pi / 2, 2 * np.pi)
        else:
            angle = line_level.astype(np.float64) - np.pi / 2
            angle = preprocess_angle(angle, img, mask=True)[0]
        angle[gradnorm < self.conf.grad_thresh] = -1024
        if self.conf.faster_lsd:
            lines = fast_lsd(
                img.astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm,
                gradangle=angle,
                grad_nfa=self.conf.grad_nfa,
            )[:, :4].reshape(-1, 2, 2)
        else:
            lines = lsd(
                img.astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm,
                gradangle=angle,
                grad_nfa=self.conf.grad_nfa,
            )[:, :4].reshape(-1, 2, 2)
        # Optionally filter out lines based on the DF and line_level
        if self.conf.filtering is not None:
            if self.conf.filtering == "strict":
                df_thresh, ang_thresh = 1.0, np.pi / 12
            else:
                df_thresh, ang_thresh = 1.5, np.pi / 9
            if self.conf.use_img_grad_angle:
                angle = img_grad_angle
            else:
                angle = line_level - np.pi / 2
            lines = filter_outlier_lines(
                img,
                lines[:, :, [1, 0]],
                df,
                angle,
                mode="inlier_thresh",
                use_grad=False,
                inlier_thresh=0.5,
                df_thresh=df_thresh,
                ang_thresh=ang_thresh,
            )[0][:, :, [1, 0]]
        # Now perform optional min length filtering and apply force num lines if needed
        lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
        to_keep = lengths >= self.conf.min_length
        lines, lengths = lines[to_keep], lengths[to_keep]

        # Keep the best lines (best lines are the shortest ones)
        scores = np.sqrt(lengths)
        lines = lines[:, :4].reshape(-1, 2, 2)
        indices = np.argsort(-scores)
        if self.conf.max_num_lines is not None:
            indices = indices[: self.conf.max_num_lines]
            lines = lines[indices]

        if self.conf.merge:
            lines = merge_lines(
                torch.from_numpy(lines), thresh=4, overlap_thresh=0
            ).numpy()

        # Pad if necessary
        n = len(lines)
        valid_mask = np.ones(n, dtype=bool)
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - n
            lines = np.concatenate(
                [lines, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
            )
            valid_mask = np.concatenate([valid_mask, np.zeros(pad, dtype=bool)], axis=0)

        return {"lines": lines, "valid_lines": valid_mask}

    def _forward(self, data):
        """
        Perform forward pass on the data. Supports batched data.
        Args:
            data: dictionary containing the data. Must contain the following keys: image, line_angle_field, line_distance_field.
        Returns: a list of tensors, containing the lines for each image: shape: [N_images x (n_lines, 2, 2)]
        """
        # Convert to the right data format
        image = data["image"]
        line_level = data["line_anglefield"]
        line_df_denormalized = data["line_distancefield"]

        # preprocess input to lsd
        np_img = (image.cpu().numpy()[:, 0] * 255).astype(np.uint8)
        np_df = line_df_denormalized.cpu().numpy()
        np_ll = line_level.cpu().numpy()
        lines = []
        # valid lines contain a line mask indicating which lines were actually predicted and which are padding
        # applied to enable batching via conf.force_num_lines
        valid_lines = []
        for img, df, ll in zip(np_img, np_df, np_ll):
            line_pred = self.detect_lines(img, df, ll)
            lines.append(torch.Tensor(line_pred["lines"]))
            valid_lines.append(torch.Tensor(line_pred["valid_lines"]))
        # Here a list of lines for each img is returned.
        outputs = {"lines": lines, "valid_lines": valid_lines}
        return outputs

    def loss(self, pred, data):
        raise NotImplementedError
