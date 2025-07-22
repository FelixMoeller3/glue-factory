import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from omegaconf import OmegaConf

from gluefactory.datasets.homographies_deeplsd import sample_homography
from gluefactory.geometry.homography import warp_points_torch
from gluefactory.geometry.kp_losses import soft_argmax_only_loss
from gluefactory.models import get_model
from gluefactory.models.backbones.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.base_model import BaseModel
from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.models.extractors.aliked import DKD, SDDH, SMH, InputPadder
from gluefactory.models.lines.fast_lsd_extractor import FastLSDLineExtractor
from gluefactory.models.utils.metrics_lines import get_rep_and_loc_error
from gluefactory.models.utils.metrics_points import (
    compute_loc_error,
    compute_pr,
    compute_repeatability,
)
from gluefactory.settings import DATA_PATH
from gluefactory.utils.misc import change_dict_key, sync_and_time

# Parameters for calculating point metrics in validation loss
default_H_params = {
    "translation": True,
    "rotation": True,
    "scaling": True,
    "perspective": True,
    "scaling_amplitude": 0.2,
    "perspective_amplitude_x": 0.2,
    "perspective_amplitude_y": 0.2,
    "patch_ratio": 0.85,
    "max_angle": 1.57,
    "allow_artifacts": True,
}

aliked_checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"  # used for training based on ALIKED weights
logger = logging.getLogger(__file__)


class JointPointLineDetectorDescriptor(BaseModel):
    """
    Pipeline to jointly detect and describe keypoints and lines given an RGB image.

    If a checkpoint is loaded, the config from the checkpoint is not. We could change that in the future.
    """

    # default checkpoint used for automatic weight loading if no other path specified
    # currently its the oxparis-800-focal checkpoint
    jpl_default_checkpoint_url = (
        "https://polybox.ethz.ch/index.php/s/IN0dxL4ljUacf9K/download"
    )

    default_conf = {
        "aliked_model_name": "aliked-n16",  # ALIKED model determining architecture of our backbone
        "use_line_anglefield": False,  # if false, model will be initialized without AF branch and AF isn't considered in inference or training
        "line_df_decoder_channels": 32,
        "line_af_decoder_channels": 32,
        "max_num_keypoints": 1024,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "nms_radius": 3,
        "subpixel_refinement": True,  # perform subpixel refinement after detection
        "force_num_keypoints": False,
        "training": {  # training settings
            "do": False,  # switch to turn off other settings regarding training = "training mode"
            "two_view": False,  # whether training is done with a two-view pipeline (True) or with a one-view pipeline (False)
            "aliked_pretrained": True,  # use pretrained ALIKED weights in backbone encoder
            "pretrain_kp_decoder": True,  # use pretrained ALIKED weights for keypoint-heatmap decoder
            "train_descriptors": {  # for train descriptors in one-view: generate gt descriptors, other losses for two_view
                "gt_aliked_model": "aliked-n32",
                "use_one_view_loss": True,  # In one view training can decide if train descriptors with this flag
                "use_two_view_loss": True,  # can only be used if two_view training activated (sparseNRE loss)
            },
            "loss": {
                "use_one_view_df_loss": True,  # one-view losses can be applied any time
                "use_one_view_af_loss": True,  # af loss is only applied if af-use is activated
                "use_two_view_df_loss": True,  # two view losses are only applied if two-view training activated
                "use_one_view_kp_loss": True,  # use one-view keypoint loss ex. focal, l1 etc
                "kp_loss_name": "focal_loss",  # other options: bce, weighted_bce or focal loss
                "kp_loss_parameters": {
                    "lambda_weighted_bce": 200,  # weighted bce parameter factor how to boost keypoint loss in map
                    "focal_gamma": 2,
                    # focal loss parameter controlling how strong to focus on hard examples (typical range 1-5)
                    "focal_alpha": 0.25,  # focal loss parameter to mitigate class imbalances
                },
                "refinement_radius": 5,  # radius for softargmax loss
                "loss_weights": {
                    "one_view_line_af_weight": 1,
                    "one_view_line_df_weight": 1,
                    "two_view_line_df_weight": 1,
                    "keypoint_weight": 1,
                    "one_view_descriptor_weight": 1,
                    "two_view_descriptor_weight": 1,
                    "softargmax_weight": 1,  # if > 0 activates calculation of soft argmax loss on keypoint detection. Only used if two_view activated
                },
            },
        },
        "line_detection": {  # by default we use the POLD2 Line Extractor (MLP with Angle Field)
            "do": True,
            "name": "lines.fast_lsd_extractor",
            "conf": FastLSDLineExtractor.default_conf,
            # following options only used for ablations
            "use_deeplsd_kp": False,  # whether we should use DeepLSD line endpoints as junction candidates. Otherwise use JPLDD keypoints
            "use_deeplsd_df_af": False,  # whether we should use Distance and Angle Field from JPLDD or DeepLSD
        },
        "checkpoint": jpl_default_checkpoint_url,  # if given and non-null, load model checkpoint if local path load locally if standard url download it.
        "line_neighborhood": 5,  # used to normalize / denormalize line distance field
        "timeit": False,  # override timeit: False from BaseModel
        "trainable": True,
    }

    # used for line detection ablation and development when we use deeplsd af/df or line endpoints instead of original net output
    deeplsd_conf = {
        "detect_lines": True,
        "line_detection_params": {
            "merge": False,
            "filtering": True,
            "grad_thresh": 3,
            "grad_nfa": True,
        },
        "weights": "DeepLSD/weights/deeplsd_md.tar",  # path to the weights of the DeepLSD model (relative to DATA_PATH)
    }

    n_limit_max = 20000  # taken from ALIKED which gives max num keypoints to detect!

    required_data_keys = ["image"]

    def _init(self, conf) -> None:
        logger.debug(f"final config dict(type={type(conf)}): {conf}")
        # set loss fn
        assert self.conf.training.loss.kp_loss_name in [
            "weighted_bce",
            "focal_loss",
            "bce",
        ]
        if self.conf.training.loss.kp_loss_name == "weighted_bce":
            self.loss_fn = self.weighted_bce_loss
        elif self.conf.training.loss.kp_loss_name == "focal_loss":
            self.loss_fn = self.focal_loss
        else:
            self.loss_fn = nn.BCELoss(reduction="none")
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        aliked_model_cfg = aliked_cfgs[conf.aliked_model_name]
        dim = aliked_model_cfg["dim"]
        K = aliked_model_cfg["K"]
        M = aliked_model_cfg["M"]
        self.lambda_valid_kp = conf.training.loss.kp_loss_parameters.lambda_weighted_bce
        # Load Network Components
        self.encoder_backbone = AlikedEncoder(aliked_model_cfg)
        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here
        self.dkd = DKD(  # heuristic point-detection with subpixel refinement from ALIKE (remove border points, nms, refinement)
            radius=conf.nms_radius,
            top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
            scores_th=conf.detection_threshold,
            n_limit=(
                conf.max_num_keypoints
                if conf.max_num_keypoints > 0
                else self.n_limit_max
            ),
        )
        # Keypoint descriptor module "SDDH" from ALIKED
        self.descriptor_branch = SDDH(
            dim, K, M, gate=nn.SELU(inplace=True), conv2D=False, mask=False
        )
        ## Line Attraction Field information (Line Distance Field and Angle Field) ##
        # Line distance field decoder similar to that in DeepLSD
        self.distance_field_branch = nn.Sequential(
            nn.Conv2d(dim, conf.line_df_decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(conf.line_df_decoder_channels),
            nn.Conv2d(
                conf.line_df_decoder_channels,
                conf.line_df_decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conf.line_df_decoder_channels),
            nn.Conv2d(conf.line_df_decoder_channels, 1, kernel_size=1),
            nn.ReLU(),
        )
        # only use line angle-field if configured
        if conf.use_line_anglefield:
            # Angle branch similar to angle field decoder in DeepLSD
            self.angle_field_branch = nn.Sequential(
                nn.Conv2d(dim, conf.line_af_decoder_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(conf.line_af_decoder_channels),
                nn.Conv2d(
                    conf.line_af_decoder_channels,
                    conf.line_af_decoder_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(conf.line_af_decoder_channels),
                nn.Conv2d(conf.line_af_decoder_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            logger.warning("-- USE OF ANGLE FIELD IS DEACTIVATED! --")

        self.timings = None
        if conf.timeit:
            self.timings = {
                "total-makespan": [],
                "encoder": [],
                "keypoint-and-junction-heatmap": [],
                "line-df": [],
                "descriptor-branch": [],
                "keypoint-detection": [],
            }
            if conf.line_detection.do:
                self.timings["line-detection"] = []
            if conf.use_line_anglefield:
                self.timings["line-af"] = []

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.training.do and conf.training.aliked_pretrained:
            logger.warning("Load pretrained weights for aliked parts...")
            self.load_pretrained_aliked_elements()

        # Initialize Lightweight ALIKED model to perform on-the-fly ground-truth generation for descriptors if training in one-view setting
        if conf.training.do and conf.training.train_descriptors.use_one_view_loss:
            logger.warning("Load ALiked Lightweight model for descriptor training...")
            aliked_gt_cfg = {
                "model_name": self.conf.training.train_descriptors.gt_aliked_model,
                "max_num_keypoints": self.conf.max_num_keypoints,
                "detection_threshold": self.conf.detection_threshold,
                "force_num_keypoints": False,
                "pretrained": True,
                "nms_radius": self.conf.nms_radius,
            }
            self.aliked_lw = get_model("extractors.aliked_light")(aliked_gt_cfg).eval()

        # load model checkpoint if given -> only load weights
        if conf.checkpoint is not None:
            if Path(conf.checkpoint).exists():
                logger.warning(
                    f"Load model parameters from local checkpoint {conf.checkpoint}"
                )
                chkpt_statedict = torch.load(
                    conf.checkpoint, map_location=torch.device("cpu")
                )
            else:
                logger.warning(
                    f"Try Load model parameters from URL checkpoint {conf.checkpoint}"
                )
                chkpt_statedict = torch.hub.load_state_dict_from_url(
                    conf.checkpoint, map_location="cpu"
                )

            # Extract from two-view
            chkpt_statedict["model"] = {
                k.split("extractor.")[-1]: v
                for k, v in chkpt_statedict["model"].items()
            }

            # remove mlp weights from line detection
            chkpt_statedict["model"] = {
                k: v for k, v in chkpt_statedict["model"].items() if not ("mlp" in k)
            }
            # if angle field is not wanted we filter out its weights if existent so we can also load old checkpoints including this branch
            if not self.conf.use_line_anglefield:
                chkpt_statedict["model"] = {
                    k: v
                    for k, v in chkpt_statedict["model"].items()
                    if not ("angle_field_branch" in k)
                }

            self.load_state_dict(
                chkpt_statedict["model"], strict=True
            )  # set to True to check if all keys are present (mlp weights are not present as we removed them above)
        elif conf.checkpoint is not None:
            chkpt_statedict = torch.hub.load_state_dict_from_url(
                conf.checkpoint, map_location=torch.device("cpu")
            )
            # Extract from two-view
            chkpt_statedict["model"] = {
                k.split("extractor.")[-1]: v
                for k, v in chkpt_statedict["model"].items()
            }
            self.load_state_dict(chkpt_statedict["model"], strict=False)

        # Load line extractor and import line metrics if line detection is used
        if self.conf.line_detection.do:
            logger.info(f"Load line extractor: {self.conf.line_detection.name}")
            self.line_extractor = get_model(self.conf.line_detection.name)(
                self.conf.line_detection.conf
            )

        # only load deeplsd model if we perform ablation or development
        if self.conf.line_detection.do and (
            self.conf.line_detection.use_deeplsd_kp
            or self.conf.line_detection.use_deeplsd_df_af
        ):
            deeplsd_conf = {
                "detect_lines": True,
                "line_detection_params": {
                    "merge": True,
                    "filtering": True,
                    "grad_thresh": 3,
                    "grad_nfa": True,
                },
                "weights": "DeepLSD/weights/deeplsd_md.tar",  # path to the weights of the DeepLSD model (relative to DATA_PATH)
            }
            deeplsd_conf = OmegaConf.create(deeplsd_conf)
            ckpt_path = DATA_PATH / deeplsd_conf.weights
            ckpt = torch.load(
                str(ckpt_path), map_location=torch.device("cpu"), weights_only=False
            )
            deeplsd_net = DeepLSD(deeplsd_conf)
            deeplsd_net.load_state_dict(ckpt["model"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.deeplsd = deeplsd_net.to(device).eval()

    # Utility methods for line distance-field for (de)normalization
    def normalize_df(self, df: torch.Tensor) -> torch.Tensor:
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm: torch.Tensor) -> torch.Tensor:
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data: dict) -> torch.Tensor:
        """
        Perform a forward pass. Certain things are only executed NOT in training mode.
        Returned:
            - Probabilistic Keypoint Heatmap
            - Detected Keypoints
            - Keypoint descriptors (sparse, do one for every detected keypoint)
            - DeepLSD like Distance field (denormalized)
            - DeepLSD like Angle Field (between -Pi and Pi as radians)
            - Detected Lines (if line detection activated)
        """
        if self.conf.timeit:
            total_start = sync_and_time()
        # output container definition
        output = {}

        # pad image
        image = data["image"]
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        padded_img = padder.pad(image)

        # pass through encoder
        if self.conf.timeit:
            start_encoder = sync_and_time()
        feature_map_padded = self.encoder_backbone(padded_img)
        if self.conf.timeit:
            self.timings["encoder"].append(sync_and_time() - start_encoder)

        # pass through keypoint & junction decoder
        if self.conf.timeit:
            start_keypoints = sync_and_time()
        score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)
        if self.conf.timeit:
            self.timings["keypoint-and-junction-heatmap"].append(
                sync_and_time() - start_keypoints
            )

        # normalize and remove padding and format dimensions
        feature_map_padded_normalized = torch.nn.functional.normalize(
            feature_map_padded, p=2, dim=1
        )
        feature_map = padder.unpad(feature_map_padded_normalized)
        logger.debug(
            f"Image size: {image.shape}\nFeatureMap-unpadded: {feature_map.shape}\nFeatureMap-padded: {feature_map_padded.shape}"
        )
        keypoint_and_junction_score_map = padder.unpad(
            score_map_padded
        )  # B x 1 x H x W

        # For storing, remove additional dimension but keep batch dimension even if its 1
        # but keep additional dimension for variable -> needed by dkd
        if keypoint_and_junction_score_map.shape[0] == 1:
            output["keypoint_and_junction_score_map"] = keypoint_and_junction_score_map[
                :, 0, :, :
            ]  # B x H x W
        else:
            output["keypoint_and_junction_score_map"] = (
                keypoint_and_junction_score_map.squeeze()
            )  # B x H x W

        ## Line DF Decoder ##
        if self.conf.timeit:
            start_line_df = sync_and_time()
        line_distance_field = self.denormalize_df(
            self.distance_field_branch(feature_map)
        )  # denormalize as NN outputs normalized version
        # remove additional dimensions of size 1 if not having batchsize one
        line_distance_field = (
            line_distance_field.squeeze(1)
            if line_distance_field.shape[0] == 1
            else line_distance_field.squeeze()
        )
        if self.conf.timeit:
            self.timings["line-df"].append(sync_and_time() - start_line_df)
        output["line_distancefield"] = line_distance_field

        ## Line AF Decoder ##
        if self.conf.use_line_anglefield:
            if self.conf.timeit:
                start_line_af = sync_and_time()
            line_angle_field = (
                self.angle_field_branch(feature_map) * torch.pi
            )  # multipy with pi as output is in [0, 1] and we want angle
            # remove additional dimensions of size 1 if not having batchsize one
            line_angle_field = (
                line_angle_field.squeeze(1)
                if line_distance_field.shape[0] == 1
                else line_angle_field.squeeze()
            )
            if self.conf.timeit:
                self.timings["line-af"].append(sync_and_time() - start_line_af)
            output["line_anglefield"] = line_angle_field

        # Keypoint detection
        if self.conf.timeit:
            start_keypoints = sync_and_time()

        # Keypoint detection also removes kp at border. it can return topk keypoints or set of thresholded kp.
        keypoints, _, kptscores = self.dkd(
            keypoint_and_junction_score_map,
            sub_pixel=bool(self.conf.subpixel_refinement),
        )
        if self.conf.timeit:
            self.timings["keypoint-detection"].append(sync_and_time() - start_keypoints)

        # raw output of DKD needed to generate GT-Descriptors (ONLY done if one-view-loss used)
        if (
            self.conf.training.do
            and self.conf.training.train_descriptors.use_one_view_loss
        ):
            output["keypoints_raw"] = keypoints

        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required, can set detection_threshold=-1 and conf.max_num_keypoints -> HERE WE SET THESE VALUES
        # SO WE CAN EXPECT SAME NUM!
        rescaled_kp = wh * (torch.stack(keypoints) + 1.0) / 2.0
        output["keypoints"] = rescaled_kp
        output["keypoint_scores"] = torch.stack(kptscores)

        # Keypoint descriptors
        if self.conf.timeit:
            start_desc = sync_and_time()

        keypoint_descriptors, _ = self.descriptor_branch(feature_map, keypoints)

        if self.conf.timeit:
            self.timings["descriptor-branch"].append(sync_and_time() - start_desc)

        output["descriptors"] = torch.stack(keypoint_descriptors)  # B N D

        ## Line Detection ##
        # Only Perform line detection when NOT in training mode
        if self.conf.line_detection.do and not self.training:
            if self.conf.timeit:
                start_lines = sync_and_time()

            if output.get("line_anglefield", None) is None:
                # create dummy so that zipping works
                line_angle_field = torch.zeros_like(line_distance_field)

            # Perform forward pass for line detector, batching handled internally
            line_data = {
                "line_anglefield": line_angle_field.detach().clone(),
                "line_distancefield": line_distance_field.detach().clone(),
                "image": image,
                "keypoints": rescaled_kp.clone(),
                "kp_descriptors": output["descriptors"].clone(),
            }
            pred_line_data = self.line_extractor(line_data)

            output["lines"] = torch.stack(pred_line_data["lines"], dim=0)
            if self.conf.line_detection.conf.return_line_descriptors:
                output["line_descriptors"] = torch.stack(
                    pred_line_data["line_descriptors"], dim=0
                )
            output["valid_lines"] = torch.stack(pred_line_data["valid_lines"], dim=0)

            # Use aliked points sampled from inbetween Line endpoints?
            if self.conf.timeit:
                self.timings["line-detection"].append(sync_and_time() - start_lines)

        if self.conf.timeit:
            self.timings["total-makespan"].append(sync_and_time() - total_start)
        return output

    def weighted_bce_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of the weighted BCE loss to cope with class imbalance between keypoint- and non-keypoint pixels.
        We use this loss for leaning the keypoint heatmap.
        """
        epsilon = 1e-6
        return -self.lambda_valid_kp * target * torch.log(prediction + epsilon) - (
            1 - target
        ) * torch.log(1 - prediction + epsilon)

    def focal_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of the full-focal loss to cope with class imbalance between keypoint- and non-keypoint pixels and
        to focus more on hard examples. We use this loss for leaning the keypoint heatmap.
        """
        alpha = self.conf.training.loss.kp_loss_parameters.focal_alpha
        gamma = self.conf.training.loss.kp_loss_parameters.focal_gamma
        epsilon = 1e-6  # Small value to avoid log(0)

        # Compute the positive and negative parts of the focal loss
        pos_part = (
            -alpha * torch.pow(1 - prediction, gamma) * torch.log(prediction + epsilon)
        )
        neg_part = (
            -(1 - alpha)
            * torch.pow(prediction, gamma)
            * torch.log(1 - prediction + epsilon)
        )

        # Combine the parts to get the total loss
        loss = target * pos_part + (1 - target) * neg_part
        return loss

    def warp_data(self, df, angle, H, ps: list):
        h, w = df.shape[1:3]
        ps = tuple(ps)

        # Warp the closest point on a line
        pix_loc = (
            torch.stack(
                torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1
            )
            .to(df.device)
            .float()
        )

        warped_dfs = []

        for i in range(df.shape[0]):
            with torch.no_grad():  # TODO: could be batched or made simpler to warp the distance field?
                offset = df[i][:, :, None] * torch.stack(
                    [torch.sin(angle[i]), torch.cos(angle[i])], dim=-1
                )
            closest = pix_loc + offset
            warped_closest = warp_points_torch(
                closest.reshape(-1, 2).unsqueeze(0), H[i], inverse=False
            ).reshape(h, w, 2)
            warped_pix_loc = warp_points_torch(
                pix_loc.reshape(-1, 2).unsqueeze(0), H[i], inverse=False
            ).reshape(h, w, 2)

            offset_norm = torch.linalg.norm(offset, dim=-1)
            zero_offset = offset_norm < 1e-3
            offset_norm[zero_offset] = 1
            scaling = (
                torch.linalg.norm(warped_closest - warped_pix_loc, dim=-1) / offset_norm
            )
            scaling[zero_offset] = 0

            # Warp the DF
            warped_df = warp_perspective(
                df[i][None, None], H[i].unsqueeze(0), ps, mode="bilinear"
            ).squeeze()
            warped_scaling = warp_perspective(
                scaling[None, None], H[i].unsqueeze(0), ps, mode="bilinear"
            ).squeeze()
            warped_df *= warped_scaling

            warped_dfs.append(warped_df)

        return torch.stack(warped_dfs)

    def loss(self, pred: dict, data: dict) -> dict:
        """
        format of data: B x H x W
        perform loss calculation based on prediction and data(=groundtruth) for a batch.
        If predictions contain padding_mask we consider this on loss calculation
        1. On Keypoint-ScoreMap:        weighted BCE Loss / BCE Loss / Focal Loss
        2. On Keypoint-Descriptors:     L1 loss
        3. On Line-Angle Field:         use angle loss from deepLSD paper (ONLY IF AF ACTIVATED!)
        4. On Line-Distance Field:      use L1 loss on normalized versions of Distance field (as in deepLSD paper)
        """

        losses = {}
        metrics = {}
        losses["total"] = 0

        prediction_dict = {}
        if self.conf.training.two_view:
            for k, v in pred.items():
                if k.endswith("0"):
                    prediction_dict[k[:-1]] = v
                else:
                    prediction_dict[k] = v
        else:
            prediction_dict = pred

        # Load view0 ground truth if two view
        gt_dict_view0 = data["view0"]["cache"] if self.conf.training.two_view else data
        gt_dict_view1 = data["view1"]["cache"] if self.conf.training.two_view else None
        H = (
            data["H_0to1"] if self.conf.training.two_view else None
        )  # for each sample in batch there is a separate homography
        H_inv = (
            torch.linalg.inv(H) if self.conf.training.two_view else None
        )  # inv calc supports batch of matrix

        img = data["view0"]["image"] if self.conf.training.two_view else data["image"]
        # define padding mask which is only ones if no padding is used -> makes loss compatible with any scaling technique and whether padding is used or not
        padding_mask_view0 = gt_dict_view0.get(
            "padding_mask", torch.ones_like(img)
        )[  # TODO: padding is not the same
            :, 0, :, :
        ].int()
        df_gt_mask_view0 = (
            gt_dict_view0["deeplsd_distance_field"] < self.conf.line_neighborhood
        )
        df_gt_mask_view1 = (
            gt_dict_view1["deeplsd_distance_field"] < self.conf.line_neighborhood
            if self.conf.training.two_view
            else None
        )

        # Use BCE, WeightedBCE or Focal Loss for point position loss
        if self.conf.training.loss.use_one_view_kp_loss:
            keypoint_scoremap_loss = self.loss_fn(
                prediction_dict["keypoint_and_junction_score_map"] * padding_mask_view0,
                gt_dict_view0["superpoint_heatmap"] * padding_mask_view0,
            ).mean(dim=(1, 2))
            losses["one_view_kp_scoremap"] = keypoint_scoremap_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.keypoint_weight
                * keypoint_scoremap_loss
            )

        # If training descriptors: decide between one-view and two-view node
        if self.conf.training.train_descriptors.use_one_view_loss:
            # in case of one view: generate gt descriptors to directly supervise using l1 loss
            data = {
                **data,
                **self.get_groundtruth_descriptors(
                    {
                        "keypoints": prediction_dict["keypoints_raw"],
                        "image": gt_dict_view0["image"],
                    }
                ),
            }
            keypoint_descriptor_loss = F.l1_loss(
                prediction_dict["descriptors"],
                data["aliked_descriptors"],
                reduction="none",
            ).mean(dim=(1, 2))
            losses["one_view_kp_descriptors"] = keypoint_descriptor_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.one_view_descriptor_weight
                * keypoint_descriptor_loss
            )

        # Calculate two view loss (Sparse NRE for descriptors if wanted)
        if (
            self.conf.training.train_descriptors.use_two_view_loss
            and self.conf.training.two_view
        ):
            # Two Way sparse NRE computation

            # best match in kp of A - for each kp projected from B to A
            matches_B_to_A = compute_matches(
                prediction_dict["keypoints"],
                prediction_dict["keypoints1"],
                H,
                best_match_only=True,
            )

            # best match in kp of B - for each kp projected from A to B
            matches_A_to_B = compute_matches(
                prediction_dict["keypoints1"],
                prediction_dict["keypoints"],
                H_inv,
                best_match_only=True,
            )

            # returns overall mean scalar, need to repeat on batch dim for total loss.
            keypoint_descriptor_lossBA = sparse_nre_loss(
                prediction_dict["descriptors"],  # (B, N_A, Dim)
                prediction_dict["descriptors1"],  # (B, N_B, Dim)
                matches_B_to_A,  # (M, 3)
            )

            # returns overall mean scalar, need to repeat on batch dim for total loss.
            keypoint_descriptor_lossAB = sparse_nre_loss(
                prediction_dict["descriptors1"],  # (B, N_B, Dim)
                prediction_dict["descriptors"],  # (B, N_A, Dim)
                matches_A_to_B,  # (M, 3)
            )
            overall_mean = (keypoint_descriptor_lossBA + keypoint_descriptor_lossAB) / 2
            # repeat mean loss across batch dimension to have same mean later on consolidation
            losses["two_view_kp_descriptors"] = overall_mean.repeat(img.shape[0])
            losses["total"] += (
                self.conf.training.loss.loss_weights.two_view_descriptor_weight
                * losses["two_view_kp_descriptors"]
            )

        # use angular loss for anglefield, only if use of af is activated
        if (
            self.conf.use_line_anglefield
            and self.conf.training.loss.use_one_view_af_loss
        ):
            af_diff = (
                gt_dict_view0["deeplsd_angle_field"]
                - prediction_dict["line_anglefield"]
            )
            line_af_loss = (
                torch.minimum(af_diff**2, (torch.pi - af_diff.abs()) ** 2)
                * padding_mask_view0
            ).mean(
                dim=(1, 2)
            )  # pixelwise minimum
            losses["one_view_line_af"] = line_af_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.one_view_line_af_weight
                * line_af_loss
            )

        # Distance field loss. Depends on the pipeline (two-view or one-view)
        # use normalized versions for loss
        if self.conf.training.loss.use_one_view_df_loss:
            line_df_loss = F.l1_loss(
                self.normalize_df(prediction_dict["line_distancefield"])
                * df_gt_mask_view0
                * padding_mask_view0,
                self.normalize_df(gt_dict_view0["deeplsd_distance_field"])
                * df_gt_mask_view0
                * padding_mask_view0,
                # only supervise in line neighborhood
                reduction="none",
            ).mean(dim=(1, 2))
            losses["one_view_line_df"] = line_df_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.one_view_line_df_weight
                * line_df_loss
            )

        # Two view df consistency loss
        if self.conf.training.two_view and self.conf.training.loss.use_two_view_df_loss:
            # img1 to img0
            warped_df_1_to_0 = self.warp_data(
                df=prediction_dict["line_distancefield1"],
                angle=data["view1"]["cache"]["deeplsd_angle_field"],
                H=H_inv,
                ps=tuple(prediction_dict["line_distancefield"].shape[1:]),
            ).squeeze(1)
            # valid mask - warp image of ones from view1 to view0. Padding with 0 around warped part gives mask
            valid_mask_1_to_0 = warp_perspective(
                torch.ones_like(
                    prediction_dict["line_distancefield1"].unsqueeze(1),
                    device=prediction_dict["line_distancefield1"].device,
                ),
                H_inv,
                tuple(prediction_dict["line_distancefield1"].shape[1:]),
                mode="nearest",
            ).squeeze(1)

            loss_1to0 = F.l1_loss(  # TODO: use df-gt-mask in two-view consistency loss? (could introduce bias)
                self.normalize_df(prediction_dict["line_distancefield"])
                * df_gt_mask_view0
                * padding_mask_view0
                * valid_mask_1_to_0,
                self.normalize_df(warped_df_1_to_0)
                * df_gt_mask_view0
                * padding_mask_view0
                * valid_mask_1_to_0,
                reduction="none",
            ).mean(
                dim=(1, 2)
            )

            # img0 to img1
            warped_df_0to1 = self.warp_data(
                df=prediction_dict["line_distancefield"],
                angle=gt_dict_view0["deeplsd_angle_field"],  # Note: view0 angle field
                H=H,  # Note: H instead of H_inv
                ps=tuple(prediction_dict["line_distancefield1"].shape[1:]),
            ).squeeze(1)
            # valid mask for 0->1 warping
            valid_mask_0to1 = warp_perspective(
                torch.ones_like(
                    prediction_dict["line_distancefield"].unsqueeze(1),
                    device=prediction_dict["line_distancefield"].device,
                ),
                H,  # Note: H instead of H_inv
                tuple(prediction_dict["line_distancefield"].shape[1:]),
                mode="nearest",
            ).squeeze(1)
            # compute loss
            loss_0to1 = F.l1_loss(
                self.normalize_df(prediction_dict["line_distancefield1"])
                * df_gt_mask_view1
                * padding_mask_view0
                * valid_mask_0to1,
                self.normalize_df(warped_df_0to1)
                * df_gt_mask_view1
                * padding_mask_view0
                * valid_mask_0to1,
                reduction="none",
            ).mean(dim=(1, 2))

            losses["two_view_line_df"] = (loss_0to1 + loss_1to0) / 2
            losses["total"] += (
                self.conf.training.loss.loss_weights.two_view_line_df_weight
                * losses["two_view_line_df"]
            )

        # soft argmax loss (Only applicable with two-view pipeline)
        if (
            self.conf.training.two_view
            and self.conf.training.loss.refinement_radius > 0
            and self.conf.training.loss.loss_weights.softargmax_weight > 0
        ):
            loc_loss = soft_argmax_only_loss(
                pred["keypoint_and_junction_score_map0"],
                pred["keypoint_and_junction_score_map1"],
                gt_dict_view0["keypoints"],
                gt_dict_view0["keypoint_scores"] > 0,
                H,
                self.conf.training.loss.refinement_radius,
            )
            losses["soft_argmax_kp_loss"] = loc_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.softargmax_weight * loc_loss
            )

        # add metrics if in validation mode
        if not self.training:
            metrics = self.metrics(pred, data)
        return losses, metrics

    def get_groundtruth_descriptors(self, pred: dict) -> torch.Tensor:
        """
        Takes keypoints from predictions + computes ground-truth descriptors for it.
        """
        assert (
            pred.get("image", None) is not None
            and pred.get("keypoints", None) is not None
        )
        with torch.no_grad():
            descriptors = self.aliked_lw(pred)
        return descriptors

    def load_pretrained_aliked_elements(self) -> None:
        """
        Loads ALIKED weights for backbone encoder, score_head(SMH) and SDDH
        """
        # Load state-dict of wanted aliked-model
        aliked_state_url = aliked_checkpoint_url.format(self.conf.aliked_model_name)
        aliked_state_dict = torch.hub.load_state_dict_from_url(
            aliked_state_url, map_location="cpu"
        )
        # change keys
        for k, _ in list(aliked_state_dict.items()):
            if k.startswith("block") or k.startswith("conv"):
                change_dict_key(aliked_state_dict, k, f"encoder_backbone.{k}")
            elif k.startswith("score_head"):
                if not self.conf.training.pretrain_kp_decoder:
                    del aliked_state_dict[k]
                else:
                    change_dict_key(
                        aliked_state_dict, k, f"keypoint_and_junction_branch.{k}"
                    )
            elif k.startswith("desc_head"):
                change_dict_key(aliked_state_dict, k, f"descriptor_branch.{k[10:]}")
            else:
                continue

        # load values
        self.load_state_dict(aliked_state_dict, strict=False)

    def state_dict(self, *args, **kwargs):
        """
        Custom state dict to exclude aliked_lw module from checkpoint.
        """
        sd = super().state_dict(*args, **kwargs)
        # don't store lightweight aliked model for descriptor gt computation
        if self.conf.training.train_descriptors.use_one_view_loss:
            for k in list(sd.keys()):
                if k.startswith("aliked_lw"):
                    del sd[k]
        return sd

    def get_current_timings(self, reset: bool = False) -> dict:
        """
        It returns the median of the current forward-pass timings in a dictionary for
        all the single network parts. Raises ValueError if timings are not activated.

        reset: if True deletes all collected times until now
        """
        if self.timings is None:
            raise ValueError(
                "Inner Timings not activated/initialized. Hence, cannot get current timings"
            )
        results = {}
        for k, v in self.timings.items():
            results[k] = np.median(v)
            if reset:
                self.timings[k] = []
        return results

    @staticmethod
    def get_pr(
        pred_kp: torch.Tensor, gt_kp: torch.Tensor, tol: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the precision and recall, based on GT KP.
        """
        if len(gt_kp) == 0:
            precision = float(len(pred_kp) == 0)
            recall = 1.0
        elif len(pred_kp) == 0:
            precision = 1.0
            recall = float(len(gt_kp) == 0)
        else:
            dist = torch.norm(pred_kp[:, None] - gt_kp[None], dim=2)
            close = (dist < tol).float()
            precision = close.max(dim=1)[0].mean()
            recall = close.max(dim=0)[0].mean()
        return precision, recall

    def metrics(self, pred: dict, data: dict) -> dict:
        """
        Compute evaluation metrics for points. Also for lines if they are contained in the output
        Args:
            pred: dict, containing predictions made by the model
            data: dict containing image data and ground truth

        Returns: dict, containing the computed metrics
        """
        return {}
        device = pred["keypoint_and_junction_score_map"].device
        gt = data["superpoint_heatmap"].cpu().numpy()
        predictions = pred["keypoint_and_junction_score_map"].cpu().numpy()
        # Compute the precision and recall
        warped_outputs, Hs = self._get_warped_outputs(data)
        warped_predictions = (
            warped_outputs["keypoint_and_junction_score_map"].cpu().numpy()
        )

        precision, recall, _ = compute_pr(gt, predictions)
        loc_error_points = compute_loc_error(gt, predictions)
        rep_points = compute_repeatability(predictions, warped_predictions, Hs)
        out = {
            "precision": torch.tensor(
                precision.copy(), dtype=torch.float, device=device
            ),
            "recall": torch.tensor(recall.copy(), dtype=torch.float, device=device),
            "repeatability_points": torch.tensor(
                [rep_points], dtype=torch.float, device=device
            ),
            "loc_error_points": torch.tensor(
                [loc_error_points], dtype=torch.float, device=device
            ),
        }
        if "lines" in warped_outputs:
            lines = pred["lines"]
            warped_lines = warped_outputs["lines"]
            rep_lines, loc_error_lines = get_rep_and_loc_error(
                lines, warped_lines, Hs, predictions[0].shape, [50], [3]
            )
            out["repeatability_lines"] = torch.tensor(
                rep_lines, dtype=torch.float, device=device
            )
            out["loc_error_lines"] = torch.tensor(
                loc_error_lines, dtype=torch.float, device=device
            )

        return out

    def _get_warped_outputs(self, data: dict) -> tuple[dict, list[torch.Tensor]]:
        """
        given image data. Generate random homographies, warp the input images with them and run the joint
        point line detection on the warped images.

        Returns: tuple - (jpl output from warped images, homographies used to warp original image)
        """
        imgs = data["image"]
        device = data["image"].device
        batch_size = imgs.shape[0]
        data_shape = imgs.shape[2:]
        warped_imgs = torch.empty(imgs.shape, dtype=torch.float, device=device)
        Hs = torch.empty((batch_size, 3, 3), dtype=torch.float, device=device)
        for i in range(batch_size):
            H = torch.tensor(
                sample_homography(data_shape, **default_H_params),
                dtype=torch.float,
                device=device,
            )
            Hs[i] = H
            warped_imgs[i] = warp_perspective(
                imgs[i].unsqueeze(0), H.unsqueeze(0), data_shape, mode="bilinear"
            )
        with torch.no_grad():
            warped_outputs = self({"image": warped_imgs})
        return warped_outputs, Hs


def compute_matches(
    keypoints_imA: torch.Tensor,
    keypoints_imB: torch.Tensor,
    H: torch.Tensor,
    matching_threshold: float = 3.0,
    best_match_only: bool = False,
) -> torch.Tensor:
    """
    Projects keypoints from image B to image A using a homography matrix H.
    Returns a list of matches for kp projected from B to A to all Points in A.
    if best match only - find best matching keypoint in img A for projected kp from B to A
    Args:
        keypoints_imA: keypoints in image A (B, N_A, 2)
        keypoints_imB: keypoints in image B (B, N_B, 2)
        H: homography matrix from image A to image B (B, 3, 3)
        matching_threshold: the threshold for a keypoint to be considered a match for another
    Returns:
        Tensor: (M, 3) where M is the number of matches (0 or 1 for best match). Each row (batch_idx, keypoint_idx_imA, keypoint_idx_imB)
    """
    # Warp detected Kp from Image B to Image A
    warped_points = warp_points_torch(keypoints_imB, H, inverse=True)  # (B, N_B, 2)
    # Compute distance for each warped point p_BA to all points P_A
    # keypoints_imA[:, :, None, :] -> (B, N_A, 1, 2)
    # warped_points[:, None, :, :] -> (B, 1, N_B, 2)
    # Broadcasting results in -> (B, N_A, N_B, 2)
    # After norm computation -> (B, N_A, N_B)
    dists = torch.linalg.norm(
        keypoints_imA[:, :, None, :] - warped_points[:, None, :, :], axis=-1
    )  # (B, N_A, N_BA)

    if best_match_only:
        # For each keypoint in B, find the closest keypoint in A
        min_dists, best_matches_A = torch.min(
            dists, dim=1
        )  # both (B, N_B) idx + value of best match
        valid_matches = min_dists < matching_threshold  # (B, N_B)

        # Get batch indices, and kp indices, of kp in B and corresponding kp indices in imA indices
        batch_idx, idx_B = torch.where(
            valid_matches
        )  # both: (M,) where M = number of valid matches (here its only 0 or 1)
        idx_A = best_matches_A[batch_idx, idx_B]  # (M,)

        matches = torch.stack([batch_idx, idx_A, idx_B], dim=1)
    else:
        batch_idx, idx_A, idx_B = torch.where(
            dists < matching_threshold
        )  # all: (M,), M=num-matches
        matches = torch.stack([batch_idx, idx_A, idx_B], dim=1)  # (M, 3)

    return matches  # (M, 3)


def sparse_nre_loss(
    descriptors1: torch.Tensor,
    descriptors2: torch.Tensor,
    matches: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Compute the Sparse Neural Reprojection Error (NRE) loss for batched input.
    For each keypoint in A (descr 1), get softmax out of best matching keypoint projected from B (descr 2) to A.
    For this to work, the number of keypoints and tus matches must be the same for all samples in the batch.

    Args:
        descriptors1 (torch.Tensor): Descriptors from image 1 (B, N1, D).
        descriptors2 (torch.Tensor): Descriptors from image 2 (B, N2, D).
        matches (torch.Tensor): shape (M, 3) where each row contains
                               (batch_idx, idx_kp_desc1, idx_kp_desc2).
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        torch.Tensor: Computed Sparse NRE loss (scalar), here we return scalar loss as per sample calculation would be
                      way less efficient (need to build loss vector for later after return)
    """
    # if there is no match return 0
    if matches.shape[0] == 0:
        return torch.tensor(0.0, device=descriptors1.device, requires_grad=True)

    # Extract batch indices and keypoint indices
    batch_idx = matches[:, 0]  # (M,)
    kp_idx1 = matches[:, 1]  # (M,)
    kp_idx2 = matches[:, 2]  # (M,)

    # Batch wise similarity calculation (need softmax prob over all descr of img 2 for kp in img 1)
    # Extract matched descriptors from image 1 -> For each kp in Im1 for that a match exists, get descriptor
    desc1_matched = descriptors1[batch_idx, kp_idx1]  # (M, D)
    M = matches.shape[0]

    # For each match, we need descriptors for whole image of respective sample -> (M, N2, D)
    desc2_expanded = descriptors2[batch_idx]  # (M, N2, D)

    # Expand desc1_matched to match: (M, D) -> (M, 1, D)
    desc1_expanded = desc1_matched.unsqueeze(1)  # (M, 1, D)

    # Compute similarities: (M, 1, D) @ (M, D, N2) -> (M, 1, N2)
    similarities = torch.bmm(
        desc1_expanded, desc2_expanded.transpose(-2, -1)
    )  # (M, 1, N2)
    similarities = similarities.squeeze(1)  # (M, N2)

    # Subtract 1 and apply temperature scaling, then apply softmax along N2 dimension
    softmax_probs = torch.softmax((similarities - 1) / temperature, dim=1)  # (M, N2)

    # Get probabilities for correct matches using advanced indexing
    correct_match_probs = softmax_probs[torch.arange(M), kp_idx2]  # (M,)

    # Compute negative log likelihood
    loss = -torch.log(correct_match_probs + 1e-8).mean()
    return loss
