import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from omegaconf import OmegaConf
from gluefactory.models.utils.metrics_lines import get_rep_and_loc_error
from gluefactory.datasets.homographies_deeplsd import sample_homography
from gluefactory.models import get_model
from gluefactory.models.backbones.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.base_model import BaseModel
from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.models.extractors.aliked import DKD, SDDH, SMH, InputPadder
from gluefactory.models.lines.pold2_extractor import LineExtractor
from gluefactory.models.utils.metrics_points import (
    compute_loc_error,
    compute_pr,
    compute_repeatability,
)
from gluefactory.settings import DATA_PATH
from gluefactory.utils.misc import change_dict_key, sync_and_time
from gluefactory.geometry.homography import warp_points_torch
from gluefactory.geometry.kp_losses import soft_argmax_only_loss

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
    default_conf = {
        "aliked_model_name": "aliked-n16",  # aliked model determining structure of our encoder
        "use_line_anglefield": True,  # if set to false, model will be initialized without AF branch and AF will not be output or considered in inference or training
        "line_df_decoder_channels": 64,
        "line_af_decoder_channels": 64,
        "max_num_keypoints": 1024,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "nms_radius": 3,
        "subpixel_refinement": True,  # perform subpixel refinement after detection
        "force_num_keypoints": False,
        "training": {  # training settings
            "two_view": False, # whether training is done with a two-view pipeline (True) or with a one-view pipeline (False)
            "do": False,  # switch to turn off other settings regarding training = "training mode"
            "aliked_pretrained": True,
            "pretrain_kp_decoder": True,
            "train_descriptors": { # for train decriptors in one-view: generate gt descriptrs, in two-view: use caps loss
                "do": True,  # if train is True, initialize ALIKED Light model form OTF Descriptor GT
                "gt_aliked_model": "aliked-n32",
            },  # if train is True, initialize ALIKED Light model form OTF Descriptor GT
            "loss": {
                "kp_loss_name": "weighted_bce",  # other options: bce, weighted_bce or focal loss
                "kp_loss_parameters": {
                    "lambda_weighted_bce": 200,  # weighted bce parameter factor how to boost keypoint loss in map
                    "focal_gamma": 5,
                    # focal loss parameter controlling how strong to focus on hard examples (typical range 1-5)
                    "focal_alpha": 0.8,  # focal loss parameter to mitigate class imbalances
                },
                "refinement_radius": 5,  # radius for softargmax loss
                "loss_weights": {
                    "line_af_weight": 10,
                    "line_df_weight": 10,
                    "keypoint_weight": 1,
                    "descriptor_weight": 1,
                    "softargmax_weight": 1,
                },
            },
        },
        "line_detection": {  # by default we use the POLD2 Line Extractor (MLP with Angle Field)
            "do": True,
            "conf": LineExtractor.default_conf,
            "use_deeplsd_kp": False,  # whether we should use DeepLSD line endpoints as junction candidates. Otherwise use JPLDD keypoints
            "use_deeplsd_df_af": False,  # whether we should use Distance and Angle Field from JPLDD or DeepLSD
        },
        "checkpoint": None,  # if given and non-null, load model checkpoint if local path load locally if standard url download it.
        "line_neighborhood": 5,  # used to normalize / denormalize line distance field
        "timeit": True,  # override timeit: False from BaseModel
    }

    # used for line detection ablation and development when we use deeplsd af/df or line endpoints
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
        self.dkd = DKD(  # Not learned Point detection with subpixel refinement (remove border points, nms, refinement)
            radius=conf.nms_radius,
            top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
            scores_th=conf.detection_threshold,
            n_limit=(
                conf.max_num_keypoints
                if conf.max_num_keypoints > 0
                else self.n_limit_max
            ),
        )  # Differentiable Keypoint Detection from ALIKE
        # Keypoint and line descriptors
        self.descriptor_branch = SDDH(
            dim, K, M, gate=nn.SELU(inplace=True), conv2D=False, mask=False
        )
        # Line Attraction Field information (Line Distance Field and Angle Field)
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
        if self.conf.use_line_anglefield:
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

        if conf.timeit:
            self.timings = {
                "total-makespan": [],
                "encoder": [],
                "keypoint-and-junction-heatmap": [],
                "line-af": [],
                "line-df": [],
                "descriptor-branch": [],
                "keypoint-detection": [],
            }
            if conf.line_detection.do:
                self.timings["line-detection"] = []

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.training.do and conf.training.aliked_pretrained:
            logger.warning("Load pretrained weights for aliked parts...")
            old_test_val1 = self.encoder_backbone.conv1.weight.data.clone()
            self.load_pretrained_aliked_elements()
            assert not torch.all(
                torch.eq(self.encoder_backbone.conv1.weight.data.clone(), old_test_val1)
            ).item()  # test if weights really loaded!

        # Initialize Lightweight ALIKED model to perform OTF GT generation for descriptors if training in one-view setting
        if conf.training.do and conf.training.train_descriptors.do and not conf.training.two_view:
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
        if conf.checkpoint is not None and Path(conf.checkpoint).exists():
            logger.warning(f"Load model parameters from checkpoint {conf.checkpoint}")
            chkpt = torch.load(conf.checkpoint, map_location=torch.device("cpu"))

            # remove mlp weights from line detection TODO: remove them when storing instead of filter on load
            chkpt["model"] = {
                k: v for k, v in chkpt["model"].items() if not ("mlp" in k)
            }
            # if angle field is not wanted we filter out its weights if existent so we can also load old checkpoints including this branch
            if not self.conf.use_line_anglefield:
                chkpt["model"] = {
                    k: v
                    for k, v in chkpt["model"].items()
                    if not ("angle_field_branch" in k)
                }

            self.load_state_dict(
                chkpt["model"], strict=True
            )  # set to True to check if all keys are present (mlp weights are not present as we removed them above)
        elif conf.checkpoint is not None:
            chkpt = torch.hub.load_state_dict_from_url(
                conf.checkpoint, map_location=torch.device("cpu")
            )
            self.load_state_dict(chkpt["model"], strict=False)

        # Load line extractor and import line metrics if line detection is used
        if self.conf.line_detection.do:

            self.line_extractor = LineExtractor(
                self.conf.line_detection.conf,
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

    # Utility methods for line df and af with deepLSD
    def normalize_df(self, df: torch.Tensor) -> torch.Tensor:
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm: torch.Tensor) -> torch.Tensor:
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data: dict) -> torch.Tensor:
        """
        Perform a forward pass. Certain things are only executed NOT in training mode.
        Returned:
            - Probabilistic Keypoint Heatmap
            - DeepLSD like Distance field (denormalized)
            - DeepLSD like Angle Field (between -Pi and Pi as radians)
        """
        if self.conf.timeit:
            total_start = sync_and_time()
        # output container definition
        output = {}

        # load image and padder
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
        assert (feature_map.shape[2], feature_map.shape[3]) == (
            image.shape[2],
            image.shape[3],
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

        # Keypoint detection also removes kp at border. it can return topk keypoints or threshold.
        keypoints, _, kptscores = self.dkd(
            keypoint_and_junction_score_map,
            sub_pixel=bool(self.conf.subpixel_refinement),
        )
        if self.conf.timeit:
            self.timings["keypoint-detection"].append(sync_and_time() - start_keypoints)

        # raw output of DKD needed to generate GT-Descriptors (ONLY done in ONE_VIEW training)
        if self.conf.training.train_descriptors.do and not self.conf.training.two_view:
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

        # Extract Lines from Learned Part of the Network
        # Only Perform line detection when NOT in training mode
        if self.conf.line_detection.do and not self.training: # TODO: we might need to do line detect during training for an end to end train setting
            if self.conf.timeit:
                start_lines = sync_and_time()
            lines = []
            valid_lines = []
            line_descs = []
            line_indices = []

            if output.get("line_anglefield", None) is None:
                # create dummy so that zipping works
                line_angle_field = torch.zeros_like(line_distance_field)

            for df, af, kp, desc in zip(
                line_distance_field, line_angle_field, rescaled_kp, keypoint_descriptors
            ):
                """
                "line_detection": {  # by default we use the POLD2 Line Extractor (MLP with Angle Field)
                    "do": True,
                    "conf": LineExtractor.default_conf,
                    "use_deeplsd_kp": False, # whether we should use DeepLSD line endpoints as junction candidates. Otherwise use JPLDD keypoints
                    "use_deeplsd_df_af": False # whether we should use Distance and Angle Field from JPLDD or DeepLSD
                },
                """
                # Only use deeplsd if explicitly activated
                if (
                    self.conf.line_detection.use_deeplsd_kp
                    or self.conf.line_detection.use_deeplsd_df_af
                ):
                    img = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    c_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
                    inputs = {
                        "image": torch.tensor(
                            gray_img, dtype=torch.float, device=padded_img.device
                        )[None, None]
                        / 255.0
                    }
                    with torch.no_grad():
                        deeplsd_output = self.deeplsd(inputs)
                    deeplsd_lines = np.array(deeplsd_output["lines"][0]).astype(int)

                    deeplsd_lines_torch = torch.tensor(deeplsd_lines).cuda()
                    deeplsd_lines_torch[:, :, 0] = torch.clamp(
                        deeplsd_lines_torch[:, :, 0], 0, img.shape[1]
                    )
                    deeplsd_lines_torch[:, :, 1] = torch.clamp(
                        deeplsd_lines_torch[:, :, 1], 0, img.shape[0]
                    )
                    keypoints_deeplsd = torch.cat(
                        (deeplsd_lines_torch[:, 0], deeplsd_lines_torch[:, 1])
                    )
                # prepare line data for line detection!
                line_data = {
                    "points": (
                        torch.clone(kp)
                        if not self.conf.line_detection.use_deeplsd_kp
                        else keypoints_deeplsd
                    ),
                    "distance_map": (
                        torch.clone(df)
                        if not self.conf.line_detection.use_deeplsd_df_af
                        else deeplsd_output["df"][0]
                    ),
                    "descriptors": (
                        torch.clone(desc)
                        if not self.conf.line_detection.use_deeplsd_kp
                        else torch.zeros((keypoints_deeplsd.shape[0], 128)).cuda()
                    ),
                    "angle_map": None,
                }
                if self.conf.use_line_anglefield:
                    line_data["angle_map"] = (
                        torch.clone(af)
                        if not self.conf.line_detection.use_deeplsd_df_af
                        else deeplsd_output["line_level"][0]
                    )

                line_pred = self.line_extractor(line_data)
                lines.append(line_pred["lines"])
                line_descs.append(line_pred["line_descriptors"])
                line_indices.append(line_pred["line_endpoint_indices"])
                # Line matchers expect the lines to be stored as line endpoints where line endpoint = coordinate of respective keypoint
                if len(lines) == 0:
                    print("NO LINES DETECTED")

                valid_lines.append(
                    torch.ones(len(lines[-1])).to(line_distance_field[-1].device)
                )
            output["lines"] = torch.stack(lines, dim=0)
            output["line_descriptors"] = torch.stack(line_descs, dim=0)
            output["valid_lines"] = torch.stack(valid_lines, dim=0)

            # Use aliked points sampled from inbetween Line endpoints?
            if self.conf.timeit:
                self.timings["line-detection"].append(sync_and_time() - start_lines)

        if self.conf.timeit:
            self.timings["total-makespan"].append(sync_and_time() - total_start)
        return output

    def weighted_bce_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-6
        return -self.lambda_valid_kp * target * torch.log(prediction + epsilon) - (
            1 - target
        ) * torch.log(1 - prediction + epsilon)

    def focal_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    
    def warp_data(self, df, offset, H, ps: list):
        h, w = offset.shape[1:3]
        ps = tuple(ps)

        # Warp the closest point on a line
        pix_loc = torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1
        ).to(offset.device).float()

        warped_dfs = []

        for i in range(df.shape[0]):
            
            closest = pix_loc + offset[i]
            warped_closest = warp_points_torch(closest.reshape(-1, 2), H, inverse=False).reshape(h, w, 2)
            warped_pix_loc = warp_points_torch(pix_loc.reshape(-1, 2), H, inverse=False).reshape(h, w, 2)
            
            offset_norm = torch.linalg.norm(offset[i], dim=-1)
            zero_offset = offset_norm < 1e-3
            offset_norm[zero_offset] = 1
            scaling = torch.linalg.norm(warped_closest - warped_pix_loc, dim=-1) / offset_norm
            scaling[zero_offset] = 0

            # Warp the DF
            warped_df = warp_perspective(df[i][None, None], H, ps, mode="bilinear").squeeze()
            warped_scaling = warp_perspective(scaling[None, None], H, ps, mode="bilinear").squeeze()
            warped_df *= warped_scaling

            warped_dfs.append(warped_df)

        return torch.stack(warped_dfs)

    def loss(self, pred: dict, data:dict) -> dict:
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

        prediction_dict = {}
        if self.conf.training.two_view:
            for k, v in pred.items():
                if k.endswith("0"):
                    prediction_dict[k[:-1]] = v 
        else:
            prediction_dict = pred

        gt_dict = data["view0"]["cache"] if self.conf.training.two_view else data
        H = data["H_0to1"] if self.conf.training.two_view else None

        img = data["view0"]["image"] if self.conf.training.two_view else data["image"]
        # define padding mask which is only ones if no padding is used -> makes loss compatible with any scaling technique and whether padding is used or not
        padding_mask = gt_dict.get("padding_mask", torch.ones_like(img))[
            :, 0, :, :
        ].int()

        # Use BCE, WeightedBCE or Focal Loss for point position loss
        keypoint_scoremap_loss = self.loss_fn(
            prediction_dict["keypoint_and_junction_score_map"] * padding_mask,
            gt_dict["superpoint_heatmap"] * padding_mask,
        ).mean(dim=(1, 2))

        losses["keypoint_and_junction_score_map"] = keypoint_scoremap_loss
        # If training descriptors: decide between one-view and two-view node
        if self.conf.training.train_descriptors.do:
            if not self.conf.training.two_view:
                # in case of one view: generate gt descriptors to directly supervise using l1 loss
                data = {
                **data,
                **self.get_groundtruth_descriptors(
                    {"keypoints": prediction_dict["keypoints_raw"], "image": gt_dict["image"]}
                ),
                }
                keypoint_descriptor_loss = F.l1_loss(
                    prediction_dict["descriptors"], gt_dict["aliked_descriptors"], reduction="none"
                ).mean(dim=(1, 2))
            else:
                # in case of two-view: use the caps window loss for descriptors
                matches = compute_matches(prediction_dict["keypoints"], pred["keypoints1"], H)
                keypoint_descriptor_loss = 0
                for b_idx in range(len(matches)):
                    keypoint_descriptor_loss += sparse_nre_loss(prediction_dict["descriptors"][b_idx],pred["descriptors1"][b_idx], matches[b_idx])
                keypoint_descriptor_loss /= len(matches)
            losses["descriptors"] = keypoint_descriptor_loss.unsqueeze(0)

        # use angular loss for anglefield, if use of af is activated
        if self.conf.use_line_anglefield:
            af_diff = gt_dict["deeplsd_angle_field"] - prediction_dict["line_anglefield"]
            line_af_loss = (
                torch.minimum(af_diff**2, (torch.pi - af_diff.abs()) ** 2)
                * padding_mask
            ).mean(
                dim=(1, 2)
            )  # pixelwise minimum
            losses["line_anglefield"] = line_af_loss

        # Distance field loss. Depends on the pipeline (two-view or one-view)
        # use normalized versions for loss
        gt_mask = gt_dict["deeplsd_distance_field"] < self.conf.line_neighborhood
        line_df_loss = F.l1_loss(
            self.normalize_df(pred["line_distancefield0"]) * gt_mask * padding_mask,
            self.normalize_df(data["view0"]["cache"]["deeplsd_distance_field"]) * gt_mask * padding_mask,
            # only supervise in line neighborhood
            reduction="none",
        ).mean(dim=(1, 2))
        losses["line_distancefield"] = line_df_loss
        if self.conf.training.two_view:
            # In case of two-view, add df consistency loss
            warped_df = self.warp_data(
                df = pred["line_distancefield1"],
                offset = data["view1"]["cache"]["deeplsd_offset_field"],
                H = torch.linalg.inv(H),
                ps = tuple(pred["line_distancefield0"].shape[1:])
            )
            valid_mask = warp_perspective(
                torch.ones_like(pred["line_distancefield0"][None], device=pred["line_distancefield0"].device),
                torch.linalg.inv(H),
                tuple(pred["line_distancefield0"].shape[1:]),
                mode="nearest",
            ).squeeze(1)

            # warped_df = warp_perspective(pred["line_distancefield1"][:,None,:,:],torch.linalg.inv(H), tuple(pred["line_distancefield0"].shape[1:]))
            warped_df = warped_df.squeeze(1)
            losses["line_distancefield"] += F.l1_loss(
                self.normalize_df(pred["line_distancefield0"]) * gt_mask * padding_mask * valid_mask,
                self.normalize_df(warped_df) * gt_mask * padding_mask * valid_mask,
                # only supervise in line neighborhood
                reduction="none",
            ).mean(dim=(1, 2))

        # Compute overall loss
        overall_loss = (
            self.conf.training.loss.loss_weights.keypoint_weight
            * losses["keypoint_and_junction_score_map"]
            + self.conf.training.loss.loss_weights.line_df_weight * losses["line_distancefield"]
        )
        if self.conf.use_line_anglefield:
            overall_loss += (
                self.conf.training.loss.loss_weights.line_af_weight * losses["line_anglefield"]
            )
        if self.conf.training.train_descriptors.do:
            overall_loss += (
                self.conf.training.loss.loss_weights.descriptor_weight
                * losses["descriptors"]
            )

        # soft argmax loss
        if self.conf.training.loss.refinement_radius > 0 and self.conf.training.loss.loss_weights.softargmax_weight > 0:
            loc_loss = soft_argmax_only_loss(
                pred["keypoint_and_junction_score_map0"],
                pred["keypoint_and_junction_score_map1"],
                data["view0"]["cache"]["keypoints"],
                data["view0"]["cache"]["keypoint_scores"] > 0,
                H,
                self.conf.training.loss.refinement_radius,
            )
            losses["loc_loss"] = loc_loss
            overall_loss += self.conf.training.loss.loss_weights.softargmax_weight * loc_loss


        losses["total"] = overall_loss

        # add metrics if not in training mode
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
        if self.conf.training.train_descriptors.do:
            for k in list(sd.keys()):
                if k.startswith("aliked_lw"):
                    del sd[k]
        return sd

    def get_current_timings(self, reset: bool=False) -> dict:
        """
        ONLY USE IF TIMEIT ACTIVATED. It returns the average of the current times in a dictionary for
        all the single network parts.

        reset: if True deletes all collected times until now
        """
        results = {}
        for k, v in self.timings.items():
            results[k] = np.median(v)
            if reset:
                self.timings[k] = []
        return results

    @staticmethod
    def get_pr(pred_kp: torch.Tensor, gt_kp: torch.Tensor, tol:int=3) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _get_warped_outputs(self, data):
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


def compute_matches(keypoints_im1: torch.Tensor, keypoints_im2: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    warped_points = warp_points_torch(keypoints_im2, H, inverse=True)
    dists = torch.linalg.norm(keypoints_im1[:,:,None,:] - warped_points[:,None,:,:],axis=-1)
    
    bs = keypoints_im1.shape[0]
    matches = []
    for b_idx in range(bs):
        matches.append(torch.stack(torch.where(dists[b_idx] < 3.0)).T)
    return matches
    

def sparse_nre_loss(descriptors1: torch.Tensor, descriptors2: torch.Tensor, matches: torch.Tensor, temperature: float=0.1):
    """
    Compute the Sparse Neural Reprojection Error (NRE) loss.

    Args:
        descriptors1 (torch.Tensor): Descriptors from image 1 (N1 x D).
        descriptors2 (torch.Tensor): Descriptors from image 2 (N2 x D).
        matches (list of tuples): List of matched keypoint indices [(i, j), ...].
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        torch.Tensor: Computed Sparse NRE loss.
    """
    # Extract matched descriptors
    desc1 = torch.stack([descriptors1[i] for i, _ in matches])  # (M x D)
    desc2 = torch.stack([descriptors2[j] for _, j in matches])  # (M x D)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(desc1, desc2.t())  # (M x M)

    # subtract 1 (as in paper)
    similarity_matrix -= 1

    # Apply temperature scaling
    similarity_matrix /= temperature

    # Create ground truth labels
    labels = torch.arange(len(matches), device=descriptors1.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

