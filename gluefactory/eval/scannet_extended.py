import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import torch
import torch.utils
import torch.utils.data
import multiprocessing as mp
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.eval.eval_pipeline import (
    EvalPipeline,
    exists_eval,
    load_eval,
    save_eval,
)
from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args
from gluefactory.models import BaseModel
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.models.utils.metrics_lines import (
    compute_loc_error,
    compute_repeatability,
)
from gluefactory.settings import EVAL_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.utils.tensor import map_tensor
from gluefactory.visualization.viz2d import plot_images, plot_lines, save_plot


class ScannetPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "scannet_1500",
            "num_workers": 1,
            "preprocessing": {
                "resize": 480,  # we also resize during eval to have comparable metrics
                "side": "short",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
        "use_points": True,
        "use_lines": False,
        "repeatability_th": [1, 3, 5],
        "num_lines_th": [10, 50, 300],
        "ransac_thresholds": [
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
        ],
        "pose_thresholds": [5, 10, 20]
    }
    export_keys = []

    optional_export_keys = [
        "lines0",
        "lines1",
        "orig_lines0",
        "orig_lines1",
        "line_matches0",
        "line_matches1",
        "line_matching_scores0",
        "line_matching_scores1",
        "line_distances",
    ]

    def _init(self, conf):
        if conf.use_points:
            self.export_keys += [
                "keypoints0",
                "keypoints1",
                "keypoint_scores0",
                "keypoint_scores1",
                "matches0",
                "matches1",
                "matching_scores0",
                "matching_scores1",
            ]
        if conf.use_lines:
            self.export_keys += [
                "lines0",
                "lines1",
                "line_matches0",
                "line_matches1",
                "line_matching_scores0",
                "line_matching_scores1",
            ]

    def intrinsics_to_camera(self, K):
        px, py = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        return {
            "model": "PINHOLE",
            "width": int(2 * px),
            "height": int(2 * py),
            "params": [fx, fy, px, py],
        }

    def angle_error_vec(self, v1, v2):
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


    def angle_error_mat(self, R1, R2):
        cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
        cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
        return np.rad2deg(np.abs(np.arccos(cos)))


    def compute_pose_error(self, T_0to1, R, t):
        R_gt = T_0to1[:3, :3]
        t_gt = T_0to1[:3, 3]
        error_t = self.angle_error_vec(t, t_gt)
        error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
        error_R = self.angle_error_mat(R, R_gt)
        return error_t, error_R


    def estimate_pose(self, kpts0, kpts1, K0, K1, thresh, conf=0.99999, type="poselib"):
        if len(kpts0) < 5:
            return None
        if type == "poselib":
            import poselib

            (pose, details) = poselib.estimate_relative_pose(
                kpts0.tolist(),
                kpts1.tolist(),
                self.intrinsics_to_camera(K0),
                self.intrinsics_to_camera(K1),
                ransac_opt={
                    "max_iterations": 10000,  # default 100000
                    "success_prob": conf,  # default 0.99999
                    "max_epipolar_error": thresh,  # default 1.0
                },
                bundle_opt={},  # all defaults
            )
            ret = (pose.R, pose.t, details["inliers"])

        elif type == "opencv":
            f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
            norm_thresh = thresh / f_mean

            kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
            kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

            E, mask = cv2.findEssentialMat(
                kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.RANSAC
            )

            assert E is not None

            best_num_inliers = 0
            ret = None
            for _E in np.split(E, len(E) / 3):
                n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
                if n > best_num_inliers:
                    best_num_inliers = n
                    ret = (R, t[:, 0], mask.ravel() > 0)
        else:
            raise NotImplementedError

        return ret


    def estimate_pose_parallel(self, args):
        return self.estimate_pose(*args)


    def pose_auc(self, errors, thresholds):
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]
        aucs = []
        for t in thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index - 1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.trapz(r, x=e) / t)
        return aucs


    def pose_accuracy(self, errors, thresholds):
        return [np.mean(errors < t) * 100 for t in thresholds]


    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        print(data_conf)
        dataset = get_dataset("scannet_1500")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(
        self,
        experiment_dir: Path,
        model: BaseModel | None = None,
        overwrite: bool = False,
    ) -> Path:
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run(
        self,
        experiment_dir: Path,
        model: BaseModel | None = None,
        overwrite=False,
        overwrite_eval=False,
        plot=False,
    ):
        """Run export+eval loop"""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(), pred_file, plot)
            # save_eval(experiment_dir, s, f, r)
        # s, r = load_eval(experiment_dir)
        return s, f, r

    def run_eval(
        self, loader: torch.utils.data.DataLoader, pred_file: Path, plot: bool
    ):
        assert pred_file.exists()
        results = defaultdict(list)

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        all_matches = []

        for i, data in enumerate(tqdm(loader)):

            pred = cache_loader(data)
            # pred['K0'] = data['K0']
            # pred['K1'] = data['K1']
            all_matches.append({"mkpts0": pred['keypoints0'][pred['matches0']],
                                "mkpts1": pred['keypoints1'][pred['matches1']],
                                "K0": data['K0'], "K1": data['K1'],
                                "T_0to1": data["T_0to1"]})
        
        aucs_by_thresh = {}
        accs_by_thresh = {}
        for ransac_thresh in self.conf["ransac_thresholds"]:

            # fname = os.path.join(
            #     self.conf["output"],
            #     f'{name}_{self.conf["eval"]["estimator"]}_{ransac_thresh}.txt',
            # )
            
            errors = []
            # pairs = self.pairs

            # do the benchmark in parallel
            if self.conf["data"]["num_workers"] != 1:

                pool = mp.Pool(self.conf["data"]["num_workers"])
                pool_args = [
                    (
                        all_matches[pair_idx]["mkpts0"],
                        all_matches[pair_idx]["mkpts1"],
                        all_matches[pair_idx]["K0"][0],
                        all_matches[pair_idx]["K1"][0],
                        ransac_thresh,
                    )
                    for pair_idx in range(len(all_matches))
                ]
                results = list(
                    tqdm(
                        pool.imap(self.estimate_pose_parallel, pool_args),
                        total=len(pool_args),
                        desc=f"Running benchmark for th={ransac_thresh}",
                        leave=False,
                    )
                )
                pool.close()

                for pair_idx, ret in enumerate(results):
                    if ret is None:
                        err_t, err_R = np.inf, np.inf
                    else:
                        R, t, inliers = ret
                        pair = all_matches[pair_idx]
                        err_t, err_R = self.compute_pose_error(pair["T_0to1"][0], R, t)
                    # errors_file.write(f"{err_t} {err_R}\n")
                    errors.append([err_t, err_R])
            # do the benchmark in serial
            else:
                for pair_idx, pair in tqdm(
                    enumerate(all_matches),
                    desc=f"Running benchmark for th={ransac_thresh}",
                    leave=False,
                    total=len(all_matches),
                ):
                    mkpts0 = pair["mkpts0"]
                    mkpts1 = pair["mkpts1"]
                    ret = self.estimate_pose(
                        mkpts0, mkpts1, pair["K0"][0], pair["K1"][0], ransac_thresh
                    )
                    if ret is None:
                        err_t, err_R = np.inf, np.inf
                    else:
                        R, t, inliers = ret
                        err_t, err_R = self.compute_pose_error(pair["T_0to1"][0], R, t)
                    # errors_file.write(f"{err_t} {err_R}\n")
                    # errors_file.flush()
                    errors.append([err_t, err_R])

                # errors_file.close()

            # compute AUCs
            errors = np.array(errors)
            errors = errors.max(axis=1)
            aucs = self.pose_auc(errors, self.conf["pose_thresholds"])
            accs = self.pose_accuracy(errors, self.conf["pose_thresholds"])
            aucs = {k: v * 100 for k, v in zip(self.conf["pose_thresholds"], aucs)}
            accs = {k: v for k, v in zip(self.conf["pose_thresholds"], accs)}
            aucs_by_thresh[ransac_thresh] = aucs
            accs_by_thresh[ransac_thresh] = accs

            # dump summary for this method
            summary = {
                "name": name,
                "aucs_by_thresh": aucs_by_thresh,
                "accs_by_thresh": accs_by_thresh,
            }
            # json.dump(
            #     summary,
            #     open(
            #         os.path.join(
            #             self.config["output"],
            #             f'{name}_{self.config["pose_estimator"]}_summary.json',
            #         ),
            #         "w",
            #     ),
            #     indent=2,
            # )
        figures = {}
        return summary, figures, figures
        for i, data in enumerate(tqdm(loader)):
            # if i in range(360,365):
            #     continue
            pred = cache_loader(data)
            # Remove batch dimension
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            # add custom evaluations here

            results_i = {}

            # # we also store the names for later reference
            # results_i["names"] = data["name"][0]
            # results_i["scenes"] = data["scene"][0]

            if "lines0" in pred:
                lines0 = pred["lines0"].cpu()
                lines1 = pred["lines1"].cpu()

                if plot:
                    plot_images(
                        [
                            data["view0"]["image"].permute(1, 2, 0),
                            data["view1"]["image"].permute(1, 2, 0),
                        ],
                        ["H0", "H1"],
                    )
                    plot_lines(lines=[pred["orig_lines0"], pred["orig_lines1"]])
                    save_plot(os.path.join("./match_score/", f"{i}.jpg"))
                    plt.close()

                results_i["repeatability"] = compute_repeatability(
                    lines0,
                    lines1,
                    pred["line_matches0"].cpu(),
                    pred["line_matches1"].cpu(),
                    pred["line_matching_scores0"].cpu(),
                    self.conf.repeatability_th,
                    rep_type="num",
                )
                results_i["loc_error"] = compute_loc_error(
                    pred["line_matching_scores0"].cpu(), self.conf.num_lines_th
                )
                results_i["num_lines"] = (lines0.shape[0] + lines1.shape[0]) / 2

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.median(arr), 3)

        if "repeatability" in results.keys():
            for i, th in enumerate(self.conf.repeatability_th):
                cur_nums = list(map(lambda x: x[i], results["repeatability"]))
                summaries[f"repeatability@{th}px"] = round(np.median(cur_nums), 3)
        if "loc_error" in results.keys():
            for i, th in enumerate(self.conf.num_lines_th):
                cur_nums = list(map(lambda x: x[i], results["loc_error"]))
                summaries[f"loc_error@{th}lines"] = round(np.median(cur_nums), 3)

        figures = {}

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ScannetPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = ScannetPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
        plot=args.plot,
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
