
import os
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

import cv2
import numpy as np
import torch

from gluefactory.models.deeplsd_inference import DeepLSD

DATA_PATH = Path("/local/home/Point-Line/data")
OXPA_PATH = DATA_PATH / "revisitop1m_POLD2/jpg/"
LINE_WIDTH = 1

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

def initialize_model(device, conf):

    ## DeepLSD Model
    deeplsd_conf = OmegaConf.create(conf)

    ckpt_path = DATA_PATH / deeplsd_conf.weights
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    deeplsd_net = DeepLSD(deeplsd_conf)
    deeplsd_net.load_state_dict(ckpt["model"])
    deeplsd_net = deeplsd_net.to(device).eval()

    return deeplsd_net

def get_binary_line_heatmap(image_path, model, device):
    gray_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    with torch.no_grad():
        inputs = {
            "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                None, None
            ]
            / 255.0
        }
        deeplsd_output = model(inputs)
    deeplsd_lines = deeplsd_output["lines"][0].numpy().astype(int)

    # Create a binary heatmap of the detected lines
    heatmap = np.zeros_like(gray_img, dtype=np.uint8)
    for line in deeplsd_lines:
        cv2.line(heatmap, line[0], line[1], 255, LINE_WIDTH)

    ## DEBUG
    """
    for l in deeplsd_lines:
        cv2.line(gray_img, l[0], l[1], 255, LINE_WIDTH)

    print(gray_img.shape, heatmap.shape)
    joint_img = np.concatenate([gray_img, heatmap], axis=1)
    joint_img = cv2.cvtColor(joint_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("joint_img.jpg", joint_img)
    """

    return heatmap

def save_heatmaps_oxpa(model, device):
    fps = list(OXPA_PATH.glob("**/base_image.jpg"))
    for i, fp in enumerate(tqdm(fps)):
        # print(f"Processing image {i+1}/{len(fps)}: {fp}")
        save_path = fp.parent / "line_binary_heatmap.jpg"
        df_path = fp.parent / "df.jpg"

        if save_path.exists() or not df_path.exists():
            continue

        heatmap = get_binary_line_heatmap(fp, model, device)
        cv2.imwrite(str(save_path), heatmap)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(device, deeplsd_conf)
    save_heatmaps_oxpa(model, device)

if __name__ == "__main__":
    main()