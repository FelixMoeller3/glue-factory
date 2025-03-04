"""
CNN classifier for candidate line endpoints using DF and AF values sampled along the line.
Use the following command to train the CNN:
    python -m gluefactory.train pold2_cnn_test --conf gluefactory/configs/pold2_cnn_train.yaml
Use the following command to plot the confusion matrix):
    python -m gluefactory.models.lines.pold2_cnn \
        --conf gluefactory/configs/pold2_cnn_train.yaml \
        --weights outputs/training/pold2_cnn_test/checkpoint_best.tar
Use the following command to test the dataloader:
    python -m gluefactory.datasets.pold2_cnn_dataset --conf gluefactory/configs/pold2_cnn_dataloader_test.yaml
"""

import argparse
import logging

import torch
from omegaconf import OmegaConf
from torch import nn

from gluefactory.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class POLD2_CNN(BaseModel):

    default_conf = {
        "name": "lines.pold2_cnn",
        "has_angle_field": True,
        "has_distance_field": True,
        "num_line_samples": 30,  # number of sampled points between line endpoints
        "num_bands": 1,  # number of bands to sample along the line
        "band_width": 1,  # width of the band to sample along the line
        "cnn_hidden_dims": [256, 128, 128, 64, 32],
        "pred_threshold": 0.9,
        "weights": None,
        "device": None,
        "brute_force_samples": False,
    }

    def _init(self, conf):
        if conf.device is not None:
            self.device = conf.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        input_dim = 0
        if conf.has_angle_field:
            input_dim += conf.num_line_samples
        if conf.has_distance_field:
            input_dim += conf.num_line_samples
        input_dim *= conf.num_bands
        if input_dim == 0:
            raise ValueError("No input features selected for CNN")

        cnn_layers = []
        # cnn_layers.append(nn.Conv1d(1,5,3,1,1))
        cnn_layers.append(nn.Conv2d(1, 4, 3, 1, 1))
        cnn_layers.append(nn.ReLU())
        # cnn_layers.append(nn.Conv1d(5,5,3,1,1))
        cnn_layers.append(nn.Conv2d(4, 8, 3, 1, 1))
        cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Conv2d(8, 16, 3, 1, 1))
        cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Flatten())
        cnn_layers.append(nn.Linear(16 * input_dim, conf.cnn_hidden_dims[0]))
        for i in range(1, len(conf.cnn_hidden_dims)):
            cnn_layers.append(
                nn.Linear(conf.cnn_hidden_dims[i - 1], conf.cnn_hidden_dims[i])
            )
            cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Linear(conf.cnn_hidden_dims[-1], 1))
        self.cnn = nn.Sequential(*cnn_layers)

        if self.conf.weights is not None:
            ckpt = torch.load(
                str(self.conf.weights), map_location=self.device, weights_only=True
            )
            self.load_state_dict(ckpt["model"], strict=True)
            logger.info(f"Successfully loaded model weights from {self.conf.weights}")
        self.cnn.to(self.device)

        self.set_initialized()

    def _forward(self, data: torch.Tensor):
        x = data["input"]  # B x Num_Bands*Num_Samples
        df = x[:, : self.conf.num_bands * self.conf.num_line_samples].reshape(
            -1, self.conf.num_bands, self.conf.num_line_samples
        )
        af = x[:, self.conf.num_bands * self.conf.num_line_samples :].reshape(
            -1, self.conf.num_bands, self.conf.num_line_samples
        )
        x = torch.cat([df, af], dim=2)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return {"line_probs": torch.sigmoid(self.cnn(x))}

    def loss(self, pred, data):

        losses = {}

        # Compute the loss (BCE loss between predictions and labels)
        labels = data["label"]
        x_pred = pred["line_probs"]

        loss = nn.BCELoss()(x_pred.reshape(-1), labels.float())
        losses["total"] = loss.unsqueeze(0)

        metrics = self.metrics(pred, data)
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                metrics[k] = v.unsqueeze(0)

        return losses, metrics

    def metrics(self, pred, data, eps=1e-7):
        labels = data["label"].flatten()
        x_pred = pred["line_probs"].flatten()

        device = labels.device

        x_pred_th = (x_pred > self.conf.pred_threshold).float()

        tp = (x_pred_th * labels).sum().item()
        tn = ((1 - x_pred_th) * (1 - labels)).sum().item()
        fp = (x_pred_th * (1 - labels)).sum().item()
        fn = ((1 - x_pred_th) * labels).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)

        return {
            "accuracy": torch.tensor(accuracy, dtype=torch.float, device=device),
            "precision": torch.tensor(precision, dtype=torch.float, device=device),
            "recall": torch.tensor(recall, dtype=torch.float, device=device),
            "f1": torch.tensor(f1, dtype=torch.float, device=device),
            "f1_inv": torch.tensor(1 - f1, dtype=torch.float, device=device),
        }


__main_model__ = POLD2_CNN

# Run the model and plot the confusion matrix
if __name__ == "__main__":
    from ... import logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    conf = (
        OmegaConf.load(args.conf) if args.conf is not None else POLD2_CNN.default_conf
    )
    conf.weights = args.weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = POLD2_CNN(conf).to(device)
    model.eval()

    # Load the data
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import metrics

    from gluefactory.datasets.pold2_mlp_dataset import POLD2_MLP_Dataset

    dataset = POLD2_MLP_Dataset(conf.data)
    dataloader = dataset.get_data_loader("val")

    actual = []
    predicted = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == len(dataloader) - 1:
            break

        y = batch["label"]
        batch["input"] = batch["input"].to(device)
        batch["label"] = batch["label"].to(device)

        with torch.no_grad():
            x_pred = model(batch)
            x_pred = x_pred["line_probs"]

            actual.append(y.cpu().numpy())
            predicted.append(x_pred.cpu().numpy())

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    predicted = predicted > conf.model.pred_threshold

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[0, 1]
    )

    cm_display.plot()
    output_path = Path(args.weights).parent / "confusion_matrix.png"
    plt.savefig(output_path)
    logger.info(f"Confusion matrix saved to {output_path}")
