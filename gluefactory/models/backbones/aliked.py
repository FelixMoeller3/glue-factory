from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet

from gluefactory.models.base_model import BaseModel

# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]


class InputPadder(object):
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        mask=False,
    ):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.mask = mask

        self.channel_num = (
            3 * kernel_size * kernel_size if mask else 2 * kernel_size * kernel_size
        )
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.channel_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0

        out = self.offset_conv(x)
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
        else:
            offset = out
            mask = None
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=mask,
        )
        return x


def get_conv(
    inplanes,
    planes,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    conv_type="conv",
    mask=False,
):
    if conv_type == "conv":
        conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "dcn":
        conv = DeformableConv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=_pair(padding),
            bias=bias,
            mask=mask,
        )
    else:
        raise TypeError
    return conv


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = get_conv(
            in_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(
            out_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


# modified based on torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ResBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = get_conv(
            inplanes, planes, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = get_conv(
            planes, planes, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class ALIKED(BaseModel):
    default_conf = {
        "model_name": "aliked-n16",
        "pretrained": True,
    }

    checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"

    n_limit_max = 20000

    cfgs = {
        "aliked-t16": {
            "c1": 8,
            "c2": 16,
            "c3": 32,
            "c4": 64,
            "dim": 64,
            "K": 3,
            "M": 16,
        },
        "aliked-n16": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "K": 3,
            "M": 16,
        },
        "aliked-n16rot": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "K": 3,
            "M": 16,
        },
        "aliked-n32": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "K": 3,
            "M": 32,
        },
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        # get configurations
        c1, c2, c3, c4, dim, K, M = [v for _, v in self.cfgs[conf.model_name].items()]
        conv_types = ["conv", "conv", "dcn", "dcn"]
        conv2D = False
        mask = False

        # build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = ResBlock(
            c1,
            c2,
            1,
            nn.Conv2d(c1, c2, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[1],
        )
        self.block3 = ResBlock(
            c2,
            c3,
            1,
            nn.Conv2d(c2, c3, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[2],
            mask=mask,
        )
        self.block4 = ResBlock(
            c3,
            c4,
            1,
            nn.Conv2d(c3, c4, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[3],
            mask=mask,
        )
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )

        # load pretrained
        if conf.pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                self.checkpoint_url.format(conf.model_name), map_location="cpu"
            )
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("desc_head")
            }
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("score_head")
            }
            self.load_state_dict(state_dict, strict=True)

    def extract_dense_map(self, image):
        # Pads images such that dimensions are divisible by
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32
        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1) # B x dim x H x W

        # Unpads images
        feature_map = padder.unpad(x1234)
        return feature_map

    def _forward(self, data):
        image = data["image"]
        feature_map = self.extract_dense_map(image)
        
        return feature_map
    
    def loss(self, pred, data):
        raise NotImplementedError
