import os
from typing import Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def load_seg_model(ckpt_path: str, device: str = "cpu") -> Optional[Tuple[nn.Module, torch.device]]:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None

    dev = torch.device(device if device in ("cpu", "cuda") else "cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=dev)
    except Exception:
        return None

    state = None
    if isinstance(ckpt, dict):
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state = ckpt["model_state"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state = ckpt

    if state is None:
        return None

    model = UNet(3, 1).to(dev)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        return None

    model.eval()
    return model, dev


def _preprocess_image_for_unet(bgr: np.ndarray, size: int = 256) -> torch.Tensor:
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    chw = resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    return tensor


def infer_mask(model: nn.Module, device: torch.device, bgr_image: np.ndarray, input_size: int = 256) -> Optional[np.ndarray]:

    if bgr_image is None or bgr_image.size == 0:
        return None

    import cv2

    h, w = bgr_image.shape[:2]
    try:
        with torch.no_grad():
            x = _preprocess_image_for_unet(bgr_image, size=input_size).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0]  # HxW
            mask_small = (prob > 0.5).float().cpu().numpy()
            mask_small = (mask_small * 255).astype(np.uint8)
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            return mask
    except Exception:
        return None

