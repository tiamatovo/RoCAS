# -*- coding: utf-8 -*-
import os
from typing import Tuple, Optional
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 2) -> None:  # 修改为2分类输出
        super().__init__()

        self.enc1 = self.conv_block(in_ch, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final(dec1)


def load_seg_model(ckpt_path: str, device: str = "cpu") -> Optional[Tuple[nn.Module, torch.device]]:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"model path is not exist: {ckpt_path}")
        return None

    dev = torch.device(device if device in ("cpu", "cuda") else "cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=dev)

        print(f"model keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else '非字典类型'}")

    except Exception as e:
        print(f"load model failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    state = None
    if isinstance(ckpt, dict):
        #  {'model_state_dict': state_dict, ...}
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
            print("load model_state_dict ")
        # 兼容旧版本 {"model_state": state_dict}
        elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state = ckpt["model_state"]
            print("load model_state ")
        elif all(isinstance(k, str) for k in ckpt.keys()):
            if any(k.startswith(('enc', 'dec', 'bottleneck', 'final', 'upconv', 'pool', 'conv', 'model.')) for k in
                   list(ckpt.keys())[:10]):
                state = ckpt
                print("load state_dict ")
            else:
                possible_keys = ['state_dict', 'model', 'net', 'network', 'module', 'state_dict', 'model_state_dict']
                for key in possible_keys:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state = ckpt[key]
                        print(f"load {key}")
                        break
    elif isinstance(ckpt, OrderedDict):
        state = ckpt
        print("load OrderedDict ")
    else:
        print(f"not support type: {type(ckpt)}")
        return None

    if state is None:
        print(
            f" {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
        return None

    model = UNet(3, out_ch=2).to(dev) 

    try:
        model.load_state_dict(state, strict=True)
        print(f"UNet(input_channels=3, output_channels=2)")
    except RuntimeError as e:
        try:
            model.load_state_dict(state, strict=False)
            print(f"UNet(input_channels=3, output_channels=2)")
        except Exception as e2:
            print(f"{e2}")
            import traceback
            traceback.print_exc()
            return None

    model.eval()
    return model, dev


def _preprocess_image_for_unet(bgr: np.ndarray, size: int = 256) -> torch.Tensor:

    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std


    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    # HWC -> CHW
    chw = resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    return tensor


def infer_mask(model: nn.Module, device: torch.device, bgr_image: np.ndarray, input_size: int = 256) -> Optional[
    np.ndarray]:
    if bgr_image is None or bgr_image.size == 0:
        print("image is none")
        return None

    h, w = bgr_image.shape[:2]
    try:
        with torch.no_grad():
            x = _preprocess_image_for_unet(bgr_image, size=input_size).to(device)
            logits = model(x)

            print(f"模型输出形状: {logits.shape}")

            pred = torch.argmax(logits, dim=1)

            pred_single = pred[0].cpu().numpy() 

            print(f" {pred_single.shape}")
            print(f" {np.unique(pred_single)}")


            mask_small = (pred_single == 1).astype(np.uint8) * 255

            print(f" {mask_small.shape}")
            print(f" {np.min(mask_small)} - {np.max(mask_small)}")

            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

            print(f" {mask.shape}")
            print(f" {np.min(mask)} - {np.max(mask)}")

            return mask
    except Exception as e:
        print(f"infer_mask failed: {e}")
        import traceback
        traceback.print_exc()
        return None