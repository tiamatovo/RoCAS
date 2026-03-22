# -*- coding: utf-8 -*-
import sys
import os

if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import glob

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)


def log(s):
    print(s, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image as PILImage
    import torch
    torch.set_num_threads(1)
    try:
        torch.backends.mkldnn.set_flags(False)
    except Exception:
        pass
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_ch=3, out_ch=1):
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
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(self.pool3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    MASK_EXTS = ('.png', '.tif', '.tiff')
    size = 128 if args.cpu else 256
    if args.cpu:
        log("CPU 训练使用输入尺寸 128×128 以降低内存占用")

    imgs, masks = [], []
    search_dirs = [args.mask_dir] if args.mask_dir else []
    if args.img_dir and args.img_dir not in search_dirs:
        search_dirs.append(args.img_dir)
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        for p in glob.glob(os.path.join(args.img_dir, ext)):
            name = os.path.splitext(os.path.basename(p))[0]
            found = False
            for mask_dir_i in search_dirs:
                for suf in ('', '_mask'):
                    for mext in MASK_EXTS:
                        m = os.path.join(mask_dir_i, name + suf + mext)
                        if os.path.isfile(m):
                            masks.append(m)
                            imgs.append(p)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    if len(imgs) < 10:
        log(f"有效样本不足（需至少10对），当前: {len(imgs)}。")
        sys.exit(1)

    class DS(Dataset):
        def __len__(self):
            return len(imgs)
        def __getitem__(self, i):
            try:
                pil_img = PILImage.open(imgs[i]).convert("RGB")
                pil_mask = PILImage.open(masks[i]).convert("L")
                img = np.array(pil_img.resize((size, size), PILImage.BILINEAR), dtype=np.float32) / 255.0
                mask = np.array(pil_mask.resize((size, size), PILImage.NEAREST), dtype=np.float32)
                mask = (mask > 127).astype(np.float32)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
                img = img.transpose(2, 0, 1)
                return torch.from_numpy(img).float(), torch.from_numpy(mask).unsqueeze(0).float()
            except Exception:
                return torch.zeros(3, size, size), torch.zeros(1, size, size)

    if args.cpu:
        device = torch.device("cpu")
        log("训练使用 CPU（已忽略 GPU）")
        batch_size = min(args.batch_size, 2)
        if batch_size < args.batch_size:
            log(f"CPU 训练为防内存不足，batch_size 已由 {args.batch_size} 降为 {batch_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = args.batch_size
    log("正在创建 DataLoader...")
    loader = DataLoader(DS(), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    log("正在创建模型...")
    model = UNet(3, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    log("开始训练...")
    it = iter(loader)
    imgs_b, masks_b = next(it)
    imgs_b, masks_b = imgs_b.to(device), masks_b.to(device)
    with torch.no_grad():
        _ = model(imgs_b)
    log("预热(仅前向)完成，开始正式训练...")
    import json
    train_losses = []
    for ep in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for imgs_b, masks_b in loader:
            imgs_b, masks_b = imgs_b.to(device), masks_b.to(device)
            opt.zero_grad()
            out = model(imgs_b)
            loss = F.binary_cross_entropy_with_logits(out, masks_b)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
        avg = total_loss / max(n, 1)
        train_losses.append(round(avg, 6))
        log(f"Epoch {ep+1}/{args.epochs} loss={avg:.4f}")
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save({"model_state": model.state_dict()}, args.save_path)
    log(f"模型已保存: {args.save_path}")
    history_path = args.save_path.rsplit(".", 1)[0] + "_history.json"
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({"train_loss": train_losses, "epochs": len(train_losses)}, f, indent=2)
        log(f"训练曲线已保存: {history_path}")
    except Exception as e:
        log(f"保存训练曲线失败: {e}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("U-Net training error:", e, flush=True)
        traceback.print_exc()
        sys.exit(1)
