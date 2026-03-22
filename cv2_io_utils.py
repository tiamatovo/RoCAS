# -*- coding: utf-8 -*-
"""
OpenCV 图像读写与中文路径兼容
Windows 下 cv2.imread/imwrite 不支持含中文路径，此处通过「先读/写字节再解码/编码」统一处理。
"""
import os
import numpy as np
import cv2


def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """
    读取图像，支持中文等 Unicode 路径（如 Windows 下含中文的路径）。
    与 cv2.imread 行为一致，返回 numpy 数组或 None。
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, flags)
        return img
    except Exception:
        return None


def cv2_imwrite(path, img, params=None):
    """
    写入图像，支持中文等 Unicode 路径。
    与 cv2.imwrite 行为一致，返回 True/False。
    """
    if img is None:
        return False
    try:
        ext = os.path.splitext(path)[1].lower()
        if not ext or ext not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
            ext = '.png'
        success, buf = cv2.imencode(ext, img, params or [])
        if not success:
            return False
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False
