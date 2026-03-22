# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

def cv2_imread(path, flags=cv2.IMREAD_COLOR):
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
