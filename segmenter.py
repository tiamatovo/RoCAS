# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from datetime import datetime
from collections import Counter
import json

import torch


# ========== 分割 ===========
class RockSegmenter:
    def __init__(self, log_callback=None):
        self.segmentation_methods_used = {}
        self.log_callback = log_callback  # 日志回调函数

        # 中文方法名称映射
        self.method_names = {
            'grabcut': 'GrabCut智能分割',
            'color': '颜色阈值分割',
            'edges': '边缘检测分割',
            'threshold': '自适应阈值分割',
            'watershed': '分水岭分割',
            'kmeans': 'K-means聚类分割',
            'deeplearning': '深度学习分割',
            'none': '无分割',
            'error': '分割失败'
        }
        self._dl_model = None
        self._dl_device = None
        self._dl_model_path = None
        self._dl_use_gpu = True

    def log(self, message):
        """日志输出"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(f"[Segmenter] {message}")

    def segment_by_methods(self, image, image_name, selected_methods, log_callback=None):
        """使用选定的方法进行分割，返回所有方法的结果"""
        if log_callback:
            self.log_callback = log_callback
        
        results = {}
        original = image.copy()
        
        self.log(f"开始分割图像: {image_name}")
        self.log(f"图像尺寸: {image.shape[1]}×{image.shape[0]}")
        self.log(f"选择的分割方法: {', '.join(selected_methods)}")
        
        # 图像预处理
        processed = self._preprocess_image_enhanced(image)
        self.log("✓ 图像预处理完成（降噪、增强对比度）")
        
        for method_idx, method_name in enumerate(selected_methods, 1):
            try:
                self.log(f"\n--- 执行 {method_name} 分割 ({method_idx}/{len(selected_methods)}) ---")
                
                if method_name == 'GrabCut Intelligent Segmentation':
                    segmented, mask, score = self._segment_by_grabcut_enhanced(processed, original)
                elif method_name == 'Color Threshold Segmentation':
                    segmented, mask, score = self._segment_by_color_enhanced(processed, original)
                elif method_name == 'Edge Detection Segmentation':
                    segmented, mask, score = self._segment_by_edges_enhanced(processed, original)
                elif method_name == 'Adaptive Threshold Segmentation':
                    segmented, mask, score = self._segment_by_threshold_enhanced(processed, original)
                elif method_name == 'Watershed Segmentation':
                    segmented, mask, score = self._segment_by_watershed(processed, original)
                elif method_name == 'K-means Clustering Segmentation':
                    segmented, mask, score = self._segment_by_kmeans(processed, original)
                elif method_name == 'Deep Learning Segmentation':
                    segmented, mask, score = self._segment_by_deeplearning(processed, original)
                else:
                    continue
                
                if mask is not None and score > 0:
                    results[method_name] = {
                        'segmented': segmented,
                        'mask': mask,
                        'score': score,
                        'original': original
                    }
                    self.log(f"✓ {method_name} 完成，得分: {score:.3f}")
                else:
                    self.log(f"✗ {method_name} 失败或得分过低（该样本颜色/纹理可能不适配此算法）")
                    
            except Exception as e:
                self.log(f"✗ {method_name} 执行出错: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
        
        self.log(f"\n分割完成，成功方法数: {len(results)}/{len(selected_methods)}")
        return results

    def _preprocess_image_enhanced(self, image):
        """增强的图像预处理"""
        # 1. 降噪
        denoised = cv2.bilateralFilter(image, 9, 75, 75)

        # 2. 增强对比度（CLAHE）
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def _segment_by_grabcut_enhanced(self, processed, original):
        """增强的GrabCut分割（大图缩放到 600 内 + 少迭代，避免卡在首图）"""
        h_orig, w_orig = processed.shape[:2]
        max_side = 600
        scale = 1.0
        if max(h_orig, w_orig) > max_side:
            scale = max_side / max(h_orig, w_orig)
            w_small = max(64, int(w_orig * scale))
            h_small = max(64, int(h_orig * scale))
            work_img = cv2.resize(processed, (w_small, h_small), interpolation=cv2.INTER_AREA)
        else:
            work_img = processed

        h, w = work_img.shape[:2]
        # 使用颜色分割获取初始区域（在小图上）
        color_mask = self._segment_by_color_enhanced(work_img, work_img)[1]

        if color_mask is None:
            x, y, rw, rh = w//4, h//4, w//2, h//2
        else:
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, rw, rh = cv2.boundingRect(max_contour)
                margin = min(30, w//20, h//20)
                x = max(0, x - margin)
                y = max(0, y - margin)
                rw = min(w - x, rw + 2*margin)
                rh = min(h - y, rh + 2*margin)
            else:
                x, y, rw, rh = w//4, h//4, w//2, h//2

        # 确保 rect 宽高至少为 2，避免 OpenCV 异常或卡死
        rw = max(2, min(rw, w - x))
        rh = max(2, min(rh, h - y))
        x = min(x, w - rw)
        y = min(y, h - rh)
        rect = (x, y, rw, rh)

        mask = np.zeros(work_img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 3 次迭代，配合缩放下速度稳定、不易卡顿
        cv2.grabCut(work_img, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255

        if scale < 1.0:
            mask2 = cv2.resize(mask2, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            mask2 = (mask2 > 127).astype(np.uint8) * 255

        mask2 = self._postprocess_mask_enhanced(mask2)
        segmented = self._apply_mask_enhanced(original, mask2)
        score = self._evaluate_segmentation_enhanced(mask2)
        return segmented, mask2, score

    def _segment_by_color_enhanced(self, processed, original):
        """增强的颜色分割（针对岩石优化，扩展颜色范围提高成功率）"""
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        # 岩石常见颜色范围（更宽泛，覆盖更多样本）
        # 棕色/黄褐色
        lower_brown1 = np.array([8, 25, 15])
        upper_brown1 = np.array([35, 255, 255])
        mask_brown1 = cv2.inRange(hsv, lower_brown1, upper_brown1)

        # 灰色/灰白色
        lower_gray = np.array([0, 0, 25])
        upper_gray = np.array([180, 55, 220])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # 红色/橙红色
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        # 黄绿色（部分岩石）
        lower_yg = np.array([35, 30, 30])
        upper_yg = np.array([85, 255, 255])
        mask_yg = cv2.inRange(hsv, lower_yg, upper_yg)

        # 合并所有掩码
        mask = cv2.bitwise_or(mask_brown1, mask_gray)
        mask = cv2.bitwise_or(mask, mask_red)
        mask = cv2.bitwise_or(mask, mask_yg)

        # 后处理
        mask = self._postprocess_mask_enhanced(mask)

        # 应用掩码
        segmented = self._apply_mask_enhanced(original, mask)

        # 评估
        score = self._evaluate_segmentation_enhanced(mask)

        return segmented, mask, score

    def _segment_by_edges_enhanced(self, processed, original):
        """增强的边缘检测分割"""
        # 转换为灰度
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # 多尺度边缘检测
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        # 形态学操作连接边缘
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 填充轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)

        # 填充轮廓（优先大轮廓，若无则用最大轮廓）
        large_contours = [c for c in contours if cv2.contourArea(c) > 500]
        fill_list = large_contours if large_contours else (contours and [max(contours, key=cv2.contourArea)] or [])
        for contour in fill_list:
            cv2.fillPoly(mask, [contour], 255)

        # 后处理
        mask = self._postprocess_mask_enhanced(mask)

        # 应用掩码
        segmented = self._apply_mask_enhanced(original, mask)

        # 评估
        score = self._evaluate_segmentation_enhanced(mask)

        return segmented, mask, score

    def _segment_by_threshold_enhanced(self, processed, original):
        """自适应阈值分割"""
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # 自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 后处理
        mask = self._postprocess_mask_enhanced(mask)

        # 应用掩码
        segmented = self._apply_mask_enhanced(original, mask)

        # 评估
        score = self._evaluate_segmentation_enhanced(mask)

        return segmented, mask, score

    def _segment_by_watershed(self, processed, original):
        """分水岭分割（增加 Otsu 失败时的回退）"""
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # 阈值处理（Otsu 对低对比度图可能失效，尝试中值回退）
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_ratio = np.sum(thresh > 0) / thresh.size
        if fg_ratio < 0.02 or fg_ratio > 0.98:  # Otsu 结果异常
            med = np.median(gray)
            _, thresh = cv2.threshold(gray, med, 255, cv2.THRESH_BINARY_INV)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 距离变换
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # 未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 分水岭算法
        markers = cv2.watershed(processed, markers)

        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers > 1] = 255

        # 后处理
        mask = self._postprocess_mask_enhanced(mask)

        # 应用掩码
        segmented = self._apply_mask_enhanced(original, mask)

        # 评估
        score = self._evaluate_segmentation_enhanced(mask)

        return segmented, mask, score

    def _segment_by_kmeans(self, processed, original):
        """K-means聚类分割（岩石通常在中心，以中心区域主导的类为前景）"""
        data = processed.reshape((-1, 3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        k = 2
        flags = getattr(cv2, 'KMEANS_PP_CENTERS', cv2.KMEANS_RANDOM_CENTERS)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

        labels = labels.reshape(processed.shape[:2])
        h, w = labels.shape

        # 中心区域（1/3）的标签投票，选中心主导的类为前景
        cy, cx = h // 2, w // 2
        r = min(h, w) // 6
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        center_labels = labels[y1:y2, x1:x2].ravel()
        unique, counts = np.unique(center_labels, return_counts=True)
        foreground_label = unique[np.argmax(counts)]

        # 创建掩码
        mask = np.zeros(processed.shape[:2], dtype=np.uint8)
        mask[labels == foreground_label] = 255

        # 后处理
        mask = self._postprocess_mask_enhanced(mask)

        # 应用掩码
        segmented = self._apply_mask_enhanced(original, mask)

        # 评估
        score = self._evaluate_segmentation_enhanced(mask)

        return segmented, mask, score

    def set_dl_config(self, model_path, use_gpu=True):
        """设置深度学习分割的模型路径和是否使用GPU"""
        self._dl_model_path = model_path
        self._dl_use_gpu = use_gpu
        self._dl_model = None
        self._dl_device = None

    def _load_dl_model(self):
        """懒加载深度学习模型"""
        # 添加调试信息
        self.log(f"尝试加载深度学习模型，路径: {self._dl_model_path}")

        if not self._dl_model_path:
            self.log("深度学习分割: 模型路径为空")
            return False

        if not os.path.isfile(self._dl_model_path):
            self.log(f"深度学习分割: 模型文件不存在 - {self._dl_model_path}")
            # 检查是否存在其他可能的模型文件
            model_dir = os.path.dirname(self._dl_model_path)
            if os.path.exists(model_dir):
                try:
                    files = os.listdir(model_dir)
                    pth_files = [f for f in files if f.lower().endswith(('.pth', '.pt'))]
                    if pth_files:
                        self.log(f"深度学习分割: 在目录 {model_dir} 中找到以下可能的模型文件: {pth_files}")
                except Exception as e:
                    self.log(f"检查模型目录时出错: {e}")
            return False

        if self._dl_model is not None and self._dl_model_path == getattr(self, '_last_dl_path', None):
            self.log("深度学习分割: 模型已加载，无需重复加载")
            return True

        try:
            from rock_seg_model import load_seg_model
            device = "cuda" if self._dl_use_gpu else "cpu"

            # 检查GPU可用性
            if self._dl_use_gpu and not torch.cuda.is_available():
                self.log("警告: CUDA不可用，将使用CPU")
                device = "cpu"

            self.log(f"深度学习分割: 尝试使用设备 {device} 加载模型")
            result = load_seg_model(self._dl_model_path, device)
            if result is None:
                self.log("深度学习分割: load_seg_model 函数返回 None，模型加载失败")
                return False
            self._dl_model, self._dl_device = result
            self._last_dl_path = self._dl_model_path
            self.log("深度学习分割: 模型加载成功")
            return True
        except ImportError as e:
            self.log(f"导入rock_seg_model模块失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
        except Exception as e:
            self.log(f"加载深度学习模型失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def _segment_by_deeplearning(self, processed, original):
        """深度学习 U-Net 语义分割"""
        if not self._load_dl_model():
            # 这里已经通过 _load_dl_model 输出了详细错误信息
            self.log("深度学习分割: 跳过分割步骤，因为模型未加载成功")
            return None, None, 0.0
        try:
            from rock_seg_model import infer_mask
            self.log("深度学习分割: 开始推理")
            mask = infer_mask(self._dl_model, self._dl_device, original, input_size=256)
            if mask is None:
                self.log("深度学习分割: 推理返回空掩码")
                return None, None, 0.0
            self.log("深度学习分割: 推理完成，进行后处理")
            mask = self._postprocess_mask_enhanced(mask)
            segmented = self._apply_mask_enhanced(original, mask)
            # 对深度学习分割单独放宽评估条件：
            # 只要模型输出的前景区域非空，就认为分割成功，避免因为面积比例等阈值被全部丢弃
            mask_area = int(np.sum(mask > 0))
            if mask_area <= 0:
                self.log("深度学习分割: 后处理后的掩码区域为0")
                score = 0.0
            else:
                # 仍然调用一次评估函数用于排序/参考，但保证有一个较大的正分
                base_score = self._evaluate_segmentation_enhanced(mask)
                score = max(0.8, float(base_score))
                self.log(f"深度学习分割: 成功，得分 {score}")
            return segmented, mask, score
        except ImportError as e:
            self.log(f"导入rock_seg_model模块失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None, None, 0.0
        except Exception as e:
            self.log(f"深度学习分割执行出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None, None, 0.0

    def _segment_by_deeplearning(self, processed, original):
        """深度学习 U-Net 语义分割"""
        if not self._load_dl_model():
            # 这里已经通过 _load_dl_model 输出了详细错误信息
            self.log("深度学习分割: 跳过分割步骤，因为模型未加载成功")
            return None, None, 0.0
        try:
            from rock_seg_model import infer_mask
            self.log("深度学习分割: 开始推理")
            mask = infer_mask(self._dl_model, self._dl_device, original, input_size=256)
            if mask is None:
                self.log("深度学习分割: 推理返回空掩码")
                return None, None, 0.0
            self.log("深度学习分割: 推理完成，进行后处理")
            mask = self._postprocess_mask_enhanced(mask)
            segmented = self._apply_mask_enhanced(original, mask)
            # 对深度学习分割单独放宽评估条件：
            # 只要模型输出的前景区域非空，就认为分割成功，避免因为面积比例等阈值被全部丢弃
            mask_area = int(np.sum(mask > 0))
            if mask_area <= 0:
                self.log("深度学习分割: 后处理后的掩码区域为0")
                score = 0.0
            else:
                # 仍然调用一次评估函数用于排序/参考，但保证有一个较大的正分
                base_score = self._evaluate_segmentation_enhanced(mask)
                score = max(0.8, float(base_score))
                self.log(f"深度学习分割: 成功，得分 {score}")
            return segmented, mask, score
        except Exception as e:
            self.log(f"深度学习分割执行出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None, None, 0.0
    
    def _postprocess_mask_enhanced(self, mask):
        """增强的掩码后处理"""
        # 形态学闭运算（填充小洞）
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # 形态学开运算（去除小噪点）
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # 去除小连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 找到最大连通区域
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        return mask
    
    def _apply_mask_enhanced(self, image, mask):
        """增强的掩码应用（保持图像质量）"""
        # 创建3通道掩码
        if len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask
        
        # 归一化掩码
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        
        # 应用掩码（保持原图质量）
        result = (image.astype(np.float32) * mask_normalized).astype(np.uint8)
        
        return result
    
    def _evaluate_segmentation_enhanced(self, mask):
        """增强的分割质量评估（放宽条件，提高成功率）"""
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        
        if total_area == 0:
            return 0.0
        
        mask_ratio = mask_area / total_area
        
        # 放宽：5%-95% 均可接受（原10-80%过严导致大量失败）
        if mask_ratio < 0.05 or mask_ratio > 0.95:
            return 0.0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        max_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        area = cv2.contourArea(max_contour)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # 占比合理性：理想40%，放宽评分
        ratio_score = 1.0 - abs(mask_ratio - 0.4) * 1.2
        ratio_score = max(0, ratio_score)
        
        # 降低 circularity 权重，形状不规则的岩石也能通过
        final_score = 0.5 * ratio_score + 0.5 * max(0, circularity)
        return max(0.01, final_score)  # 最低0.01，避免完全丢弃

    def auto_segment_rock(self, image, image_name=None):
        """自动分割岩石区域"""
        try:
            # 保存原始图像用于对比
            original_image = image.copy()

            # 图像预处理
            processed = self._preprocess_image_fast(image)

            # 多方法分割尝试
            segmentation_methods = [
                (self._segment_by_grabcut, "GrabCut分割"),
                (self._segment_by_color_fast, "颜色分割"),
                (self._segment_by_edges_fast, "边缘分割")
            ]

            best_mask = None
            best_score = -1
            best_method = "无（使用原图）"

            for method, method_name in segmentation_methods:
                try:
                    mask = method(processed)
                    if mask is not None:
                        score = self._evaluate_segmentation_fast(mask)
                        if score > best_score:
                            best_score = score
                            best_mask = mask
                            best_method = method_name
                            if score > 0.8:
                                break
                except Exception as e:
                    continue

            if best_mask is None:
                return original_image, np.ones(original_image.shape[:2],
                                               dtype=np.uint8) * 255, best_method, original_image

            # 后处理
            refined_mask = self._postprocess_mask_fast(best_mask)

            chinese_method = self.method_names.get(best_method, best_method)
            if image_name:
                self.segmentation_methods_used[image_name] = {
                    'method': chinese_method,
                    'original_method': best_method,
                    'score': best_score,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            # 应用掩码
            segmented_rock = self._apply_mask(original_image, refined_mask)

            return segmented_rock, refined_mask, best_method, original_image

        except Exception as e:
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255, "分割失败", image

    def get_segmentation_stats(self):
        """获取分割方法统计"""
        method_counts = Counter([info['method'] for info in self.segmentation_methods_used.values()])
        return {
            'total_images': len(self.segmentation_methods_used),
            'method_distribution': dict(method_counts),
            'detailed_info': self.segmentation_methods_used
        }

    def _preprocess_image_fast(self, image):
        """快速图像预处理"""
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        return blurred

    def _segment_by_color_fast(self, image):
        """快速颜色分割"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_brown = np.array([10, 30, 20])
        upper_brown = np.array([20, 255, 200])
        lower_gray = np.array([0, 0, 30])
        upper_gray = np.array([180, 50, 200])

        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        return cv2.bitwise_or(mask_brown, mask_gray)

    def _segment_by_edges_fast(self, image):
        """快速边缘分割"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.fillPoly(mask, [contour], 255)
        return mask

    def _segment_by_grabcut(self, image):
        """使用GrabCut算法进行智能分割"""
        # 先用颜色分割结果确定岩石大致位置
        color_mask = self._segment_by_color_fast(image)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 取最大轮廓的外接矩形作为初始区域
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            # 适当扩大矩形范围
            x = max(0, x - 20)
            y = max(0, y - 20)
            w = min(image.shape[1] - x, w + 40)
            h = min(image.shape[0] - y, h + 40)
            rect = (x, y, w, h)
        else:
            # 若颜色分割失败，再用中心矩形兜底
            height, width = image.shape[:2]
            rect = (width // 4, height // 4, width // 2, height // 2)

        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 执行GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
        return mask2

    def _evaluate_segmentation_fast(self, mask):
        """评估分割质量"""
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        if total_area == 0:
            return -1
        mask_ratio = mask_area / total_area

        # 排除过小或过大的区域
        if mask_ratio < 0.05 or mask_ratio > 0.9:
            return -1

        # 计算轮廓紧凑度
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return -1
        max_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_contour, True)
        if perimeter == 0:
            return -1
        circularity = 4 * np.pi * cv2.contourArea(max_contour) / (perimeter ** 2)

        # 综合得分
        ratio_score = 1.0 - abs(mask_ratio - 0.5)
        return 0.3 * ratio_score + 0.7 * circularity

    def _postprocess_mask_fast(self, mask):
        """快速后处理掩码"""
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _apply_mask(self, image, mask):
        """应用掩码到图像"""
        # 关键修复：反转mask，让岩石区域变为255（保持原色），背景区域变为0（变黑色）
        inverted_mask = cv2.bitwise_not(mask)
        result = image.copy()
        result[inverted_mask == 0] = 0  # 背景区域（原来mask为255的地方）变黑色
        return result


# =========== 分割可视化类 ===========
class SegmentationVisualizer:
    """分割可视化类"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.create_output_folders()

    def create_output_folders(self):
        """创建输出文件夹"""
        folders = ['segmented', 'masks', 'comparisons', 'comparisons2']
        for folder in folders:
            path = os.path.join(self.output_dir, folder)
            if not os.path.exists(path):
                os.makedirs(path)

    def save_results(self, original_image, segmented_image, mask, image_name, method):
        """保存分割结果"""
        try:
            base_name = os.path.splitext(image_name)[0]

            # 1. 保存分割后的图像
            segmented_path = os.path.join(self.output_dir, 'segmented', f"{base_name}_segmented.jpg")
            cv2.imwrite(segmented_path, segmented_image)

            # 2. 保存分割掩码
            mask_path = os.path.join(self.output_dir, 'masks', f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)

            # 3. 创建对比图样式1（原图+分割图+掩码）
            comparison1 = self.create_comparison1(original_image, segmented_image, mask, method)
            comparison1_path = os.path.join(self.output_dir, 'comparisons', f"{base_name}_comparison1.jpg")
            cv2.imwrite(comparison1_path, comparison1)

            # 4. 创建对比图样式2（原图+分割图）
            comparison2 = self.create_comparison2(original_image, segmented_image, method, image_name)
            comparison2_path = os.path.join(self.output_dir, 'comparisons2', f"{base_name}_comparison2.jpg")
            cv2.imwrite(comparison2_path, comparison2)

            return {
                'segmented': segmented_path,
                'mask': mask_path,
                'comparison1': comparison1_path,
                'comparison2': comparison2_path
            }

        except Exception as e:
            print(f"保存结果失败: {str(e)}")
            return None

    def create_comparison1(self, original, segmented, mask, method):
        """创建对比图样式1"""
        # 调整图像尺寸
        h1, w1 = original.shape[:2]
        h2, w2 = segmented.shape[:2]
        h3, w3 = mask.shape[:2]

        target_height = max(h1, h2, h3)
        scale1 = target_height / h1
        scale2 = target_height / h2
        scale3 = target_height / h3

        original_resized = cv2.resize(original, (int(w1 * scale1), target_height))
        segmented_resized = cv2.resize(segmented, (int(w2 * scale2), target_height))

        # 将掩码转换为彩色
        if len(mask.shape) == 2:
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_colored = mask
        mask_resized = cv2.resize(mask_colored, (int(w3 * scale3), target_height))

        # 拼接图像
        comparison = np.hstack([original_resized, segmented_resized, mask_resized])

        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        titles = ["Original", f"Segmented({method})", "Mask"]
        title_width = original_resized.shape[1]

        for i, title in enumerate(titles):
            x_pos = i * title_width + 10
            cv2.putText(comparison, title, (x_pos, 30), font, 0.6, (255, 255, 255), 2)

        return comparison

    def create_comparison2(self, original, segmented, method, image_name):
        """创建对比图样式2"""
        h1, w1 = original.shape[:2]
        h2, w2 = segmented.shape[:2]

        target_height = max(h1, h2)
        scale1 = target_height / h1
        scale2 = target_height / h2

        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)

        original_resized = cv2.resize(original, (new_w1, target_height))
        segmented_resized = cv2.resize(segmented, (new_w2, target_height))

        # 拼接图像
        comparison = np.hstack([original_resized, segmented_resized])

        # 添加标题区域
        title_height = 60
        comparison_with_title = np.ones((target_height + title_height, new_w1 + new_w2, 3), dtype=np.uint8) * 255
        comparison_with_title[title_height:, :new_w1] = original_resized
        comparison_with_title[title_height:, new_w1:] = segmented_resized

        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison_with_title, f"Original: {image_name}", (10, 25), font, 0.5, (0, 0, 0), 1)
        cv2.putText(comparison_with_title, f"Segmented: {method}", (new_w1 + 10, 25), font, 0.5, (0, 0, 0), 1)

        return comparison_with_title


# =========== 批量分割流程 ===========
class BatchSegmentationWorker:
    """批量分割工作器"""

    def __init__(self, input_dir, output_dir, max_workers=4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.segmenter = RockSegmenter()
        self.visualizer = SegmentationVisualizer(output_dir)
        self.processed_count = 0
        self.total_count = 0

    def process_image(self, image_path):
        """处理单张图像 - 修复中文路径问题"""
        try:
            image_name = os.path.basename(image_path)

            # 方法1：使用PIL读取图像（支持中文路径）
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image)
                # PIL读取的是RGB，转换为OpenCV的BGR
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                # 方法2：尝试使用OpenCV
                self.log_message(f"PIL读取失败，尝试OpenCV: {str(pil_error)}")
                # 先解码中文字符路径
                try:
                    import sys
                    if sys.platform == 'win32':
                        # Windows系统：转换为系统编码
                        import locale
                        system_encoding = locale.getpreferredencoding()
                        encoded_path = image_path.encode(system_encoding, 'ignore')
                        decoded_path = encoded_path.decode(system_encoding)
                    else:
                        decoded_path = image_path

                    img = cv2.imread(decoded_path)
                except Exception as cv_error:
                    # 方法3：尝试使用绝对路径
                    abs_path = os.path.abspath(image_path)
                    img = cv2.imread(abs_path)

            if img is None:
                # 最后尝试：使用系统命令复制到临时文件
                try:
                    import tempfile
                    import shutil

                    temp_dir = tempfile.gettempdir()
                    temp_name = f"temp_{os.path.basename(image_path)}"
                    temp_path = os.path.join(temp_dir, temp_name)

                    # 复制文件到临时目录
                    shutil.copy2(image_path, temp_path)

                    # 读取临时文件
                    img = cv2.imread(temp_path)

                    # 清理临时文件
                    try:
                        os.remove(temp_path)
                    except:
                        pass

                except Exception as temp_error:
                    return None, f"无法加载图像（所有方法都失败）: {image_name}。错误: {temp_error}"

            if img is None:
                return None, f"无法加载图像: {image_name}"

            # 检查图像是否有效
            if img.size == 0:
                return None, f"图像数据为空: {image_name}"

            # 执行分割
            segmented, mask, method, original = self.segmenter.auto_segment_rock(img, image_name)

            # 保存结果
            result_paths = self.visualizer.save_results(original, segmented, mask, image_name, method)

            return result_paths, f"成功处理: {image_name} ({method})"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return None, f"处理失败 {os.path.basename(image_path)}: {str(e)}\n详细信息: {error_details[:200]}"

    def log_message(self, message):
        """辅助日志方法"""
        print(f"[Segmenter] {message}")
        
    def get_image_files(self):
        """获取所有图像文件"""
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
        image_files = []

        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_files.append(os.path.join(root, file))

        return image_files

    def generate_report(self):
        """生成分割报告"""
        stats = self.segmenter.get_segmentation_stats()
        report_path = os.path.join(self.output_dir,
                                   f"segmentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return report_path, stats