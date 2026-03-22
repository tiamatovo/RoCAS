import io
import platform
import subprocess
import sys
import os
import cv2
import socket
import traceback
import datetime
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from segmenter import RockSegmenter
from cv2_io_utils import cv2_imread, cv2_imwrite
from matplotlib.figure import Figure
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFileDialog, QMessageBox,
                             QFrame, QProgressBar, QTextEdit, QScrollArea,
                             QPushButton, QLineEdit, QSizePolicy, QInputDialog, QToolBar, QMenu,
                             QGroupBox, QComboBox, QSlider, QDialog, QProgressDialog, QDoubleSpinBox, QCheckBox,
                             QRadioButton, QButtonGroup, QTabWidget, QTreeWidget, QTreeWidgetItem, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGridLayout,)
from PyQt6.QtGui import QPixmap, QImage, QAction, QDesktopServices, QIcon, QCloseEvent, QPainter, QPen, QColor, QFont, \
    QBrush
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QUrl, QBuffer, QIODevice, QSettings, QRect, QPoint, QTimer


class ColorAnalysisWorker(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(str, list, int)
    error_occurred = pyqtSignal(str)

    def __init__(self, image, grid_size, standard_vectors, color_names, color_codes):
        super().__init__()
        self.image = image
        self.grid_size = grid_size
        self.standard_vectors = standard_vectors
        self.color_names = color_names
        self.color_codes = color_codes

    def run(self):
        try:
            self.progress_updated.emit(0)

            img_normalized = self.normalize_brightness(self.image)
            self.progress_updated.emit(10)

            scale_factor = 0.5
            img_resized = cv2.resize(
                img_normalized,
                (int(img_normalized.shape[1] * scale_factor),
                 int(img_normalized.shape[0] * scale_factor))
            )
            self.progress_updated.emit(20)

            height, width, _ = img_resized.shape
            half_grid = self.grid_size // 2
            x_range = width - half_grid
            y_range = height - half_grid

            color_vectors = []
            total_steps = (x_range - half_grid) * (y_range - half_grid)
            current_step = 0

            for i in range(half_grid, x_range):
                for j in range(half_grid, y_range):
                    grid_pixels = img_resized[j - half_grid:j + half_grid + 1,
                                  i - half_grid:i + half_grid + 1]

                    if self.grid_size < 15:
                        color_values = self.arithmetic_mean(grid_pixels)
                    else:
                        color_values = self.weighted_mean(grid_pixels, self.grid_size)

                    color_vectors.append(color_values)

                    current_step += 1
                    if current_step % 100 == 0:
                        progress = 25 + int((current_step / total_steps) * 40)
                        self.progress_updated.emit(progress)

            self.progress_updated.emit(65)

            color_frequency, color_matches = self.find_closest_color(color_vectors)
            total_vectors = len(color_vectors)

            structured_stats = []
            sorted_colors = sorted(color_frequency.items(), key=lambda x: x[1], reverse=True)

            for idx, ((color_name, color_code), count) in enumerate(sorted_colors):
                percentage = (count / total_vectors) * 100

                relevant_matches = [m for m in color_matches
                                    if m["color_name"] == color_name and m["color_code"] == color_code]

                if relevant_matches:
                    vectors_arr = np.array([m["target_vector"] for m in relevant_matches])
                    avg_target_vec = np.mean(vectors_arr, axis=0).astype(int)
                    std_vec = relevant_matches[0]["standard_vector"].astype(int)
                else:
                    avg_target_vec = np.array([0, 0, 0])
                    std_vec = np.array([0, 0, 0])

                structured_stats.append({
                    "name": color_name,
                    "code": color_code,
                    "count": count,
                    "percent": percentage,
                    "std_vec": std_vec,
                    "target_vec": avg_target_vec
                })

                if idx % 10 == 0:
                    progress = 65 + int((idx / len(sorted_colors)) * 30)
                    self.progress_updated.emit(progress)

            self.progress_updated.emit(95)

            report_text = "=== color analysis report=\n\n"
            for stat in structured_stats:
                report_text += f"{stat['name']} ({stat['code']}): {stat['percent']:.2f}%\n"

            self.result_ready.emit(report_text, structured_stats, total_vectors)
            self.progress_updated.emit(100)

        except Exception as e:
            error_msg = f"error:\n{str(e)}\n\ndetail:\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

    def normalize_brightness(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def arithmetic_mean(grid_pixels):
        """计算算术平均法"""
        mean_r = np.mean(grid_pixels[:, :, 2])
        mean_g = np.mean(grid_pixels[:, :, 1])
        mean_b = np.mean(grid_pixels[:, :, 0])
        return [int(round(mean_r)), int(round(mean_g)), int(round(mean_b))]

    @staticmethod
    def weighted_mean(grid_pixels, grid_size):
        """计算加权平均法"""
        half_grid = grid_size // 2
        height, width, _ = grid_pixels.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        dist = np.sqrt((y - half_grid) ** 2 + (x - half_grid) ** 2)
        weights = np.exp(-dist ** 2 / (2 * (half_grid ** 2)))

        mean_r = np.round(np.sum(grid_pixels[:, :, 2] * weights) / np.sum(weights))
        mean_g = np.round(np.sum(grid_pixels[:, :, 1] * weights) / np.sum(weights))
        mean_b = np.round(np.sum(grid_pixels[:, :, 0] * weights) / np.sum(weights))
        return [int(mean_r), int(mean_g), int(mean_b)]

    def find_closest_color(self, color_vectors):
        """精确的颜色匹配算法，特别处理黑白等极端颜色"""
        color_matches = []
        color_frequency = {}

        # 将输入转换为numpy数组
        color_vectors = np.array(color_vectors)

        # 特别处理黑色和白色的阈值
        black_threshold = 50  # RGB值都低于这个阈值认为是黑色
        white_threshold = 200  # RGB值都高于这个阈值认为是白色

        for target_color in color_vectors:
            # 提取RGB分量（后三个元素）
            r, g, b = target_color[0], target_color[1], target_color[2]

            # 首先检查是否为黑色
            if r <= black_threshold and g <= black_threshold and b <= black_threshold:
                # 直接匹配为黑色
                best_index = 8  # 黑色在数据中的索引是8
            # 然后检查是否为白色
            elif r >= white_threshold and g >= white_threshold and b >= white_threshold:
                # 直接匹配为白色
                best_index = 4  # 白色在数据中的索引是4
            else:
                # 对于其他颜色，使用改进的匹配算法
                target_rgb = np.array([r, g, b])

                # 计算与所有标准颜色的距离（只使用RGB分量）
                distances = np.linalg.norm(self.standard_vectors - target_rgb, axis=1)

                # 找到最小距离的索引
                best_index = np.argmin(distances)

            color_name = self.color_names[best_index]
            color_code = self.color_codes[best_index]

            # 记录匹配详情
            color_matches.append({
                "target_vector": target_color,
                "color_name": color_name,
                "color_code": color_code,
                "standard_vector": self.standard_vectors[best_index],
                "distance": np.linalg.norm(self.standard_vectors[best_index] - np.array([r, g, b]))
            })

            color_key = (color_name, color_code)
            color_frequency[color_key] = color_frequency.get(color_key, 0) + 1

        return color_frequency, color_matches


class MatplotlibWindow(QMainWindow):
    """通用的 Matplotlib 独立窗口，自带工具栏（保存、缩放、移动）"""

    def __init__(self, title="Chart", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 500)
        self.parent = parent

        # 核心部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # 1. 创建 Figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)

        # 2. 创建原生工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 拦截保存事件，添加我们的交互对话框
        for action in self.toolbar.actions():
            if action.text() == 'Save':
                action.triggered.disconnect()
                action.triggered.connect(self.custom_save)
                break

        # 3. 布局
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def custom_save(self):
        """自定义保存功能，添加交互对话框"""
        # 使用主窗口记忆的最近目录
        initial_path = "chart.png"
        if hasattr(self.parent, "get_last_directory"):
            last_dir = self.parent.get_last_directory()
            if last_dir:
                initial_path = os.path.join(last_dir, "chart.png")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", initial_path,
            "PNG Files (*.png);;PDF Files (*.pdf);;TIFF Files (*.tif)"
        )

        if file_path:
            # 记忆保存目录
            if hasattr(self.parent, "set_last_directory"):
                self.parent.set_last_directory(file_path)

            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图表时发生错误：\n{e}")

    def get_figure(self):
        return self.figure

    def draw(self):
        self.canvas.draw()


class ImageViewerWindow(QMainWindow):

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(image_path))
        self.resize(700, 500)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        v.addWidget(self.scroll)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)

        self._zoom = 1.0
        self._pixmap = QPixmap(image_path)
        self.apply_zoom()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self._zoom *= 1.1
        else:
            self._zoom /= 1.1
        self._zoom = max(0.1, min(self._zoom, 10))
        self.apply_zoom()

    def apply_zoom(self):
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self._pixmap.size() * self._zoom,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(scaled)

    def custom_save(self):
        """自定义保存功能，添加交互对话框"""
        # 使用主窗口记忆的最近目录
        initial_path = "chart.png"
        if hasattr(self.parent, "get_last_directory"):
            last_dir = self.parent.get_last_directory()
            if last_dir:
                initial_path = os.path.join(last_dir, "chart.png")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", initial_path,
            "PNG Files (*.png);;PDF Files (*.pdf);;TIFF Files (*.tif)"
        )

        if file_path:
            # 记忆保存目录
            if hasattr(self.parent, "set_last_directory"):
                self.parent.set_last_directory(file_path)
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.show_export_success_dialog(file_path)
            except Exception as e:
                QMessageBox.critical(self, "保存失败", str(e))

    def show_export_success_dialog(self, file_path):
        """显示导出成功的交互对话框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("保存成功")
        msg_box.setText(f"图表已保存到：\n{file_path}")
        msg_box.setIcon(QMessageBox.Icon.Information)

        open_file_btn = msg_box.addButton("打开文件", QMessageBox.ButtonRole.ActionRole)
        open_folder_btn = msg_box.addButton("打开所在文件夹", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)

        msg_box.exec()

        clicked_btn = msg_box.clickedButton()
        if clicked_btn == open_file_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
        elif clicked_btn == open_folder_btn:
            folder_path = os.path.dirname(file_path)
            if platform.system() == "Windows":
                subprocess.run(['explorer', '/select,', os.path.normpath(file_path)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))


class SegmentationMethodDialog(QDialog):
    """分割算法选择对话框"""

    def __init__(self, parent=None, default_methods=None, show_reminder=True):
        super().__init__(parent)
        self.setWindowTitle("选择分割算法")
        self.setModal(True)
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # 说明文字
        info_label = QLabel("请选择要使用的分割算法（可多选）：")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px 0;")
        layout.addWidget(info_label)

        # 算法选择区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        self.method_checks = {}
        methods = [
            ('GrabCut智能分割', '使用GrabCut算法，适合复杂背景的岩石区域分割。'),
            ('颜色阈值分割', '基于HSV颜色范围，适合颜色与背景差异明显的岩石。'),
            ('边缘检测分割', '基于边缘检测和形态学操作，适合边界清晰的岩石颗粒。'),
            ('自适应阈值分割', '自适应阈值，适合光照不均或亮度变化较大的图像。'),
            ('分水岭分割', '分水岭算法，适合颗粒粘连、需要分离相邻岩石区域的情况。'),
            ('K-means聚类分割', '基于颜色聚类，适合颜色分布明显但边界不规则的图像。')
        ]

        for method_name, description in methods:
            check = QCheckBox(method_name)
            check.setToolTip(description)
            if default_methods and method_name in default_methods:
                check.setChecked(True)

            # 为每个算法创建一个卡片：名称 + 描述
            method_frame = QFrame()
            method_frame.setStyleSheet("""
                QFrame {
                    border: 1px solid #dcdfe6;
                    border-radius: 4px;
                    background-color: #ffffff;
                }
            """)
            method_layout = QVBoxLayout(method_frame)
            method_layout.setContentsMargins(8, 6, 8, 6)
            method_layout.setSpacing(3)

            top_row = QHBoxLayout()
            top_row.addWidget(check)
            top_row.addStretch()
            method_layout.addLayout(top_row)

            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
            desc_label.setWordWrap(True)
            method_layout.addWidget(desc_label)

            scroll_layout.addWidget(method_frame)
            self.method_checks[method_name] = check

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # 不再提醒选项
        self.dont_remind_check = QCheckBox("使用默认设置且不再提醒")
        self.dont_remind_check.setChecked(not show_reminder)
        layout.addWidget(self.dont_remind_check)

        # 选择操作按钮：全选 / 全不选 / 反选
        op_layout = QHBoxLayout()
        op_layout.addStretch()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all)
        select_none_btn = QPushButton("全不选")
        select_none_btn.clicked.connect(self.unselect_all)
        invert_btn = QPushButton("反选")
        invert_btn.clicked.connect(self.inverse_selection)
        op_layout.addWidget(select_all_btn)
        op_layout.addWidget(select_none_btn)
        op_layout.addWidget(invert_btn)
        layout.addLayout(op_layout)

        # 确认 / 取消按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def select_all(self):
        """全选"""
        for check in self.method_checks.values():
            check.setChecked(True)

    def unselect_all(self):
        """全不选"""
        for check in self.method_checks.values():
            check.setChecked(False)

    def inverse_selection(self):
        """反选"""
        for check in self.method_checks.values():
            check.setChecked(not check.isChecked())

    def get_selected_methods(self):
        """获取选中的方法"""
        return [name for name, check in self.method_checks.items() if check.isChecked()]

    def should_remind(self):
        """是否应该提醒"""
        return not self.dont_remind_check.isChecked()


class ImageEditorApp(QMainWindow):

    # ========== 初始化界面和方法 ==========
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoCAS v2.20 (By TianXiaoying, Email: dachang0220@163.com)")
        self.setWindowIcon(QIcon(r".\resources\assets\images\logo\logo.png"))
        self.resize(800, 525)

        # 初始化变量
        self.image = None
        self.image_path = None
        self.stats_data = []  # 存储最新的分析结果
        self.total_samples = 0
        self.original_image = None  # 用于裁剪恢复

        # 分割相关属性
        self.segmenter = RockSegmenter(log_callback=self.log)
        self.exporter = HighDPIExporter()
        self.segmented_image = None
        self.segmentation_mask = None
        self.original_image_for_seg = None
        self.segmentation_method = None

        # 加载分割设置
        self.load_segmentation_settings()
        # 加载日志设置
        self.load_log_settings()

        # 初始化日志自动保存
        self.init_log_auto_save()

        # 设置全局样式（扁平化、现代风）
        self.setup_styles()

        # 初始化界面布局
        self.init_ui()
        self.create_menus()
        self.add_copyright_footer()
        self.load_csv_data()  # 预加载数据
        self.load_dpi_settings()
        self.load_export_settings()

        # 通用大图查看窗口引用
        self._image_viewer = None

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(
            self,
            "退出",
            "确定要退出程序吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    # ========== 界面、布局、样式 ==========
    def init_ui(self):
        # 1. 创建顶部工具栏 (替代原来的按钮堆)
        self.create_toolbar()

        # 2. 中央主区域 - 使用标签页
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: #f5f6f7;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #34495e;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #bdc3c7;
            }
        """)

        # 标签页1: Single Image (现有功能)
        single_image_tab = self.create_single_image_tab()
        self.tab_widget.addTab(single_image_tab, "Single Image")

        # 标签页2: Dataset (新增功能)
        dataset_tab = self.create_dataset_tab()
        self.tab_widget.addTab(dataset_tab, "Dataset")

        main_layout.addWidget(self.tab_widget)

    def create_single_image_tab(self):
        """创建单图处理标签页"""
        tab_widget = QWidget()
        main_layout = QHBoxLayout(tab_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- 左侧：图像显示区 ---
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.NoFrame)
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(5, 5, 5, 5)

        # A. 增加标题栏
        header_layout = QHBoxLayout()
        preview_title = QLabel("Image Preview")
        preview_title.setStyleSheet("font-weight: bold; color: #34495e; font-size: 14px;")
        header_layout.addWidget(preview_title)
        header_layout.addStretch()
        image_layout.addLayout(header_layout)

        # B. 图片显示 Label
        self.image_label = QLabel(
            "No image available.\n\nClick here or the button in the upper left corner to load an image.")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 设置样式：虚线边框、圆角、鼠标悬停变色
        self.image_label.setStyleSheet("""
            QLabel#ImageLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 10px;
                background-color: #f8f9fa;
                color: #7f8c8d;
            }
            QLabel#ImageLabel:hover {
                background-color: #ecf0f1;
                border: 2px dashed #3498db;
                color: #3498db;
            }
        """)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # C. 核心：开启鼠标交互并绑定函数
        self.image_label.setCursor(Qt.CursorShape.PointingHandCursor)  # 鼠标移上去变小手
        # 自定义鼠标点击，左键放大 / 加载，右键清除
        self.image_label.mousePressEvent = self.on_preview_clicked

        image_layout.addWidget(self.image_label)

        # D. 显示当前样本名（仅在有图片时显示）
        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_name_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding-top: 4px;")
        self.image_name_label.setVisible(False)
        image_layout.addWidget(self.image_name_label)
        main_layout.addWidget(image_frame, stretch=7)  # 稍微调大一点占比

        # --- 右侧：参数与日志区 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 参数设置组
        settings_group = QGroupBox("Parameter settings")
        settings_layout = QVBoxLayout()

        # 翻转模式选择
        settings_layout.addWidget(QLabel("Flip mode:"))
        self.flip_combo = QComboBox()
        self.flip_combo.addItems(["None", "Vertical flip", "Horizontal flip", "Vertical + Horizontal"])
        self.flip_combo.setFont(QFont("Microsoft YaHei", 10))
        settings_layout.addWidget(self.flip_combo)

        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)

        # 日志输出组
        log_group = QGroupBox("Process Log")
        log_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.output_text)

        # 进度条放在日志区域内
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        log_layout.addWidget(self.progress_bar)

        log_group.setLayout(log_layout)

        # 日志区域自定义右键菜单
        self.output_text.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.output_text.customContextMenuRequested.connect(self.show_log_menu)

        right_layout.addWidget(settings_group, stretch=2)
        right_layout.addWidget(log_group, stretch=8)

        main_layout.addWidget(right_panel, stretch=3)  # 右侧占比 30%

        return tab_widget

    def update_dataset_tab_info(self, dataset_path):
        """更新Dataset标签页的信息显示（统计 / 结构 / 预览）"""
        if not os.path.isdir(dataset_path):
            return

        self.dataset_summary_label.setText(
            f"当前数据集: {os.path.basename(dataset_path)}  |  路径: {dataset_path}"
        )

        # 扫描图片
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
        self.dataset_image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith(supported_formats):
                    self.dataset_image_files.append(os.path.join(root, f))

        # 统计信息
        total_images = len(self.dataset_image_files)
        formats = {}
        folders = {}
        for img_path in self.dataset_image_files:
            ext = os.path.splitext(img_path)[1].lower()
            formats[ext] = formats.get(ext, 0) + 1
            folder = os.path.dirname(img_path)
            folder_name = "Root" if folder == dataset_path else os.path.relpath(folder, dataset_path)
            folders[folder_name] = folders.get(folder_name, 0) + 1

        self.dataset_stats_table.setRowCount(6 + len(formats) + len(folders))
        row = 0
        self.dataset_stats_table.setItem(row, 0, QTableWidgetItem("Dataset Path"))
        self.dataset_stats_table.setItem(row, 1, QTableWidgetItem(dataset_path))
        row += 1
        self.dataset_stats_table.setItem(row, 0, QTableWidgetItem("Total Images"))
        self.dataset_stats_table.setItem(row, 1, QTableWidgetItem(str(total_images)))
        row += 1
        self.dataset_stats_table.setItem(row, 0, QTableWidgetItem("Image Formats"))
        self.dataset_stats_table.setItem(row, 1, QTableWidgetItem(", ".join(formats.keys())))
        row += 1
        self.dataset_stats_table.setItem(row, 0, QTableWidgetItem("Format Distribution"))
        self.dataset_stats_table.setItem(
            row, 1,
            QTableWidgetItem(", ".join([f"{k}: {v}" for k, v in formats.items()]))
        )
        row += 1
        self.dataset_stats_table.setItem(row, 0, QTableWidgetItem("Number of Folders"))
        self.dataset_stats_table.setItem(row, 1, QTableWidgetItem(str(len(folders))))
        row += 1
        self.dataset_stats_table.setItem(row, 0, QTableWidgetItem("Folder Distribution"))
        folder_str = ", ".join([f"{k}: {v}" for k, v in sorted(folders.items(), key=lambda x: x[1], reverse=True)[:10]])
        if len(folders) > 10:
            folder_str += f" ... (and {len(folders) - 10} more)"
        self.dataset_stats_table.setItem(row, 1, QTableWidgetItem(folder_str))

        # 构建树结构
        self.dataset_tree.clear()
        root_item = QTreeWidgetItem(self.dataset_tree)
        root_item.setText(0, os.path.basename(dataset_path))
        root_item.setExpanded(True)
        folder_dict = {}
        for img_path in self.dataset_image_files:
            folder = os.path.dirname(img_path)
            folder_dict.setdefault(folder, []).append(os.path.basename(img_path))

        for folder, files in sorted(folder_dict.items()):
            rel_path = os.path.relpath(folder, dataset_path)
            parent = root_item
            if rel_path != ".":
                parts = rel_path.split(os.sep)
                for part in parts:
                    found = False
                    for i in range(parent.childCount()):
                        if parent.child(i).text(0) == part:
                            parent = parent.child(i)
                            found = True
                            break
                    if not found:
                        new_item = QTreeWidgetItem(parent)
                        new_item.setText(0, part)
                        parent = new_item
            for f in files:
                file_item = QTreeWidgetItem(parent)
                file_item.setText(0, f)

        # 重置分页并刷新预览
        self.dataset_current_page = 0
        self.dataset_show_all = False
        self.update_dataset_preview()

    def update_dataset_preview(self):
        """根据分页状态刷新 Dataset 标签页中的图片预览"""
        # 清空当前格子
        for i in reversed(range(self.dataset_preview_grid.count())):
            w = self.dataset_preview_grid.itemAt(i).widget()
            if w:
                w.setParent(None)

        total = len(self.dataset_image_files)
        if total == 0:
            self.dataset_preview_label.setText("第 0 页 / 共 0 页 (共 0 张)")
            self.dataset_prev_btn.setEnabled(False)
            self.dataset_next_btn.setEnabled(False)
            return

        if self.dataset_show_all:
            start_index, end_index = 0, total
            current_page, total_pages = 1, 1
        else:
            page_size = self.dataset_preview_page_size
            total_pages = (total + page_size - 1) // page_size
            self.dataset_current_page = max(0, min(self.dataset_current_page, total_pages - 1))
            current_page = self.dataset_current_page + 1
            start_index = self.dataset_current_page * page_size
            end_index = min(start_index + page_size, total)

        self.dataset_preview_label.setText(f"第 {current_page} 页 / 共 {total_pages} 页 (共 {total} 张)")

        # 按钮可用性
        if self.dataset_show_all or total_pages <= 1:
            self.dataset_prev_btn.setEnabled(False)
            self.dataset_next_btn.setEnabled(False)
        else:
            self.dataset_prev_btn.setEnabled(current_page > 1)
            self.dataset_next_btn.setEnabled(current_page < total_pages)

        cols = 4
        for idx, img_path in enumerate(self.dataset_image_files[start_index:end_index]):
            try:
                pil_img = Image.open(img_path)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                thumb_size = 150
                pil_img.thumbnail((thumb_size, thumb_size))
                img_array = np.array(pil_img)
                h, w, ch = img_array.shape
                bytes_per_line = ch * w
                q_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                # 容器：图片 + 文件名
                container = QWidget()
                v = QVBoxLayout(container)
                v.setContentsMargins(0, 0, 0, 0)
                v.setSpacing(3)

                img_label = QLabel()
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #ddd;
                        border-radius: 5px;
                        padding: 5px;
                        background-color: white;
                    }
                    QLabel:hover {
                        border: 2px solid #3498db;
                    }
                """)
                img_label.setCursor(Qt.CursorShape.PointingHandCursor)
                img_label.mousePressEvent = lambda e, p=img_path: self.open_image_viewer(p)

                name_label = QLabel(os.path.basename(img_path))
                name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                name_label.setStyleSheet("color: #555; font-size: 10px;")

                v.addWidget(img_label)
                v.addWidget(name_label)

                row = idx // cols
                col = idx % cols
                self.dataset_preview_grid.addWidget(container, row, col)
            except Exception as e:
                self.log(f"Error loading dataset preview: {img_path} - {e}")

    def dataset_prev_page(self):
        """Dataset 预览上一页"""
        if self.dataset_show_all:
            self.dataset_show_all = False
        if self.dataset_current_page > 0:
            self.dataset_current_page -= 1
        self.update_dataset_preview()

    def dataset_next_page(self):
        """Dataset 预览下一页"""
        if self.dataset_show_all:
            self.dataset_show_all = False
            self.dataset_current_page = 0
        total = len(self.dataset_image_files)
        if total > 0:
            total_pages = (total + self.dataset_preview_page_size - 1) // self.dataset_preview_page_size
            if self.dataset_current_page < total_pages - 1:
                self.dataset_current_page += 1
        self.update_dataset_preview()

    def dataset_load_all(self):
        """Dataset 预览加载全部"""
        self.dataset_show_all = True
        self.dataset_current_page = 0
        self.update_dataset_preview()

    def create_dataset_tab(self):
        """创建数据集管理标签页（集成 DatasetInfo 功能，简洁风格）"""
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 顶部：数据集概要 + 加载按钮
        header_layout = QHBoxLayout()
        self.dataset_summary_label = QLabel("尚未加载数据集")
        self.dataset_summary_label.setStyleSheet("font-weight: bold; color: #34495e; font-size: 13px;")
        header_layout.addWidget(self.dataset_summary_label)
        header_layout.addStretch()

        load_dataset_btn = QPushButton("加载数据集文件夹")
        load_dataset_btn.setIcon(QIcon(r".\resources\assets\images\button\addFolder.png"))
        load_dataset_btn.clicked.connect(self.addFolder)
        header_layout.addWidget(load_dataset_btn)
        main_layout.addLayout(header_layout)

        # 中部：标签页（统计 / 结构 / 预览）
        self.dataset_tabs = QTabWidget()

        # 统计信息
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        self.dataset_stats_table = QTableWidget()
        self.dataset_stats_table.setColumnCount(2)
        self.dataset_stats_table.setHorizontalHeaderLabels(["属性", "数值"])
        header = self.dataset_stats_table.horizontalHeader()
        # 默认给第一列一个稍宽的初始宽度，但保持“可拖动调整”
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.dataset_stats_table.setColumnWidth(0, 180)
        header.setStretchLastSection(True)
        self.dataset_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self.dataset_stats_table)
        self.dataset_tabs.addTab(stats_widget, "统计信息")

        # 文件夹结构
        structure_widget = QWidget()
        structure_layout = QVBoxLayout(structure_widget)
        structure_layout.setContentsMargins(10, 10, 10, 10)
        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setHeaderLabel("数据集结构")
        structure_layout.addWidget(self.dataset_tree)
        self.dataset_tabs.addTab(structure_widget, "文件结构")

        # 图片预览
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        inner = QWidget()
        self.dataset_preview_grid = QGridLayout(inner)
        self.dataset_preview_grid.setSpacing(10)
        scroll_area.setWidget(inner)
        preview_layout.addWidget(scroll_area)

        # 分页控制
        ctrl_layout = QHBoxLayout()
        self.dataset_preview_label = QLabel("第 0 页 / 共 0 页 (共 0 张)")
        ctrl_layout.addWidget(self.dataset_preview_label)
        ctrl_layout.addStretch()
        self.dataset_prev_btn = QPushButton("上一页")
        self.dataset_next_btn = QPushButton("下一页")
        self.dataset_all_btn = QPushButton("加载全部")
        ctrl_layout.addWidget(self.dataset_prev_btn)
        ctrl_layout.addWidget(self.dataset_next_btn)
        ctrl_layout.addWidget(self.dataset_all_btn)
        preview_layout.addLayout(ctrl_layout)

        self.dataset_tabs.addTab(preview_widget, "图片预览")

        main_layout.addWidget(self.dataset_tabs)

        # 初始化分页相关变量
        self.dataset_image_files = []
        self.dataset_preview_page_size = 20
        self.dataset_current_page = 0
        self.dataset_show_all = False

        # 绑定按钮到槽函数
        self.dataset_prev_btn.clicked.connect(self.dataset_prev_page)
        self.dataset_next_btn.clicked.connect(self.dataset_next_page)
        self.dataset_all_btn.clicked.connect(self.dataset_load_all)

        return tab_widget

    def setup_styles(self):
        """设置现代化的扁平风格 QSS"""
        style_sheet = """
            QMainWindow {
                background-color: #f5f6f7;
            }
            /* 工具栏样式 */
            QToolBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
                padding: 5px;
                spacing: 10px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                min-width: 35px;      /* 设置统一的最小宽度，整齐排列 */
                padding: 1px;
                margin: 1px;
                color: #333;
                font-family: "Microsoft YaHei";
            }
            QToolButton:hover {
                background-color: #e6f7ff; /* 悬浮时的浅蓝色背景 */
                border: 1px solid #1890ff; /* 悬浮时的细边框 */
            }
            QToolButton:pressed {
                background-color: #bae7ff;
            }

            /* 右侧面板样式 */
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #ffffff;
                font-family: "Microsoft YaHei";
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
            }

            /* 状态显示区 */
            QLabel#ImageLabel {
                background-color: #eef2f5;
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                color: #a0aec0;
            }

            /* Tooltip 样式 */
            QToolTip {
                border: 1px solid #333;
                background-color: #333;
                color: white;
                padding: 4px;
                opacity: 200;
            }

            QTextEdit {
                border: 1px solid #dcdfe6;
                border-radius: 4px;
                background-color: #fafafa;
                selection-background-color: #1890ff;
            }

            /* 日志区域右键菜单样式*/
            QMenu {
                background-color: #ffffff;
                border: 1px solid #d1d4d8;
            }

            QMenu::item {
                padding: 6px 30px 6px 20px;
                background-color: transparent;
            }

            QMenu::item:selected {
                background-color: #3b8ee0;
                color: #ffffff;
            }

            /* 菜单栏样式：保持白色背景，精致的悬浮感 */
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #dcdfe6;
            }

            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
            }

            /* 强制隐藏菜单中的小图标 */
            QMenu::icon {
                width: 0px;
            }

            QMenuBar::item:selected {
                background-color: #f0f7ff; /* 浅蓝色选中效果 */
                color: #1890ff;
            }

            QMenu {
                background-color: #ffffff;
                border: 1px solid #dcdfe6;
            }

            QMenu::item:selected {
                background-color: #1890ff;
                color: white;
            }
             """
        self.setStyleSheet(style_sheet)

    def create_menus(self):
        menubar = self.menuBar()
        # 统一菜单栏样式：纯白背景，无边框感
        menubar.setStyleSheet("QMenuBar { background-color: #ffffff; border: none; }")

        # --- File 菜单 ---
        file_menu = menubar.addMenu('&File')

        # 统一菜单栏样式：纯白背景，无边框感
        menubar.setStyleSheet("QMenuBar { background-color: #ffffff; border: none; }")

        # 创建纯文字 Action
        open_menu_act = QAction('Open Image...', self)
        open_menu_act.setShortcut('Ctrl+O')
        open_menu_act.triggered.connect(self.load_image)
        file_menu.addAction(open_menu_act)

        save_menu_act = QAction('Save Result', self)
        save_menu_act.setShortcut('Ctrl+S')
        save_menu_act.triggered.connect(self.save_image)
        file_menu.addAction(save_menu_act)

        # --- Settings 菜单 ---
        settings_menu = menubar.addMenu('&Settings')

        # 添加DPI设置菜单项
        settings_menu.addSeparator()
        dpi_settings_action = QAction('Export DPI Settings', self)
        dpi_settings_action.triggered.connect(self.show_dpi_settings_dialog)
        dpi_settings_action.setShortcut('Ctrl+D')
        settings_menu.addAction(dpi_settings_action)

        # 导出设置菜单项
        export_settings_action = QAction('Export Settings', self)
        export_settings_action.triggered.connect(self.show_export_settings_dialog)
        export_settings_action.setShortcut('Ctrl+E')
        settings_menu.addAction(export_settings_action)

        # 添加分割算法设置
        settings_menu.addSeparator()
        seg_settings_action = QAction('Segmentation Algorithm Settings', self)
        seg_settings_action.triggered.connect(self.show_segmentation_settings_dialog)
        seg_settings_action.setShortcut('Ctrl+M')
        settings_menu.addAction(seg_settings_action)

        # 添加日志设置
        log_settings_action = QAction('Log Settings', self)
        log_settings_action.triggered.connect(self.show_log_settings_dialog)
        log_settings_action.setShortcut('Ctrl+L')
        settings_menu.addAction(log_settings_action)

        # --- Analysis 菜单 ---
        analysis_menu = menubar.addMenu('&Analysis')
        run_menu_act = QAction('Start Recognition', self)
        run_menu_act.setShortcut('F5')
        run_menu_act.triggered.connect(self.start_processing)
        analysis_menu.addAction(run_menu_act)

        # --- Help 菜单 ---
        help_menu = menubar.addMenu('&Help')
        # 修复：使用正确的方法调用方式
        help_menu_action = QAction('About RoCAS', self)
        help_menu_action.setShortcut('F1')
        help_menu_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(help_menu_action)

    # ========== 添加DPI设置对话框方法 ==========
    def show_dpi_settings_dialog(self):
        """显示DPI设置对话框"""
        dialog = DPISettingsDialog(self)

        # 加载当前设置
        dialog.dpi_group.button(self.dpi).setChecked(True)
        dialog.width_spin.setValue(self.fig_width)
        dialog.height_spin.setValue(self.fig_height)
        dialog.save_individual.setChecked(self.save_individual)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self.save_dpi_settings(
                settings['dpi'],
                settings['width'],
                settings['height'],
                settings['save_individual']
            )
            # 更新当前设置
            self.dpi = settings['dpi']
            self.fig_width = settings['width']
            self.fig_height = settings['height']
            self.save_individual = settings['save_individual']

            QMessageBox.information(
                self,
                "设置已保存",
                f"DPI设置已保存：\nDPI: {self.dpi}\n尺寸: {self.fig_width}×{self.fig_height}英寸\n单独保存: {'是' if self.save_individual else '否'}"
            )

    def create_toolbar(self):
        """创建顶部图标工具栏"""
        toolbar = QToolBar("功能工具栏")
        toolbar.setIconSize(QSize(22, 22))
        toolbar.setMovable(False)  # 禁止拖动
        toolbar.layout().setSpacing(4)  # 图标贴得近
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        self.addToolBar(toolbar)

        # 定义统一的字体大小
        btn_font = QFont("Microsoft YaHei", 9)

        # 1. 加载图片
        self.load_action = QAction(QIcon(r".\resources\assets\images\button\load.png"), "Open", self)
        self.load_action.setStatusTip("Tips：从本地文件夹打开一张图片")
        self.load_action.setToolTip("加载图片\n点击选择一张 png 或 jpg 图片进行分析")
        self.load_action.triggered.connect(self.load_image)
        toolbar.addAction(self.load_action)
        toolbar.widgetForAction(self.load_action).setFont(btn_font)

        # 2. 翻转图片
        self.flip_action = QAction(QIcon(r".\resources\assets\images\button\flip.png"), "Flip", self)
        self.flip_action.setStatusTip("Tips:从右侧选中一种翻转方式,之后点击翻转按钮")
        self.flip_action.setToolTip("执行翻转\n根据右侧设置的模式对图片进行翻转")
        self.flip_action.triggered.connect(self.flip_image)
        toolbar.addAction(self.flip_action)

        # 3. 裁剪图片
        self.crop_action = QAction(QIcon(r".\resources\assets\images\button\crop.png"), "Crop", self)
        self.crop_action.setStatusTip("Tips：手动对图片进行裁剪，选择想要识别的目标区域")
        self.crop_action.setToolTip("裁剪图片\n手动框选目标区域")
        self.crop_action.triggered.connect(self.crop_image)
        toolbar.addAction(self.crop_action)
        toolbar.widgetForAction(self.crop_action).setFont(btn_font)

        # 5. 保存结果
        self.save_action = QAction(QIcon(r".\resources\assets\images\button\save.png"), "Save", self)
        self.save_action.setStatusTip("Tips：保存当前显示的图片")
        self.save_action.setToolTip("保存图片\n将处理后的结果保存到本地")
        self.save_action.triggered.connect(self.save_image)
        toolbar.addAction(self.save_action)
        toolbar.widgetForAction(self.save_action).setFont(btn_font)

        toolbar.addSeparator()

        # 6. 自适应分割-添加文件夹
        self.addFolder_action = QAction(QIcon(r".\resources\assets\images\button\addFolder.png"), "Dataset",
                                        self)
        self.addFolder_action.setStatusTip("Tips：选择图像数据集所在文件夹")
        self.addFolder_action.setToolTip("加载所有需要图像分割的图片")
        self.addFolder_action.triggered.connect(self.addFolder)
        toolbar.addAction(self.addFolder_action)
        toolbar.widgetForAction(self.addFolder_action).setFont(btn_font)

        # 7. 一键开启自适应分割
        self.seg_action = QAction(QIcon(r".\resources\assets\images\button\segmentation.png"), "seg", self)
        self.seg_action.setStatusTip("Tips：一键开启自适应分割算法；最后生成原图、目标区域图、分割图以及分割掩码")
        self.seg_action.setToolTip("自适应算法包括：\n\n1.边缘检测算法\n2.阈值分割算法\n3.GrabCut分割算法")
        self.seg_action.triggered.connect(self.seg)
        toolbar.addAction(self.seg_action)
        toolbar.widgetForAction(self.seg_action).setFont(btn_font)

        # 4. 核心功能：识别颜色
        self.process_action = QAction(QIcon(r".\resources\assets\images\button\recognize.png"), "Analyze",
                                      self)
        self.process_action.setStatusTip("Tips：开启颜色智能识别")
        self.process_action.setToolTip("分析图片中的岩石颜色分布")
        self.process_action.triggered.connect(self.start_processing)
        toolbar.addAction(self.process_action)
        toolbar.widgetForAction(self.process_action).setFont(btn_font)

        # 6. 高清截图
        self.screenshot_action = QAction(QIcon(r".\resources\assets\images\button\screenshot-fill.png"),
                                         "Snapshot",
                                         self)
        self.screenshot_action.triggered.connect(self.capture_high_res_screenshot)

        # 将截图功能放在右侧
        empty = QWidget()
        empty.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(empty)
        toolbar.addAction(self.screenshot_action)
        toolbar.widgetForAction(self.screenshot_action).setFont(btn_font)

    # 设置-导出对话框
    def show_export_settings_dialog(self):
        """显示导出设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("导出设置")
        dialog.setModal(True)
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 标题区域
        title_label = QLabel("导出设置")
        title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            color: #2c3e50;
            padding: 10px;
        """)
        layout.addWidget(title_label)

        # 标题显示选项
        title_group = QGroupBox("图片标题设置")
        title_layout = QVBoxLayout()

        self.title_check = QCheckBox("在导出的图片中显示标题（样本名称）")
        self.title_check.setChecked(self.show_title_in_export)
        self.title_check.setToolTip("勾选后，导出的对比图会显示标题；不勾选则只显示图片")
        title_layout.addWidget(self.title_check)

        title_hint = QLabel("提示：单独保存的原图、分割图、掩码默认不显示标题")
        title_hint.setStyleSheet("color: #7f8c8d; font-size: 11px; padding-left: 20px;")
        title_layout.addWidget(title_hint)

        title_group.setLayout(title_layout)
        layout.addWidget(title_group)

        # 文件名格式选项
        filename_group = QGroupBox("文件名格式设置")
        filename_layout = QVBoxLayout()

        self.filename_time_radio = QRadioButton("时间+样本名 (例如: 20250101_120000_样本名_方法_comparison_style1.png)")
        self.filename_custom_radio = QRadioButton("自定义前缀 (例如: Rock_样本名_方法_comparison_style1.png)")

        if self.filename_format == 'time_name':
            self.filename_time_radio.setChecked(True)
        else:
            self.filename_custom_radio.setChecked(True)

        filename_layout.addWidget(self.filename_time_radio)
        filename_layout.addWidget(self.filename_custom_radio)

        # 自定义前缀输入
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("自定义前缀:"))
        self.custom_prefix_edit = QLineEdit(self.custom_filename_prefix)
        self.custom_prefix_edit.setPlaceholderText("例如: Rock, Sample, Test")
        self.custom_prefix_edit.setEnabled(self.filename_custom_radio.isChecked())
        prefix_layout.addWidget(self.custom_prefix_edit)

        # 当选择自定义前缀时，启用输入框
        self.filename_custom_radio.toggled.connect(self.custom_prefix_edit.setEnabled)

        filename_layout.addLayout(prefix_layout)
        filename_group.setLayout(filename_layout)
        layout.addWidget(filename_group)

        # 示例显示
        example_label = QLabel("文件名示例:")
        example_label.setStyleSheet("font-weight: bold; color: #34495e;")
        layout.addWidget(example_label)

        self.example_label = QLabel()
        self.example_label.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 8px;
            color: #495057;
            font-family: 'Consolas', monospace;
            font-size: 11px;
        """)
        self.update_filename_example()
        layout.addWidget(self.example_label)

        # 连接信号更新示例
        self.filename_time_radio.toggled.connect(self.update_filename_example)
        self.filename_custom_radio.toggled.connect(self.update_filename_example)
        self.custom_prefix_edit.textChanged.connect(self.update_filename_example)

        layout.addStretch()

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            show_title = self.title_check.isChecked()
            filename_format = 'time_name' if self.filename_time_radio.isChecked() else 'custom'
            custom_prefix = self.custom_prefix_edit.text().strip() or "Rock"

            self.save_export_settings(show_title, filename_format, custom_prefix)
            QMessageBox.information(
                self,
                "设置已保存",
                f"导出设置已保存：\n"
                f"显示标题: {'是' if show_title else '否'}\n"
                f"文件名格式: {'时间+样本名' if filename_format == 'time_name' else '自定义前缀'}\n"
                f"自定义前缀: {custom_prefix}"
            )

        # 清理临时控件
        self.title_check = None
        self.filename_time_radio = None
        self.filename_custom_radio = None
        self.custom_prefix_edit = None
        self.example_label = None

    def update_filename_example(self):
        """更新文件名示例"""
        if not hasattr(self, 'example_label') or not self.example_label:
            return

        if self.filename_time_radio.isChecked():
            example = "20250101_120000_样本名_方法名_comparison_style1.png"
        else:
            prefix = self.custom_prefix_edit.text().strip() or "Rock"
            example = f"{prefix}_样本名_方法名_comparison_style1.png"

        self.example_label.setText(example)

    # ========== 自适应分割 ==========
    def addFolder(self):
        """选择图像数据集文件夹"""
        last_dir = self.get_last_directory()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择图像数据集文件夹",
            last_dir
        )

        if not folder_path:
            return

        self.segmentation_folder = folder_path
        self.log(f"已选择数据集文件夹: {folder_path}")
        # 记忆数据集所在目录
        self.set_last_directory(folder_path)

        # 更新Dataset标签页的信息显示（不再弹出独立窗口）
        self.update_dataset_tab_info(folder_path)

    def seg(self):
        """单张图像分割功能（带进度条）"""
        if self.image is None:
            QMessageBox.warning(self, "警告", "请先加载图片！")
            return

        # 选择分割算法
        selected_methods = self.default_segmentation_methods.copy()
        show_reminder = self.show_segmentation_reminder

        if show_reminder:
            dialog = SegmentationMethodDialog(
                self,
                default_methods=self.default_segmentation_methods,
                show_reminder=show_reminder
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            selected_methods = dialog.get_selected_methods()
            if not selected_methods:
                QMessageBox.warning(self, "警告", "请至少选择一个分割算法！")
                return

            # 保存"不再提醒"设置
            if not dialog.should_remind():
                self.save_segmentation_settings(selected_methods, False)

        # 创建进度对话框
        progress_dialog = QProgressDialog("正在分割图像，请稍候...", "取消", 0, 100, self)
        progress_dialog.setWindowTitle("图像分割")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.setValue(0)

        try:
            self.log("=" * 60)
            self.log("开始图像分割处理")
            self.log("=" * 60)

            # 保存原始图像
            self.original_image_for_seg = self.image.copy()
            image_name = os.path.basename(self.image_path) if self.image_path else "unknown.jpg"

            # 更新进度
            progress_dialog.setLabelText("正在初始化分割算法...")
            progress_dialog.setValue(10)
            QApplication.processEvents()  # 更新UI

            # 创建进度回调函数
            total_methods = len(selected_methods)
            current_method = 0

            def progress_callback(message):
                """进度回调，更新进度条"""
                nonlocal current_method
                if "完成" in message or "成功" in message:
                    current_method += 1
                    progress = 10 + int((current_method / total_methods) * 80)
                    progress_dialog.setValue(progress)
                    progress_dialog.setLabelText(f"正在处理: {message}")
                else:
                    progress_dialog.setLabelText(f"正在处理: {message}")
                QApplication.processEvents()

            # 执行分割（使用选定的方法）
            results = self.segmenter.segment_by_methods(
                self.image.copy(),
                image_name,
                selected_methods,
                log_callback=progress_callback
            )

            progress_dialog.setValue(90)
            progress_dialog.setLabelText("正在准备预览窗口...")
            QApplication.processEvents()

            if not results:
                progress_dialog.close()
                QMessageBox.warning(self, "分割失败", "所有分割方法都失败了，请检查图像质量。")
                return

            progress_dialog.setValue(100)
            progress_dialog.close()

            self.log(f"\n分割完成，成功方法数: {len(results)}")

            # 显示分割结果预览窗口（修复：只传2个参数）
            self.show_segmentation_results_preview(results, image_name)

        except Exception as e:
            progress_dialog.close()
            self.log(f"分割失败: {str(e)}")
            error_msg = f"分割过程中发生错误:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "分割错误", error_msg)

    def show_segmentation_result(self, original, segmented, mask, method, image_name):
        """显示分割结果并询问是否导出"""
        # 更新显示为分割结果
        self.display_image_on_label(segmented)
        self.log(f"分割结果已显示在预览区，使用的分割方法: {method}")

        # 创建分割结果预览窗口
        preview_window = SegmentationPreviewWindow(
            original, segmented, mask, method, image_name, self
        )
        preview_window.show()

    def show_segmentation_results_preview(self, results, image_name):
        """显示多算法分割结果预览窗口"""
        preview_window = MultiMethodSegmentationPreviewWindow(
            results, image_name, self
        )
        preview_window.show()

    def export_segmentation_high_dpi(self, original, segmented, mask, method, image_name):
        """导出高DPI分割对比图（使用保存的设置，带进度条）"""
        # 从设置中读取DPI和尺寸
        dpi = self.dpi
        width = self.fig_width
        height = self.fig_height
        save_individual = self.save_individual
        show_title = self.show_title_in_export  # 从设置读取

        # 选择保存目录
        last_dir = self.get_last_directory()
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "选择保存目录",
            last_dir
        )

        if not output_dir:
            return

        # 记忆导出目录
        self.set_last_directory(output_dir)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 创建进度对话框
        progress_dialog = QProgressDialog("正在导出高DPI图片，请稍候...", "取消", 0, 100, self)
        progress_dialog.setWindowTitle("导出图片")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.setValue(0)

        exported_files = []

        try:
            self.log(f"开始导出高DPI图片 (DPI={dpi})...")
            progress_dialog.setLabelText("正在初始化导出器...")
            progress_dialog.setValue(5)
            QApplication.processEvents()

            # 创建导出器（传入标题显示设置）
            exporter = HighDPIExporter(show_title=show_title)

            # 生成文件名
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            method_clean = method.replace(' ', '_').replace('/', '_')

            # 导出样式1（三列）
            self.log("正在生成样式1对比图...")
            progress_dialog.setLabelText("正在生成样式1对比图...")
            progress_dialog.setValue(20)
            QApplication.processEvents()

            fig1 = exporter.create_comparison_style1(
                original, segmented, mask, method, dpi, width, height, show_title=show_title
            )
            style1_filename = self.generate_export_filename(base_name, method_clean, "comparison_style1")
            style1_path = os.path.join(output_dir, style1_filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(style1_path), exist_ok=True)
            fig1.savefig(style1_path, dpi=dpi, bbox_inches='tight', format='PNG')
            plt.close(fig1)
            exported_files.append(style1_path)
            self.log(f"样式1已保存: {style1_path}")

            # 导出样式2（两列）
            self.log("正在生成样式2对比图...")
            progress_dialog.setLabelText("正在生成样式2对比图...")
            progress_dialog.setValue(50)
            QApplication.processEvents()

            fig2 = exporter.create_comparison_style2(
                original, segmented, method, image_name, dpi, width, height, show_title=show_title
            )
            style2_filename = self.generate_export_filename(base_name, method_clean, "comparison_style2")
            style2_path = os.path.join(output_dir, style2_filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(style2_path), exist_ok=True)
            fig2.savefig(style2_path, dpi=dpi, bbox_inches='tight', format='PNG')
            plt.close(fig2)
            exported_files.append(style2_path)
            self.log(f"样式2已保存: {style2_path}")

            # 单独保存原图、分割图、掩码（不显示标题）
            if save_individual:
                self.log("正在单独保存原图、分割图、掩码...")
                progress_dialog.setLabelText("正在单独保存原图、分割图、掩码...")
                progress_dialog.setValue(70)
                QApplication.processEvents()

                # 生成文件夹名
                if self.filename_format == 'time_name':
                    
                    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    individual_dir_name = f"{time_str}_{base_name}_{method_clean}_individual"
                else:
                    individual_dir_name = f"{self.custom_filename_prefix}_{base_name}_{method_clean}_individual"

                individual_dir = os.path.join(output_dir, individual_dir_name)
                os.makedirs(individual_dir, exist_ok=True)

                paths = exporter.save_individual_images(
                    original, segmented, mask, image_name, individual_dir, dpi, show_title=False
                )
                exported_files.extend([paths['original'], paths['segmented'], paths['mask']])
                self.log(f"原图已保存: {paths['original']}")
                self.log(f"分割图已保存: {paths['segmented']}")
                self.log(f"掩码已保存: {paths['mask']}")

            progress_dialog.setValue(100)
            progress_dialog.close()

            # 使用统一的导出成功对话框
            self.show_export_success_dialog(
                exported_files,
                title="导出成功",
                message_prefix="图片已成功导出"
            )

        except Exception as e:
            progress_dialog.close()
            self.log(f"导出失败: {str(e)}")
            
            QMessageBox.critical(
                self,
                "导出错误",
                f"导出过程中发生错误:\n{str(e)}\n\n{traceback.format_exc()}"
            )

    # ========== DPI设置相关方法 ==========
    def load_dpi_settings(self):
        """从QSettings加载DPI设置"""
        settings = QSettings("RockAnalysisTool", "DPI")
        self.dpi = settings.value("dpi", 300, type=int)
        self.fig_width = settings.value("fig_width", 15.0, type=float)
        self.fig_height = settings.value("fig_height", 5.0, type=float)
        self.save_individual = settings.value("save_individual", True, type=bool)

    def save_dpi_settings(self, dpi, width, height, save_individual):
        """保存DPI设置到QSettings"""
        settings = QSettings("RockAnalysisTool", "DPI")
        settings.setValue("dpi", dpi)
        settings.setValue("fig_width", width)
        settings.setValue("fig_height", height)
        settings.setValue("save_individual", save_individual)

    def show_export_success_dialog(self, file_paths, title="导出成功", message_prefix="文件已成功导出"):
        """
        统一的导出成功对话框
        file_paths: 可以是单个文件路径（字符串）或文件路径列表
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)

        # 判断是单个文件还是多个文件
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            is_single = True
        else:
            is_single = len(file_paths) == 1

        # 构建消息文本
        if is_single:
            msg = f"{message_prefix}：\n{os.path.basename(file_paths[0])}"
            msg += f"\n\n保存位置:\n{os.path.dirname(file_paths[0])}"
        else:
            msg = f"{message_prefix} {len(file_paths)} 个文件：\n\n"
            for i, path in enumerate(file_paths[:5], 1):  # 最多显示5个
                msg += f"{i}. {os.path.basename(path)}\n"
            if len(file_paths) > 5:
                msg += f"... 还有 {len(file_paths) - 5} 个文件\n"
            msg += f"\n保存位置:\n{os.path.dirname(file_paths[0])}"

        msg_box.setText(msg)
        msg_box.setIcon(QMessageBox.Icon.Information)

        # 添加按钮
        if is_single:
            # 单个文件：显示"打开图片"和"打开文件夹"
            open_file_btn = msg_box.addButton("打开图片", QMessageBox.ButtonRole.ActionRole)
            open_folder_btn = msg_box.addButton("打开文件夹", QMessageBox.ButtonRole.ActionRole)
        else:
            # 多个文件：只显示"打开文件夹"
            open_folder_btn = msg_box.addButton("打开文件夹", QMessageBox.ButtonRole.ActionRole)
            open_file_btn = None

        msg_box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)

        msg_box.exec()

        # 处理按钮点击
        clicked_btn = msg_box.clickedButton()
        if clicked_btn == open_file_btn and is_single:
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_paths[0]))
        elif clicked_btn == open_folder_btn:
            folder_path = os.path.dirname(file_paths[0])
            if platform.system() == "Windows":
                if is_single:
                    # 打开文件夹并选中文件
                    subprocess.run(['explorer', '/select,', os.path.normpath(file_paths[0])])
                else:
                    # 只打开文件夹
                    subprocess.run(['explorer', os.path.normpath(folder_path)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))

    # ========== 按钮、信息、日志显示等相关操作 ==========
    def load_csv_data(self):
        """程序启动时加载一次 CSV，避免重复读取"""
        csv_path = r".\resources\files\color.csv"
        try:
            if not os.path.exists(csv_path):
                self.log(f"Error: CSV file not found at {csv_path}")
                # 禁用分析按钮防止崩溃
                self.process_action.setEnabled(False)
                return

            df = pd.read_csv(csv_path, encoding='GBK', sep=',')

            # 检查缺失值
            if df.iloc[:, 6:9].isnull().any().any():
                self.log("Warning: CSV contains missing values in vector columns.")

            # 提取数据存为类属性
            self.standard_vectors = df.iloc[:, 6:9].values
            self.color_names = df['岩石颜色'].values
            self.color_codes = df['Munsell颜色代码'].values
            self.log("Color database loaded successfully.")

        except Exception as e:
            self.log(f"Failed to load CSV: {e}")
            QMessageBox.critical(self, "Data Error", f"Cannot load color database:\n{e}")

    def show_log_menu(self, position):
        """创建并显示日志框的右键菜单"""
        menu = QMenu(self)

        clear_action = QAction("Clear Log", self)
        clear_action.triggered.connect(self.output_text.clear)

        copy_action = QAction("Copy Selected", self)
        copy_action.triggered.connect(self.output_text.copy)

        menu.addAction(copy_action)
        menu.addSeparator()
        menu.addAction(clear_action)

        menu.exec(self.output_text.mapToGlobal(position))

    def log(self, message):
        """同时输出详细日志到文本框和底部状态栏，并自动保存"""
        time_str = datetime.datetime.now().strftime("%H:%M:%S")

        # 1. 更新底部状态小字
        if hasattr(self, 'status_label'):
            self.status_label.setText(message[:50])

        # 2. 更新详细历史日志
        if hasattr(self, 'output_text'):
            log_entry = f"[{time_str}] {message}"
            self.output_text.append(log_entry)
            self.output_text.ensureCursorVisible()

            # 添加到缓冲区
            self.log_buffer.append(log_entry)

            # 自动保存（每10条或每5秒）
            if self.log_auto_save and len(self.log_buffer) >= 10:
                self.save_log_file()

        # 定期保存（每5秒）
        if self.log_auto_save and hasattr(self, 'log_timer'):
            if not hasattr(self, 'log_timer'):
                self.log_timer = QTimer()
                self.log_timer.timeout.connect(self.save_log_file)
                self.log_timer.start(5000)  # 5秒

    def save_log_file(self):
        """保存日志文件"""
        if not self.log_auto_save or not self.log_buffer:
            return

        try:
            # 获取计算机名
            computer_name = socket.gethostname()
            date_str = datetime.datetime.now().strftime("%Y%m%d")

            # 生成文件名
            log_filename = f"{computer_name}_{date_str}.log"
            log_filepath = os.path.join(self.log_save_path, log_filename)

            # 追加写入
            with open(log_filepath, 'a', encoding='utf-8') as f:
                for entry in self.log_buffer:
                    f.write(entry + '\n')

            # 清空缓冲区
            self.log_buffer.clear()

        except Exception as e:
            print(f"保存日志失败: {e}")

    def load_export_settings(self):
        """加载导出设置"""
        settings = QSettings("RockAnalysisTool", "Export")
        # 是否显示标题
        self.show_title_in_export = settings.value("show_title", False, type=bool)
        # 文件名格式：'time_name' 或 'custom'
        self.filename_format = settings.value("filename_format", "time_name", type=str)
        # 自定义文件名前缀
        self.custom_filename_prefix = settings.value("custom_prefix", "Rock", type=str)

    def save_export_settings(self, show_title, filename_format, custom_prefix):
        """保存导出设置"""
        settings = QSettings("RockAnalysisTool", "Export")
        settings.setValue("show_title", show_title)
        settings.setValue("filename_format", filename_format)
        settings.setValue("custom_prefix", custom_prefix)
        self.show_title_in_export = show_title
        self.filename_format = filename_format
        self.custom_filename_prefix = custom_prefix

    # 3. 添加分割设置和日志设置相关方法
    def load_segmentation_settings(self):
        """加载分割设置"""
        settings = QSettings("RockAnalysisTool", "Segmentation")
        default_methods_str = settings.value("default_methods", "GrabCut智能分割,颜色阈值分割", type=str)
        self.default_segmentation_methods = default_methods_str.split(',') if default_methods_str else ['GrabCut智能分割']
        self.show_segmentation_reminder = settings.value("show_reminder", True, type=bool)

    def save_segmentation_settings(self, methods, show_reminder):
        """保存分割设置"""
        settings = QSettings("RockAnalysisTool", "Segmentation")
        settings.setValue("default_methods", ','.join(methods))
        settings.setValue("show_reminder", show_reminder)
        self.default_segmentation_methods = methods
        self.show_segmentation_reminder = show_reminder

    def load_log_settings(self):
        """加载日志设置"""
        settings = QSettings("RockAnalysisTool", "Log")
        self.log_auto_save = settings.value("auto_save", False, type=bool)
        self.log_save_path = settings.value("save_path", os.path.join(os.getcwd(), "logs"), type=str)
        # 确保日志目录存在
        os.makedirs(self.log_save_path, exist_ok=True)

    def save_log_settings(self, auto_save, save_path):
        """保存日志设置"""
        settings = QSettings("RockAnalysisTool", "Log")
        settings.setValue("auto_save", auto_save)
        settings.setValue("save_path", save_path)
        self.log_auto_save = auto_save
        self.log_save_path = save_path
        # 确保日志目录存在
        os.makedirs(self.log_save_path, exist_ok=True)

    def init_log_auto_save(self):
        """初始化日志自动保存"""
        self.log_buffer = []
        if self.log_auto_save:
            self.log("日志自动保存已启用")

    def show_log_settings_dialog(self):
        """显示日志设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("日志设置")
        dialog.setModal(True)
        dialog.resize(500, 300)

        layout = QVBoxLayout(dialog)

        # 自动保存选项
        auto_save_check = QCheckBox("启用日志自动保存")
        auto_save_check.setChecked(self.log_auto_save)
        layout.addWidget(auto_save_check)

        # 保存路径
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("日志保存路径:"))
        path_edit = QLineEdit(self.log_save_path)
        path_edit.setReadOnly(True)
        path_layout.addWidget(path_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(lambda: self.browse_log_path(path_edit))
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # 说明
        info_label = QLabel("日志文件命名规则: 计算机名_日期.log\n例如: DESKTOP-ABC123_20250101.log")
        info_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        layout.addWidget(info_label)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.save_log_settings(auto_save_check.isChecked(), path_edit.text())
            QMessageBox.information(self, "设置已保存", "日志设置已保存")

    def browse_log_path(self, path_edit):
        """浏览日志保存路径"""
        last_dir = path_edit.text() or self.get_last_directory()
        path = QFileDialog.getExistingDirectory(self, "选择日志保存目录", last_dir)
        if path:
            path_edit.setText(path)
            # 记忆目录
            self.set_last_directory(path)

    def show_segmentation_settings_dialog(self):
        """显示分割算法设置对话框"""
        dialog = SegmentationMethodDialog(
            self,
            default_methods=self.default_segmentation_methods,
            show_reminder=self.show_segmentation_reminder
        )
        dialog.setWindowTitle("分割算法默认设置")
        dialog.dont_remind_check.setText("不再提醒选择算法（使用当前选择作为默认）")

        if dialog.exec() == QDialog.DialogCode.Accepted:
            methods = dialog.get_selected_methods()
            if methods:
                self.save_segmentation_settings(methods, dialog.should_remind())
                QMessageBox.information(self, "设置已保存", f"默认分割算法已设置为: {', '.join(methods)}")

    def on_preview_clicked(self, event):
        """当用户点击预览区域时触发"""
        # 右键：清除当前图片（带一次性提示）
        if event.button() == Qt.MouseButton.RightButton:
            self.on_clear_image_requested()
            return

        # 左键：如果已有图片，则打开大图查看；否则触发加载图片
        if event.button() == Qt.MouseButton.LeftButton:
            if self.image is not None and self.image_path:
                self.open_image_viewer(self.image_path)
            else:
                self.load_image()

    def on_clear_image_requested(self):
        """右键清除当前加载的图片"""
        if self.image is None:
            return

        settings = QSettings("RockAnalysisTool", "MainWindow")
        ask = settings.value("ask_clear_image", True, type=bool)

        if ask:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("清除图片")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setText("右键预览区域将清除当前加载的图片。\n\n是否清除当前图片？")
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            checkbox = QCheckBox("不再提醒")
            msg_box.setCheckBox(checkbox)
            result = msg_box.exec()

            if checkbox.isChecked():
                settings.setValue("ask_clear_image", False)

            if result != QMessageBox.StandardButton.Yes:
                return

        # 执行清除操作
        self.image = None
        self.image_path = None
        self.display_pixmap = None
        self.image_label.clear()
        self.image_label.setText(
            "No image available.\n\nClick here or the button in the upper left corner to load an image."
        )

        if hasattr(self, "image_name_label"):
            self.image_name_label.setText("")
            self.image_name_label.setVisible(False)

        # 禁用与当前图片相关的操作
        if hasattr(self, "process_action"):
            self.process_action.setEnabled(False)
        if hasattr(self, "save_action"):
            self.save_action.setEnabled(False)

    def show_about_dialog(self):
        """关于对话框"""
        about_text = """
            <h3>RoCAS v2.20</h3>
            <p><b>Rock Color Analysis System</b></p>
            <hr>
            <p>Developed by: <b>Tian Xiaoying</b></p>
            <p>Affiliation: Your Lab / University Name</p>
            <p>Email: <a href='mailto:dachang0220@163.com'>dachang0220@163.com</a></p>
            <br>
            <p><i>A professional tool for rock surface chromaticity recognition 
            and petrological data analysis.</i></p>
        """
        QMessageBox.about(self, "About RoCAS", about_text)

    def add_copyright_footer(self):
        """添加状态栏：左下角显示状态，右下角显示版权信息"""
        status_bar = self.statusBar()

        # 左侧：状态信息标签
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; margin-right: 5px;")
        status_bar.addWidget(self.status_label)  # 左侧添加状态标签

        # 右侧：版权信息（使用addPermanentWidget使其保持在右侧）
        copyright_label = QLabel("Copyright © 2025-2026 TianXiaoying.All Rights Reserved.")
        copyright_label.setStyleSheet("font-size: 12px; color: #909399;")
        status_bar.addPermanentWidget(copyright_label)  # 右侧添加版权标签

    def show_image_info(self):
        """显示当前图片信息"""
        if self.image is not None:
            height, width = self.image.shape[:2]
            self.log(f"Current image size: {width}x{height}")

            # 计算推荐的网格大小
            recommended = min(25, min(width, height) // 10)
            if recommended % 2 == 0:
                recommended += 1
            self.log(f"Recommended grid size: {recommended} (odd num)")
        else:
            self.log("未加载图片")

    def display_image_on_label(self, img_array):
        """将 OpenCV 图片转换为 QPixmap 并显示"""
        if img_array is None:
            return

        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        # OpenCV 是 BGR，Qt 是 RGB，需要转换
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 适应 Label 大小保持比例
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    # ========== 颜色识别单一操作：图像的加载、裁剪、翻转及识别 ==========
    def load_image(self):
        last_dir = self.get_last_directory()
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "choose rock image",
                last_dir,
                "Image files (*.jpg *.png *.jpeg *.bmp *.tif);;All(*)")

            if not file_path:
                return

            # 保存图片路径
            self.image_path = file_path
            self._last_image_path = file_path

            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Error", "The file does not exist or is missing.")
                return
            file_size = os.path.getsize(file_path)

            if file_size > 10 * 1024 * 1024:
                QMessageBox.warning(self, "warn", "Large files may cause program lag.")

            self.log(f"Try loading images: {file_path}")

            try:
                pil_image = Image.open(file_path)
                self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                self.log("Image successfully loaded using PIL.")
                self.set_last_directory(file_path)
                self._display_image_centered()
                self.log(f"Image loaded successfully: {file_path}")
                self.show_image_info()
                # 更新样本名显示
                if hasattr(self, "image_name_label"):
                    self.image_name_label.setText(f"{os.path.basename(file_path)}")
                    self.image_name_label.setVisible(True)

            except Exception as e:
                self.log(f"PIL failed to load; try OpenCV instead.: {str(e)}")
                self.image = cv2_imread(file_path)
                if self.image is None:
                    raise Exception("Unable to load images using OpenCV")

            if self.image is None or self.image.size == 0:
                QMessageBox.critical(self, "错误", "无法解码图片文件!")
                return

            self._display_image_centered()
            self.log(f"Image loaded successfully: {file_path}")
            self.crop_action.setEnabled(True)
            self.process_action.setEnabled(True)
            self.save_action.setEnabled(True)
        except Exception as e:
            self.log(f"An error occurred while loading the image.: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred while loading the image.: {str(e)}")

    def save_image(self):
        """保存处理后的岩石分析结果图"""
        if self.image is None:
            QMessageBox.warning(self, "Warning", "No image to save!")
            return

        try:
            # 处理裁剪后的内存布局问题
            temp_img = np.ascontiguousarray(self.image)

            h, w, c = temp_img.shape
            bytes_per_line = c * w

            # 转换为 QImage
            qimg = QImage(temp_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()

            # 调用核心导出函数
            self._execute_export(QPixmap.fromImage(qimg), "Rock_Analysis")

        except Exception as e:
            self.log(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "Error", f"转换图像数据失败: {e}")

    def flip_image(self):
        """修复：翻转功能 - 使用英文匹配"""
        if self.image is None:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        mode = self.flip_combo.currentText()
        if mode == "None":
            QMessageBox.warning(self, "Warn", "请先在右侧参数设置中选择一个翻转模式")
            self.log("未选择翻转模式")
            return
        elif mode == "Vertical flip":
            self.image = cv2.flip(self.image, 0)
        elif mode == "Horizontal flip":
            self.image = cv2.flip(self.image, 1)
        elif mode == "Vertical + Horizontal":
            self.image = cv2.flip(self.image, -1)

        self.display_image_on_label(self.image)
        self.log(f"已执行: {mode}")

    def crop_image(self):
        """
        使用增强版裁剪窗口手动选择裁剪区域，并更新显示图像
        """
        if self.image is None:
            QMessageBox.critical(self, "错误", "请先加载图片!")
            return

        # 保存原始图像用于取消操作
        self.original_image = self.image.copy()

        try:
            # 创建增强版裁剪窗口
            self.crop_window = EnhancedCropWindow(self)

            # 检查图像尺寸
            height, width = self.image.shape[:2]
            if height <= 0 or width <= 0:
                raise Exception("图像尺寸无效")

            # 注意：传递原始图像的副本，避免修改原始数据
            self.crop_window.set_image(self.image.copy())
            self.crop_window.cropConfirmed.connect(self.on_crop_confirmed)
            self.crop_window.cropCancelled.connect(self.on_crop_cancelled)

            # 居中显示
            screen_geometry = QApplication.primaryScreen().geometry()
            window_geometry = self.crop_window.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            self.crop_window.move(window_geometry.topLeft())

            self.crop_window.show()

        except Exception as e:
            self.log(f"裁剪窗口创建失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"裁剪窗口创建失败: {str(e)}")

    def on_crop_confirmed(self, rect):
        """裁剪确认回调"""
        try:
            # 执行裁剪操作
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            # 验证裁剪区域
            if w <= 0 or h <= 0:
                raise Exception("无效的裁剪区域")

            # 裁剪图像 - 注意：确保使用原始图像副本
            cropped_image = self.original_image[y:y + h, x:x + w]

            # 验证裁剪后图像
            if cropped_image.size == 0:
                raise Exception("裁剪后图像为空")

            # 更新主图像
            self.image = cropped_image

            # 更新显示
            self._display_image_centered()

            # 更新日志
            self.log(f"图片已裁剪: 位置({x},{y}), 大小{w}x{h}")
            self.log(f"当前图片尺寸: {self.image.shape[1]}x{self.image.shape[0]}")

            # 启用相关功能
            self.process_action.setEnabled(True)
            self.save_action.setEnabled(True)

            # 显示成功消息
            QMessageBox.information(self, "成功", f"图片裁剪完成! \n新尺寸: {w}x{h}")

        except Exception as e:
            # 裁剪失败恢复原始图像
            self.image = self.original_image.copy() if hasattr(self, 'original_image') else None
            self.log(f"裁剪失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"裁剪失败: {str(e)}")
            if self.image is not None:
                self._display_image_centered()

    def on_crop_cancelled(self):
        """裁剪取消回调"""
        self.log("用户取消了裁剪操作")
        # 恢复原始图像显示
        if hasattr(self, 'original_image'):
            self.image = self.original_image.copy()
            self._display_image_centered()

    # 在 EnhancedCropWindow 类中添加异常处理：
    def set_image(self, cv_image):
        """设置OpenCV图像"""
        if cv_image is None:
            return

        try:
            # 检查图像尺寸
            height, width, channel = cv_image.shape
            if height <= 0 or width <= 0:
                raise Exception("图像尺寸无效")

            # 转换OpenCV图像为QImage
            bytes_per_line = 3 * width
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.qimage = QImage(img_rgb.data, width, height,
                                 bytes_per_line, QImage.Format.Format_RGB888).copy()

            if self.qimage.isNull():
                raise Exception("QImage创建失败")

            # 设置到裁剪控件
            self.crop_widget.set_image(self.qimage)
            self.crop_widget.original_size = QPoint(width, height)

            # 更新缩放滑块
            current_scale = int(self.crop_widget.scale_factor * 100)
            self.zoom_slider.setValue(current_scale)
            self.zoom_value_label.setText(f"{current_scale}%")

        except Exception as e:
            print(f"设置图像错误: {e}")
            QMessageBox.critical(self, "错误", f"设置图像失败: {str(e)}")

    def on_crop_confirmed(self, rect):
        """裁剪确认回调"""
        try:
            # 执行裁剪操作
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            # 验证裁剪区域
            if w <= 0 or h <= 0:
                raise Exception("无效的裁剪区域")

            # 裁剪图像
            self.image = self.image[y:y + h, x:x + w]

            # 验证裁剪后图像
            if self.image.size == 0:
                raise Exception("裁剪后图像为空")

            # 更新显示
            self._display_image_centered()
            self.log(f"图片已裁剪: 位置({x},{y}), 大小{w}x{h}")
            self.log(f"当前图片尺寸: {self.image.shape[1]}x{self.image.shape[0]}")

            self.process_action.setEnabled(True)
            self.save_action.setEnabled(True)

        except Exception as e:
            # 裁剪失败恢复原始图像
            if hasattr(self, 'original_image'):
                self.image = self.original_image
            self.log(f"裁剪失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"裁剪失败: {str(e)}")

    def on_crop_cancelled(self):
        """裁剪取消回调"""
        self.log("用户取消了裁剪操作")

    def start_processing(self):
        """开始处理图像逻辑"""
        if self.image is None:
            QMessageBox.critical(self, "错误", "请先加载图片！")
            return

        # 获取当前图片尺寸
        height, width = self.image.shape[:2]

        # 计算合适的网格大小范围
        min_grid = 5
        max_grid = min(width, height) // 4  # 网格大小不超过图片尺寸的1/4
        if max_grid < min_grid:
            max_grid = min_grid + 10

        # 根据图片尺寸动态设置默认值
        default_grid = min(25, max_grid)

        grid_size, ok = QInputDialog.getInt(
            self, "输入网格大小",
            f"请输入网格大小, 如 3、7、9、15 (当前图片大小：{width}x{height}={width * height})",
            value=default_grid, min=min_grid, max=max_grid, step=1
        )
        if not ok:  # 用户点了取消
            return

        # 确保网格大小为奇数
        if grid_size % 2 == 0:
            grid_size += 1
            self.log(f"网格大小已调整为奇数: {grid_size}")
            QMessageBox.information(self, "提示", f"网格大小已调整为奇数: {grid_size}")

        self.log(f"开始分析，网格大小: {grid_size}px, 图片尺寸: {width}x{height}")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 启动后台线程
        self.worker = ColorAnalysisWorker(
            self.image,
            grid_size,
            self.standard_vectors,
            self.color_names,
            self.color_codes
        )

        # 连接信号
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.error_occurred.connect(lambda err: (
            QMessageBox.critical(self, "错误", err),
            self.progress_bar.setVisible(False)
        ))
        self.worker.result_ready.connect(self.on_analysis_finished)

        self.worker.start()

    def capture_high_res_screenshot(self):
        """捕捉软件全貌截图（含边框与标题栏）"""
        # 捕捉包含标题栏的完整窗口几何体
        rect = self.frameGeometry()
        screenshot = self.screen().grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())

        # 传入截图，前缀设为 RoCAS_UI
        self._execute_export(screenshot, "RoCAS_Full_View")

    # ========== 导出图片、导出表格等、识别结果、分析报告等其他功能 ==========
    def _execute_export(self, pixmap_to_save, default_prefix="RoCAS"):
        """
        内部核心导出逻辑：
        整合了 TIFF-300DPI、交互对话框、自动命名、中文兼容
        """
        # 1. 自动生成带时间戳的文件名
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        join_path = os.path.join(self.get_last_directory(), f"{now_str}_{default_prefix}.png")
        # 2. 弹出保存对话框
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            join_path,
            "PNG Files (*.png);;TIFF Files (*.tif *.tiff)",
        )
        if not save_path:
            return

        try:

            # 3. 将 QPixmap 转换为 PIL 对象进行处理 (兼容所有格式)
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.ReadWrite)
            pixmap_to_save.save(buffer, "PNG")
            pil_img = Image.open(io.BytesIO(buffer.data()))

            # 4. 执行保存逻辑
            if save_path.lower().endswith(('.tif', '.tiff')):
                # TIFF 模式：无损压缩 + 300 DPI
                pil_img.save(save_path, format='TIFF', dpi=(300.0, 300.0), compression="tiff_lzw")
                log_msg = f"TIFF (300DPI) Exported: {save_path}"
            else:
                # PNG 模式：原生保存
                pil_img.save(save_path, format='PNG')
                log_msg = f"PNG Exported: {save_path}"

            # 保存图片保存目录
            self.set_last_directory(os.path.dirname(save_path))

            # 5. 更新日志
            if hasattr(self, 'output_text'):
                self.log(log_msg)

            # 6. 弹出交互式对话框
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Export Success")
            msg_box.setText(f"Successfully saved to:\n{os.path.basename(save_path)}")
            msg_box.setIcon(QMessageBox.Icon.Information)

            open_file_btn = msg_box.addButton("Open Image", QMessageBox.ButtonRole.ActionRole)
            open_folder_btn = msg_box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
            msg_box.addButton("OK", QMessageBox.ButtonRole.AcceptRole)

            msg_box.exec()

            # 7. 交互逻辑处理
            clicked_btn = msg_box.clickedButton()
            if clicked_btn == open_file_btn:
                QDesktopServices.openUrl(QUrl.fromLocalFile(save_path))
            elif clicked_btn == open_folder_btn:
                if platform.system() == "Windows":
                    # 打开文件夹并选中该文件
                    subprocess.run(['explorer', '/select,', os.path.normpath(save_path)])
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(save_path)))

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred:\n{str(e)}")

    def on_analysis_finished(self, report_text, stats_list, total_count):
        """分析完成，隐藏进度条并弹出独立的结果分析窗口"""
        self.progress_bar.setVisible(False)
        self.log(f"分析完成。总采样点: {total_count}")

        # 保存数据到实例变量中
        self.stats_data = stats_list
        self.total_samples = total_count

        # 获取文件名
        file_name = "未知样本"
        if hasattr(self, 'image_path') and self.image_path:
            file_name = os.path.basename(self.image_path)
        elif hasattr(self, '_last_image_path') and self._last_image_path:
            file_name = os.path.basename(self._last_image_path)

        try:
            # 如果已存在结果窗口，先关闭它
            if hasattr(self, 'result_window') and self.result_window:
                self.result_window.close()
                self.result_window = None

            self.result_window = AnalysisResultWindow(stats_list, total_count, file_name, self)
            self.result_window.show()
        except Exception as e:
            self.log(f"弹窗失败: {str(e)}")
            
            print(f"弹窗失败详情: {traceback.format_exc()}")
            QMessageBox.warning(self, "提示", "分析已完成，但结果窗口显示失败，请检查日志。")

    def get_last_directory(self):
        """获取上次使用的目录"""
        settings = QSettings("RockAnalysisTool", "MainWindow")
        return settings.value("last_directory", "", type=str)

    def set_last_directory(self, file_path):
        """设置上次使用的目录"""
        # 既支持文件路径也支持文件夹路径
        if file_path and os.path.isdir(file_path):
            directory = file_path
        else:
            directory = os.path.dirname(file_path)
        settings = QSettings("RockAnalysisTool", "MainWindow")
        settings.setValue("last_directory", directory)

    def _display_image_centered(self):
        """图片居中显示"""
        if self.image is not None:
            canvas_width = self.image_label.width()
            canvas_height = self.image_label.height()
            self.display_pixmap = self.convert_to_display(self.image, canvas_width, canvas_height)
            self.image_label.setPixmap(self.display_pixmap)
            self.image_label.setText("")

    def open_image_viewer(self, image_path):
        """打开可缩放的大图查看窗口"""
        if not os.path.exists(image_path):
            return
        self._image_viewer = ImageViewerWindow(image_path, self)
        self._image_viewer.show()

    def convert_to_display(self, img, canvas_width, canvas_height):
        try:
            if canvas_width <= 0 or canvas_height <= 0:
                canvas_width = 600
                canvas_height = 500

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width, channel = img_rgb.shape

            if img_width <= 0 or img_height <= 0:
                raise Exception("无效的图像尺寸")

            original_ratio = img_width / img_height
            canvas_ratio = canvas_width / canvas_height

            if original_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = max(1, int(canvas_width / original_ratio))
            else:
                new_height = canvas_height
                new_width = max(1, int(canvas_height * original_ratio))

            resized_img = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

            q_image = QImage(
                resized_img.data,
                new_width,
                new_height,
                new_width * channel,
                QImage.Format.Format_RGB888
            )

            return QPixmap.fromImage(q_image)
        except Exception as e:
            self.log(f"图像转换错误: {str(e)}")
            return QPixmap()

    def export_single_segmentation_result(self, original, segmented, mask, method, image_name, output_dir):
        """导出单个分割结果（支持指定目录，返回文件列表）"""
        dpi = self.dpi
        width = self.fig_width
        height = self.fig_height
        save_individual = self.save_individual
        show_title = self.show_title_in_export

        exported_files = []

        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            exporter = HighDPIExporter(show_title=show_title)
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            method_clean = method.replace(' ', '_').replace('/', '_')

            # 导出样式1
            fig1 = exporter.create_comparison_style1(
                original, segmented, mask, method, dpi, width, height, show_title=show_title
            )
            style1_filename = self.generate_export_filename(base_name, method_clean, "comparison_style1")
            style1_path = os.path.join(output_dir, style1_filename)
            os.makedirs(os.path.dirname(style1_path), exist_ok=True)
            fig1.savefig(style1_path, dpi=dpi, bbox_inches='tight', format='PNG')
            plt.close(fig1)
            exported_files.append(style1_path)

            # 导出样式2
            fig2 = exporter.create_comparison_style2(
                original, segmented, method, image_name, dpi, width, height, show_title=show_title
            )
            style2_filename = self.generate_export_filename(base_name, method_clean, "comparison_style2")
            style2_path = os.path.join(output_dir, style2_filename)
            os.makedirs(os.path.dirname(style2_path), exist_ok=True)
            fig2.savefig(style2_path, dpi=dpi, bbox_inches='tight', format='PNG')
            plt.close(fig2)
            exported_files.append(style2_path)

            # 单独保存
            if save_individual:
                if self.filename_format == 'time_name':
                    
                    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    individual_dir_name = f"{time_str}_{base_name}_{method_clean}_individual"
                else:
                    individual_dir_name = f"{self.custom_filename_prefix}_{base_name}_{method_clean}_individual"

                individual_dir = os.path.join(output_dir, individual_dir_name)
                os.makedirs(individual_dir, exist_ok=True)
                paths = exporter.save_individual_images(
                    original, segmented, mask, image_name, individual_dir, dpi, show_title=False
                )
                exported_files.extend([paths['original'], paths['segmented'], paths['mask']])

            return exported_files

        except Exception as e:
            self.log(f"导出失败: {str(e)}")
            
            self.log(traceback.format_exc())
            return []

    def generate_export_filename(self, base_name, method_clean, suffix, extension='.png'):
        """根据设置生成导出文件名"""
        if self.filename_format == 'time_name':
            # 时间+样本名格式
            
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{time_str}_{base_name}_{method_clean}_{suffix}{extension}"
        else:
            # 自定义前缀格式
            filename = f"{self.custom_filename_prefix}_{base_name}_{method_clean}_{suffix}{extension}"
        return filename


class HighDPIExporter:
    """高DPI图片导出工具类"""

    def __init__(self, show_title=False):
        self.supported_dpi = [300, 600, 1200]
        self.supported_formats = ['PNG', 'TIFF', 'PDF']
        # 是否显示标题
        self.show_title = show_title

    def create_comparison_style1(self, original, segmented, mask, method_name,
                                 dpi=300, fig_width=15, fig_height=5, show_title=True):
        """创建样式1对比图：原图+分割图+掩码（三列）"""

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi)

        # 原图
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        if show_title:
            axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 分割图
        axes[1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        if show_title:
            axes[1].set_title(f'分割结果 ({method_name})', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 掩码
        if len(mask.shape) == 2:
            axes[2].imshow(mask, cmap='gray')
        else:
            axes[2].imshow(mask)
        if show_title:
            axes[2].set_title('分割掩码', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout(pad=2.0)
        return fig

    def create_comparison_style2(self, original, segmented, method_name, image_name,
                                 dpi=300, fig_width=12, fig_height=6, show_title=True):
        """创建样式2对比图：原图+分割图（两列）"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)

        # 原图
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        if show_title:
            axes[0].set_title(f'原始图像: {os.path.basename(image_name)}', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 分割图
        axes[1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        if show_title:
            axes[1].set_title(f'分割结果 - {method_name}', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout(pad=2.0)
        return fig

    def save_individual_images(self, original, segmented, mask, image_name,
                               output_dir, dpi=300, show_title=False):
        """单独保存原图、分割图、掩码到指定文件夹（不显示标题）"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_name))[0]

        # 保存原图（无标题）
        original_path = os.path.join(output_dir, f"{base_name}_original.png")
        fig_orig = plt.figure(figsize=(10, 10), dpi=dpi)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        if show_title:
            plt.title(f'原始图像: {base_name}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        fig_orig.savefig(original_path, dpi=dpi, bbox_inches='tight', format='PNG', pad_inches=0)
        plt.close(fig_orig)

        # 保存分割图（无标题）
        segmented_path = os.path.join(output_dir, f"{base_name}_segmented.png")
        fig_seg = plt.figure(figsize=(10, 10), dpi=dpi)
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        if show_title:
            plt.title(f'分割结果: {base_name}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        fig_seg.savefig(segmented_path, dpi=dpi, bbox_inches='tight', format='PNG', pad_inches=0)
        plt.close(fig_seg)

        # 保存掩码（无标题）
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        fig_mask = plt.figure(figsize=(10, 10), dpi=dpi)
        if len(mask.shape) == 2:
            plt.imshow(mask, cmap='gray')
        else:
            plt.imshow(mask)
        plt.axis('off')
        if show_title:
            plt.title(f'分割掩码: {base_name}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        fig_mask.savefig(mask_path, dpi=dpi, bbox_inches='tight', format='PNG', pad_inches=0)
        plt.close(fig_mask)

        return {
            'original': original_path,
            'segmented': segmented_path,
            'mask': mask_path
        }

    def generate_export_filename(self, base_name, method_clean, suffix, extension='.png'):
        """根据设置生成导出文件名"""
        if self.filename_format == 'time_name':
            # 时间+样本名格式
            
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{time_str}_{base_name}_{method_clean}_{suffix}{extension}"
        else:
            # 自定义前缀格式
            filename = f"{self.custom_filename_prefix}_{base_name}_{method_clean}_{suffix}{extension}"
        return filename


class DPISettingsDialog(QDialog):
    """DPI和尺寸设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("高DPI导出设置")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        # DPI设置
        dpi_group = QGroupBox("DPI设置")
        dpi_layout = QVBoxLayout()

        self.dpi_300 = QRadioButton("300 DPI (标准论文)")
        self.dpi_600 = QRadioButton("600 DPI (高清)")
        self.dpi_1200 = QRadioButton("1200 DPI (超高清)")
        self.dpi_300.setChecked(True)

        self.dpi_group = QButtonGroup()
        self.dpi_group.addButton(self.dpi_300, 300)
        self.dpi_group.addButton(self.dpi_600, 600)
        self.dpi_group.addButton(self.dpi_1200, 1200)

        dpi_layout.addWidget(self.dpi_300)
        dpi_layout.addWidget(self.dpi_600)
        dpi_layout.addWidget(self.dpi_1200)
        dpi_group.setLayout(dpi_layout)

        # 图像尺寸设置
        size_group = QGroupBox("图像尺寸 (英寸)")
        size_layout = QVBoxLayout()

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("宽度:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 50.0)
        self.width_spin.setValue(15.0)
        self.width_spin.setSingleStep(0.5)
        width_layout.addWidget(self.width_spin)
        size_layout.addLayout(width_layout)

        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("高度:"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 50.0)
        self.height_spin.setValue(5.0)
        self.height_spin.setSingleStep(0.5)
        height_layout.addWidget(self.height_spin)
        size_layout.addLayout(height_layout)

        size_group.setLayout(size_layout)

        # 保存选项
        save_group = QGroupBox("保存选项")
        save_layout = QVBoxLayout()

        self.save_individual = QCheckBox("单独保存原图、分割图、掩码")
        self.save_individual.setChecked(True)
        save_layout.addWidget(self.save_individual)

        save_group.setLayout(save_layout)

        layout.addWidget(dpi_group)
        layout.addWidget(size_group)
        layout.addWidget(save_group)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def get_settings(self):
        """获取设置值"""
        dpi = self.dpi_group.checkedId()
        return {
            'dpi': dpi if dpi > 0 else 300,
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'save_individual': self.save_individual.isChecked()
        }


class AnalysisResultWindow(QMainWindow):
    def __init__(self, stats_data, total_samples, file_name, parent=None):
        super().__init__(parent)
        self.stats_data = stats_data
        self.total_samples = total_samples
        # 对filename进行分割，将文件后缀去掉
        # Path(file_name).stem
        self.file_name = file_name.split('.')[0]

        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self.setWindowTitle(f"识别结果分析 - {file_name}")
        self.resize(850, 650)
        self.init_ui()

    def init_ui(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 设置图标
        self.setWindowIcon(QIcon(r".\resources\assets\images\button\results.png"))

        # 顶部：简洁标题 + 按钮行（风格参考 Dataset）
        header_layout = QHBoxLayout()
        title_lbl = QLabel(f"样本: {self.file_name} | 总采样点: {self.total_samples}")
        title_lbl.setStyleSheet("font-weight: bold; color: #34495e; font-size: 13px;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        btn_layout = QHBoxLayout()
        self.btn_3d = QPushButton("3D颜色空间图")
        self.btn_hist = QPushButton("颜色分布直方图")
        self.btn_export = QPushButton("导出数据")

        for btn in [self.btn_3d, self.btn_hist, self.btn_export]:
            btn.setFixedHeight(28)
            btn.setMinimumWidth(110)
            btn_layout.addWidget(btn)
        header_layout.addLayout(btn_layout)

        layout.addLayout(header_layout)

        # [中间内容区 - 简洁表格样式]
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.card_layout = QVBoxLayout(container)
        self.card_layout.setContentsMargins(10, 10, 10, 10)
        self.card_layout.setSpacing(5)
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 生成简单的数据行
        for i, item in enumerate(self.stats_data):
            if isinstance(item, dict):
                row = self.create_simple_row(i + 1, item)
            else:
                continue
            self.card_layout.addWidget(row)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # 绑定按钮事件
        self.btn_3d.clicked.connect(self.show_3d_view)
        self.btn_hist.clicked.connect(self.show_hist_view)
        self.btn_export.clicked.connect(self.export_data)

    def create_simple_row(self, idx, data):
        row_frame = QFrame()
        row_frame.setStyleSheet("""
            QFrame { 
                background: white; border: none;
                border-bottom: 1px solid #f2f2f2; padding: 12px 5px;
            }
            QFrame:hover { background: #f9f9f9; }
        """)

        main_layout = QHBoxLayout(row_frame)
        main_layout.setSpacing(25)

        # 1. 序号
        idx_label = QLabel(f"{idx:02d}")
        idx_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #bdc3c7;")
        main_layout.addWidget(idx_label)

        # 2. 颜色名称和代码区 (增加固定文字标签)
        name_code_container = QWidget()
        nc_layout = QVBoxLayout(name_code_container)
        nc_layout.setContentsMargins(0, 0, 0, 0)
        nc_layout.setSpacing(4)

        # 使用辅助函数创建带标签的行
        nc_layout.addWidget(self._make_labeled_text("ColorName:", data.get('name', '未知'), "#2c3e50", True))
        nc_layout.addWidget(self._make_labeled_text("ColorCode:", data.get('code', '未知'), "#2c3e50", True))

        main_layout.addWidget(name_code_container, stretch=2)

        # 3. 向量对比区 (上下排列 + 颜色预览)
        vector_container = QWidget()
        v_layout = QVBoxLayout(vector_container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(4)

        std_v = [int(x) for x in data.get('std_vec', [0, 0, 0])]
        act_v = [int(x) for x in data.get('target_vec', [0, 0, 0])]

        # v_layout.addWidget(self._make_vector_line("标准向量:", std_v, "#3498db"))
        # v_layout.addWidget(self._make_vector_line("实测向量:", act_v, "#e74c3c"))
        v_layout.addWidget(self._make_vector_line("STD", std_v, "#2c3e50"))
        v_layout.addWidget(self._make_vector_line("ACT", act_v, "#2c3e50"))

        main_layout.addWidget(vector_container, stretch=2)

        # --- 4. 统计信息 (占比和频次) ---
        stat_frame = QWidget()
        stat_layout = QVBoxLayout(stat_frame)
        stat_layout.setContentsMargins(0, 0, 0, 0)
        stat_layout.setSpacing(4)
        stat_layout.setAlignment(Qt.AlignmentFlag.AlignRight)  # 整体靠右对齐

        # 准备数据
        percent_val = f"{data.get('percent', 0.0):.2f}%"
        count_val = f"{data.get('count', 0)} px"

        # 使用辅助函数创建带标签的行，颜色统一为 #2c3e50
        # 提示：如果希望标签也靠右，可以在 _make_labeled_text 里调整，或者直接在这里添加
        stat_layout.addWidget(self._make_labeled_text("占比:", percent_val, "#2c3e50", True))
        stat_layout.addWidget(self._make_labeled_text("频次:", count_val, "#2c3e50", True))

        main_layout.addWidget(stat_frame, stretch=2)

        return row_frame

    def _make_labeled_text(self, label_text, value_text, color, is_bold):
        """辅助函数：创建 [标签: 变量] 结构的横向部件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # 固定文字标签
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #95a5a6; font-size: 11px; font-weight: normal; min-width: 70px;")

        # 变量内容 (使用 QLineEdit 方便复制)
        entry = QLineEdit(value_text)
        entry.setReadOnly(True)
        bold_style = "font-weight: bold;" if is_bold else ""
        entry.setStyleSheet(f"border: none; background: transparent; color: {color}; {bold_style} font-size: 12px;")

        layout.addWidget(lbl)
        layout.addWidget(entry)
        layout.addStretch()
        return widget

    def _make_vector_line(self, label_text, vec, text_color):
        """辅助函数：创建 [向量标签: 坐标值 [色块]]"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #7f8c8d; font-size: 11px; min-width: 60px;")

        vec_str = f"{vec[0]}, {vec[1]}, {vec[2]}"
        entry = QLineEdit(vec_str)
        entry.setReadOnly(True)
        entry.setFixedWidth(110)
        entry.setStyleSheet(f"""
            border: 1px solid #f0f0f0; background: #fafafa; 
            color: {text_color}; font-family: 'Consolas'; font-size: 11px;
            padding: 1px 4px; border-radius: 2px;
        """)

        # 颜色预览色块
        color_swatch = QFrame()
        color_swatch.setFixedSize(35, 16)
        color_swatch.setStyleSheet(f"""
            background-color: rgb({vec[0]}, {vec[1]}, {vec[2]});
            border: 1px solid #ddd; border-radius: 3px;
        """)

        layout.addWidget(lbl)
        layout.addWidget(entry)
        layout.addWidget(color_swatch)
        layout.addStretch()
        return widget

    def _make_vector_line(self, label_text, vec, text_color):
        """辅助函数：创建 单行标签+坐标+色块 的部件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # 引导文字
        lbl = QLabel(f"{label_text}:")
        lbl.setStyleSheet(f"color: #7f8c8d; font-size: 11px; min-width: 55px;")

        # RGB数值文本 (可复制)
        vec_str = f"{vec[0]}, {vec[1]}, {vec[2]}"
        entry = QLineEdit(vec_str)
        entry.setReadOnly(True)
        entry.setStyleSheet(f"""
            border: none; 
            background: #f8f9fa; 
            color: {text_color}; 
            font-family: 'Consolas'; 
            font-size: 11px;
            padding: 2px 5px;
            border-radius: 2px;
        """)
        entry.setFixedWidth(100)

        # 颜色预览色块
        color_swatch = QFrame()
        color_swatch.setFixedSize(30, 14)
        # 绘制对应的颜色
        color_swatch.setStyleSheet(f"""
            background-color: rgb({vec[0]}, {vec[1]}, {vec[2]});
            border: 1px solid #ddd;
            border-radius: 2px;
        """)

        layout.addWidget(lbl)
        layout.addWidget(entry)
        layout.addWidget(color_swatch)
        layout.addStretch()

        return widget

    def show_3d_view(self):
        """弹出带工具栏的独立 3D 颜色空间窗口"""
        # 从父窗口获取DPI设置，如果没有则使用默认值300
        dpi = 300
        if self.parent() and hasattr(self.parent(), 'dpi'):
            dpi = self.parent().dpi

        self.plot_3d_win = MatplotlibWindow(f"3D 颜色空间 - {self.file_name}", self)

        fig = self.plot_3d_win.get_figure()
        fig.set_dpi(dpi)  # 设置DPI
        ax = fig.add_subplot(111, projection='3d')

        # 2. 准备数据（取前 50 个主要颜色）
        top_stats = self.stats_data[:50]

        # 收集数据
        xs, ys, zs, colors, sizes = [], [], [], [], []

        for item in top_stats:
            if isinstance(item, dict):
                target_vec = item.get('target_vec', [0, 0, 0])
                percent = item.get('percent', 0.0)
            else:
                # 兼容旧格式
                target_vec = item[5] if len(item) > 5 else [0, 0, 0]
                percent = item[3] if len(item) > 3 else 0.0

            if isinstance(target_vec, np.ndarray):
                target_vec = target_vec.tolist()

            xs.append(target_vec[0])
            ys.append(target_vec[1])
            zs.append(target_vec[2])
            colors.append(np.array(target_vec) / 255.0)
            sizes.append(percent * 20 + 50)

        if not xs:  # 没有数据
            QMessageBox.warning(self, "警告", "没有可显示的颜色数据")
            return

        # 3. 绘图
        ax.scatter(xs, ys, zs, c=colors, s=sizes,
                   alpha=0.8, edgecolors='black', linewidth=0.5)

        # 4. 设置轴标签和样式
        ax.set_xlabel('Red (R)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Green (G)', fontsize=9, fontweight='bold')
        ax.set_zlabel('Blue (B)', fontsize=9, fontweight='bold')
        ax.set_title(f'RGB 坐标系下的 3D 颜色空间分布\n{self.file_name}',
                     fontsize=11, fontweight='bold', pad=20)

        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])
        ax.grid(True, alpha=0.3)

        # 5. 刷新画布并显示窗口
        self.plot_3d_win.draw()
        self.plot_3d_win.show()

    def export_data(self):
        """导出结果数据 (整合 TXT 报告、PNG 表格和 Excel 表格)"""
        default_name = f"{os.path.splitext(self.file_name)[0]}_识别结果"

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "导出识别结果", default_name,
            "Text Files (*.txt);;PNG Images (*.png);;Excel Files (*.xlsx);;CSV Files (*.csv)"
        )

        if not save_path:
            return

        try:
            # 根据文件扩展名确定格式
            ext = os.path.splitext(save_path)[1].lower()

            if ext == '.txt':
                self.export_txt(save_path)
            elif ext == '.png':
                self.export_png(save_path)
            elif ext == '.xlsx':
                self.export_excel(save_path)
            elif ext == '.csv':
                self.export_csv(save_path)
            else:
                # 默认使用TXT
                save_path = save_path + '.txt'
                self.export_txt(save_path)

            # 导出成功，弹出交互对话框
            self.show_export_success_dialog(save_path)

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"错误详情: {str(e)}")

    def export_png(self, save_path):
        # --- 导出 PNG (绘图表格) ---
        if save_path.endswith('.png'):
            # 动态计算高度
            fig_h = max(6, len(self.stats_data) * 0.5 + 2)
            fig = Figure(figsize=(12, fig_h), dpi=200)
            ax = fig.add_subplot(111)
            ax.axis('off')

            # 准备表格数据
            col_labels = ["序号", "颜色名称", "代码", "占比(%)", "频数", "标准RGB", "实测RGB"]
            table_vals = []

            for i, item in enumerate(self.stats_data):
                if isinstance(item, dict):
                    name = item.get('name', '未知')
                    code = item.get('code', '未知')
                    percent = item.get('percent', 0.0)
                    count = item.get('count', 0)
                    std_vec = item.get('std_vec', [0, 0, 0])
                    target_vec = item.get('target_vec', [0, 0, 0])
                else:
                    # 兼容旧格式
                    name = item[0] if len(item) > 0 else '未知'
                    code = item[1] if len(item) > 1 else '未知'
                    count = item[2] if len(item) > 2 else 0
                    percent = item[3] if len(item) > 3 else 0.0
                    std_vec = item[4] if len(item) > 4 else [0, 0, 0]
                    target_vec = item[5] if len(item) > 5 else [0, 0, 0]

                if isinstance(std_vec, np.ndarray):
                    std_vec = std_vec.tolist()
                if isinstance(target_vec, np.ndarray):
                    target_vec = target_vec.tolist()

                std_s = f"{int(std_vec[0])},{int(std_vec[1])},{int(std_vec[2])}"
                act_s = f"{int(target_vec[0])},{int(target_vec[1])},{int(target_vec[2])}"
                table_vals.append([
                    str(i + 1), name, code, f"{percent:.2f}%",
                    str(count), std_s, act_s
                ])

            # 绘制表格
            the_table = ax.table(cellText=table_vals, colLabels=col_labels,
                                 loc='center', cellLoc='center', colColours=["#e6e6e6"] * 7)
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1, 1.8)

            ax.set_title(f"岩石颜色识别结果表 - {self.file_name}\n(总采样点: {self.total_samples})",
                         fontsize=14, pad=20)

            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.5)

    def export_excel(self, save_path):
        """导出为Excel：增加两列专门用于颜色填充预览"""
        # 1. 准备数据：增加 '标准颜色预览' 和 '实测颜色预览' 两列空占位
        rows = []
        for i, item in enumerate(self.stats_data):
            # 处理字典或列表格式
            if isinstance(item, dict):
                std_v = [int(x) for x in item.get('std_vec', [0, 0, 0])]
                act_v = [int(x) for x in item.get('target_vec', [0, 0, 0])]
                name, code, perc, count = item.get('name', ''), item.get('code', ''), item.get('percent', 0), item.get(
                    'count', 0)
            else:
                name, code, count, perc, std_v, act_v = item[0], item[1], item[2], item[3], item[4], item[5]

            rows.append([
                i + 1, name, code, round(perc, 2), count,
                std_v[0], std_v[1], std_v[2],
                act_v[0], act_v[1], act_v[2],
                "", ""  # 最后两列留空，准备填色
            ])

        cols = ['序号', '颜色名称', '代码', '占比(%)', '频数',
                '标准R', '标准G', '标准B', '实测R', '实测G', '实测B',
                '标准RGB预览', '实测RGB预览']
        df = pd.DataFrame(rows, columns=cols)

        # 2. 写入 Excel 并进行单元格染色
        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='识别结果')
                worksheet = writer.sheets['识别结果']

                # --- 调整样式 ---
                # 设置色块预览列（L列和M列）宽一点，方便观察
                worksheet.column_dimensions['L'].width = 18
                worksheet.column_dimensions['M'].width = 18

                for row_idx in range(2, len(df) + 2):
                    # 获取该行的 RGB
                    sr, sg, sb = rows[row_idx - 2][5:8]  # 标准
                    ar, ag, ab = rows[row_idx - 2][8:11]  # 实测

                    # 转换为 HEX (注意：openpyxl 的 PatternFill 需要 RRGGBB)
                    hex_std = f"{int(sr):02x}{int(sg):02x}{int(sb):02x}"
                    hex_act = f"{int(ar):02x}{int(ag):02x}{int(ab):02x}"

                    # 填充最后两列 (Column 12 和 13)
                    cell_std = worksheet.cell(row=row_idx, column=12)
                    cell_act = worksheet.cell(row=row_idx, column=13)

                    cell_std.fill = PatternFill(start_color=hex_std, end_color=hex_std, fill_type="solid")
                    cell_act.fill = PatternFill(start_color=hex_act, end_color=hex_act, fill_type="solid")

                # 设置标题行颜色（灰色）
                header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
                for cell in worksheet[1]:
                    cell.fill = header_fill

            QMessageBox.information(self, "成功", f"Excel已生成！\n请查看最后两列的颜色对比。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存Excel失败: {str(e)}")

    def export_csv(self, save_path):
        """导出为CSV文件"""
        # 准备数据
        data = []
        for i, item in enumerate(self.stats_data):
            if isinstance(item, dict):
                name = item.get('name', '未知')
                code = item.get('code', '未知')
                percent = item.get('percent', 0.0)
                count = item.get('count', 0)
                std_vec = item.get('std_vec', [0, 0, 0])
                target_vec = item.get('target_vec', [0, 0, 0])
            else:
                # 兼容旧格式
                name = item[0] if len(item) > 0 else '未知'
                code = item[1] if len(item) > 1 else '未知'
                count = item[2] if len(item) > 2 else 0
                percent = item[3] if len(item) > 3 else 0.0
                std_vec = item[4] if len(item) > 4 else [0, 0, 0]
                target_vec = item[5] if len(item) > 5 else [0, 0, 0]

            if isinstance(std_vec, np.ndarray):
                std_vec = std_vec.tolist()
            if isinstance(target_vec, np.ndarray):
                target_vec = target_vec.tolist()

            data.append({
                '序号': i + 1,
                '颜色名称': name,
                '代码': code,
                '占比(%)': f"{percent:.2f}",
                '频数': count,
                '标准R': std_vec[0],
                '标准G': std_vec[1],
                '标准B': std_vec[2],
                '实测R': target_vec[0],
                '实测G': target_vec[1],
                '实测B': target_vec[2]
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')

    def show_export_success_dialog(self, file_path):
        """显示导出成功的交互对话框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("导出成功")
        msg_box.setText(f"文件已成功导出到：\n{file_path}")
        msg_box.setIcon(QMessageBox.Icon.Information)

        # 添加按钮
        open_file_btn = msg_box.addButton("打开文件", QMessageBox.ButtonRole.ActionRole)
        open_folder_btn = msg_box.addButton("打开所在文件夹", QMessageBox.ButtonRole.ActionRole)
        ok_btn = msg_box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)

        msg_box.exec()

        # 处理按钮点击
        clicked_btn = msg_box.clickedButton()
        if clicked_btn == open_file_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
        elif clicked_btn == open_folder_btn:
            folder_path = os.path.dirname(file_path)
            if platform.system() == "Windows":
                # 打开文件夹并选中该文件
                subprocess.run(['explorer', '/select,', os.path.normpath(file_path)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))

    def export_txt(self, save_path):
        # --- 导出 TXT ---
        if save_path.endswith('.txt'):
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("岩石颜色识别报告\n")
                f.write(f"样本名称: {self.file_name}\n")
                f.write(f"导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总采样点: {self.total_samples}\n")
                f.write("=" * 60 + "\n\n")

                header = f"{'序号':<5}{'颜色名称':<12}{'代码':<12}{'占比(%)':<10}{'频数':<10}{'实测RGB':<20}\n"
                f.write(header)
                f.write("-" * 80 + "\n")

                for i, item in enumerate(self.stats_data):
                    if isinstance(item, dict):
                        name = item.get('name', '未知')
                        code = item.get('code', '未知')
                        percent = item.get('percent', 0.0)
                        count = item.get('count', 0)
                        target_vec = item.get('target_vec', [0, 0, 0])
                    else:
                        # 兼容旧格式
                        name = item[0] if len(item) > 0 else '未知'
                        code = item[1] if len(item) > 1 else '未知'
                        count = item[2] if len(item) > 2 else 0
                        percent = item[3] if len(item) > 3 else 0.0
                        target_vec = item[5] if len(item) > 5 else [0, 0, 0]

                    if isinstance(target_vec, np.ndarray):
                        target_vec = target_vec.tolist()

                    rgb_str = f"({int(target_vec[0])},{int(target_vec[1])},{int(target_vec[2])})"
                    f.write(f"{i + 1:<5}{name:<12}{code:<12}{percent:<10.2f}{count:<10}{rgb_str:<20}\n")

    def show_hist_view(self):
        """显示统计直方图窗口"""
        if not self.stats_data:
            return

        # 从父窗口获取DPI设置，如果没有则使用默认值300
        dpi = 300
        if self.parent() and hasattr(self.parent(), 'dpi'):
            dpi = self.parent().dpi

        self.hist_win = MatplotlibWindow(f"颜色分布直方图 - {self.file_name}", self)
        fig = self.hist_win.get_figure()
        fig.set_dpi(dpi)  # 设置DPI
        ax = fig.add_subplot(111)

        # 2. 准备数据
        plot_stats = self.stats_data[:12]
        names = []
        values = []
        colors = []

        for item in plot_stats:
            if isinstance(item, dict):
                name = item.get('name', '未知')
                code = item.get('code', '未知')
                percent = item.get('percent', 0.0)
                std_vec = item.get('std_vec', [0, 0, 0])
            else:
                # 兼容旧格式
                name = item[0] if len(item) > 0 else '未知'
                code = item[1] if len(item) > 1 else '未知'
                percent = item[3] if len(item) > 3 else 0.0
                std_vec = item[4] if len(item) > 4 else [0, 0, 0]

            # 直方图x轴颜色代码，不显示颜色名称
            # names.append(f"{name}\n({code})")
            names.append(f"{code}")
            values.append(percent)

            # 使用标准RGB值作为颜色
            if isinstance(std_vec, np.ndarray):
                std_vec = std_vec.tolist()
            rgb_normalized = [c / 255.0 for c in std_vec[:3]]
            colors.append(rgb_normalized)

        if not names:
            QMessageBox.warning(self, "警告", "没有可显示的数据")
            return

        # 3. 绘制柱状图
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1.2)

        # 4. 设置轴标签和样式
        ax.set_xlabel('颜色类别', fontsize=12, fontweight='bold')
        ax.set_ylabel('占比 (%)', fontsize=12, fontweight='bold')
        # ax.set_title(f'颜色分布直方图\n{self.file_name}',
        #             fontsize=14, fontweight='bold', pad=20)

        # 设置x轴刻度
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)

        # 设置y轴范围
        ax.set_ylim([0, max(values) * 1.2 if values else 100])

        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 设置网格
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 设置字体和样式以提高清晰度
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # 设置刻度标签
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()

        # 5. 刷新画布并显示窗口
        self.hist_win.draw()
        self.hist_win.show()


class ColorRecognizeMethods():
    """颜色识别核心算法"""

    def __init__(self):
        self.progress_var = None
        self.df = self._create_color_database()
        self.standard_vectors = None
        self.color_names = None
        self.color_codes = None
        self._prepare_color_data()

    def _create_color_database(self):
        """创建颜色数据库"""
        df = pd.read_csv(r".\resources\files\color.csv", encoding='GBK', sep=',')
        if df.iloc[:, 6:9].isnull().any().any():
            print("数据中存在缺失值，请检查数据。")

        # 提前提取标准颜色向量、颜色名称和颜色代码，避免在循环中重复提取
        standard_vectors = df.iloc[:, 6:9].values
        color_names = df['岩石颜色'].values
        color_codes = df['Munsell颜色代码'].values

        return df

    def _prepare_color_data(self):
        """准备颜色数据"""
        if self.df is not None:
            self.standard_vectors = self.df.iloc[:, 6:9].values  # 修复：使用正确的列索引
            self.color_names = self.df['岩石颜色'].values
            self.color_codes = self.df['Munsell颜色代码'].values

    # 计算标准颜色向量
    @staticmethod
    def calculate_color_vector(row):
        return row['R'], row['G'], row['B']

    # 计算欧氏距离
    @staticmethod
    def euclidean_distance(vector1, vector2):
        return np.linalg.norm(np.array(vector1) - np.array(vector2))

    # 计算余弦相似度
    @staticmethod
    def cosine_similarity(vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0


class CropRectWidget(QWidget):
    """自定义裁剪区域选择控件"""
    rectChanged = pyqtSignal(QRect)
    rectConfirmed = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # 状态变量
        self.image = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.original_size = QPoint(0, 0)

        # 裁剪矩形
        self.crop_rect = QRect()
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.dragging = False
        self.resizing = False
        self.resize_edge = None
        self.dragging_image = False
        self.last_mouse_pos = QPoint()
        self.drawing_new_rect = False

        # 显示选项
        self.show_grid = True
        self.grid_size = 50
        self.show_coords = True
        self.show_pixel_info = True

        # 颜色配置
        self.rect_color = QColor(255, 255, 255)  # 白色边框
        self.rect_fill_color = QColor(255, 255, 255, 30)  # 轻微半透明
        self.grid_color = QColor(200, 200, 200, 100)
        self.text_color = QColor(255, 255, 255)

        # 边缘调整大小区域宽度
        self.edge_margin = 8

        # 调试标志
        self.debug_mode = False

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(100, 100)
        self.setStyleSheet("background-color: #2c2c2c;")

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 如果已经有图片，重新适应窗口
        if self.image is not None:
            self.fit_to_view()

    def fit_to_view(self):
        """自适应窗口大小，确保图片居中显示"""
        if self.image is None or self.width() == 0 or self.height() == 0:
            return

        img_w, img_h = self.image.width(), self.image.height()
        view_w, view_h = self.width(), self.height()

        # 计算保持长宽比的最大缩放比例
        scale_w = view_w / img_w if img_w > 0 else 1.0
        scale_h = view_h / img_h if img_h > 0 else 1.0
        self.scale_factor = min(scale_w, scale_h, 1.0)  # 最大1:1显示

        # 计算缩放后的图像尺寸
        scaled_w = img_w * self.scale_factor
        scaled_h = img_h * self.scale_factor

        # 计算居中偏移
        self.offset = QPoint(
            int((view_w - scaled_w) // 2),
            int((view_h - scaled_h) // 2)
        )

        self.update()

    def get_image_point(self, widget_point):
        """将控件坐标转换为图像坐标"""
        if self.scale_factor <= 0:
            return QPoint(-1, -1)

        x = int(round((widget_point.x() - self.offset.x()) / self.scale_factor))
        y = int(round((widget_point.y() - self.offset.y()) / self.scale_factor))

        return QPoint(x, y)

    def get_widget_point(self, image_point):
        """将图像坐标转换为控件坐标"""
        x = int(round(image_point.x() * self.scale_factor + self.offset.x()))
        y = int(round(image_point.y() * self.scale_factor + self.offset.y()))
        return QPoint(x, y)

    def get_scaled_rect(self, rect):
        """将图像矩形转换为控件矩形"""
        if rect.isNull():
            return QRect()

        top_left = self.get_widget_point(rect.topLeft())
        bottom_right = self.get_widget_point(rect.bottomRight())

        return QRect(top_left, bottom_right)

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor(45, 45, 45))  # 深灰色背景

        if self.image is None:
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont("Microsoft YaHei", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "请加载图像...")
            return

        # 计算缩放后的图像尺寸
        scaled_w = int(self.original_size.x() * self.scale_factor)
        scaled_h = int(self.original_size.y() * self.scale_factor)

        if scaled_w <= 0 or scaled_h <= 0:
            return

        # 缩放图像并保持长宽比
        scaled_image = self.image.scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # 居中绘制图像
        painter.drawImage(self.offset.x(), self.offset.y(), scaled_image)

        # 如果图片较小，绘制边界框
        if scaled_w < self.width() and scaled_h < self.height():
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.offset.x() - 1, self.offset.y() - 1,
                             scaled_w + 2, scaled_h + 2)

        if self.show_grid and self.scale_factor > 0.1:
            self.draw_grid(painter)

        if not self.crop_rect.isNull():
            scaled_rect = self.get_scaled_rect(self.crop_rect)
            self.draw_crop_rect(painter, scaled_rect)

        if self.show_coords:
            self.draw_coordinate_info(painter)

    def debug_print(self, message):
        """调试打印"""
        if self.debug_mode:
            print(f"[CropRectWidget] {message}")

    def set_image(self, image):
        """设置要裁剪的图像"""
        self.image = image
        if self.image is not None:
            self.original_size = QPoint(self.image.width(), self.image.height())
            self.scale_factor = 1.0
            self.offset = QPoint(0, 0)
            self.fit_to_view()
        self.update()

    def draw_grid(self, painter):
        """绘制网格"""
        if self.scale_factor <= 0:
            return

        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.DotLine))

        grid_spacing = int(self.grid_size * self.scale_factor)
        if grid_spacing < 10:
            return

        start_x = self.offset.x()
        end_x = start_x + int(self.original_size.x() * self.scale_factor)
        x = start_x
        while x <= end_x:
            painter.drawLine(x, self.offset.y(), x,
                             self.offset.y() + int(self.original_size.y() * self.scale_factor))
            x += grid_spacing

        start_y = self.offset.y()
        end_y = start_y + int(self.original_size.y() * self.scale_factor)
        y = start_y
        while y <= end_y:
            painter.drawLine(self.offset.x(), y,
                             self.offset.x() + int(self.original_size.x() * self.scale_factor), y)
            y += grid_spacing

    def draw_crop_rect(self, painter, rect):
        """绘制裁剪矩形 - 简化大方版"""
        if rect.isNull():
            return

        # 轻微半透明填充（几乎看不见）
        painter.setBrush(QBrush(self.rect_fill_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(rect)

        # 白色实线边框（比虚线更清晰）
        pen = QPen(self.rect_color, 2, Qt.PenStyle.SolidLine)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawRect(rect)

        # 显示尺寸信息（白色文字，简洁大方）
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        # 使用实际的裁剪矩形尺寸，不是缩放后的rect
        width = self.crop_rect.width()
        height = self.crop_rect.height()
        text = f"{width} × {height} px"

        # 计算文本位置（在矩形上方居中显示）
        text_width = painter.fontMetrics().horizontalAdvance(text)
        text_x = rect.center().x() - text_width // 2
        text_y = rect.top() - 8

        # 绘制白色文字
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(text_x, text_y, text)

    def draw_coordinate_info(self, painter):
        """绘制坐标信息"""
        painter.setFont(QFont("Consolas", 10))

        mouse_pos = self.mapFromGlobal(self.cursor().pos())
        if self.rect().contains(mouse_pos):
            img_point = self.get_image_point(mouse_pos)

            if (0 <= img_point.x() < self.original_size.x() and
                    0 <= img_point.y() < self.original_size.y()):

                info_text = f"X: {img_point.x():4d}  Y: {img_point.y():4d}"

                if not self.crop_rect.isNull():
                    if self.crop_rect.contains(img_point):
                        rel_x = img_point.x() - self.crop_rect.left()
                        rel_y = img_point.y() - self.crop_rect.top()
                        info_text += f"  (ΔX: {rel_x:+4d}, ΔY: {rel_y:+4d})"

                # 在右下角显示坐标信息
                text_width = painter.fontMetrics().horizontalAdvance(info_text)
                text_x = self.width() - text_width - 10
                text_y = self.height() - 10

                # 绘制文本
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(text_x, text_y - 5, info_text)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.image is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            widget_pos = event.pos()
            img_pos = self.get_image_point(widget_pos)

            # 确保在图像范围内
            if not (0 <= img_pos.x() < self.original_size.x() and
                    0 <= img_pos.y() < self.original_size.y()):
                return

            # 检查是否在调整大小的边缘
            if not self.crop_rect.isNull():
                self.resize_edge = self.get_resize_edge(widget_pos)
                if self.resize_edge:
                    self.resizing = True
                    self.start_point = img_pos
                    return

                # 检查是否在裁剪矩形内
                scaled_rect = self.get_scaled_rect(self.crop_rect)
                if scaled_rect.contains(widget_pos):
                    self.dragging = True
                    self.start_point = img_pos
                    return

            # 检查是否按住Ctrl/Shift键拖动图像
            modifiers = event.modifiers()
            if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                self.dragging_image = True
                self.last_mouse_pos = widget_pos
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

            # 开始绘制新矩形
            self.drawing_new_rect = True
            self.dragging = True
            self.crop_rect = QRect(img_pos, img_pos)
            self.start_point = img_pos
            self.end_point = img_pos

        elif event.button() == Qt.MouseButton.RightButton:
            # 右键清除选区
            self.crop_rect = QRect()
            self.update()
            self.rectChanged.emit(QRect())

        self.update()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.image is None:
            return

        widget_pos = event.pos()
        img_pos = self.get_image_point(widget_pos)

        # 处理图像拖动
        if self.dragging_image:
            delta = widget_pos - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = widget_pos
            self.update()
            return

        # 确保在图像范围内
        img_pos.setX(max(0, min(img_pos.x(), self.original_size.x() - 1)))
        img_pos.setY(max(0, min(img_pos.y(), self.original_size.y() - 1)))

        # 更新鼠标光标
        cursor_set = False

        if not self.crop_rect.isNull():
            resize_edge = self.get_resize_edge(widget_pos)
            if resize_edge:
                self.set_resize_cursor(resize_edge)
                cursor_set = True
            else:
                scaled_rect = self.get_scaled_rect(self.crop_rect)
                if scaled_rect.contains(widget_pos):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    cursor_set = True

        # 检查是否按住Ctrl/Shift键显示手型光标
        modifiers = event.modifiers()
        if not cursor_set and modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            cursor_set = True

        if not cursor_set:
            self.setCursor(Qt.CursorShape.CrossCursor)

        # 处理调整大小
        if self.resizing and self.resize_edge:
            self.resize_crop_rect(img_pos)
            self.update()
            return

        # 处理拖拽
        if self.dragging:
            self.end_point = img_pos

            if self.drawing_new_rect:
                # 绘制新矩形
                self.crop_rect = QRect(self.start_point, self.end_point).normalized()
            else:
                # 移动现有矩形
                delta = img_pos - self.start_point
                new_rect = self.crop_rect.translated(delta)

                # 确保不超出图像边界
                if (new_rect.left() >= 0 and new_rect.top() >= 0 and
                        new_rect.right() < self.original_size.x() and
                        new_rect.bottom() < self.original_size.y()):
                    self.crop_rect = new_rect
                    self.start_point = img_pos

            self.update()
            self.rectChanged.emit(self.crop_rect)
            return

        self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.resizing = False
            self.drawing_new_rect = False
            self.resize_edge = None
            self.dragging_image = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

            # 如果矩形太小，清除它
            if self.crop_rect.width() < 5 and self.crop_rect.height() < 5:
                self.crop_rect = QRect()
                self.rectChanged.emit(QRect())

        self.update()

    def wheelEvent(self, event):
        """滚轮缩放事件"""
        if self.image is None or self.scale_factor <= 0:
            return

        try:
            # 获取鼠标位置
            mouse_pos = event.position().toPoint()

            # 获取缩放前的图像坐标
            old_img_pos = self.get_image_point(mouse_pos)

            # 计算缩放因子
            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta > 0 else 0.9

            # 计算新的缩放比例
            new_scale = self.scale_factor * zoom_factor

            # 限制缩放范围 (5% 到 500%)
            if new_scale < 0.05:
                new_scale = 0.05
            elif new_scale > 5.0:
                new_scale = 5.0

            # 如果缩放比例没有变化，直接返回
            if abs(new_scale - self.scale_factor) < 0.001:
                return

            # 更新缩放比例
            self.scale_factor = new_scale

            # 调整偏移量，使鼠标位置对应的图像点保持不变
            new_offset_x = mouse_pos.x() - old_img_pos.x() * self.scale_factor
            new_offset_y = mouse_pos.y() - old_img_pos.y() * self.scale_factor

            # 确保偏移量是有效的
            img_width = self.original_size.x()
            img_height = self.original_size.y()

            # 计算缩放后的图像尺寸
            scaled_width = img_width * self.scale_factor
            scaled_height = img_height * self.scale_factor

            # 限制偏移量，防止图像被拖出视图
            max_offset_x = max(0, int(scaled_width - self.width()))
            max_offset_y = max(0, int(scaled_height - self.height()))

            # 确保偏移量在合理范围内
            new_offset_x = max(-max_offset_x, min(0, int(new_offset_x)))
            new_offset_y = max(-max_offset_y, min(0, int(new_offset_y)))

            # 更新偏移量
            self.offset = QPoint(new_offset_x, new_offset_y)

            # 刷新显示
            self.update()

        except Exception as e:
            self.debug_print(f"滚轮事件错误: {e}")

    def get_resize_edge(self, widget_point):
        """获取调整大小的边缘"""
        if self.crop_rect.isNull():
            return None

        scaled_rect = self.get_scaled_rect(self.crop_rect)
        margin = self.edge_margin

        left_edge = abs(widget_point.x() - scaled_rect.left()) <= margin
        right_edge = abs(widget_point.x() - scaled_rect.right()) <= margin
        top_edge = abs(widget_point.y() - scaled_rect.top()) <= margin
        bottom_edge = abs(widget_point.y() - scaled_rect.bottom()) <= margin

        if left_edge and top_edge:
            return 'nw'
        elif left_edge and bottom_edge:
            return 'sw'
        elif right_edge and top_edge:
            return 'ne'
        elif right_edge and bottom_edge:
            return 'se'
        elif left_edge:
            return 'w'
        elif right_edge:
            return 'e'
        elif top_edge:
            return 'n'
        elif bottom_edge:
            return 's'

        return None

    def set_resize_cursor(self, edge):
        """设置调整大小的光标"""
        cursors = {
            'n': Qt.CursorShape.SizeVerCursor,
            's': Qt.CursorShape.SizeVerCursor,
            'w': Qt.CursorShape.SizeHorCursor,
            'e': Qt.CursorShape.SizeHorCursor,
            'nw': Qt.CursorShape.SizeFDiagCursor,
            'se': Qt.CursorShape.SizeFDiagCursor,
            'ne': Qt.CursorShape.SizeBDiagCursor,
            'sw': Qt.CursorShape.SizeBDiagCursor
        }
        self.setCursor(cursors.get(edge, Qt.CursorShape.ArrowCursor))

    def resize_crop_rect(self, img_pos):
        """调整裁剪矩形大小"""
        rect = self.crop_rect

        if 'n' in self.resize_edge:
            rect.setTop(int(max(0, min(img_pos.y(), rect.bottom() - 1))))
        if 's' in self.resize_edge:
            rect.setBottom(int(min(self.original_size.y() - 1, max(img_pos.y(), rect.top() + 1))))
        if 'w' in self.resize_edge:
            rect.setLeft(int(max(0, min(img_pos.x(), rect.right() - 1))))
        if 'e' in self.resize_edge:
            rect.setRight(int(min(self.original_size.x() - 1, max(img_pos.x(), rect.left() + 1))))

        self.crop_rect = rect.normalized()
        self.rectChanged.emit(self.crop_rect)

    def reset_view(self):
        """重置视图"""
        self.fit_to_view()
        self.update()

    def clear_selection(self):
        """清除选区"""
        self.crop_rect = QRect()
        self.update()
        self.rectChanged.emit(QRect())


class MultiMethodSegmentationPreviewWindow(QMainWindow):
    """多算法分割结果预览窗口"""

    def __init__(self, results, image_name, parent=None):
        super().__init__(parent)
        self.results = results
        self.image_name = image_name
        self.parent_app = parent

        self.setWindowTitle(f"分割结果预览 - {os.path.basename(image_name)}")
        self.setWindowIcon(QIcon(r".\resources\assets\images\button\segmentation.png"))
        self.resize(900, 600)

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 顶部：简洁标题行（参考 Dataset）
        header_layout = QHBoxLayout()
        title_label = QLabel(f"图像: {os.path.basename(self.image_name)} | 成功方法数: {len(self.results)}")
        title_label.setStyleSheet("font-weight: bold; color: #34495e; font-size: 13px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # 滚动区域显示所有算法的结果
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: 1px solid #dcdfe6; border-radius: 5px;")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(15)

        # 为每个算法创建显示区域
        for method_name, result_data in self.results.items():
            method_group = self.create_method_result_group(method_name, result_data)
            scroll_layout.addWidget(method_group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # 按钮区域（简单扁平风格）
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        export_all_btn = QPushButton("导出所有结果")
        export_all_btn.setIcon(QIcon(r".\resources\assets\images\button\save.png"))
        export_all_btn.clicked.connect(self.export_all_results)
        btn_layout.addWidget(export_all_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def create_method_result_group(self, method_name, result_data):
        """为单个算法创建结果显示组"""
        group = QGroupBox(f"{method_name} (得分: {result_data['score']:.3f})")
        layout = QHBoxLayout(group)
        layout.setSpacing(10)

        # 原图
        orig_label = QLabel()
        orig_pixmap = self.cv2_to_pixmap(result_data['original'])
        orig_label.setPixmap(orig_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_label.setStyleSheet("border: 2px solid #3498db; border-radius: 5px; padding: 5px;")

        orig_frame = QFrame()
        orig_layout = QVBoxLayout(orig_frame)
        orig_layout.addWidget(QLabel("原图"))
        orig_layout.addWidget(orig_label)
        layout.addWidget(orig_frame)

        # 分割图
        seg_label = QLabel()
        seg_pixmap = self.cv2_to_pixmap(result_data['segmented'])
        seg_label.setPixmap(
            seg_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        seg_label.setStyleSheet("border: 2px solid #2ecc71; border-radius: 5px; padding: 5px;")

        seg_frame = QFrame()
        seg_layout = QVBoxLayout(seg_frame)
        seg_layout.addWidget(QLabel("分割图"))
        seg_layout.addWidget(seg_label)
        layout.addWidget(seg_frame)

        # 掩码
        mask_label = QLabel()
        if len(result_data['mask'].shape) == 2:
            mask_rgb = cv2.cvtColor(result_data['mask'], cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = result_data['mask']
        mask_pixmap = self.cv2_to_pixmap(mask_rgb)
        mask_label.setPixmap(mask_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 5px; padding: 5px;")

        mask_frame = QFrame()
        mask_layout = QVBoxLayout(mask_frame)
        mask_layout.addWidget(QLabel("分割掩码"))
        mask_layout.addWidget(mask_label)
        layout.addWidget(mask_frame)

        # 导出单个按钮
        export_btn = QPushButton(f"导出\n{method_name}")
        export_btn.clicked.connect(lambda checked, m=method_name, d=result_data: self.export_single_result(m, d))
        layout.addWidget(export_btn)

        return group

    def cv2_to_pixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        if len(cv_img.shape) == 2:
            height, width = cv_img.shape
            q_img = QImage(cv_img.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def export_single_result(self, method_name, result_data):
        """导出单个算法的结果"""
        if self.parent_app:
            self.parent_app.export_segmentation_high_dpi(
                result_data['original'],
                result_data['segmented'],
                result_data['mask'],
                method_name,
                self.image_name
            )

    # def export_all_results(self):
    #     """导出所有算法的结果"""
    #     if self.parent_app:
    #         for method_name, result_data in self.results.items():
    #             self.parent_app.export_segmentation_high_dpi(
    #                 result_data['original'],
    #                 result_data['segmented'],
    #                 result_data['mask'],
    #                 method_name,
    #                 self.image_name
    #             )

    def export_all_results(self):
        """导出所有算法的结果（带进度条）"""
        if not self.parent_app:
            return

        # 创建进度对话框
        progress_dialog = QProgressDialog("正在导出所有结果，请稍候...", "取消", 0, len(self.results), self)
        progress_dialog.setWindowTitle("批量导出")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.setValue(0)

        all_exported_files = []

        try:
            for idx, (method_name, result_data) in enumerate(self.results.items(), 1):
                progress_dialog.setLabelText(f"正在导出: {method_name} ({idx}/{len(self.results)})")
                progress_dialog.setValue(idx - 1)
                QApplication.processEvents()

                # 选择保存目录（第一次询问，后续使用相同目录）
                if idx == 1:
                    last_dir = self.parent_app.get_last_directory()
                    output_dir = QFileDialog.getExistingDirectory(
                        self,
                        "选择保存目录",
                        last_dir
                    )
                    if not output_dir:
                        progress_dialog.close()
                        return
                else:
                    # 使用第一次选择的目录
                    pass

                # 导出单个结果（需要修改export_segmentation_high_dpi支持指定目录）
                # 这里需要调用一个支持指定目录的版本
                exported = self.parent_app.export_single_segmentation_result(
                    result_data['original'],
                    result_data['segmented'],
                    result_data['mask'],
                    method_name,
                    self.image_name,
                    output_dir
                )

                if exported:
                    all_exported_files.extend(exported)

            progress_dialog.setValue(len(self.results))
            progress_dialog.close()

            # 显示统一的导出成功对话框
            if all_exported_files:
                self.parent_app.show_export_success_dialog(
                    all_exported_files,
                    title="批量导出成功",
                    message_prefix=f"已成功导出 {len(all_exported_files)} 个文件"
                )

        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(self, "导出错误", f"导出过程中发生错误:\n{str(e)}")


class SegmentationPreviewWindow(QMainWindow):
    """分割结果预览窗口，包含导出按钮"""

    def __init__(self, original, segmented, mask, method, image_name, parent=None):
        super().__init__(parent)
        self.original = original
        self.segmented = segmented
        self.mask = mask
        self.method = method
        self.image_name = image_name
        self.parent_app = parent

        self.setWindowTitle(f"分割结果预览 - {os.path.basename(image_name)}")
        self.setWindowIcon(QIcon(r".\resources\assets\images\button\segmentation.png"))
        self.resize(900, 600)

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 标题栏
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
        title_layout = QHBoxLayout(title_frame)

        title_label = QLabel(f"分割方法: {self.method}")
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        title_layout.addWidget(title_label)

        image_name_label = QLabel(f"图像: {os.path.basename(self.image_name)}")
        image_name_label.setStyleSheet("color: white; font-size: 12px;")
        title_layout.addStretch()
        title_layout.addWidget(image_name_label)

        layout.addWidget(title_frame)

        # 预览区域（使用滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: 1px solid #dcdfe6; border-radius: 5px;")

        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(10)

        # 原图预览
        orig_label = QLabel()
        orig_pixmap = self.cv2_to_pixmap(self.original)
        orig_label.setPixmap(orig_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_label.setStyleSheet("border: 2px solid #3498db; border-radius: 5px; padding: 5px;")

        orig_group = QGroupBox("原始图像")
        orig_layout = QVBoxLayout()
        orig_layout.addWidget(orig_label)
        orig_group.setLayout(orig_layout)

        # 分割图预览
        seg_label = QLabel()
        seg_pixmap = self.cv2_to_pixmap(self.segmented)
        seg_label.setPixmap(
            seg_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        seg_label.setStyleSheet("border: 2px solid #2ecc71; border-radius: 5px; padding: 5px;")

        seg_group = QGroupBox("分割结果")
        seg_layout = QVBoxLayout()
        seg_layout.addWidget(seg_label)
        seg_group.setLayout(seg_layout)

        # 掩码预览
        mask_label = QLabel()
        if len(self.mask.shape) == 2:
            mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = self.mask
        mask_pixmap = self.cv2_to_pixmap(mask_rgb)
        mask_label.setPixmap(mask_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 5px; padding: 5px;")

        mask_group = QGroupBox("分割掩码")
        mask_layout = QVBoxLayout()
        mask_layout.addWidget(mask_label)
        mask_group.setLayout(mask_layout)

        preview_layout.addWidget(orig_group)
        preview_layout.addWidget(seg_group)
        preview_layout.addWidget(mask_group)

        scroll_area.setWidget(preview_widget)
        layout.addWidget(scroll_area)

        # 按钮区域
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.addStretch()

        # 导出按钮
        export_btn = QPushButton("导出高DPI图片")
        export_btn.setIcon(QIcon(r".\resources\assets\images\button\save.png"))
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        export_btn.clicked.connect(self.export_images)
        btn_layout.addWidget(export_btn)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addWidget(btn_frame)

        # 显示当前DPI设置提示
        dpi_info = QLabel(
            f"当前DPI设置: {self.parent_app.dpi} DPI | 尺寸: {self.parent_app.fig_width}×{self.parent_app.fig_height}英寸")
        dpi_info.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 5px;")
        dpi_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(dpi_info)

    def cv2_to_pixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        if len(cv_img.shape) == 2:
            # 灰度图
            height, width = cv_img.shape
            q_img = QImage(cv_img.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            # 彩色图
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def export_images(self):
        """导出图片"""
        if self.parent_app:
            self.parent_app.export_segmentation_high_dpi(
                self.original, self.segmented, self.mask,
                self.method, self.image_name
            )


class EnhancedCropWindow(QMainWindow):
    """增强版裁剪窗口（完整实现）"""
    cropConfirmed = pyqtSignal(QRect)
    cropCancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像裁剪 - 选择目标区域")

        # 添加确认状态标志
        self.confirmed = False

        # 图像数据
        self.image = None
        self.qimage = None
        self.original_cv_image = None  # 存储原始OpenCV图像

        # 窗口尺寸相关
        self.screen_size = QApplication.primaryScreen().size()
        self.min_window_size = QSize(800, 600)  # 最小窗口大小
        self.max_window_size = QSize(1920, 1080)  # 最大窗口大小
        self.margin = 20  # 图片四周留白

        # 窗口尺寸比例因子
        self.width_factor = 0.8  # 窗口宽度占屏幕的比例
        self.height_factor = 0.8  # 窗口高度占屏幕的比例

        # 添加窗口图标
        try:
            self.setWindowIcon(QIcon(r".\resources\assets\images\button\crop.png"))
        except:
            pass

        # 初始化UI
        self.init_ui()

        # 设置窗口样式
        self.setup_window_style()

    def keyPressEvent(self, event):
        """键盘快捷键：主键盘 Enter 和小键盘 Enter 都触发确认裁剪，Esc 取消"""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.confirm_crop()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_crop()
        else:
            super().keyPressEvent(event)

    def init_ui(self):
        """初始化UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 1. 工具栏
        toolbar_layout = QHBoxLayout()

        # 缩放控制
        self.zoom_label = QLabel("缩放:")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(5, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)

        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setFixedWidth(50)

        self.fit_button = QPushButton("适应窗口")
        self.fit_button.clicked.connect(self.fit_to_window)
        self.fit_button.setFixedWidth(80)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_button.setFixedWidth(60)

        # 提示标签
        self.hint_label = QLabel("提示: 按住Ctrl/Shift拖动图像 | 滚轮缩放 | 右键清除选区")
        self.hint_label.setStyleSheet("color: #666; font-size: 12px; font-family: 'Microsoft YaHei';")

        toolbar_layout.addWidget(self.zoom_label)
        toolbar_layout.addWidget(self.zoom_slider)
        toolbar_layout.addWidget(self.zoom_value_label)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.hint_label)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.fit_button)
        toolbar_layout.addWidget(self.reset_button)

        main_layout.addLayout(toolbar_layout)

        # 2. 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dcdfe6;
                border-radius: 4px;
                background-color: #2c2c2c;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                border: none;
                background: #f1f1f1;
                width: 10px;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #c1c1c1;
                min-height: 20px;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: #a8a8a8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
                height: 0px;
                width: 0px;
            }
        """)

        # 创建裁剪控件
        self.crop_widget = CropRectWidget()

        # 设置裁剪控件的背景色
        self.crop_widget.setStyleSheet("background-color: #2c2c2c;")

        # 设置裁剪控件的大小策略
        self.crop_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 将裁剪控件放入滚动区域
        scroll_area.setWidget(self.crop_widget)

        main_layout.addWidget(scroll_area, stretch=1)

        # 3. 信息面板
        info_group = QGroupBox("选区信息")
        info_layout = QHBoxLayout()

        self.pos_label = QLabel("位置: (0, 0)")
        self.size_label = QLabel("尺寸: 0 × 0 px")
        self.area_label = QLabel("面积: 0 px²")

        for label in [self.pos_label, self.size_label, self.area_label]:
            label.setStyleSheet("font-family: 'Consolas'; font-size: 12px;")

        info_layout.addWidget(self.pos_label)
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.area_label)
        info_layout.addStretch()

        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # 4. 控制按钮
        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("清除选区")
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setFixedWidth(100)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: "Microsoft YaHei";
            }
            QPushButton:hover {
                background-color: #ffebee;
                border-color: #f44336;
                color: #f44336;
            }
        """)

        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_crop)
        self.cancel_button.setFixedWidth(100)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: "Microsoft YaHei";
            }
            QPushButton:hover {
                background-color: #fff3e0;
                border-color: #ff9800;
                color: #ff9800;
            }
        """)

        self.confirm_button = QPushButton("确认裁剪")
        self.confirm_button.clicked.connect(self.confirm_crop)
        self.confirm_button.setFixedWidth(100)
        self.confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #45a049;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: "Microsoft YaHei";
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
                border-color: #3d8b40;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)

        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.confirm_button)

        main_layout.addLayout(button_layout)

        # 连接信号
        self.crop_widget.rectChanged.connect(self.update_info)

        # 设置快捷键
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """设置快捷键"""
        self.cancel_button.setShortcut("Esc")
        self.confirm_button.setShortcut("Return")
        self.fit_button.setShortcut("Space")
        self.clear_button.setShortcut("Delete")

        # 添加最大化/最小化快捷键
        self.minimize_action = QAction(self)
        self.minimize_action.setShortcut("Ctrl+M")
        self.minimize_action.triggered.connect(self.showMinimized)
        self.addAction(self.minimize_action)

        self.maximize_action = QAction(self)
        self.maximize_action.setShortcut("Ctrl+Shift+M")
        self.maximize_action.triggered.connect(self.toggle_maximize)
        self.addAction(self.maximize_action)

    def toggle_maximize(self):
        """切换最大化状态"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def setup_window_style(self):
        """设置窗口样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6f7;
            }
            QGroupBox {
                border: 1px solid #dcdfe6;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #ffffff;
                font-family: "Microsoft YaHei";
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
            }
            QLabel {
                font-family: "Microsoft YaHei";
            }
        """)

    def calculate_optimal_window_size(self, img_width, img_height):
        """计算最优窗口大小"""
        # 工具栏、信息面板、按钮的大致高度
        toolbar_height = 40
        info_height = 60
        button_height = 50
        margins = self.margin * 2  # 上下左右都有边距

        # 计算显示图片所需的最小窗口尺寸
        min_content_width = img_width + margins
        min_content_height = img_height + margins

        # 加上其他UI元素的高度
        min_window_width = min_content_width
        min_window_height = min_content_height + toolbar_height + info_height + button_height

        # 限制最小窗口大小
        window_width = max(min_window_width, self.min_window_size.width())
        window_height = max(min_window_height, self.min_window_size.height())

        # 限制最大窗口大小（不超过屏幕的80%）
        max_allowed_width = int(self.screen_size.width() * self.width_factor)
        max_allowed_height = int(self.screen_size.height() * self.height_factor)

        window_width = min(window_width, max_allowed_width)
        window_height = min(window_height, max_allowed_height)

        return QSize(window_width, window_height)

    def set_image(self, cv_image):
        """设置OpenCV图像"""
        if cv_image is None:
            return

        try:
            self.original_cv_image = cv_image.copy()
            height, width, channel = cv_image.shape

            if height <= 0 or width <= 0:
                raise Exception("图像尺寸无效")

            bytes_per_line = 3 * width
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.qimage = QImage(img_rgb.data, width, height,
                                 bytes_per_line, QImage.Format.Format_RGB888).copy()

            if self.qimage.isNull():
                raise Exception("QImage创建失败")

            # 设置到裁剪控件
            self.crop_widget.set_image(self.qimage)
            self.crop_widget.original_size = QPoint(width, height)

            # 计算最优窗口大小
            optimal_size = self.calculate_optimal_window_size(width, height)

            # 调整窗口大小
            self.resize(optimal_size)

            # 居中显示
            self.center_on_screen()

            # 适应窗口显示
            self.fit_to_window()

            # 更新缩放滑块
            current_scale = int(self.crop_widget.scale_factor * 100)
            self.zoom_slider.setValue(current_scale)
            self.zoom_value_label.setText(f"{current_scale}%")

        except Exception as e:
            print(f"设置图像错误: {e}")
            QMessageBox.critical(self, "错误", f"设置图像失败: {str(e)}")

    def center_on_screen(self):
        """居中显示在屏幕上"""
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def on_zoom_changed(self, value):
        """缩放滑块改变"""
        self.crop_widget.scale_factor = value / 100.0
        self.crop_widget.offset = QPoint(0, 0)
        self.crop_widget.update()
        self.zoom_value_label.setText(f"{value}%")

    def fit_to_window(self):
        """适应窗口大小"""
        self.crop_widget.fit_to_view()
        current_scale = int(self.crop_widget.scale_factor * 100)
        self.zoom_slider.setValue(current_scale)
        self.zoom_value_label.setText(f"{current_scale}%")
        self.crop_widget.update()

    def reset_view(self):
        """重置视图"""
        self.crop_widget.reset_view()
        current_scale = int(self.crop_widget.scale_factor * 100)
        self.zoom_slider.setValue(current_scale)
        self.zoom_value_label.setText(f"{current_scale}%")

    def clear_selection(self):
        """清除选区"""
        self.crop_widget.clear_selection()
        self.update_info(QRect())

    def update_info(self, rect):
        """更新信息显示"""
        if rect.isNull():
            self.pos_label.setText("位置: (0, 0)")
            self.size_label.setText("尺寸: 0 × 0 px")
            self.area_label.setText("面积: 0 px²")
        else:
            self.pos_label.setText(f"位置: ({rect.x()}, {rect.y()})")
            self.size_label.setText(f"尺寸: {rect.width()} × {rect.height()} px")
            self.area_label.setText(f"面积: {rect.width() * rect.height()} px²")

    def cancel_crop(self):
        """取消裁剪"""
        self.cropCancelled.emit()
        self.confirmed = True
        self.close()

    def confirm_crop(self):
        """确认裁剪"""
        rect = self.crop_widget.crop_rect
        if rect.isNull():
            reply = QMessageBox.question(
                self, "确认",
                "没有选择任何区域，是否取消裁剪？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.cropCancelled.emit()
                self.confirmed = True
                self.close()
            return

        self.cropConfirmed.emit(rect)
        self.confirmed = True
        self.close()

    def closeEvent(self, event):
        """关闭事件"""
        if self.confirmed:
            event.accept()
            return

        if not self.crop_widget.crop_rect.isNull():
            reply = QMessageBox.question(
                self, "确认",
                "您有未保存的选区，是否取消裁剪并关闭窗口？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.cropCancelled.emit()
                event.accept()
            else:
                event.ignore()
        else:
            self.cropCancelled.emit()
            event.accept()


class BatchSegmentationWorker(QThread):
    """批量分割工作线程"""
    progress_updated = pyqtSignal(int, str)  # 进度, 当前文件
    file_completed = pyqtSignal(str, dict)  # 文件路径, 结果
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, image_files, output_dir, selected_methods, parent_app):
        super().__init__()
        self.image_files = image_files
        self.output_dir = output_dir
        self.selected_methods = selected_methods
        self.parent_app = parent_app
        self.is_running = True

    def run(self):
        """执行批量分割"""
        try:
            segmenter = RockSegmenter(log_callback=self.log_message)
            total = len(self.image_files)

            for idx, image_path in enumerate(self.image_files):
                if not self.is_running:
                    break

                self.progress_updated.emit(int((idx / total) * 100), os.path.basename(image_path))

                try:
                    # 读取图像
                    img = cv2_imread(image_path)
                    if img is None:
                        self.log_message(f"Failed to load: {image_path}")
                        continue

                    # 执行分割
                    results = segmenter.segment_by_methods(
                        img,
                        os.path.basename(image_path),
                        self.selected_methods,
                        log_callback=self.log_message
                    )

                    if results:
                        # 保存结果
                        self.save_results(image_path, results)
                        self.file_completed.emit(image_path, results)

                except Exception as e:
                    self.error_occurred.emit(f"Error processing {image_path}: {str(e)}")
                    self.log_message(f"Error: {str(e)}")

            self.progress_updated.emit(100, "Completed")
            self.finished_signal.emit()

        except Exception as e:
            self.error_occurred.emit(f"Batch segmentation failed: {str(e)}")

    def log_message(self, message):
        """记录日志"""
        if self.parent_app:
            self.parent_app.log(message)

    def save_results(self, image_path, results):
        """保存分割结果"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        dataset_name = os.path.basename(os.path.dirname(image_path))

        # 为每个方法创建输出目录
        for method_name, result_data in results.items():
            method_clean = method_name.replace(' ', '_').replace('/', '_')

            # 创建样本文件夹: 日期_样本名
            
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            sample_dir = os.path.join(
                self.output_dir,
                f"{dataset_name}_分割图_{method_clean}",
                f"{date_str}_{base_name}"
            )
            os.makedirs(sample_dir, exist_ok=True)

            # 保存原图、分割图、掩码
            original = result_data['original']
            segmented = result_data['segmented']
            mask = result_data['mask']

            cv2_imwrite(os.path.join(sample_dir, "original.png"), original)
            cv2_imwrite(os.path.join(sample_dir, "segmented.png"), segmented)
            cv2_imwrite(os.path.join(sample_dir, "mask.png"), mask)

            # 保存对比图（使用HighDPIExporter）
            try:
                exporter = HighDPIExporter(show_title=False)
                dpi = self.parent_app.dpi if self.parent_app else 300
                width = self.parent_app.fig_width if self.parent_app else 15
                height = self.parent_app.fig_height if self.parent_app else 5

                # 样式1
                fig1 = exporter.create_comparison_style1(
                    original, segmented, mask, method_name, dpi, width, height, show_title=False
                )
                fig1.savefig(os.path.join(sample_dir, "comparison_style1.png"),
                             dpi=dpi, bbox_inches='tight', format='PNG')
                plt.close(fig1)

                # 样式2
                fig2 = exporter.create_comparison_style2(
                    original, segmented, method_name, image_path, dpi, width, height, show_title=False
                )
                fig2.savefig(os.path.join(sample_dir, "comparison_style2.png"),
                             dpi=dpi, bbox_inches='tight', format='PNG')
                plt.close(fig2)
            except Exception as e:
                self.log_message(f"Error saving comparison images: {str(e)}")

    def stop(self):
        """停止处理"""
        self.is_running = False


class BatchSegmentationDialog(QDialog):
    """批量分割对话框"""

    def __init__(self, image_files, dataset_path, parent=None):
        super().__init__(parent)
        self.image_files = image_files
        self.dataset_path = dataset_path
        self.parent_app = parent
        self.worker = None

        self.setWindowTitle("Batch Segmentation")
        self.setModal(True)
        self.resize(600, 500)

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 标题
        title_label = QLabel("Batch Image Segmentation")
        title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            color: #2c3e50;
            padding: 10px;
        """)
        layout.addWidget(title_label)

        # 信息显示
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f"Dataset: {os.path.basename(self.dataset_path)}"))
        info_layout.addWidget(QLabel(f"Total Images: {len(self.image_files)}"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 算法选择
        method_group = QGroupBox("Select Segmentation Methods")
        method_layout = QVBoxLayout()

        # 使用现有的SegmentationMethodDialog的逻辑
        self.method_checks = {}
        methods = [
            ('GrabCut智能分割', '使用GrabCut算法，适合复杂背景'),
            ('颜色阈值分割', '基于颜色范围，适合颜色明显的岩石'),
            ('边缘检测分割', '基于边缘检测，适合边界清晰的岩石'),
            ('自适应阈值分割', '自适应阈值，适合光照不均的图像'),
            ('分水岭分割', '分水岭算法，适合重叠区域'),
            ('K-means聚类分割', 'K-means聚类，适合颜色分布明显的图像')
        ]

        for method_name, description in methods:
            check = QCheckBox(method_name)
            check.setToolTip(description)
            check.setChecked(True)  # 默认全选
            method_layout.addWidget(check)
            self.method_checks[method_name] = check

        # 操作按钮
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: [c.setChecked(True) for c in self.method_checks.values()])
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(lambda: [c.setChecked(False) for c in self.method_checks.values()])
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        btn_layout.addStretch()
        method_layout.addLayout(btn_layout)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # 输出目录选择
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout()

        output_layout_widget = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        output_layout_widget.addWidget(self.output_dir_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout_widget.addWidget(browse_btn)
        output_layout.addLayout(output_layout_widget)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # 进度显示
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        layout.addWidget(self.status_label)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.start_btn.clicked.connect(self.start_segmentation)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def browse_output_dir(self):
        """浏览输出目录"""
        last_dir = self.output_dir_edit.text() if self.output_dir_edit.text() else self.dataset_path
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            last_dir
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def start_segmentation(self):
        """开始分割"""
        # 检查选中的方法
        selected_methods = [name for name, check in self.method_checks.items() if check.isChecked()]
        if not selected_methods:
            QMessageBox.warning(self, "No Method Selected", "Please select at least one segmentation method.")
            return

        # 检查输出目录
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 禁用开始按钮
        self.start_btn.setEnabled(False)
        self.status_label.setText("Processing...")

        # 创建工作线程
        self.worker = BatchSegmentationWorker(
            self.image_files,
            output_dir,
            selected_methods,
            self.parent_app
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)

        self.worker.start()

    def update_progress(self, value, filename):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Processing: {filename}")
        QApplication.processEvents()

    def on_file_completed(self, file_path, results):
        """文件处理完成"""
        if self.parent_app:
            self.parent_app.log(f"Completed: {os.path.basename(file_path)}")

    def on_finished(self):
        """处理完成"""
        self.progress_bar.setValue(100)
        self.status_label.setText("Completed!")
        self.start_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "Batch Segmentation Complete",
            f"Successfully processed {len(self.image_files)} images."
        )

    def on_error(self, error_msg):
        """处理错误"""
        if self.parent_app:
            self.parent_app.log(f"Error: {error_msg}")
        QMessageBox.warning(self, "Error", error_msg)

    def closeEvent(self, event):
        """关闭事件"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Segmentation in Progress",
                "Segmentation is still running. Do you want to stop it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait()
            else:
                event.ignore()
                return
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    matplotlib.use('QtAgg')
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    window = ImageEditorApp()
    window.show()
    sys.exit(app.exec())
