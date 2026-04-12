# RoCAS #

![Alt text](./resources/assets/images/readme/RoCAS%20GUI.png)

## 项目介绍 ##

### 这个仓库是做什么的？

RoCAS (Rock Color Analysis System) 是一个综合的岩石图像分析工具，包括图像分割、颜色识别和模型训练。它提供了直观的 GUI 界面，让用户能够高效地分析岩石图像。

#### 如何设置？

请查看教程视频：[教程视频链接] 或按照以下说明操作：

1. 克隆仓库

   ```bash
   git clone https://github.com/yourusername/RoCAS.git
   cd RoCAS
   ```

2. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 运行应用程序

   ```bash
   python RoCAS.py
   ```

使用方法

1) 启动应用程序：
   - 对于 Windows 用户：双击 RoCAS.py 或从终端运行 `python RoCAS.py`
   - 对于其他用户：从终端运行 `python RoCAS.py`

2) 应用程序打开后，点击 "Load Image" 选择要分析的岩石图像。为了测试，您可以选择 `samples/` 目录中的样本图像。

3) 根据需要使用图像预处理工具（翻转、裁剪、旋转）来准备图像进行分析。

4) 点击 "Image Segmentation" 打开分割对话框。选择所需的分割算法，然后点击 "OK" 运行分割。

5) 分割完成后，将出现一个预览窗口，显示原始图像、分割结果和分割掩码。

6) 点击 "Color Analysis" 对分割后的岩石进行颜色识别。颜色分析结果将显示在结果面板中。

7) 点击 "Export High DPI Images" 将分析结果导出为高分辨率图像，用于 publication。

8) 要训练自定义 U-Net 分割模型，点击 "Model Training" 并选择 "U-Net Segmentation"。按照说明选择训练数据和参数。

9) 要训练自定义颜色识别模型，点击 "Model Training" 并选择 "Color Recognition"。按照说明选择训练数据和参数。

10) 要使用U-net模型，需要先下载我们已经训练好的特定数据模型结果：https://github.com/tiamatovo/RoCAS/releases ， 然后从 /samples 文件夹中加载实例样本图，可快速上手查看效果。

输出文件

- **分割结果**：保存在输入图像的同一目录中，包括原始图像、分割结果、分割掩码和黑色背景的目标。
- **训练模型**：保存在 `models/` 目录中，包括 U-Net 分割模型和颜色识别模型。
- **分析日志**：保存在 `logs/` 目录中，包括分析过程的详细日志。

联系方式

Contact: XiaoyingTian.Xinjiang University

Email: dachang0220@163.com

许可证

Copyright (C) 2026 Xiaoying, Tian

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see `https://www.gnu.org/licenses/`.