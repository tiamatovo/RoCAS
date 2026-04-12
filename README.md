# RoCAS #

![Alt text](./resources/assets/images/readme/RoCAS%20GUI.png)

## What is this repository for?

RoCAS (Rock Color Analysis System) is a comprehensive tool for rock image analysis, including image segmentation, color recognition, and model training. It provides an intuitive GUI interface for users to analyze rock images efficiently.

#### How do I get set up?

See the tutorial video: [Link to tutorial video] or follow the instructions:

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/RoCAS.git
   cd RoCAS
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application

   ```bash
   python RoCAS.py
   ```

Usage

1) Launch the application:
   - For Windows users: double click on RoCAS.py or run `python RoCAS.py` from a terminal
   - For other users: run `python RoCAS.py` from a terminal

2) After the application opens, click "Load Image" to select a rock image for analysis. For testing purposes, choose sample images from the `samples/` directory.

3) Use the image preprocessing tools (flip, crop, rotate) if needed to prepare the image for analysis.

4) Click "Image Segmentation" to open the segmentation dialog. Select the desired segmentation algorithms and click "OK" to run segmentation.

5) After segmentation is complete, a preview window will appear showing the original image, segmentation result, and segmentation mask.

6) Click "Color Analysis" to perform color recognition on the segmented rock. The color analysis results will be displayed in the results panel.

7) Click "Export High DPI Images" to export the analysis results as high-resolution images for publication.

8) To train a custom U-Net segmentation model, click "Model Training" and select "U-Net Segmentation". Follow the instructions to select training data and parameters.

9) To train a custom color recognition model, click "Model Training" and select "Color Recognition". Follow the instructions to select training data and parameters.

OUTPUT FILES

- **Segmentation results**: Saved in the same directory as the input image, including original image, segmentation result, segmentation mask, and subject with black background.
- **Trained models**: Saved in the `models/` directory, including U-Net segmentation models and color recognition models.
- **Analysis logs**: Saved in the `logs/` directory, including detailed logs of the analysis process.

Who do I talk to?

Contact: Xiaoying,Tian. Xinjiang University

Email: dachang0220@163.com

License

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