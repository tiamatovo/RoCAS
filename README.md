# RoCAS #

![Alt text](./resources/assets/images/readme/RoCAS%20GUI.png)

### What is this repository for? ###

RoCAS (Rock Color Analysis System) is an open‑source software for standardized rock color identification from digital images. It integrates a pre‑trained U‑Net model for automatic rock segmentation from complex backgrounds, and a weighted multi‑feature fusion algorithm (cosine similarity, Euclidean distance, channel divergence, and luminance equalization) to match the segmented rock to the nearest standard color in the SY/T 5751—2012 digital benchmark (112 colors, aligned with the Munsell Rock Color Chart). The system provides a user‑friendly PyQt6 GUI and is designed for geological field surveys, core description, and digital logging.

### How do I get set up? ###

**Hardware requirements:**  

- General‑purpose computer (Windows 10/11, Linux, or macOS)  
- NVIDIA GPU (RTX 2060 or higher) is recommended for faster U‑Net inference, but CPU mode is also supported.

**Software requirements:**  

- Python 3.10  
- PyTorch, OpenCV‑Python, NumPy, Matplotlib, PyQt6

**Installation (Windows 64‑bit recommended):**  

1. Download or clone this repository:  
   `git clone https://github.com/tiamatovo/RoCAS.git`  
2. (Optional but recommended) Create a conda environment:  
   `conda create -n rocas python=3.10`  
   `conda activate rocas`  
3. Install dependencies:  
   `pip install -r requirements.txt`  

### Usage ###

1) Launch the GUI:

   - **Windows executable**: double‑click `RoCAS.exe`  
   - **Python source**: open a terminal, navigate to the repository folder, and run:  
     `python rocas_gui.py`

2) The main window opens (see Fig. 1 in our paper). Click **“Load Image”** to select a rock sample image (supported formats: .jpg, .png, .tif).

3) **Segmentation (background removal):**  
   - Choose the segmentation method:  
     * `U‑Net (deep learning)` – recommended for complex backgrounds (field outcrops, heterogeneous lighting).  
     * `GrabCut` / `Thresholding` – for simple, uniform backgrounds.  
   - Click **“Run Segmentation”**. The rock mask will appear in the right panel.  
   - Use the **“Flip”** button for data augmentation if needed.

4) **Color extraction and matching:**  
   - Set the **grid size** (default = 7 pixels).  
   - Click **“Extract Colors”**. The system divides the rock area into a grid, computes the dominant color of each cell, and matches it to the 112‑color SY/T 5751—2012 database using the weighted multi‑feature fusion algorithm.

5) **View and save results:**  
   - The main color (most frequent Munsell code) is shown on the GUI.  
   - A **color abundance map** (pie chart or bar plot) is displayed – click **“Save Results”** to export:
     - A CSV table with grid‑level Munsell codes and RGB values.
     - A PDF report containing the segmented rock, the color‑coded grid, and the abundance statistics.

6) To process another image, click **“New Image”** and repeat from step 2.

**Keyboard shortcuts (inside the image viewer):**  

- `Ctrl+Q` – quit the application  
- `Ctrl+S` – save current results

**Output files (saved in the same folder as the input image, or a user‑defined folder):**  

- `<image_name>_RoCAS_report.pdf` – complete report with segmentation mask, color grid, and abundance statistics.  
- `<image_name>_color_table.csv` – per‑grid Munsell code, RGB values, and matching scores.  
- `<image_name>_mask.png` – binary mask of the extracted rock region.  
- `<image_name>_color_grid.png` – visualisation of the colour‑mapped grid over the rock.

### Who do I talk to? ###

**Maintainer:**  
Zhaohui Zhang (corresponding author) – School of Geology and Mining Engineering, Xinjiang University  
*zhangzhaohui <at> xju.edu.cn*  

**Lead developer:**  
Xiaoying Tian – same affiliation  
*tianxiaoying <at> xju.edu.cn*

For questions, bug reports, or feature requests, please open an **Issue** on this GitHub repository or contact the maintainer by email.

### License ###

Copyright (C) 2026 Xiaoying Tian,

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
