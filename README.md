# PCB Defect Detection with YOLOv3

This project implements a YOLOv3-based model to detect defects in printed circuit boards (PCBs). It supports real-time detection via webcam or video stream, as well as offline detection for uploaded images. The project includes pre-trained weights for easy deployment.

## Features
- **Real-time Detection:** Use a webcam or video stream for defect detection.
- **Image Detection:** Detect defects in uploaded images.
- **Pre-trained Model:** Includes trained YOLOv3 weights for immediate use.
- **OpenCV Integration:** Provides tools for integration with OpenCV and Raspberry Pi.

## Project Structure
```
PCB_defect_detection/
├── README.md                # Documentation
├── requirements.txt         # Python dependencies
├── detect_uploaded_files.py # Script for detection
├── realtime_PCB_defect_detection.py # Real-time detection script
├── yolov3_custom.cfg        # YOLO configuration file
├── yolov3_custom.weights    # Trained weights file
├── obj.names                # Class names file
├── obj.data                 # YOLO data file
├── data/                    # Directory for test images and videos
│   ├── test.jpg             # Example test image


```

## Installation

### Prerequisites
- Python 3.7+
- OpenCV
- NumPy

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure OpenCV and NumPy are installed:
   ```bash
   pip install opencv-python-headless numpy
   ```
   Download trained weight file for the model
   https://drive.google.com/drive/folders/1tK_NyqkUBpj7k4qPWGfxXpe2s63CKPhk?usp=drive_link
## Usage

### Download Dataset
Download the PCB defect dataset from Kaggle:
1. Ensure you have the Kaggle API set up. Refer to the [Kaggle API Documentation](https://www.kaggle.com/docs/api) for setup instructions.
2. Use the following command to download the dataset:
   ```bash
   kaggle datasets download -d akhatova/pcb-defects -p data/
   ```
3. Extract the dataset:
   ```bash
   unzip data/pcb-defects.zip -d data/
   ```

**About the Dataset**
- **Context:**
  The Open Lab on Human Robot Interaction of Peking University has released the PCB defect dataset. It contains 6 types of defects made using Photoshop: missing hole, mouse bite, open circuit, short, spur, and spurious copper.

- **Content:**
  The dataset contains 1386 images for defect detection, classification, and registration tasks.

- **Acknowledgements:**
  PCB dataset was downloaded using a public link from [Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB).

  Authors of the dataset: Huang, Weibo, and Peng Wei. "A PCB dataset for defects detection and classification." arXiv preprint arXiv:1901.08204 (2019). [Dataset on Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects)

### Image Detection
Use the `detect_uploaded_files.py` script to detect defects in an uploaded image.
```bash
python detect_uploaded_files.py
```

### Real-Time Detection
Use the `realtime_PCB_defect_detection.py` script for real-time defect detection via webcam or video stream.
```bash
python realtime_PCB_defect_detection.py
```

Ensure the trained weights (`yolov3_custom.weights`) and configuration files (`yolov3_custom.cfg`) are in the project directory.

### Example Command
To test the model on an image:
```bash
python detect_uploaded_files.py
```

## Files Included
- `yolov3_custom.weights`: Trained YOLOv3 weights.
- `yolov3_custom.cfg`: YOLOv3 configuration file.
- `obj.names`: Class names for detection.
- `obj.data`: YOLO data file specifying classes and paths.
- `detect_uploaded_files.py`: Script for detecting defects in uploaded images.
- `realtime_PCB_defect_detection.py`: Script for real-time defect detection.

## Trained Model
The trained YOLOv3 model (`yolov3_custom.weights`) is included in this repository. This model was trained with nearly zero loss, providing high accuracy for defect detection.

## Example Workflow
1. Run the real-time detection script:
   ```bash
   python realtime_PCB_defect_detection.py
   ```
2. Connect a webcam or video stream to observe live predictions.
3. Use the image detection script to process individual test images.

## Integration with Raspberry Pi
To use this project on a Raspberry Pi:
1. Transfer the repository and required files (`yolov3_custom.weights`, `yolov3_custom.cfg`, `obj.names`) to the Raspberry Pi.
2. Install Python dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-opencv
   pip install numpy Pillow
   ```
3. Run the real-time detection script on the Raspberry Pi.

## Dependencies
The project relies on the following libraries:
- OpenCV
- NumPy
- Pillow (for GUI support)

Dependencies are listed in `requirements.txt`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- YOLOv3 implementation: [AlexeyAB's Darknet](https://github.com/AlexeyAB/darknet)
- OpenCV for real-time computer vision support.

