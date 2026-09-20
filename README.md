# Tom Detection Using YOLOv8

A deep learning project for detecting the character **Tom** from *Tom & Jerry* video frames using the **YOLOv8** object detection model.

---

## 📌 Project Overview

The goal of this project is to develop a deep learning-based object detection system capable of identifying the character **Tom** in frames extracted from a *Tom & Jerry* video.

The project uses **YOLOv8n** as the object detection model.

The system receives video frames as input and predicts the location of Tom using a bounding box.

The complete workflow includes:

* Extracting frames from a video
* Preparing the dataset
* Annotating images
* Converting annotations to YOLO format
* Splitting the dataset into training and validation sets
* Configuring the dataset using `data.yaml`
* Training a YOLOv8 model
* Evaluating the trained model
* Visualizing training metrics
* Generating predictions on images

---

## 🎯 Project Objectives

The main objectives of this project are:

* Build a custom object detection dataset.
* Detect the character Tom in video frames.
* Train a YOLOv8 model for a single object class.
* Convert image annotations into YOLO format.
* Evaluate the model using standard object detection metrics.
* Visualize the model's training performance.
* Test the trained model on unseen images.

---

## 🧩 Problem Statement

In an animated video, the character Tom can appear in different positions, poses, scales, and scenes.

The objective is to train a model that can locate Tom in different video frames.

For each detected instance, the model should determine:

* Whether Tom is present in the image.
* The location of Tom.
* The bounding box surrounding Tom.

The system is therefore formulated as a **single-class object detection problem**.

---

## 🛠️ Technologies Used

The project was implemented using the following technologies:

* Python
* YOLOv8
* Ultralytics
* OpenCV
* Pandas
* Matplotlib
* JSON
* Jupyter Notebook
* Git / GitHub

---

## 🤖 Model

The project uses:

**YOLOv8n**

YOLO stands for **You Only Look Once** and is a family of real-time object detection models.

The `n` version represents the nano model, which provides a relatively lightweight architecture suitable for faster training and inference.

The model was initialized using:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

The pretrained YOLOv8n weights were then fine-tuned on the custom Tom detection dataset.

---

## 📁 Project Structure

The main repository contains:

```text
Face-Recognition-YOLOv8/
│
├── Tom_Detection_YOLOv8.ipynb
├── data.yaml
├── .gitignore
└── README.md
```

The main Jupyter Notebook contains the complete implementation of the project.

---

## 🎬 Input Video

The original input of the project is a *Tom & Jerry* video.

The video was converted into individual image frames before training.

The extracted frames were then used to create the object detection dataset.

The project used approximately **200 frames** for the dataset.

---

# 📊 Dataset Preparation

## 1. Dataset Directory Structure

The dataset was organized into separate image and label directories.

The main structure was:

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
└── val/
    ├── images/
    └── labels/
```

The `images` directories contain the extracted video frames.

The `labels` directories contain the corresponding YOLO annotation files.

---

## 2. Frame Extraction

The first step was extracting frames from the original video.

OpenCV was used to read the video:

```python
import cv2

cap = cv2.VideoCapture(video_path)
```

The total number of frames was obtained using:

```python
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

A target of approximately 200 frames was selected.

The frame sampling interval was calculated using:

```python
step = max(total_frames // target_frames, 1)
```

Selected frames were saved as JPEG images.

Example:

```text
img_0.jpg
img_1.jpg
img_2.jpg
...
```

---

# 🏷️ Data Annotation

## 3. Image Annotation

The extracted images were annotated using **AnyLabeling**.

Only one object class was used:

```text
Tom
```

A bounding box was drawn around Tom in each annotated image.

Each bounding box represents the location of Tom in the image.

---

## 4. Annotation Format

The annotations were initially saved as JSON files.

The JSON annotation files contained information such as:

* Image width
* Image height
* Object label
* Bounding box coordinates

The annotation format was then converted to the YOLO format required by YOLOv8.

---

# 🔄 JSON to YOLO Conversion

## 5. Class Definition

The project contains only one class:

```python
class_name = "tom"
class_id = 0
```

Therefore:

```text
0 = Tom
```

---

## 6. YOLO Annotation Format

YOLO annotations use the following format:

```text
class_id x_center y_center width height
```

The bounding box coordinates are normalized between 0 and 1.

For example:

```text
0 0.512 0.438 0.214 0.391
```

The values represent:

* Class ID
* Bounding box center X
* Bounding box center Y
* Bounding box width
* Bounding box height

---

## 7. Coordinate Conversion

The bounding box coordinates from the JSON files were converted into normalized YOLO coordinates.

The center coordinates were calculated as:

```python
x_center = ((x1 + x2) / 2) / w
y_center = ((y1 + y2) / 2) / h
```

The width and height were calculated as:

```python
bw = abs(x2 - x1) / w
bh = abs(y2 - y1) / h
```

The resulting annotations were saved as `.txt` files.

Each image therefore has a corresponding label file.

Example:

```text
img_25.jpg
img_25.txt
```

---

# 📂 Train / Validation Split

## 8. Dataset Splitting

The dataset was divided into training and validation sets.

Approximately:

```text
80% → Training
20% → Validation
```

The images were shuffled before splitting.

This helps the model learn from one portion of the dataset and allows its performance to be evaluated on images that were not used for training.

---

# ⚙️ Dataset Configuration

## 9. data.yaml

The dataset configuration is provided through:

```text
data.yaml
```

The configuration defines the dataset paths and class information required by YOLOv8.

The project contains one object class:

```text
Tom
```

The YAML file allows the YOLO training process to identify:

* Training images
* Validation images
* Number of classes
* Class names

---

# 🏋️ Model Training

## 10. Training Configuration

The YOLOv8n model was trained using the following configuration:

| Parameter         |     Value |
| ----------------- | --------: |
| Model             |   YOLOv8n |
| Epochs            |        50 |
| Image Size        | 640 × 640 |
| Batch Size        |         8 |
| Number of Classes |         1 |
| Class             |       Tom |

The training was performed using:

```python
model.train(
    data=r"C:/transformer_project/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs/detect",
    name="tom_jerry_final_run",
    exist_ok=True
)
```

---

# 📈 Evaluation

## 11. Evaluation Metrics

After training, the model was evaluated using several standard object detection metrics.

The main metrics analyzed were:

* Precision
* Recall
* mAP50
* mAP50-95
* Training Loss
* Validation Loss

---

## 12. Precision

Precision measures the proportion of predicted detections that were correct.

The final precision obtained in this project was:

```text
Precision = 0.9237
```

This means that the model achieved a high proportion of correct positive predictions on the evaluated data.

---

## 13. Recall

Recall measures how many of the actual objects were successfully detected.

The final recall was:

```text
Recall = 0.8947
```

This indicates that the model was able to detect most of the Tom instances in the validation data.

---

## 14. mAP50

mAP50 represents mean Average Precision at an IoU threshold of 0.50.

The final result was:

```text
mAP50 = 0.9066
```

This indicates strong detection performance under the IoU = 0.50 criterion.

---

## 15. mAP50-95

The project also reports mAP50-95.

The final value was:

```text
mAP50-95 = 0.4790
```

This metric is more demanding because it evaluates performance across multiple IoU thresholds.

The difference between mAP50 and mAP50-95 indicates that localization performance becomes more challenging under stricter overlap requirements.

---

# 📉 Loss Analysis

## 16. Training and Validation Loss

The training process records the loss values for each epoch.

The project visualizes:

```text
Train Box Loss
Validation Box Loss
```

The final values after 50 epochs were:

```text
Final Train Box Loss = 0.8053
Final Validation Box Loss = 1.8320
```

These values were obtained from the YOLO training results.

---

# 📊 Precision and Recall Curve

The project also visualizes Precision and Recall across training epochs.

The curves are generated from:

```text
results.csv
```

The following metrics are plotted:

```text
metrics/precision(B)
metrics/recall(B)
```

This visualization helps observe how the model's detection performance changes during training.

---

# 📊 mAP Curve

The mAP50 curve was generated from the training results.

The plotted metric is:

```text
metrics/mAP50(B)
```

The curve shows the change in detection performance throughout the training process.

---

# 🧮 Confusion Matrix

A confusion matrix was generated after model validation.

The obtained results included:

```text
Tom correctly detected: 18
Tom incorrectly classified as Background: 2
Background incorrectly classified as Tom: 1
```

Because the project contains only one object class, the confusion matrix mainly reflects the relationship between the Tom class and background detections.

---

# 🖼️ Prediction Results

The trained model was also used to generate predictions on validation images.

The predicted images contain bounding boxes around detected instances of Tom.

Several prediction examples were visualized using Matplotlib.

The notebook includes examples of predictions from validation images such as:

```text
img_11.jpg
img_12.jpg
img_30.jpg
img_41.jpg
```

These visualizations provide a qualitative evaluation of the trained detector.

---

# 📋 Final Results

The final results of the trained YOLOv8n model are summarized below:

| Metric              |  Value |
| ------------------- | -----: |
| Epochs              |     50 |
| Train Box Loss      | 0.8053 |
| Validation Box Loss | 1.8320 |
| Precision           | 0.9237 |
| Recall              | 0.8947 |
| mAP50               | 0.9066 |
| mAP50-95            | 0.4790 |

---

# 🔍 Results Interpretation

The obtained Precision and Recall values show that the model was able to detect the target class effectively on the evaluated dataset.

The mAP50 value of:

```text
0.9066
```

shows strong performance under the IoU = 0.50 evaluation criterion.

However, the lower mAP50-95 value:

```text
0.4790
```

shows that localization becomes more difficult when stricter IoU thresholds are applied.

This difference suggests that there is still room for improving bounding box localization and generalization.

---

# ⚠️ Limitations

The project has several limitations.

### Limited Dataset Size

The dataset contains approximately 200 extracted frames.

A larger and more diverse dataset could improve generalization.

### Similar Video Frames

Because the images were extracted from the same video, consecutive frames can have similar visual characteristics.

Therefore, the dataset may not represent all possible appearances of Tom.

### Single Character

Only one class was trained:

```text
Tom
```

The model was not trained to distinguish between multiple characters.

### Animated Content

The model was trained specifically on an animated character.

Its performance on real human faces or unrelated objects should not be assumed from these results.

### Bounding Box Localization

The difference between mAP50 and mAP50-95 suggests that more precise localization could be investigated.

---

# 🚀 Possible Improvements

Several improvements could be considered in future versions:

* Increase the number of training images.
* Use frames from different parts of the video.
* Include multiple videos and scenes.
* Increase variation in character scale and pose.
* Improve annotation quality.
* Apply data augmentation.
* Experiment with other YOLOv8 model sizes.
* Tune training hyperparameters.
* Train for additional epochs when appropriate.
* Evaluate the model on a completely separate test dataset.
* Improve bounding box localization.

---

# 💻 How to Run

## 1. Install Dependencies

Install the required Python packages:

```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
```

---

## 2. Open the Notebook

Open:

```text
Tom_Detection_YOLOv8.ipynb
```

using Jupyter Notebook or JupyterLab.

---

## 3. Prepare the Dataset

Create the required dataset structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
│
└── val/
    ├── images/
    └── labels/
```

Place the annotated images and YOLO label files in the corresponding directories.

---

## 4. Configure data.yaml

Update the dataset paths according to the location of the dataset on your computer.

The configuration should define the training and validation image directories and the single class:

```text
Tom
```

---

## 5. Train the Model

Run the training section of the notebook.

The model will train using:

```text
YOLOv8n
50 epochs
640 image size
batch size 8
```

---

## 6. Validate the Model

After training, run the validation section.

The validation process generates evaluation metrics and visualization files.

---

# 📦 Output Files

The YOLO training process generates several output files and visualizations, including:

```text
results.csv
confusion_matrix.png
results.png
```

and other training-related outputs.

These files can be used for further analysis of the model.

---

# 📚 Project Workflow

The complete workflow can be summarized as:

```text
Tom & Jerry Video
        ↓
Frame Extraction
        ↓
Image Dataset
        ↓
Image Annotation
        ↓
JSON Annotations
        ↓
JSON → YOLO TXT
        ↓
Train / Validation Split
        ↓
data.yaml
        ↓
YOLOv8n Training
        ↓
Model Validation
        ↓
Precision / Recall
        ↓
mAP50 / mAP50-95
        ↓
Prediction Visualization
```

---

# 🧑‍💻 Project Type

This project was developed as a **Deep Learning / Object Detection course project**.

The main focus was on applying a modern object detection model to a custom video-based dataset.

---

# 📌 Key Takeaways

This project demonstrates a complete workflow for building a custom YOLOv8 object detection system.

The main stages include:

1. Video processing
2. Frame extraction
3. Dataset creation
4. Image annotation
5. YOLO label generation
6. Dataset organization
7. Model training
8. Model validation
9. Metric analysis
10. Prediction visualization

The final YOLOv8n model achieved:

```text
Precision: 0.9237
Recall:    0.8947
mAP50:     0.9066
mAP50-95:  0.4790
```

These results demonstrate that the trained model was able to detect Tom effectively on the evaluated validation data.

---

# 👤 Author

**Mahdiyeh KRM**

GitHub:

`https://github.com/MahdiyehKRM`

---

# 📄 License

This repository was created for educational and academic purposes.
