# person_detect
Built an ML model using a custom dataset 

use this url to try it 
https://persondetect.streamlit.app/

The model is here: https://www.kaggle.com/models/nehakb/person_detection
The dataset is here : https://www.kaggle.com/datasets/nehakb/crowd-detect/

Usage
How to Use
The model can be loaded using the YOLOv8 Python API. Below is an example code snippet for detection:

from ultralytics import YOLO

Load the pre-trained model
model = YOLO("yolov8_person_crowd.pt")

Perform inference on an image
results = model("crowd_image.jpg")

Display results
results.show()

Extract bounding boxes and labels
for detection in results.xyxy[0]: print(f"Bounding Box: {detection[:4]}, Confidence: {detection[4]}, Class: {detection[5]}")

Input and Output Shapes
Inputs:

Image of size 640x640 pixels (recommended for best performance).
Outputs:

Bounding boxes: [x_min, y_min, x_max, y_max, confidence, class_label].
Known Limitations
Struggles with extreme occlusions.
May have difficulty in detecting individuals in low-light or overly cluttered scenarios.
System
Model Context
Standalone or System Component: This is a standalone model that can integrate into larger monitoring systems.
Input Requirements: Clear images of crowd scenes with minimal motion blur.
Downstream Dependencies: Outputs can be used for statistical crowd analysis, anomaly detection, or fed into surveillance pipelines.
Implementation Requirements
Training Setup
Hardware:

Training: 2x NVIDIA A100 GPUs
Inference: Single NVIDIA T4 GPU or equivalent
Software:

Framework: PyTorch 2.0
Library: Ultralytics YOLOv8
Compute Requirements:

Training Time: ~24 hours
Total Computation: ~60 GPU-hours
Model Characteristics
Initialization
The model was fine-tuned from a YOLOv8 pre-trained checkpoint.
Stats
Model Size: 68 MB
Parameters: ~11.2M
Layers: 53
Latency: ~246 ms/frame on NVIDIA T4 GPU
Additional Techniques
The model is not pruned or quantized. However, it employs techniques like augmentation and mosaic loading to enhance training efficiency.
Data Overview
Training Data
Source: 6,000 annotated images collected from metro and railway station crowd datasets.
Preprocessing: Resizing to 640x640, normalization, and augmentation (flipping, rotation, and hue adjustment).
Demographic Groups
Diverse data capturing varying age groups, clothing styles, and crowd densities from multiple geographic regions.
Evaluation Data
Train/Test/valid Split: 80% / 10% / 10%
Differences: Evaluation data emphasizes challenging scenarios such as occlusions and uneven lighting.
Evaluation Results
Summary
mAP: 78%
Precision: 96%
Recall: 79%
Fairness
Definition: Model ensures consistent performance across crowd densities and demographic variations.
Metrics: mAP consistency across subgroups; bias mitigation in training data collection.
Results: No significant performance drop for any demographic group.
Usage Limitations
Sensitive use cases: Surveillance and privacy concerns in crowd monitoring.
Performance may degrade in adverse weather or low-resolution imagery.
Ethics
Risks: Misuse for unethical surveillance, privacy breaches.
Mitigations: Clear data governance, ensuring compliance with privacy laws, and deploying anonymization techniques where possible.

![2](https://github.com/user-attachments/assets/69e28917-db64-4951-9c34-12723ba16193)
![7](https://github.com/user-attachments/assets/ecfece1e-c53a-417f-8bd0-5fe7141c2640)
![26](https://github.com/user-attachments/assets/4974ce4b-356e-4503-af8a-afcff31c8827)
![47](https://github.com/user-attachments/assets/74d55149-ab29-4238-9433-8fc34ca7305f)
