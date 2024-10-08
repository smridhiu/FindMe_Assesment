
# Object Detection Assignment

This project implements two object detection methods: YOLOv5 and Faster R-CNN using Python. The goal is to perform object detection on images of shelves and output bounding box coordinates for detected products.


## Introduction

This project focuses on using pre-trained models for object detection to identify products on shelves in images. The two methods implemented are:
- YOLOv5
- Faster R-CNN

## Requirements

To run this project, you will need the following packages:

- Python >= 3.6
- PyTorch
- torchvision
- OpenCV
- Matplotlib (for visualizing results)

You can install the required packages using:

```bash
pip install torch torchvision opencv-python matplotlib
Setup
Clone the repository to your local machine:
bash
Copy code
git clone <repository-url>
cd Object-Detection-Assignment
Ensure that you have the required images in the images directory.
Usage
YOLOv5 Implementation:

Navigate to the YOLOv5 implementation directory.
Run the YOLOv5 detection script:
bash
Copy code
python yolo_detect.py
The output will display the image with bounding boxes around detected objects and print their coordinates.
Faster R-CNN Implementation:

Navigate to the Faster R-CNN implementation directory.
Run the Faster R-CNN detection script:
bash
Copy code
python faster_rcnn_detect.py
The output will display the image with bounding boxes around detected objects and print their coordinates.
Methods
YOLOv5
Description: YOLO (You Only Look Once) is a real-time object detection system that is fast and efficient.
Model: YOLOv5 is used for detecting objects in images.
Main Functions:
load_yolo_model(): Loads the YOLOv5 pre-trained model.
preprocess_image(): Reads and preprocesses the input image.
detect_objects_yolo(): Runs the model on the image to detect objects.
display_results(): Displays the output image with bounding boxes.
Faster R-CNN
Description: Faster R-CNN is a two-stage object detection model that provides high accuracy.
Model: Faster R-CNN with a ResNet-50 backbone is used for detecting objects in images.
Main Functions:
load_faster_rcnn_model(): Loads the Faster R-CNN pre-trained model.
preprocess_image(): Reads and preprocesses the input image.
detect_objects_faster_rcnn(): Runs the model on the image to detect objects.
display_results(): Displays the output image with bounding boxes.
Results
The output images generated from both methods will display bounding boxes around detected objects along with their corresponding class labels and confidence scores.

Conclusion
This project successfully demonstrates object detection using both YOLOv5 and Faster R-CNN. The results can be further improved by fine-tuning the models on specific datasets or by using different architectures.