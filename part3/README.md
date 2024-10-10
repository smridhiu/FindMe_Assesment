Augmented Reality Simulation using YOLOv5 and OpenCV
Project Overview
This project simulates an augmented reality (AR) integration by overlaying detected objects on images using a pre-trained YOLOv5 model. It involves:

Object detection using YOLOv5.
Overlaying bounding boxes and text on detected objects.
Simulating AR effects using OpenCV, such as flashing bounding boxes and other visual effects.

Setup Instructions
Clone the repository:
git clone https://github.com/smridhiu/FindMe_Assesment.git
cd FindMe_Assesment/part3

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  

Install the dependencies:
pip install -r requirements.txt

Download the YOLOv5 model (automatically done when running the script).

Running the Code
Part 1: Run the object detection script:
python src/part1_object_detection.py

This will detect objects in the image and display the bounding boxes and labels.

Part 3: Run the AR simulation script:
python src/ar_simulation.py

This script will overlay the detected objects with bounding boxes and labels, along with simulating AR effects such as flashing bounding boxes.