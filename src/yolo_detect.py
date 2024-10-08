import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load image
image = cv2.imread('../images/test_image.jpg')  # Adjust path if necessary

# Perform detection
results = model(image)

# Display results
results.show()

# Save results with bounding boxes
results.save('results/')  # Saves results to results/ directory
