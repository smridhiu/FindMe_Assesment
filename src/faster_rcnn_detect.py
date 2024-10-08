import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load image
image = cv2.imread('../images/test_image.jpg')  # Adjust path if necessary
image_resized = cv2.resize(image, (640, 640))  # Resize for model input
image_tensor = F.to_tensor(image_resized)

# Perform detection
with torch.no_grad():
    predictions = model([image_tensor])

# Extract boxes, labels, and scores
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# Draw bounding boxes
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    cv2.rectangle(image_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Save or show the result
cv2.imshow('Detected Objects', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
