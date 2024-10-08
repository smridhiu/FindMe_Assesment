import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the Faster R-CNN model pre-trained on COCO dataset
def load_faster_rcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Convert image from BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to a tensor and normalize
    image_tensor = F.to_tensor(image_rgb)  # Convert to [0, 1] range tensor
    
    return image, image_tensor

# Perform object detection
def detect_objects_faster_rcnn(model, image_tensor):
    # Unsqueeze to add batch dimension, because the model expects a batch of images
    with torch.no_grad():
        detections = model([image_tensor])[0]
    return detections

# Display the results
def display_results(image, detections, threshold=0.5):
    # Extract bounding boxes, labels, and scores
    boxes = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']
    
    # Draw the bounding boxes on the original image
    for i in range(len(boxes)):
        if scores[i] >= threshold:  # Only display boxes above the confidence threshold
            box = boxes[i].numpy().astype(int)
            label = labels[i].item()
            score = scores[i].item()
            
            # Draw the bounding box and label on the image
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {label} Score: {score:.2f}', 
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the image with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Save the image with bounding boxes
def save_image_with_boxes(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Output image saved at {output_path}")

if __name__ == '__main__':
    # Set the image path
    image_path = "../images/test_image.jpg"
    
    # Load the pre-trained Faster R-CNN model
    model = load_faster_rcnn_model()
    
    # Load and preprocess the image
    image, image_tensor = preprocess_image(image_path)
    
    # Perform object detection
    detections = detect_objects_faster_rcnn(model, image_tensor)
    
    # Display the results with bounding boxes on the image
    display_results(image, detections, threshold=0.5)
    
    # Save the image with bounding boxes (optional)
    output_image_path = '../images/output_image.jpg'
    save_image_with_boxes(image, output_image_path)
