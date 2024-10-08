import cv2
import torch

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # YOLOv5 will handle resizing and normalization internally
    return image

def load_yolo_model():
    # Load the YOLOv5 small model (YOLOv5s)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects_yolo(model, image):
    # Perform object detection
    results = model(image)  # No need for preprocessing here
    return results

def display_results(results):
    # Display the detected bounding boxes
    results.show()  # This opens a window with the detection results

    # Print bounding box coordinates and class labels
    print("Bounding Box Coordinates:")
    print(results.xyxy[0])  # Bounding box in (x1, y1, x2, y2) format
    print("Detected Objects:")
    print(results.pandas().xyxy[0])  # Bounding boxes, labels, confidence scores as a pandas DataFrame

if __name__ == '__main__':
    # Load and preprocess the image
    image = preprocess_image('../images/test_image.jpg')
    
    # Load the pre-trained YOLOv5 model
    model = load_yolo_model()
    
    # Detect objects
    results = detect_objects_yolo(model, image)
    
    # Display the results
    display_results(results)
