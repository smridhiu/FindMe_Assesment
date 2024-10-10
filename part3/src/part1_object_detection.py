import cv2
import torch

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    return image

def load_yolo_model():
    # Load the YOLOv5 small model (YOLOv5s)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects_yolo(model, image):
    # Perform object detection
    results = model(image)
    return results

def display_results(image, results):
    # Extract bounding box information and labels
    detections = results.pandas().xyxy[0]  # Pandas DataFrame of detection results

    for _, row in detections.iterrows():
        # Get bounding box coordinates and label
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Overlay the label and confidence
        label_text = f'{label} ({confidence:.2f})'
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the image with bounding boxes and labels
    cv2.imshow('Detection Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load and preprocess the image
    image = preprocess_image('../images/penguin.webp')
    
    # Load the pre-trained YOLOv5 model
    model = load_yolo_model()
    
    # Detect objects
    results = detect_objects_yolo(model, image)
    
    # Display the results
    display_results(image, results)
