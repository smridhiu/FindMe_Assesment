import cv2
import torch
import time

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    return image

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects_yolo(model, image):
    results = model(image)
    return results

def simulate_ar_effect(image, detections):
    # Loop to simulate color-changing bounding boxes for animation
    for i in range(5):  # 5 cycles of color change
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            
            # Alternate between green and red bounding boxes for AR effect
            color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Overlay the label
            label_text = f'{label} ({confidence:.2f})'
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the updated image
        cv2.imshow('AR Simulation', image)
        cv2.waitKey(500)  # Wait 500ms between frames
        time.sleep(0.5)

    cv2.destroyAllWindows()

def display_results(image, results):
    detections = results.pandas().xyxy[0]

    # Call the AR effect simulation
    simulate_ar_effect(image, detections)

if __name__ == '__main__':
    image = preprocess_image('../images/penguin.webp')
    model = load_yolo_model()
    results = detect_objects_yolo(model, image)
    display_results(image, results)
