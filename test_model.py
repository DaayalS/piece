from ultralytics import YOLO
import cv2
import numpy as np

def test_image(image_path):
    # Load the best model
    model = YOLO('runs/detect/chess_pieces_outline8/weights/best.pt')
    
    # Run inference
    results = model(image_path)
    
    # Get the first result
    result = results[0]
    
    # Load the image
    img = cv2.imread(image_path)
    
    # Draw predictions
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class and confidence
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        
        # Get class name
        class_name = result.names[cls]
        
        # Draw filled square based on class
        if class_name.lower().startswith('white'):
            color = (255, 255, 255)  # White
        else:
            color = (0, 0, 0)        # Black
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)  # Filled rectangle
        
        # Optionally, add a border for visibility
        border_color = (0, 255, 0) if class_name.lower().startswith('white') else (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness=2)
        
        # Add label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
    
    # Save the result
    output_path = 'prediction_result.jpg'
    cv2.imwrite(output_path, img)
    print(f"Prediction result saved as {output_path}")
    
    # Print detection summary
    print("\nDetection Summary:")
    for box in result.boxes:
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        class_name = result.names[cls]
        print(f"Detected {class_name} with confidence: {conf:.2f}")

if __name__ == "__main__":
    test_image('boards/27.jpg') 