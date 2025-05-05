import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")
    
    # Load model
    model_path = 'runs/detect/chess_pieces4/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(img, conf=0.1)[0]  # Lower confidence threshold
    
    # Print detection information
    print(f"\nDetections in {os.path.basename(image_path)}:")
    total_detections = 0
    
    # Create a copy of the image for visualization
    img_with_boxes = img.copy()
    
    for box in results.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        
        # Only process outlined pieces (class 0)
        if class_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Print detection info
            print(f"Found outlined piece with confidence {confidence:.2f} at position ({x1}, {y1}, {x2}, {y2})")
            total_detections += 1
            
            # Draw visualization
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Black rectangle
            cv2.putText(img_with_boxes, "Outlined", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    print(f"Total detections: {total_detections}")
    
    # Save processed image
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_processed.jpg')
    cv2.imwrite(output_path, img_with_boxes)
    print(f"Processed image saved to {output_path}\n")

def main():
    # Process all images in the boards directory
    boards_dir = 'boards'
    for filename in sorted(os.listdir(boards_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(boards_dir, filename)
            print(f"\nProcessing {filename}...")
            process_image(image_path)

if __name__ == "__main__":
    main() 