import json
import os
from pathlib import Path
import cv2

def convert_labelme_to_yolo(json_file, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    image_name = os.path.basename(json_file).replace('.json', '.jpg')
    image_path = os.path.join('boards', image_name)
    if not os.path.exists(image_path):
        print(f"Warning: Image file {image_path} not found")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return
    
    height, width = img.shape[:2]
    
    # Create YOLO format labels
    yolo_labels = []
    for shape in data['shapes']:
        if shape['label'] == 'outline':
            # Get points
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Convert to YOLO format (x_center, y_center, width, height) normalized
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = abs(x2 - x1) / width
            h = abs(y2 - y1) / height
            
            # Class 0 for outline pieces
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    # Save YOLO format labels
    output_file = os.path.join(output_dir, os.path.basename(json_file).replace('.json', '.txt'))
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_labels))
    
    print(f"Converted {json_file} to {output_file}")

def main():
    # Input directory containing LabelMe JSON files
    input_dir = 'ano'
    
    # Output directory for YOLO format labels
    output_dir = 'dataset/labels/train'
    
    # Process all JSON files
    for json_file in Path(input_dir).glob('*.json'):
        convert_labelme_to_yolo(str(json_file), output_dir)

if __name__ == "__main__":
    main() 