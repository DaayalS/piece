import os
import json
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image

def create_directory_structure():
    """Create the required directory structure for YOLO dataset."""
    directories = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_bounding_box(points):
    """Calculate bounding box from polygon points."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert bounding box to YOLO format (normalized)."""
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return x_center, y_center, width, height

def process_annotations():
    """Process all images and annotations."""
    # Get all image files
    image_files = [f for f in os.listdir('boards') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly split into train and validation sets
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Process each image
    for img_file in image_files:
        # Get corresponding annotation file
        json_file = os.path.join('ano', os.path.splitext(img_file)[0] + '.json')
        if not os.path.exists(json_file):
            print(f"Warning: No annotation file found for {img_file}")
            continue
        
        # Read image to get dimensions
        img_path = os.path.join('boards', img_file)
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Read and parse JSON annotation
        with open(json_file, 'r') as f:
            annotation = json.load(f)
        
        # Process each shape in the annotation
        yolo_annotations = []
        for shape in annotation['shapes']:
            label = shape['label']
            points = shape['points']
            
            # Get class ID (0 for outlined, 1 for solid)
            class_id = 0 if label == "outlined" else 1
            
            # Get bounding box
            bbox = get_bounding_box(points)
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_to_yolo_format(bbox, img_width, img_height)
            
            # Add to annotations list
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Determine if this is a training or validation image
        is_train = img_file in train_files
        subset = 'train' if is_train else 'val'
        
        # Copy image to appropriate directory
        shutil.copy2(
            img_path,
            os.path.join('dataset', 'images', subset, img_file)
        )
        
        # Save YOLO annotations
        txt_file = os.path.join('dataset', 'labels', subset, os.path.splitext(img_file)[0] + '.txt')
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create directory structure
    create_directory_structure()
    
    # Process all annotations
    process_annotations()
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main() 