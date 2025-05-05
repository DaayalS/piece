import os
import json
import shutil
import random
from pathlib import Path

def create_yolo_directories():
    # Create dataset directories
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)

def convert_to_yolo_format(json_file, image_width, image_height):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    yolo_annotations = []
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        # Convert label to class index
        class_id = 0 if label == 'Black Outline Piece' else 1
        
        # Get bounding box coordinates
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = (x1 + x2) / (2 * image_width)
        y_center = (y1 + y2) / (2 * image_height)
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # Add to annotations
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def process_new_data():
    # Source directories
    source_images = '/Users/dayaldonadkar/Downloads/new data for training/images'
    source_labels = '/Users/dayaldonadkar/Downloads/new data for training/labels'
    
    # Create YOLO directories
    create_yolo_directories()
    
    # Get list of all image files
    image_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Split into train and validation sets (80% train, 20% val)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Process training set
    print("Processing training set...")
    for filename in train_files:
        process_file(filename, source_images, source_labels, 'train')
    
    # Process validation set
    print("\nProcessing validation set...")
    for filename in val_files:
        process_file(filename, source_images, source_labels, 'val')

def process_file(filename, source_images, source_labels, split):
    # Get image dimensions
    image_path = os.path.join(source_images, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    image_height, image_width = img.shape[:2]
    
    # Get corresponding label file
    label_filename = os.path.splitext(filename)[0] + '.json'
    label_path = os.path.join(source_labels, label_filename)
    
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {filename}")
        return
    
    # Convert to YOLO format
    yolo_annotations = convert_to_yolo_format(label_path, image_width, image_height)
    
    # Save YOLO annotations
    output_label_path = os.path.join(f'dataset/labels/{split}', os.path.splitext(filename)[0] + '.txt')
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    # Copy image to dataset
    output_image_path = os.path.join(f'dataset/images/{split}', filename)
    shutil.copy2(image_path, output_image_path)
    
    print(f"Processed {filename} for {split} set")

def update_data_yaml():
    # Update data.yaml with new class names
    yaml_content = """path: dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: Black Outline Piece
  1: White Outline Piece

nc: 2  # number of classes
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    import cv2
    process_new_data()
    update_data_yaml()
    print("\nDataset conversion completed!") 