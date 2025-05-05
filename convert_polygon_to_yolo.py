import os
import shutil
from pathlib import Path

def convert_polygon_to_yolo(polygon_coords):
    # Convert polygon coordinates to bounding box
    x_coords = [float(polygon_coords[i]) for i in range(0, len(polygon_coords), 2)]
    y_coords = [float(polygon_coords[i]) for i in range(1, len(polygon_coords), 2)]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculate YOLO format (x_center, y_center, width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def process_labels():
    # Create output directories
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    # Get all label files
    label_files = [f for f in os.listdir('dataset/labels') if f.endswith('.txt')]
    
    # Split into train and validation sets (80% train, 20% val)
    split_idx = int(len(label_files) * 0.8)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]
    
    # Process training set
    print("Processing training set...")
    for filename in train_files:
        process_file(filename, 'train')
    
    # Process validation set
    print("\nProcessing validation set...")
    for filename in val_files:
        process_file(filename, 'val')

def process_file(filename, split):
    input_path = os.path.join('dataset/labels', filename)
    output_path = os.path.join(f'dataset/labels/{split}', filename)
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 9:  # class_id + 8 coordinates
            class_id = parts[0]
            coords = parts[1:]
            x_center, y_center, width, height = convert_polygon_to_yolo(coords)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    print(f"Processed {filename} for {split} set")

def update_data_yaml():
    # Update data.yaml with new class names
    yaml_content = """path: /Users/dayaldonadkar/Desktop/Piece_Detector/dataset  # dataset root dir
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
    process_labels()
    update_data_yaml()
    print("\nDataset conversion completed!") 