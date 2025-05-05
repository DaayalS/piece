import os
import shutil
import random

def organize_dataset():
    # Create necessary directories
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir('boards') if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle the files to ensure random distribution
    random.shuffle(image_files)
    
    # Split into train and validation sets (80% train, 20% val)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Process training set
    print("Processing training set...")
    for filename in train_files:
        # Get base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Copy image
        src_img = os.path.join('boards', filename)
        dst_img = os.path.join('dataset/images/train', filename)
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding annotation
        src_anno = os.path.join('ano', f"{base_name}.txt")
        dst_anno = os.path.join('dataset/labels/train', f"{base_name}.txt")
        if os.path.exists(src_anno):
            shutil.copy2(src_anno, dst_anno)
            print(f"Processed {filename} and its annotation for training")
        else:
            print(f"Warning: No annotation found for {filename}")
    
    # Process validation set
    print("\nProcessing validation set...")
    for filename in val_files:
        # Get base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Copy image
        src_img = os.path.join('boards', filename)
        dst_img = os.path.join('dataset/images/val', filename)
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding annotation
        src_anno = os.path.join('ano', f"{base_name}.txt")
        dst_anno = os.path.join('dataset/labels/val', f"{base_name}.txt")
        if os.path.exists(src_anno):
            shutil.copy2(src_anno, dst_anno)
            print(f"Processed {filename} and its annotation for validation")
        else:
            print(f"Warning: No annotation found for {filename}")

if __name__ == "__main__":
    organize_dataset()
    print("\nDataset organization completed!") 