import os
import shutil

def organize_images():
    # Create image directories
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir('boards') if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split into train and validation sets (80% train, 20% val)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Process training set
    print("Processing training set images...")
    for filename in train_files:
        src_path = os.path.join('boards', filename)
        dst_path = os.path.join('dataset/images/train', filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to train set")
    
    # Process validation set
    print("\nProcessing validation set images...")
    for filename in val_files:
        src_path = os.path.join('boards', filename)
        dst_path = os.path.join('dataset/images/val', filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to val set")

if __name__ == "__main__":
    organize_images()
    print("\nImage organization completed!") 