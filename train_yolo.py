from ultralytics import YOLO
import os

def train_model():
    # Create a directory for saving model weights
    os.makedirs('runs/train', exist_ok=True)
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    
    # Train the model with optimized parameters
    results = model.train(
        data='data.yaml',          # path to data config file
        epochs=100,                # increased epochs for better convergence
        imgsz=640,                # image size
        batch=8,                  # reduced batch size for better stability
        name='chess_pieces_outline',  # experiment name
        device='cpu',             # use CPU
        workers=4,                # number of worker threads
        patience=20,              # early stopping patience
        save=True,                # save checkpoints
        save_period=10,           # save checkpoint every x epochs
        verbose=True,             # print verbose output
        seed=42,                  # random seed for reproducibility
        augment=True,             # enable augmentations
        mixup=0.1,               # mixup augmentation
        mosaic=1.0,              # mosaic augmentation
        degrees=10.0,            # rotation augmentation
        translate=0.1,           # translation augmentation
        scale=0.5,               # scale augmentation
        shear=2.0,               # shear augmentation
        perspective=0.0,         # perspective augmentation
        flipud=0.0,              # flip up-down augmentation
        fliplr=0.5,              # flip left-right augmentation
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
    )
    
    # Print final metrics
    print("\nTraining completed!")
    print(f"Best mAP50: {results.best_map50}")
    print(f"Best epoch: {results.best_epoch}")
    print(f"Model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_model() 