
import os
import random
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

# --- Configuration ---
# Update this path to the location of your TrashNet dataset on your computer.
# IMPORTANT: This script now assumes you have downloaded and extracted a real image dataset.
# Please install the required libraries: pip install torch torchvision Pillow
DATA_DIR = Path('C:/Users/murug/.cache/kagglehub/datasets/feyzazkefe/trashnet/versions/1/dataset-resized')
ALL_CLASSES = ['plastic', 'metal', 'glass', 'cardboard', 'paper', 'organic']
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for validation
IMAGE_SIZE = (224, 224)

# --- Data Preparation Logic ---
def prepare_data_for_model(data_directory, all_classes):
    """
    Organizes data into train/val splits and returns file paths.
    """
    print(f"\n--- Starting data preparation from directory: {data_directory} ---")

    if not data_directory.exists():
        print(f"Error: Dataset directory '{data_directory}' not found.")
        print("Please download and extract the TrashNet dataset and update the DATA_DIR path.")
        return None, None, None

    # Initialize dictionaries to hold file paths
    all_files = {cls: [] for cls in all_classes}
    valid_classes = []
    
    for cls in all_classes:
        class_path = data_directory / cls
        if not class_path.exists():
            print(f"Warning: Directory '{class_path}' not found. Skipping class '{cls}'.")
            continue
        
        # Only select files that are likely images (ending with common extensions)
        files = [f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
        
        if not files:
            print(f"Warning: No valid image files found in '{class_path}'. Skipping class '{cls}'.")
            continue
            
        all_files[cls] = files
        valid_classes.append(cls)

    if not valid_classes:
        print("\nError: No valid classes with image data were found. Please check your dataset.")
        return None, None, None
        
    # Print a summary of the loaded files
    print("\nInitial file counts per class (only including classes with data):")
    for cls in valid_classes:
        print(f"  - {cls}: {len(all_files[cls])} images")
    
    # --- Split the data into train and validation sets ---
    print("\nSplitting data into training and validation sets...")
    train_dir = data_directory.parent / 'train'
    val_dir = data_directory.parent / 'validation'
    
    # Clear existing train/val directories to avoid duplication
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    for cls in valid_classes:
        files = all_files[cls]
        random.shuffle(files) # Shuffle the files for a good mix
        
        split_index = int(len(files) * TRAIN_SPLIT_RATIO)
        train_files = files[:split_index]
        val_files = files[split_index:]
        
        # Create destination directories
        os.makedirs(train_dir / cls, exist_ok=True)
        os.makedirs(val_dir / cls, exist_ok=True)
        
        # Copy files to the new directories
        for file in train_files:
            shutil.copy(file, train_dir / cls / file.name)
        for file in val_files:
            shutil.copy(file, val_dir / cls / file.name)
    
    print("\nData splitting and organization complete!")

    return train_dir, val_dir, valid_classes

def define_transforms(image_size):
    """
    Defines the image preprocessing and augmentation pipelines.
    """
    # Augmentation for the training set
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Normalization for the validation set (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nData preprocessing and augmentation pipelines defined.")
    return train_transforms, val_transforms

# --- Exploratory Data Analysis (EDA) Functions ---
def visualize_class_distribution(train_dir, val_dir, classes):
    """
    Creates and displays a bar chart of the class distribution.
    """
    print("\nVisualizing class distribution...")
    train_counts = [len(os.listdir(train_dir / cls)) for cls in classes]
    val_counts = [len(os.listdir(val_dir / cls)) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_counts, width, label='Train')
    rects2 = ax.bar(x + width/2, val_counts, width, label='Validation')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Image Count by Class and Split')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    
    fig.tight_layout()
    plt.show()

def show_sample_images(data_directory, classes):
    """
    Displays one random image from each class.
    """
    print("\nShowing a sample image from each class...")
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 3))
    
    for i, cls in enumerate(classes):
        class_path = data_directory / cls
        files = [f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
        if files:
            sample_file = random.choice(files)
            try:
                img = Image.open(sample_file)
                axes[i].imshow(img)
                axes[i].set_title(cls)
                axes[i].axis('off')
            except Exception as e:
                axes[i].set_title(f"{cls} (Error)")
                axes[i].axis('off')
                print(f"Could not open image file {sample_file}: {e}")
        else:
            axes[i].set_title(f"{cls} (No images)")
            axes[i].axis('off')
            
    fig.tight_layout()
    plt.show()

def analyze_pixel_distribution(data_directory, classes):
    """
    Analyzes and plots the pixel intensity distribution for a sample of images.
    """
    print("\nAnalyzing pixel intensity distribution...")
    num_samples = 5 # Number of images to sample from each class for analysis
    all_pixels = {'R': [], 'G': [], 'B': []}
    
    for cls in classes:
        class_path = data_directory / cls
        files = [f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
        if not files:
            continue
            
        sampled_files = random.sample(files, min(num_samples, len(files)))
        
        for file in sampled_files:
            try:
                img = Image.open(file).convert('RGB')
                img_array = np.array(img)
                all_pixels['R'].extend(img_array[:, :, 0].flatten())
                all_pixels['G'].extend(img_array[:, :, 1].flatten())
                all_pixels['B'].extend(img_array[:, :, 2].flatten())
            except Exception as e:
                print(f"Could not process image {file}: {e}")
                
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(all_pixels['R'], bins=256, color='red', alpha=0.7)
    axes[0].set_title('Red Channel')
    axes[1].hist(all_pixels['G'], bins=256, color='green', alpha=0.7)
    axes[1].set_title('Green Channel')
    axes[2].hist(all_pixels['B'], bins=256, color='blue', alpha=0.7)
    axes[2].set_title('Blue Channel')
    
    for ax in axes:
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        
    fig.suptitle('Pixel Intensity Distribution Across All Classes')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def print_final_summary(train_dir, val_dir, classes):
    """
    Prints the final file counts for the train and validation sets.
    """
    print("\n--- Final Data Summary ---")
    print(f"\nTraining set located at: {train_dir}")
    for cls in classes:
        count = len(os.listdir(train_dir / cls))
        print(f"  - {cls}: {count} images")

    print(f"\nValidation set located at: {val_dir}")
    for cls in classes:
        count = len(os.listdir(val_dir / cls))
        print(f"  - {cls}: {count} images")
    
    print("\n--- Next Steps for Class Imbalance ---")
    print("If you notice a significant class imbalance in the visualization, you can address it during the training process by:")
    print("  - Using class weights in your loss function.")
    print("  - Applying data augmentation techniques to the minority classes more aggressively.")
    print("  - Using a different sampling strategy, such as oversampling the minority class.")
    print("\nData is now ready for model training!")

# --- Script Execution ---
if __name__ == "__main__":
    train_data_path, val_data_path, valid_classes = prepare_data_for_model(DATA_DIR, ALL_CLASSES)
    if train_data_path and val_data_path and valid_classes:
        train_transforms, val_transforms = define_transforms(IMAGE_SIZE)
        visualize_class_distribution(train_data_path, val_data_path, valid_classes)
        show_sample_images(train_data_path, valid_classes)
        analyze_pixel_distribution(train_data_path, valid_classes)
        print_final_summary(train_data_path, val_data_path, valid_classes)
    else:
        print("\nSkipping further processing due to data preparation errors. Please resolve the issues above.")
