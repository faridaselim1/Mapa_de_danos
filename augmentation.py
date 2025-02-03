import cv2
import os
import numpy as np
import albumentations as A
from tqdm import tqdm
import random

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Flip image horizontally
    A.RandomBrightnessContrast(p=0.5),  # Adjust brightness & contrast
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Apply blur
    A.MotionBlur(blur_limit=5, p=0.3),  # Simulate motion blur
    A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # Adjust gamma
    A.Rotate(limit=30, p=0.5),  # Rotate within ±30 degrees
    A.Perspective(scale=(0.05, 0.15), p=0.4),  # Perspective distortion
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # Simulate noise
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # Grid distortions
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3)  # Elastic transformations
])

# Paths
INPUT_FOLDER = "D:/5.Semester/Freie_Entwurf/2.Kolloq/Dataset/Annotation/yolov8/datasets/images/train"  # Change this to your dataset folder
OUTPUT_FOLDER = "D:/5.Semester/Freie_Entwurf/2.Kolloq/Dataset/Annotation/yolov8/datasets/augmented_images"  # Where augmented images will be stored
AUGMENTATION_PER_IMAGE = 5  # How many variations per image

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all images from dataset
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_files)} images. Applying augmentation...")

# Process images
for img_name in tqdm(image_files):
    img_path = os.path.join(INPUT_FOLDER, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Error loading {img_name}, skipping.")
        continue

    for i in range(AUGMENTATION_PER_IMAGE):
        augmented = transform(image=image)["image"]

        # Save the augmented image
        new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, new_img_name), augmented)

print(f"✅ Augmentation complete. Augmented images saved in '{OUTPUT_FOLDER}'.")
