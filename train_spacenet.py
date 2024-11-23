import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from argparse import ArgumentParser

def calculate_overlap(image_size, patch_size=256, num_patches=2):
    """Calculate the appropriate overlap."""
    overlap = (patch_size * num_patches - image_size) / (num_patches - 1)
    overlap_rate = overlap / patch_size
    return overlap_rate

def downsample(image, factor=2):
    """Downsample the image by a given factor."""
    width, height = image.size
    new_width = width // factor
    new_height = height // factor
    return image.resize((new_width, new_height), Image.BILINEAR)

def extract_patches(image, patch_size=(256, 256), overlap_w=0.5, overlap_h=0.5):
    """Extract patches with overlap from the image."""
    width, height = image.size
    patch_w, patch_h = patch_size
    stride_w = int(patch_w * (1 - overlap_w))
    stride_h = int(patch_h * (1 - overlap_h))

    patches = []
    for i in range(0, height - patch_h + 1, stride_h):
        for j in range(0, width - patch_w + 1, stride_w):
            patch = image.crop((j, i, j + patch_w, i + patch_h))
            patches.append(np.array(patch))

    return patches

def process_images(data_path, nation, num_patches_w=5, num_patches_h=6):
    """Process images and save patches."""
    labels = {"labels": []}
    output_dir = f"{data_path}-{nation}-patches"
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(data_path)):
        if not img_name.endswith(".tif"):
            continue
        img_path = os.path.join(data_path, img_name)
        image = Image.open(img_path)  # Load the image
        downsampled_image = downsample(image, factor=2)  # Downsample the image
        overlap_w = calculate_overlap(downsampled_image.size[0], patch_size=256, num_patches=num_patches_w)
        overlap_h = calculate_overlap(downsampled_image.size[1], patch_size=256, num_patches=num_patches_h)

        patches = extract_patches(downsampled_image, patch_size=(256, 256), overlap_w=overlap_w, overlap_h=overlap_h)  # Extract patches

        for i, patch in enumerate(patches):
            patch_img = Image.fromarray(patch)
            patch_filename = f"{img_name.split('.')[0]}_{i}.png"
            patch_img.save(os.path.join(output_dir, patch_filename))
            labels["labels"].append([patch_filename, f"{img_name.split('.')[0]}_{i}"])
    # Save metadata
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(labels, f)

# Create the dataset and start training
# os.system(f"python dataset_tool.py --source data/raw_data/Germany_Training_Public/PRE-event-{nation}-patches --dest datasets/PRE-event-{nation}-patches --resolution=256x256")

os.system(f"python train.py --outdir=result/ --data=datasets/PRE-event-{nation}-patches --cond=0 --arch=ncsnpp --duration=500 --batch=80 --lr=2e-4 --cbase=64 --cres=1,2,2,4,4")

def gen_patches(data_path, nation):
    args = ArgumentParser()
    args.add_argument("--data_path", type=str, default=data_path)
    args.add_argument("--nation", type=str, default=nation)
    # Process the images to extract patches
    process_images(data_path, nation)

