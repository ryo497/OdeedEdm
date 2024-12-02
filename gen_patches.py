import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from argparse import ArgumentParser
import random

"""
We downsample the original images by a
factor of 2 to produce 256 × 256 patches and apply overlap
on the training set. We obtain 5864 pre and post-event pairs
for the Germany subset and 17660 pairs for Louisiana.
"""

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

def gen_patches(data_path, nation, mode, num_patches_w=6, num_patches_h=5):
    """Process images and save patches."""
    labels = {"labels": []}
    output_dir = f"{data_path}-{nation}-patches_{mode}"
    os.makedirs(output_dir, exist_ok=True)
    img_names = os.listdir(data_path)
    random.shuffle(img_names)
    # threshold = int(0.9 * len(img_names))
    for i, img_name in tqdm(enumerate(img_names)):
        if not img_name.endswith(".tif"):
            continue
        img_path = os.path.join(data_path, img_name)
        image = Image.open(img_path)  # Load the image
        downsampled_image = downsample(image, factor=2)  # Downsample the image
        overlap_w = calculate_overlap(downsampled_image.size[0], patch_size=256, num_patches=num_patches_w)
        overlap_h = calculate_overlap(downsampled_image.size[1], patch_size=256, num_patches=num_patches_h)
        patches = extract_patches(downsampled_image, patch_size=(256, 256), overlap_w=overlap_w, overlap_h=overlap_h)  # Extract patches
        # if i < threshold:
        #     output_dir = output_dir_train
        # else:
        #     output_dir = output_dir_test
        for i, patch in enumerate(patches):
            patch_img = Image.fromarray(patch)
            patch_filename = f"{img_name.split('.')[0]}_{i}.png"
            patch_img.save(os.path.join(output_dir, patch_filename))
            labels["labels"].append([patch_filename, f"{img_name.split('.')[0]}_{i}"])
    # Save metadata
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(labels, f)

# Germany 881 imgを 5864 9倍でいいかも

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_path", type=str, default="data/raw_data")
    args.add_argument("--nation", type=str, default="Germany")
    args.add_argument("--mode", type=str, default="train")
    args = args.parse_args()
    gen_patches(args.data_path, args.nation, args.mode)
