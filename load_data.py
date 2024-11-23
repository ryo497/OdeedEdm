import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

NATION_LIST = ["Germany", "Louisiana-East", "Louisiana-West"]
HTTP_PREFIX = "https://spacenet-dataset.s3.amazonaws.com/"

def load_data():
    # Create an unsigned S3 client for public access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    # Specify bucket and file details
    bucket_name = 'spacenet-dataset'
    for nation in NATION_LIST:
        # Define constants
        data_path = f"data/raw_data/{nation}_Training_Public/PRE-event"
        # Create the data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        file_key = f'spacenet/SN8_floods/tarballs/{nation}_Training_Public.tar.gz'
        download_path = f'data/zip_data/{nation}_Training_Public.tar.gz'
        download_dir = 'data/zip_data'
        os.makedirs(download_dir, exist_ok=True)
        # Download the file
        if not os.path.exists(download_path):
            os.system(f"wget {HTTP_PREFIX}{file_key} -O {download_path}")
            print(f"Downloaded file to {download_path}")
        else:
            print(f"File already exists at {download_path}")
        # Extract the downloaded tar file
        os.system(f"tar -xvf {download_path} -C data")

if __name__ == "__main__":
    load_data()
