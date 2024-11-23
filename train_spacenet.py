# Create the dataset and start training
import os
os.system(f"python dataset_tool.py --source data/raw_data/Germany_Training_Public/PRE-event-{nation}-patches --dest datasets/PRE-event-{nation}-patches --resolution=256x256")

os.system(f"python train.py --outdir=result/ --data=datasets/PRE-event-{nation}-patches --cond=0 --arch=ncsnpp --duration=500 --batch=80 --lr=2e-4 --cbase=64 --cres=1,2,2,4,4")



