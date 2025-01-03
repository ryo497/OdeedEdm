# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
from tqdm import tqdm
import json
import click
import pickle
import numpy as np
import torch
from torch import nn
import PIL.Image
import dnnlib
from torch_utils import misc
from torch_utils import distributed as dist
import lpips
from argparse import ArgumentParser
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# その他設定, 実験由来
dataset_kwargs = {
    "class_name": "training.dataset.ImageFolderDataset",
    # "path": "datasets/Germany_Training_Public/PRE-event/test",
    "use_labels": False,
    "xflip": False,
    "cache": True,
    "resolution": 256,
    # "max_size": 1476
}

data_loader_kwargs = {
    "pin_memory": True,
    "num_workers": 1,
    "prefetch_factor": 2
}

batch_size = 80
seed = 0

#----------------------------------------------------------------------------
def odeed_sampler(
    net, x, class_labels=None,
    num_steps=18, sigma_min=0.1, sigma_max=10, rho=6,
):
    # print(f"sigma_min: {sigma_min}, sigma_max: {sigma_max}")

    # Time step discretization (symmetric steps)
    step_indices = torch.linspace(0, num_steps - 1, num_steps, dtype=torch.float64, device=x.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps_symmetric = torch.cat([t_steps[:-1], t_steps.flip(0)])

    # Main sampling loop
    x_next = x  # Initial state

    for i, (t_cur, t_next) in enumerate(zip(t_steps_symmetric[:-1], t_steps_symmetric[1:])):
        x_cur = x_next
        
        # Denoising step
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)

        d_cur = (x_cur - denoised) / (t_cur)
        x_next = x_cur + (t_next - t_cur) * d_cur

        # 2nd order correction
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / (t_next)
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    # 最後に NaN をチェックしてカウント & マスク
    nan_mask = torch.isnan(x_next)
    nan_count = nan_mask.sum().item()  # NaNの要素数をカウント

    if nan_count > 0:
        # print(f"Detected {nan_count} NaN values in the final result.")
        x_next[nan_mask] = 0  # マスクしてゼロに置き換え
    # Adjust losses based on NaN mask
    if nan_mask.any():
        # print(f"Adjusting losses for {nan_mask.sum()} NaN-affected samples")

        # Reduce nan_mask to batch size: True if any NaN exists in the batch
        batch_nan_mask = nan_mask.view(nan_mask.size(0), -1).any(dim=1).cpu().numpy()
    else:
        batch_nan_mask = None
    return x_next, batch_nan_mask


def odeed_sampler_with_realistic_net(
    net,  # The network that outputs denoised samples or score functions.
    x_init,  # Initial noisy image or latent.
    num_steps=50,
    sigma_min=0.01,
    sigma_max=50,
    rho=7,
):
    """
    ODEED sampler implementation with realistic network integration and Runge-Kutta method.
    """
    # Generate symmetric time steps.
    step_indices = torch.linspace(0, num_steps - 1, num_steps, dtype=torch.float64)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps_symmetric = torch.cat([t_steps[:-1], t_steps.flip(0)])

    # Initialize state.
    x_cur = x_init.clone().to(torch.float64)
    x_history = []

    # Runge-Kutta 4th-order method.
    def rk4_step(x, t, dt):
        """
        Performs a single Runge-Kutta 4th-order step for the PF-ODE.
        """
        denoised = net(x, t)  # Use the network's output directly.
        k1 = (x - denoised) / t**2
        k2 = (x + 0.5 * dt * k1 - net(x + 0.5 * dt * k1, t + 0.5 * dt)) / t**2
        k3 = (x + 0.5 * dt * k2 - net(x + 0.5 * dt * k2, t + 0.5 * dt)) / t**2
        k4 = (x + dt * k3 - net(x + dt * k3, t + dt)) / t**2
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Main loop.
    for t_cur, t_next in zip(t_steps_symmetric[:-1], t_steps_symmetric[1:]):
        dt = t_next - t_cur

        # Apply Runge-Kutta step.
        x_next = rk4_step(x_cur, t_cur, dt)

        # Record state for analysis.
        x_history.append((float(t_cur), float(x_cur.mean()), float(x_next.mean())))

        # Update state.
        x_cur = x_next

    # Convert history to array for analysis.
    x_history = np.array(x_history)
    return x_cur, x_history

#----------------------------------------------------------------------------
class Loss:
    def __init__(self, net, device):
        self.lpips = lpips.LPIPS(net=net)
        self.lpips.to(device)

    def calc_loss(self, x0, x1, method):
        loss_fn = None
        if method == 'mse':
            loss_fn = nn.MSELoss(reduction='none')
        elif method == 'lpips':
            loss_fn = self.lpips
            x0 = x0.to(torch.float32)
            x1 = x1.to(torch.float32)
        loss = loss_fn(x0, x1)
        if method == 'mse':
            loss = loss.mean(dim=[1,2,3])
        return loss.squeeze()

#----------------------------------------------------------------------------

# @click.command()
# @click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
# @click.option('--datadir',                  help='Where to load the input images', metavar='DIR',                   type=str, required=True)
# @click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)

# @click.option('--lpips_net',                  help='Which net to use for lpips', metavar='alex|vgg|squeeze',        type=click.Choice(['alex','vgg','squeeze']), required=True)

# @click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
# @click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
# @click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
# @click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)


def main(network_pkl, PreEventdir, PostEventdir, outdir, lpips_net, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    # dist.init()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    Preloss_log = {
        'mse': [],
        'lpips': [],
    }
    Postloss_log = {
        'mse': [],
        'lpips': [],
    }

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    # import os
    # print(os.path.exists(network_pkl))
    # print(os.path.exists("checkpoints/network-snapshot-002400.pkl"))
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # # Other ranks follow.
    # if dist.get_rank() == 0:
    #     torch.distributed.barrier()


    # torch.distributed.barrier()
    # Load dataset.
    dist.print0('Loading dataset...')
    max_size = None # アドホックなので変えたい
    # for path, dirs, files in os.walk(datadir):
    #     for f in files:
    #         if f.endswith('png'):
    #             max_size += 1
    Predataset_obj = dnnlib.util.construct_class_by_name(path=PreEventdir, max_size=max_size, **dataset_kwargs) # subclass of training.dataset.Dataset
    Predataset_iterator = iter(torch.utils.data.DataLoader(dataset=Predataset_obj, shuffle=False, batch_size=batch_size, **data_loader_kwargs))
    Postdataset_obj = dnnlib.util.construct_class_by_name(path=PostEventdir, max_size=max_size, **dataset_kwargs) # subclass of training.dataset.Dataset
    Postdataset_iterator = iter(torch.utils.data.DataLoader(dataset=Postdataset_obj, shuffle=False, batch_size=batch_size, **data_loader_kwargs))

    # Init loss func.
    dist.print0('Initiating Loss Func...')
    loss = Loss(lpips_net, device)

    # Loop over batches.
    dist.print0(f'Generating test dataset images to "{outdir}"...')

    for batch_idx, (images, labels) in tqdm(enumerate(Predataset_iterator), total=len(Predataset_iterator), desc="Processing Batches"):
        with torch.no_grad():
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            # Generate Images
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            sampler_fn = odeed_sampler
            generated_images, nan_mask = sampler_fn(net, images, **sampler_kwargs)
            # calc reconstruction loss
            mse_loss = loss.calc_loss(images, generated_images, 'mse').cpu().detach().numpy()
            lpips_loss = loss.calc_loss(images, generated_images, 'lpips').cpu().detach().numpy()
            if nan_mask is not None:
                print(f"Adjusting losses for {nan_mask.sum()} NaN-affected samples")
                # Set NaN-affected parts to a specific value (e.g., max loss)
                max_mse_loss = 1.2  # Define maximum MSE loss for NaN samples
                max_lpips_loss = 20  # Define maximum LPIPS loss for NaN samples
                mse_loss[nan_mask] = max_mse_loss
                lpips_loss[nan_mask] = max_lpips_loss
            Preloss_log['mse'] += [float(m) for m in mse_loss]
            Preloss_log['lpips'] += [float(l) for l in lpips_loss]

            # loss_log['mse'] = np.concatenate((loss_log['mse'], mse_loss))
            # loss_log['lpips'] = np.concatenate((loss_log['lpips'], lpips_loss))
            # save loss log
    Preloss_log['mse'] = sorted(Preloss_log['mse'])
    Preloss_log['lpips'] = sorted(Preloss_log['lpips'])
    os.makedirs(outdir, exist_ok=True)
    Preloss_log_path = os.path.join(outdir, 'Preloss.json')
    with open(Preloss_log_path, 'w') as f:
        json.dump(Preloss_log, f)
    for batch_idx, (images, labels) in tqdm(enumerate(Postdataset_iterator), total=len(Postdataset_iterator), desc="Processing Batches"):
        with torch.no_grad():
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            # Generate Images
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            sampler_fn = odeed_sampler
            generated_images, nan_mask = sampler_fn(net, images, **sampler_kwargs)
            # calc reconstruction loss
            mse_loss = loss.calc_loss(images, generated_images, 'mse').cpu().detach().numpy()
            lpips_loss = loss.calc_loss(images, generated_images, 'lpips').cpu().detach().numpy()
            if nan_mask is not None:
                print(f"Adjusting losses for {nan_mask.sum()} NaN-affected samples")
                # Set NaN-affected parts to a specific value (e.g., max loss)
                max_mse_loss = 1.2  # Define maximum MSE loss for NaN samples
                max_lpips_loss = 15  # Define maximum LPIPS loss for NaN samples
                mse_loss[nan_mask] = max_mse_loss
                lpips_loss[nan_mask] = max_lpips_loss
            Postloss_log['mse'] += [float(m) for m in mse_loss]
            Postloss_log['lpips'] += [float(l) for l in lpips_loss]
    Postloss_log['mse'] = sorted(Postloss_log['mse'])
    Postloss_log['lpips'] = sorted(Postloss_log['lpips'])
    os.makedirs(outdir, exist_ok=True)
    Postloss_log_path = os.path.join(outdir, 'Postloss.json')
    with open(Postloss_log_path, 'w') as f:
        json.dump(Postloss_log, f)
    # Calculate histogram (distribution) of the loss values
    Premse_hist, Premse_bins = np.histogram(Preloss_log['mse'], bins=30, density=True)
    Postmse_hist, Postmse_bins = np.histogram(Postloss_log['mse'], bins=30, density=True)

    # Plot histogram (distribution)
    plt.figure(figsize=(10, 6))
    plt.bar(Premse_bins[:-1], Premse_hist, width=np.diff(Premse_bins), alpha=0.6, label="Pre Distribution")
    plt.bar(Postmse_bins[:-1], Postmse_hist, width=np.diff(Postmse_bins), alpha=0.6, label="Post Distribution")
    plt.title("Distribution of MSE Loss Values")
    plt.xlabel("MSE Loss Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, "MSE_Loss_Distribution.png"))
    plt.close()

    Prelpips_hist, Prelpips_bins = np.histogram(Preloss_log['lpips'], bins=30, density=True)
    Postlpips_hist, Postlpips_bins = np.histogram(Postloss_log['lpips'], bins=30, density=True)

    # Plot histogram (distribution)
    plt.figure(figsize=(10, 6))
    plt.bar(Prelpips_bins[:-1], Prelpips_hist, width=np.diff(Prelpips_bins), alpha=0.6, label="Pre Distribution")
    plt.bar(Postlpips_bins[:-1], Postlpips_hist, width=np.diff(Postlpips_bins), alpha=0.6, label="Post Distribution")
    plt.title("Distribution of LPIPS Loss Values")
    plt.xlabel("LPIPS Loss Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, "LPIPS_Loss_Distribution.png"))
    plt.close()

    PreLabel = np.array([0] * len(Preloss_log['mse']))
    PostLabel = np.array([1] * len(Postloss_log['mse']))
    mse_scores = np.concatenate((Preloss_log['mse'], Postloss_log['mse']))
    lpips_scores = np.concatenate((Preloss_log['lpips'], Postloss_log['lpips']))
    labels = np.concatenate((PreLabel, PostLabel))
    mse_auc = roc_auc_score(labels, mse_scores)
    lpips_auc = roc_auc_score(labels, lpips_scores)

    # FPR95% calculation
    def calculate_fpr95(y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # Find the threshold where TPR is closest to 0.95
        idx = np.argmax(tpr >= 0.95)
        return fpr[idx]

    mse_fpr95 = calculate_fpr95(labels, mse_scores)
    lpips_fpr95 = calculate_fpr95(labels, lpips_scores)

    # Print results
    print(f"MSE AUC: {mse_auc:.4f}, FPR95%: {mse_fpr95:.4f}")
    print(f"LPIPS AUC: {lpips_auc:.4f}, FPR95%: {lpips_fpr95:.4f}")

            # save imgs and append rec loss
            # images_np = (generated_images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            # for idx, image_np in enumerate(images_np):
            #     img_idx = batch_idx * batch_size + idx
            #     image_dir = outdir
            #     os.makedirs(image_dir, exist_ok=True)
            #     image_path = os.path.join(image_dir, f'{img_idx}.png')
            #     if image_np.shape[2] == 1:
            #         PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            #     else:
            #         PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                # loss_log['mse'].append((img_idx, mse_loss[idx]))
                # loss_log['lpips'].append((img_idx, lpips_loss[idx]))


    # Done.
    # torch.distributed.barrier()
    dist.print0('Done.')
#----------------------------------------------------------------------------
"""
python test.py \
    --network_pkl checkpoints/network-snapshot-002400.pkl\
    --PreEventdir datasets/PRE-event-Germany-patches_test \
    --PostEventdir datasets/POST-event-Germany-patches_test --outdir ODEED \
    --lpips_net alex --num_steps 18 --sigma_min 0.002 --sigma_max 80 --rho 7
"""


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--network_pkl", type=str, required=True)
    args.add_argument("--PreEventdir", type=str, required=True)
    args.add_argument("--PostEventdir", type=str, required=True)
    args.add_argument("--outdir", type=str, required=True)
    args.add_argument("--lpips_net", type=str, required=True)
    args.add_argument("--num_steps", type=int, default=18)
    args.add_argument("--sigma_min", type=float)
    args.add_argument("--sigma_max", type=float)
    args.add_argument("--rho", type=float, default=7)
    args = args.parse_args()
    main(
        network_pkl=args.network_pkl,
        PreEventdir=args.PreEventdir,
        PostEventdir=args.PostEventdir,
        outdir=args.outdir,
        lpips_net=args.lpips_net,
    )
