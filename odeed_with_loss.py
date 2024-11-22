# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import tqdm
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

import pickle

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
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    t_steps = torch.cat([reversed(t_steps[1:-1]), t_steps])# t_N, ..., t_1(拡散過程), t_0, t_1, ..., t_N(復元過程)

    # Main sampling loop.
    #x_next = latents.to(torch.float64) * t_steps[0]
    x_next = x # ノイズなしの画像を初期値とする
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # PF-ODEの決定論的な実装
        # Euler step.
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

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

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--datadir',                  help='Where to load the input images', metavar='DIR',                   type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)

@click.option('--lpips_net',                  help='Which net to use for lpips', metavar='alex|vgg|squeeze',        type=click.Choice(['alex','vgg','squeeze']), required=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)


def main(network_pkl, datadir, outdir, lpips_net, device=torch.device('cuda'), **sampler_kwargs):
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
    dist.init()

    loss_log = {
        'mse': [],
        'lpips': [],
    }

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()


    torch.distributed.barrier()
    # Load dataset.
    dist.print0('Loading dataset...')
    max_size = 0 # アドホックなので変えたい
    for path, dirs, files in os.walk(datadir):
        for f in files:
            if f.endswith('png'):
                max_size += 1
    dataset_obj = dnnlib.util.construct_class_by_name(path=datadir, max_size=max_size, **dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, shuffle=False, batch_size=batch_size, **data_loader_kwargs))

    # Init loss func.
    dist.print0('Initiating Loss Func...')
    loss = Loss(lpips_net, device)

    # Loop over batches.
    dist.print0(f'Generating test dataset images to "{outdir}"...')

    for batch_idx, (images, labels) in enumerate(dataset_iterator):
        with torch.no_grad():
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)

            # Generate Images
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            sampler_fn = odeed_sampler
            generated_images = sampler_fn(net, images, **sampler_kwargs)


            # calc reconstruction loss
            mse_loss = loss.calc_loss(images, generated_images, 'mse').cpu().detach().numpy()
            lpips_loss = loss.calc_loss(images, generated_images, 'lpips').cpu().detach().numpy()
            loss_log['mse'] = np.concatenate((loss_log['mse'], mse_loss))
            loss_log['lpips'] = np.concatenate((loss_log['lpips'], lpips_loss))

            # save loss log
            os.makedirs(outdir, exist_ok=True)
            loss_log_path = os.path.join(outdir, 'loss.pkl')
            with open(loss_log_path, 'wb') as f:
                pickle.dump(loss_log, f)

            # save imgs and append rec loss
            images_np = (generated_images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for idx, image_np in enumerate(images_np):
                img_idx = batch_idx * batch_size + idx
                image_dir = outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{img_idx}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                
                # loss_log['mse'].append((img_idx, mse_loss[idx]))
                # loss_log['lpips'].append((img_idx, lpips_loss[idx]))


    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')   
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
