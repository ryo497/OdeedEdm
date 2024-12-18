# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

@persistence.persistent_class
class OODReconstructionLoss:
    def __init__(self,
                 t0=0.5,
                 P_mean=-1.2,
                 P_std=1.2,
                 sigma_data=0.5,
                 weighting_function="training",
                 num_samples=4,
                 device="cuda"):
        """
        OOD再構成誤差を計算するクラス
        :param model: 拡散モデル Dθ
        :param t0: 時間ステップの最大値
        :param weighting_function: 重み関数 ("training" または "linear")
        :param num_samples: 各画像に対して計算する時間ステップの数 N
        :param device: 計算に使用するデバイス
        """
        self.t0 = t0
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.num_samples = num_samples
        self.weighting_function = weighting_function
        self.device = device

    def lambda_t(self, t):
        """
        重み関数 λ(t) を計算
        :param t: 時間 t
        :return: λ(t) の値
        """
        if self.weighting_function == "training":
            sigma = (t * self.P_std + self.P_mean).exp()
            return 1 / sigma ** 2  # 適切なスケールの設定
        elif self.weighting_function == "linear":
            return t
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")

    def __call__(self, net, images, labels=None, augment_pipe=None):
        """
        再構成誤差を計算
        :param x0: 入力画像テンソル (バッチサイズ, チャネル, 高さ, 幅)
        :return: OOD損失
        """
        batch_size, C, H, W = images.shape
        # 時間 t ~ U(0, t0) をサンプリング
        t = torch.rand(batch_size, self.num_samples, device=self.device) * self.t0  # t ~ U(0, t0)
        t = t.view(batch_size, self.num_samples, 1, 1, 1)  # tの形を拡張
        # t = torch.rand(batch_size, device=self.device) * self.t0
        # weights = self.lambda_t(t)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y = y.unsqueeze(1).repeat(1, self.num_samples, 1, 1, 1)
        # ノイズ epsilon ~ N(0, t0^2 * I) をサンプリング
        epsilon = torch.randn_like(y) * t
        # ノイズ付き入力 x_t
        x_t = y + epsilon
        x_t = x_t.view(batch_size * self.num_samples, C, H, W)
        t = t.view(batch_size * self.num_samples, 1, 1, 1)
        x_reconstructed = net(x_t, t)
        x_reconstructed = x_reconstructed.view(batch_size, self.num_samples, C, H, W)
        loss = ((x_reconstructed - y) ** 2).view(batch_size, self.num_samples, -1).mean(dim=2)
        loss = loss.mean(dim=1)
        # 再構成誤差 || Dθ(x0 + ϵ) - x0 ||^2
        reconstruction_error = ((x_reconstructed - y) ** 2).view(batch_size, self.num_samples, -1).mean(dim=2)
        weights = self.lambda_t(t.view(batch_size, self.num_samples))  # (B, N)
        # 9. 重み付き再構成誤差の計算 (B, N)
        weighted_reconstruction_loss = weights * reconstruction_error  # (B, N)
        total_loss = torch.sum(weighted_reconstruction_loss)
        # # 10. すべてのtの損失をバッチごとに平均 (B)
        # batch_loss = weighted_reconstruction_loss.mean(dim=1)
        # # 平均化
        # total_loss = batch_loss.mean()  # スカラー (1,)
        return total_loss

#----------------------------------------------------------------------------
