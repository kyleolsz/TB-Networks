import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import pytorch_ssim
from loss_models import PerceptualVGG19


class TBNLoss(nn.modules.Module):
    def __init__(self, tensor_type='torch.cuda.FloatTensor', use_vgg=True, vgg_model_path=''):
        super(TBNLoss, self).__init__()
        self.tensor_type = tensor_type
        self.pyssim_loss_module = pytorch_ssim.SSIM(window_size=11)
        self.use_vgg = use_vgg

        if self.use_vgg:
            self.perception_loss_module = PerceptualVGG19(feature_layers=[0, 5, 10, 15], use_normalization=False,
                                                          path=vgg_model_path)
            if 'torch.cuda.FloatTensor' == tensor_type:
                self.perception_loss_module = self.perception_loss_module.cuda()

        self.l1_loss_module = nn.L1Loss()
        if 'torch.cuda.FloatTensor' == tensor_type:
            self.l1_loss_module = self.l1_loss_module.cuda()

    def grayscale_transform(self, x):
        return ((x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]) / (3.0)).unsqueeze(1)

    def forward(self, input, target):
        input_rgb = input[:, 0:3, :, :]
        gt_tgt_rgb = target[:, 0:3, :, :]

        if self.use_vgg:
            _, fake_features = self.perception_loss_module(input_rgb)
            _, tgt_features = self.perception_loss_module(gt_tgt_rgb)
            vgg_tgt = ((fake_features - tgt_features) ** 2).mean()
        else:
            vgg_tgt = None

        l1_tgt = self.l1_loss_module(input_rgb, gt_tgt_rgb)

        ssim_tgt = self.pyssim_loss_module(self.grayscale_transform(input_rgb),
                                           self.grayscale_transform(gt_tgt_rgb))
        return vgg_tgt, l1_tgt, ssim_tgt
