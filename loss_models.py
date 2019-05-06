import torch
import torch.nn as nn
from torchvision import models

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True,
                 path=None):
        super(PerceptualVGG19, self).__init__()
        if path != '' and path is not None:
            print('Loading pretrained model')
            model = models.vgg19(pretrained=False)
            model.load_state_dict(torch.load(path))
        else:
            model = models.vgg19(pretrained=True)
        model.float()
        model.eval()

        self.model = model
        self.feature_layers = feature_layers

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean_tensor = None

        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std_tensor = None

        self.use_normalization = use_normalization

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if not self.use_normalization:
            return x

        if self.mean_tensor is None:
            self.mean_tensor = self.mean.view(1, 3, 1, 1).expand(x.size())
            self.std_tensor = self.std.view(1, 3, 1, 1).expand(x.size())

        x = (x + 1) / 2
        return (x - self.mean_tensor) / self.std_tensor

    def run(self, x):
        features = []

        h = x

        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)

        return None, torch.cat(features, dim=1)

    def forward(self, x):
        h = self.normalize(x)
        return self.run(h)


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * T.FloatTensor(x.size()).normal_()
        return x


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None, num_intermediate_layers=0):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        layers = []
        layers.append(Noise(use_noise, sigma=noise_sigma))
        layers.append(nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(Noise(use_noise, sigma=noise_sigma))
        layers.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for layer_idx in range(num_intermediate_layers):
            layers.append(Noise(use_noise, sigma=noise_sigma))
            layers.append(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(Noise(use_noise, sigma=noise_sigma))
        layers.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(Noise(use_noise, sigma=noise_sigma))
        layers.append(nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False))

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None
