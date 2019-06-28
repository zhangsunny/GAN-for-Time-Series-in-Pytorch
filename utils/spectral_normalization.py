from utils.manage_import import *
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# class SNDiscriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(SNDiscriminator, self).__init__()
#         self.input_shape = input_shape
#         self.channel = self.input_shape[0]
#         self.relu_slope = 0.2

#         self.model = nn.Sequential(
#             SpectralNorm(nn.Conv1d(self.channel, 16, 3, 2, 1)),
#             nn.LeakyReLU(self.relu_slope),
#             SpectralNorm(nn.Conv1d(16, 32, 3, 2, 1)),
#             nn.LeakyReLU(self.relu_slope),
#             SpectralNorm(nn.Conv1d(32, 64, 3, 2, 1)),
#             nn.LeakyReLU(self.relu_slope),
#             SpectralNorm(nn.Conv1d(64, 128, 3, 2, 1)),
#             nn.LeakyReLU(self.relu_slope),
#         )
#         ds_size = int(np.ceil(self.input_shape[1] / 2 ** 4))
#         self.fc = SpectralNorm(nn.Linear(128*ds_size, 64))
#         self.fc2 = nn.Sequential(
#             SpectralNorm(nn.Linear(64, 1)),
#             nn.Sigmoid(),
#         )

#     def forward(self, x, feature_matching=False):
#         if x.dim() == 2:
#             x = x.view(x.size(0), 1, -1)
#         out = self.model(x)
#         out = out.view(out.size(0), -1)
#         feature = self.fc(out)
#         validity = self.fc2(feature)
#         if feature_matching:
#             return validity, feature
#         else:
#             return validity


class SNDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(SNDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
        self.relu_slope = 0.2
        self.drop_rate = 0.25

        def dis_block(in_channel, out_channel, sn=True):
            if sn:
                layers = [
                    SpectralNorm(
                        nn.Conv1d(in_channel, out_channel, 3, 2, 1))
                ]
            else:
                layers = [
                    nn.Conv1d(in_channel, out_channel, 3, 2, 1)
                ]
            layers = layers + [
                nn.LeakyReLU(self.relu_slope),
                nn.Dropout(self.drop_rate),
            ]
            return layers

        self.model = nn.Sequential(
            *dis_block(self.channel, 16, sn=False),
            *dis_block(16, 32),
            *dis_block(32, 64),
            *dis_block(64, 128),
        )
        ds_size = int(np.ceil(self.input_shape[1] / 2 ** 4))
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(128*ds_size, 64)),
            nn.LeakyReLU(self.relu_slope),
            nn.Dropout(self.drop_rate),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, feature_matching=False):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)
        validity = self.fc2(feature)
        if feature_matching:
            return validity, feature
        else:
            return validity


class SNGenerator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        """
        Args:
            input_shape: (C,W), 例如(1,28)
            latent_dim: 默认值100
        """
        super(SNGenerator, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.init_size = int(np.ceil(self.input_shape[1]/2**2))
        self.channel = self.input_shape[0]
        self.relu_slope = 0.2
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_dim, 128*self.init_size)),
            nn.LeakyReLU(self.relu_slope),
        )
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SpectralNorm(nn.Conv1d(128, 128, 3, stride=1, padding=1)),
            nn.LeakyReLU(self.relu_slope),
            nn.Upsample(scale_factor=2),
            SpectralNorm(nn.Conv1d(128, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(self.relu_slope),
            SpectralNorm(nn.Conv1d(64, self.channel, 3, stride=1, padding=1)),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size)
        x = self.conv_blocks(out)
        return x
