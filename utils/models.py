from utils.manage_import import *


class FCGenerator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(FCGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.relu_slope = 0.2
        self.drop_rate = 0.25

        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU(self.relu_slope),
                nn.Dropout(self.drop_rate),
            ]
        return layers
        self.fc = nn.Sequential(
            *block(latent_dim, 1024),
            *block(1024, 512),
            nn.Linear(512, np.prod(input_shape)),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        out = self.fc(z)
        out = out.view(out.size(0), *self.input_shape)
        return out


class FCDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(FCDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.relu_slope = 0.2
        self.drop_rate = 0.25

        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU(self.relu_slope),
                nn.Dropout(self.drop_rate),
            ]
            return layers

        self.fc = nn.Sequential(
            *block(latent_dim, 1024),
            *block(1024, 512),
            *block(512, 64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
