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
                nn.Linear(latent_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Dropout(self.drop_rate),
            ]
        return layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),

        )