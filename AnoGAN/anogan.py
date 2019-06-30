from utils.manage_import import *
import time


# 通过激活层的梯度来寻找最匹配的隐藏变量
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.fc(z)
        return out


class AnoGAN(object):
    def __init__(self, model, latent_dim):
        self.model = model
        self.latent_dim = latent_dim
        self.device = self.model.device

    def anomaly_score(self, x, lambda_score=0.1):
        x = torch.Tensor(x).view(x.shape[0], 1, -1).to(self.device)
        [m.eval() for m in self.model.models.values()]
        z_encoded = self.encode(x)
        x_decoded = self.model.generator(z_encoded)
        validity_decoded, feature_decoded = \
            self.model.discriminator(x_decoded, True)
        validity, feature = self.model.discriminator(x, True)

        x = x.view(x.size(0), -1)
        x_decoded = x_decoded.view(x_decoded.size(0), -1)

        r_score = self.compute_rasidual_score(x, x_decoded)
        d_score = self.compute_discrimination_score(feature, feature_decoded)
        anomaly_score = (1-lambda_score) * r_score + lambda_score * d_score
        return anomaly_score

    def compute_rasidual_score(self, x, x_decoded):
        return torch.sum(torch.abs(x-x_decoded), dim=1)

    def compute_discrimination_score(self, feature, feature_decoded):
        return torch.sum(torch.abs(feature-feature_decoded), dim=1)

    def encode(self, x):
        if 'encoder' not in self.model.models.keys():
            encoder = Encoder(self.latent_dim).to(self.device)
            optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
            loss_func = nn.MSELoss().to(self.device)
            batch_size = x.shape[0]
            z = self.model.gen_noise(0, 1, (batch_size, self.latent_dim))
            for i in range(500):
                optimizer.zero_grad()
                z_encoded = encoder(z)
                x_encoded = self.model.generator(z_encoded)
                loss = loss_func(x_encoded, x)
                loss.backward()
                optimizer.step()
            z_encoded = encoder(z)
        else:
            z_encoded = self.model.encoder(x)
        return z_encoded
