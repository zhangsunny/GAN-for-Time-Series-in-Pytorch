from utils.manage_import import *
from utils.data_process import load_npz
from utils.losses import LogLoss, ItselfLoss
from DCGAN.dcgan import DCGAN


class BiDCEncoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(BiDCEncoder, self).__init__()
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
        self.latent_dim = latent_dim
        self.relu_slope = 0.2
        self.drop_rate = 0.25
        self.bn_eps = 0.8

        def dis_block(in_channel, out_channel, bn=True):
            layers = [
                nn.Conv1d(in_channel, out_channel, 3, 2, 1),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Dropout(self.drop_rate),
            ]
            if bn:
                layers.append(
                    nn.BatchNorm1d(out_channel, self.bn_eps))
            return layers

        self.model = nn.Sequential(
            *dis_block(self.channel, 16, bn=False),
            *dis_block(16, 32),
            *dis_block(32, 64),
            *dis_block(64, 128),
        )
        ds_size = int(np.ceil(self.input_shape[1] / 2 ** 4))
        self.fc = nn.Sequential(
            nn.Linear(128*ds_size, 128),
            nn.LeakyReLU(self.relu_slope),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)
        return feature


class BiDCDiscriminator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(BiDCDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
        self.latent_dim = latent_dim
        self.relu_slope = 0.2
        self.drop_rate = 0.25
        self.bn_eps = 1e-5

        def dis_block(in_channel, out_channel, bn=True):
            layers = [
                nn.Conv1d(in_channel, out_channel, 3, 2, 1),
                nn.LeakyReLU(self.relu_slope, inplace=True),
                nn.Dropout(self.drop_rate),
            ]
            if bn:
                layers.append(
                    nn.BatchNorm1d(out_channel, self.bn_eps))
            return layers

        self.model = nn.Sequential(
            *dis_block(self.channel, 16, bn=False),
            *dis_block(16, 32),
            *dis_block(32, 64),
            *dis_block(64, 128),
        )
        ds_size = int(np.ceil((self.input_shape[1]+self.latent_dim) / 2 ** 4))
        self.fc = nn.Linear(128*ds_size, 64)
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Dropout(self.drop_rate),
            nn.Sigmoid(),
        )

    def forward(self, x, z, feature_matching=False):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)
        if z.dim() == 2:
            z = z.view(z.size(0), 1, -1)
        x = torch.cat([x, z], dim=2)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)
        validity = self.fc2(feature)
        if feature_matching:
            return validity, feature
        else:
            return validity


class BiWeakDiscriminator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
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
            *block(latent_dim + input_shape[-1], 1024),
            *block(1024, 512),
            *block(512, 64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, z, feature_matching=False):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        x = torch.cat([x, z], dim=1)
        out = self.fc(x)
        return out


class BiGAN(DCGAN):
    def __init__(self, input_shape, latent_dim, lr,
                 optimizer, opt_args, noise_type):
        super(BiGAN, self).__init__(
            input_shape, latent_dim, lr, optimizer, opt_args, noise_type)
        self.encoder = None

    def build_model(self, gen_cls, dis_cls, gen_args={},
                    dis_args={}, enc_cls=None, enc_args={}):
        self.generator = gen_cls(self.latent_dim,
                                 self.input_shape,
                                 **gen_args).to(self.device)
        self.discriminator = dis_cls(self.latent_dim, self.input_shape,
                                     **dis_args).to(self.device)
        self.encoder = enc_cls(self.latent_dim, self.input_shape,
                               **enc_args).to(self.device)
        self.optimizer_g = self.optimizer(
            [{'params': self.encoder.parameters()},
             {'params': self.generator.parameters()}],
            lr=self.lr, **self.opt_args)
        self.optimizer_d = self.optimizer(
            self.discriminator.parameters(), lr=self.lr, **self.opt_args)
        # 初始化网络权重
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.encoder.apply(self.weights_init)
        self.models = {
            'generator': self.generator,
            'discriminator': self.discriminator,
            'encoder': self.encoder,
        }

    def train_on_epoch(self, loader):
        EPS = 1e-12
        local_history = dict()
        tmp_history = defaultdict(list)
        for i, (x_batch, _) in enumerate(loader):
            # x_batch = x_batch + np.random.normal(0.0, 0.1)
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim))
            x_gen = self.generator(z)
            real = self.gen_tensor(np.ones([batch_size, 1]))
            fake = self.gen_tensor(np.zeros([batch_size, 1]))
            z_decoded = self.encoder(x_batch)
            self.optimizer_d.zero_grad()
            # d_loss = self.criterion(
            #     self.discriminator(x_batch, z_decoded.detach()), real)\
            #     + self.criterion(self.discriminator(x_gen.detach(), z), fake)
            d_loss = self.criterion_log(
                self.discriminator(x_batch, z_decoded.detach())+EPS)\
                + self.criterion_log(
                    1-self.discriminator(x_gen.detach(), z)+EPS)
            d_loss.backward()
            self.optimizer_d.step()
            self.optimizer_g.zero_grad()
            # g_loss = self.criterion(self.discriminator(x_gen, z), real)
            g_loss = self.criterion_log(
                1-self.discriminator(x_gen, z)+EPS, False)
            g_loss.backward()
            self.optimizer_g.step()
            tmp_history['d_loss'].append(d_loss.item())
            tmp_history['g_loss'].append(g_loss.item())
        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history
