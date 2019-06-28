from utils.manage_import import *
from utils.data_process import load_npz
from utils.losses import LogLoss, ItselfLoss
from DCGAN.dcgan import DCGAN


class D2Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(D2Discriminator, self).__init__()
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
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
            nn.Linear(128*ds_size, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, -1)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity


class D2GAN(DCGAN):
    def __init__(self, input_shape, latent_dim, lr,
                 optimizer, opt_args, noise_type, alpha=0.2, beta=0.1):
        super(D2GAN, self).__init__(
            input_shape, latent_dim, lr, optimizer, opt_args, noise_type)
        self.alpha = alpha
        self.beta = beta
        self.discriminator2 = None
        self.optimizer_d2 = None

    def build_model(self, gen_cls, dis_cls, gen_args={}, dis_args={}):
        self.generator = gen_cls(self.latent_dim,
                                 self.input_shape,
                                 **gen_args).to(self.device)
        self.discriminator = dis_cls(self.input_shape,
                                     **dis_args).to(self.device)
        self.discriminator2 = dis_cls(self.input_shape,
                                      **dis_args).to(self.device)
        self.optimizer_g = self.optimizer(self.generator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d = self.optimizer(self.discriminator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d2 = self.optimizer(self.discriminator2.parameters(),
                                           lr=self.lr, **self.opt_args)
        # # 初始化网络权重
        # self.generator.apply(self.weights_init)
        # self.discriminator.apply(self.weights_init)
        # self.discriminator2.apply(self.weights_init)
        self.models = {
            'generator': self.generator,
            'discriminator': self.discriminator,
            'discriminator2': self.discriminator2,
        }

    def train_on_epoch(self, loader):
        local_history = dict()
        tmp_history = defaultdict(list)
        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim))
            x_gen = self.generator(z)
            self.optimizer_d.zero_grad()
            self.optimizer_d2.zero_grad()
            d1_loss = self.alpha * \
                self.criterion_log(self.discriminator(x_batch)) \
                + self.criterion_itself(
                    self.discriminator(x_gen.detach()), False)
            d2_loss = self.criterion_itself(
                self.discriminator2(x_batch), False) \
                + self.beta * self.criterion_log(
                    self.discriminator2(x_gen.detach()))
            d1_loss.backward()
            d2_loss.backward()
            self.optimizer_d.step()
            self.optimizer_d2.step()
            self.optimizer_g.zero_grad()
            g_loss = self.criterion_itself(self.discriminator(x_gen)) \
                + self.beta * self.criterion_log(
                    self.discriminator2(x_gen), False)
            g_loss.backward()
            self.optimizer_g.step()
            tmp_history['d1_loss'].append(d1_loss.item())
            tmp_history['d2_loss'].append(d2_loss.item())
            tmp_history['g_loss'].append(g_loss.item())
        local_history['d1_loss'] = np.mean(tmp_history['d1_loss'])
        local_history['d2_loss'] = np.mean(tmp_history['d2_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history
