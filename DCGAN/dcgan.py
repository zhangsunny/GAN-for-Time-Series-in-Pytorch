from utils.manage_import import *
from utils.data_process import load_npz
from utils.losses import LogLoss, ItselfLoss


class DCGenerator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        """
        Args:
            input_shape: (C,W), 例如(1,28)
            latent_dim: 默认值100
        """
        super(DCGenerator, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.init_size = int(np.ceil(self.input_shape[1]/2**2))
        self.channel = self.input_shape[0]
        self.relu_slope = 0.2
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128*self.init_size)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(self.relu_slope, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(self.relu_slope, inplace=True),
            nn.Conv1d(64, self.channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size)
        x = self.conv_blocks(out)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(DCDiscriminator, self).__init__()
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
        self.fc = nn.Linear(128*ds_size, 64)
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


class DCGAN(object):
    def __init__(self, input_shape=(1, 500), latent_dim=100,
                 lr=2e-4, optimizer=torch.optim.Adam,
                 opt_args={'betas': (0.5, 0.999)},
                 noise_type='normal'):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lr = lr
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCELoss().to(self.device)
        self.criterion_log = LogLoss().to(self.device)
        self.criterion_itself = ItselfLoss().to(self.device)
        self.noise_type = noise_type
        self.generator = None
        self.discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.history = None
        self.models = dict()
        self.img_path = './image/'
        self.save_path = './ckpt/'

    def build_model(self, gen_cls, dis_cls, gen_args={}, dis_args={}):
        self.generator = gen_cls(self.latent_dim,
                                 self.input_shape,
                                 **gen_args).to(self.device)
        self.discriminator = dis_cls(self.input_shape,
                                     **dis_args).to(self.device)
        self.optimizer_g = self.optimizer(self.generator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d = self.optimizer(self.discriminator.parameters(),
                                          lr=self.lr, **self.opt_args)
        # 初始化网络权重
        # self.generator.apply(self.weights_init)
        # self.discriminator.apply(self.weights_init)
        self.models = {
            'generator': self.generator,
            'discriminator': self.discriminator,
        }

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0)

    def train(self, name='ECG200', batch_size=64,
              epochs=1e3, sample_cycle=100, target=None):
        if not self.generator:
            raise ValueError("model doesn't be initialized,\
                             please call build_model() before train()")
        self.history = defaultdict(list)
        loader = self.load_data(name, batch_size, target)
        for epoch in range(epochs):
            local_history = self.train_on_epoch(loader)
            self.update_history(local_history)
            if (epoch + 1) % sample_cycle == 0 or (epoch + 1) == epochs:
                # self.sample(epoch + 1)
                # self.save_checkpoint(name=target)
                self.print_local_history(
                    epoch+1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)

    def train_on_epoch(self, loader):
        local_history = dict()
        tmp_history = defaultdict(list)
        for x_batch, _ in loader:
            # 添加高斯噪声以防止过拟合
            # x_batch = x_batch + np.random.normal(0.0, 0.1)
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim))
            x_gen = self.generator(z)
            real = self.gen_tensor(np.ones([batch_size, 1]))
            fake = self.gen_tensor(np.zeros([batch_size, 1]))
            # 训练判别器
            self.optimizer_d.zero_grad()
            # d_loss_real = self.criterion(self.discriminator(x_batch), real)
            # d_loss_fake = self.criterion(
            #     self.discriminator(x_gen.detach()), fake)
            # d_loss = 0.5 * (d_loss_fake + d_loss_real)
            d_loss = self.criterion_log(self.discriminator(x_batch))\
                + self.criterion_log(1-self.discriminator(x_gen.detach()))
            d_loss.backward()
            self.optimizer_d.step()
            # 训练生成器
            self.optimizer_g.zero_grad()
            # g_loss = self.criterion(self.discriminator(x_gen), real)
            g_loss = self.criterion_log(self.discriminator(x_gen))
            g_loss.backward()
            self.optimizer_g.step()
            tmp_history['d_loss'].append(d_loss.item())
            tmp_history['g_loss'].append(g_loss.item())
        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history

    def sample(self, epoch, r=5, c=5):
        os.makedirs(self.img_path, exist_ok=True)
        noise = self.gen_noise(0, 1, (r*c, self.latent_dim))
        x_fake = self.generator(noise)
        x_fake = x_fake.data.cpu().numpy().reshape(x_fake.shape[0], -1)
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['xtick.top'] = False
        plt.rcParams['ytick.left'] = False
        plt.rcParams['ytick.right'] = False
        fig, axes = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axes[i, j].plot(x_fake[count, :])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                count += 1
        fig.savefig(self.img_path+'%s.png' % epoch)
        plt.close()

    def save_checkpoint(self, name=None):
        if name is None:
            name = '{:s}.pkl'.format(self.__class__.__name__)
        else:
            name = '{:s}_{:d}.pkl'.format(self.__class__.__name__, name)
        os.makedirs(self.save_path, exist_ok=True)
        model_state = dict()
        for k, v in self.models.items():
            model_state[k] = v.state_dict()
        model_state['history'] = self.history
        torch.save(model_state, self.save_path+name)

    def load_model(self, path=None):
        if not self.generator:
            raise NameError("model doesn't be initialized")
        path = self.save_path + \
            '{:s}.pkl'.format(self.__class__.__name__) if not path else path
        states = torch.load(path)
        for k, v in self.models.items():
            v.load_state_dict(states[k])

    def update_history(self, local_history):
        for k, v in local_history.items():
            self.history[k].append(v)

    def plot_history(self, name=None):
        os.makedirs(self.img_path, exist_ok=True)
        r = len(self.history.keys())
        plt.figure(figsize=(15, int(r*3)))
        for i, k in enumerate(self.history.keys()):
            plt.subplot(r, 1, i+1)
            plt.plot(range(1, len(self.history[k]) + 1), self.history[k])
            plt.title(k)
        if name is None:
            plt.savefig(self.img_path +
                        'history_{:s}.png'.format(self.__class__.__name__))
        else:
            plt.savefig(
                self.img_path +
                'history_{:s}_{:d}.png'.format(self.__class__.__name__, name))
        plt.close()

    @staticmethod
    def print_local_history(epoch, local_history, max_epoch=10000):
        num = len(str(max_epoch))
        s = 'Epoch-{:0>{}d}:  '.format(epoch, num)
        for k, v in local_history.items():
            s = s + '{}={:.4f}  '.format(k, np.mean(v))
        print(s)

    def gen_tensor(self, x, astype='float', requires_grad=False):
        if isinstance(x, torch.Tensor):
            t = x.clone().requires_grad_(requires_grad)
        else:
            t = torch.tensor(x, requires_grad=requires_grad)
        if astype == 'float':
            t = t.float()
        elif astype == 'long':
            t = t.long()
        else:
            raise ValueError('input correct astype')
        return t.to(self.device)

    # 统一产生噪声
    def gen_noise(self, *args, **kws):
        tmp = None
        if self.noise_type == 'normal':
            tmp = np.random.normal(*args, **kws)
        elif self.noise_type == 'uniform':
            tmp = np.random.uniform(*args, **kws)
        return self.gen_tensor(tmp)

    # 从不同的数据集加载数据
    def load_data(self, name, batch_size, target, npz=True):
        dataset = None
        if npz:
            dataset = self.load_npz(name, target)
        loader = Data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
        return loader

    def load_npz(self, name='ECG200', target=None):
        data, label = load_npz(name)
        if not target:
            inx = range(len(data))
        else:
            inx = label == target
        dataset = Data.TensorDataset(
            torch.tensor(data[inx], dtype=torch.float32),
            torch.tensor(label[inx], dtype=torch.long))
        return dataset
