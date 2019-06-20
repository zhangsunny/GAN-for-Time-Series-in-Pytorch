"""
使用D2GAN实现时间序列生成
参考：https://github.com/KangBK0120/D2GAN
"""
from utils.manage_import import *
from utils.data_process import load_npz
from utils.losses import LogLoss, ItselfLoss


class LSTMGenerator(nn.Module):
    def __init__(self, input_shape, output_size,
                 hidden_size=64, num_layers=3):
        """
        Args:
            input_shape: (time_step, input_size)
            output_size: 生成数据的维度
            hidden_size: RNN Cell的size
            num_layers: LSTM层数
        """
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_step, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.time_step, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=self.time_step*self.output_size),
            nn.BatchNorm1d(self.time_step*self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """输入数据为(batch_size, time_step, input_size)
        如果输入为二维数据，会被转换为三维，input_size=1
        """
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=2)
        batch_size = x.shape[0]
        h_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        c_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        pred = self.fc(output.contiguous().view(batch_size, -1))
        pred = pred.view(batch_size, self.time_step, self.output_size)
        return pred

    def init_variable(self, *args):
        return torch.randn(*args).to(self.device)


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_shape, hidden_size=64, num_layers=3):
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_step, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h = None
        self.c = None
        self.relu_slope = 0.2
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.time_step, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(in_features=256,
                      out_features=1),
            nn.BatchNorm1d(1),
            nn.Softplus(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=2)
        batch_size = x.shape[0]
        self.h = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        self.c = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(x, (self.h, self.c))
        pred = self.fc(output.contiguous().view(batch_size, -1))
        return pred
        # validity = torch.mean(pred, dim=1)
        # return validity.view(-1, 1)

    def init_variable(self, *args):
        return torch.randn(*args).to(self.device)


class D2GAN(object):
    def __init__(self, input_shape=(96, 1), latent_dim=1, hidden_size=64,
                 num_layers=3, lr=2e-4, optimizer=torch.optim.Adam,
                 opt_args={'betas': (0.5, 0.999)},
                 noise_type='normal', alpha=0.2, beta=0.1):
        """
        Args:
            input_shape: 真实数据size, (time_step, input_size)
            latent_dim: 隐变量维度, int
        """
        super().__init__()
        self.input_shape = input_shape
        self.time_step, self.input_size = input_shape
        self.latent_dim = latent_dim
        self.latent_shape = (self.time_step, self.latent_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # 使用自定义的两种损失函数，注意是否取负数
        self.criterion_log = LogLoss().to(self.device)
        self.criterion_itself = ItselfLoss().to(self.device)
        self.noise_type = noise_type
        self.alpha = alpha
        self.beta = beta
        self.generator = None
        self.discriminator1 = None
        self.discriminator2 = None
        self.optimizer_g = None
        self.optimizer_d1 = None
        self.optimizer_d2 = None
        self.history = None
        self.models = dict()
        self.img_path = './image/'
        self.save_path = './ckpt/'

    def build_model(self, gen_cls, dis_cls, gen_args={}, dis_args={}):
        self.generator = gen_cls(self.latent_shape,
                                 self.input_size,
                                 self.hidden_size,
                                 self.num_layers, **gen_args).to(self.device)
        self.discriminator1 = dis_cls(self.input_shape,
                                      self.hidden_size,
                                      self.num_layers,
                                      **dis_args).to(self.device)
        self.discriminator2 = dis_cls(self.input_shape,
                                      self.hidden_size,
                                      self.num_layers,
                                      **dis_args).to(self.device)
        self.optimizer_g = self.optimizer(self.generator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d1 = self.optimizer(self.discriminator1.parameters(),
                                           lr=self.lr, **self.opt_args)
        self.optimizer_d2 = self.optimizer(self.discriminator2.parameters(),
                                           lr=self.lr, **self.opt_args)
        self.generator.apply(self.weights_init)
        self.discriminator1.apply(self.weights_init)
        self.discriminator2.apply(self.weights_init)
        self.models = {
            'generator': self.generator,
            'discriminator1': self.discriminator1,
            'discriminator2': self.discriminator2,
        }

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self, name='ECG200', batch_size=64,
              epochs=1e3, sample_cycle=100, target=None):
        if not self.generator:
            raise NameError("model doesn't be initialized,\
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
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            time_step = x_batch.size(1)
            z = self.gen_noise(0, 1, (batch_size, time_step, self.latent_dim))
            x_gen = self.generator(z)
            # 训练两个判别器
            self.optimizer_d1.zero_grad()
            self.optimizer_d2.zero_grad()
            d1_loss = self.alpha * \
                self.criterion_log(self.discriminator1(x_batch)) \
                + self.criterion_itself(
                    self.discriminator1(x_gen.detach()), False)
            d2_loss = self.criterion_itself(
                self.discriminator2(x_batch), False) \
                + self.beta * self.criterion_log(
                    self.discriminator2(x_gen.detach()))
            d1_loss.backward()
            d2_loss.backward()
            self.optimizer_d1.step()
            self.optimizer_d2.step()
            # 训练生成器
            self.optimizer_g.zero_grad()
            g_loss = self.criterion_itself(self.discriminator1(x_gen)) \
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

    def sample(self, epoch, r=5, c=5):
        os.makedirs(self.img_path, exist_ok=True)
        noise = self.gen_noise(0, 1, (r*c, self.time_step, self.latent_dim))
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
        os.makedirs(self.save_path, exist_ok=True)
        if name is None:
            name = 'model.pkl'
        else:
            name = 'model_{:d}.pkl'.format(name)
        model_state = dict()
        for k, v in self.models.items():
            model_state[k] = v.state_dict()
        # model_state['history'] = self.history
        torch.save(model_state, self.save_path+name)

    def load_model(self, path=None):
        if not self.generator:
            raise NameError("model doesn't be initialized")
        path = self.save_path+'model.pkl' if not path else path
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
            plt.savefig(self.img_path + 'history.png')
        else:
            plt.savefig(self.img_path + 'history_%s.png' % name)
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
