from utils.manage_import import *
from DCGAN.dcgan import DCGAN


class D2BiGAN(DCGAN):
    def __init__(self, input_shape, latent_dim, lr,
                 optimizer, opt_args, noise_type, alpha=0.2, beta=0.1):
        super(D2BiGAN, self).__init__(
            input_shape, latent_dim, lr, optimizer, opt_args, noise_type)
        self.alpha = alpha
        self.beta = beta
        self.discriminator2 = None
        self.optimizer_d2 = None
        self.encoder = None

    def build_model(self, gen_cls, dis_cls,
                    gen_args={}, dis_args={}, enc_cls=None, enc_args={}):
        self.generator = gen_cls(self.latent_dim,
                                 self.input_shape,
                                 **gen_args).to(self.device)
        self.discriminator = dis_cls(self.latent_dim, self.input_shape,
                                     **dis_args).to(self.device)
        self.discriminator2 = dis_cls(self.latent_dim, self.input_shape,
                                      **dis_args).to(self.device)
        self.encoder = enc_cls(self.latent_dim, self.input_shape,
                               **enc_args).to(self.device)
        self.optimizer_g = self.optimizer(
            [{'params': self.encoder.parameters()},
             {'params': self.generator.parameters()}],
            lr=self.lr, **self.opt_args)
        self.optimizer_d = self.optimizer(self.discriminator.parameters(),
                                          lr=self.lr, **self.opt_args)
        self.optimizer_d2 = self.optimizer(self.discriminator2.parameters(),
                                           lr=self.lr, **self.opt_args)
        # 初始化网络权重
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.discriminator2.apply(self.weights_init)
        self.encoder.apply(self.weights_init)
        self.models = {
            'generator': self.generator,
            'discriminator': self.discriminator,
            'discriminator2': self.discriminator2,
            'encoder': self.encoder,
        }

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
                print(self.alpha, self.beta)
                self.print_local_history(
                    epoch+1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)

    def train_on_epoch(self, loader):
        EPS = 1e-12
        local_history = dict()
        tmp_history = defaultdict(list)
        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim))
            x_gen = self.generator(z)
            z_decoded = self.encoder(x_batch)
            self.optimizer_d.zero_grad()
            self.optimizer_d2.zero_grad()
            d1_loss = self.alpha * \
                self.criterion_log(
                    self.discriminator(x_batch, z_decoded.detach())+EPS) \
                + self.criterion_itself(
                    self.discriminator(x_gen.detach(), z)+EPS, False)
            d2_loss = self.criterion_itself(
                self.discriminator2(x_batch, z_decoded.detach())+EPS, False) \
                + self.beta * self.criterion_log(
                    self.discriminator2(x_gen.detach(), z)+EPS)
            d1_loss.backward()
            d2_loss.backward()
            self.optimizer_d.step()
            self.optimizer_d2.step()
            self.optimizer_g.zero_grad()
            g_loss = self.criterion_itself(self.discriminator(x_gen, z)+EPS) \
                + self.beta * self.criterion_log(
                    self.discriminator2(x_gen, z)+EPS, False)
            g_loss.backward()
            self.optimizer_g.step()
            tmp_history['d1_loss'].append(d1_loss.item())
            tmp_history['d2_loss'].append(d2_loss.item())
            tmp_history['g_loss'].append(g_loss.item())
        local_history['d1_loss'] = np.mean(tmp_history['d1_loss'])
        local_history['d2_loss'] = np.mean(tmp_history['d2_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history
