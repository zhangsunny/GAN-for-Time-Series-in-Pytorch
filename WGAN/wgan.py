from utils.manage_import import *
from utils.data_process import load_npz
from utils.losses import LogLoss, ItselfLoss
from DCGAN.dcgan import DCGAN


class DCCritic(nn.Module):
    def __init__(self, input_shape):
        super(DCCritic, self).__init__()
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
        self.relu_slope = 0.2
        self.drop_rate = 0.25

        def dis_block(in_channel, out_channel, bn=True):
            layers = [
                nn.Conv1d(in_channel, out_channel, 3, 2, 1),
                # nn.BatchNorm1d(out_channel),
                nn.InstanceNorm1d(out_channel),
                nn.LeakyReLU(self.relu_slope),
                nn.Dropout(self.drop_rate),
            ]
            if not bn:
                layers.pop(1)
            return layers

        self.model = nn.Sequential(
            *dis_block(self.channel, 16, bn=False),
            *dis_block(16, 32),
            *dis_block(32, 64),
            *dis_block(64, 128),
        )
        ds_size = int(np.ceil(self.input_shape[1] / 2 ** 4))
        self.fc = nn.Sequential(
            nn.Linear(128*ds_size, 64),
            nn.LeakyReLU(self.relu_slope),
            nn.Dropout(self.drop_rate),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
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


class WGAN(DCGAN):
    def __init__(self, input_shape, latent_dim, lr,
                 optimizer, opt_args, noise_type, clip_val=0.5):
        super(WGAN, self).__init__(
            input_shape, latent_dim, lr, optimizer, opt_args, noise_type)
        self.clip_val = clip_val

    def train(self, name='ECG200', batch_size=64,
              epochs=1e3, sample_cycle=100, target=None):
        if not self.generator:
            raise ValueError("model doesn't be initialized,\
                             please call build_model() before train()")
        self.history = defaultdict(list)
        loader = self.load_data(name, batch_size, target)
        clip_val = self.clip_val
        for epoch in range(epochs):
            local_history = self.train_on_epoch(loader)
            self.update_history(local_history)
            # self.clip_val = clip_val - 0.5 * ((epoch+1) // 2000)
            # self.clip_val = clip_val + 0.5 * ((epoch+1) // 2000)
            if (epoch + 1) % sample_cycle == 0 or (epoch + 1) == epochs:
                # self.sample(epoch + 1)
                # self.save_checkpoint(name=target)
                print(self.clip_val)
                self.print_local_history(
                    epoch+1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)

    def train_on_epoch(self, loader):
        local_history = dict()
        tmp_history = defaultdict(list)
        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim))
            x_gen = self.generator(z)
            self.optimizer_d.zero_grad()
            d_loss = self.criterion_itself(self.discriminator(x_batch))\
                + self.criterion_itself(
                    self.discriminator(x_gen.detach()), False)
            d_loss.backward()
            self.optimizer_d.step()
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_val, self.clip_val)
            self.optimizer_g.zero_grad()
            g_loss = self.criterion_itself(self.discriminator(x_gen))
            g_loss.backward()
            self.optimizer_g.step()
            tmp_history['d_loss'].append(d_loss.item())
            tmp_history['g_loss'].append(g_loss.item())
        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history


class WGANGP(DCGAN):
    def __init__(self, input_shape, latent_dim, lr,
                 optimizer, opt_args, noise_type, lambda_gp=10, n_critic=5):
        super(WGANGP, self).__init__(
            input_shape, latent_dim, lr, optimizer, opt_args, noise_type)
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

    def compute_gradient_penalty(self, x_real, x_fake):
        alpha = np.random.random()
        interpolate = alpha*x_real + (1-alpha)*x_fake
        interpolate.requires_grad_(True)
        d_inter = self.discriminator(interpolate)
        fake = self.gen_tensor(np.ones((x_real.size(0), 1)))
        gradients = torch.autograd.grad(
            outputs=d_inter,
            inputs=interpolate,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
        return gradients_penalty

    def train(self, name='ECG200', batch_size=64,
              epochs=1e3, sample_cycle=100, target=None):
        if not self.generator:
            raise ValueError("model doesn't be initialized,\
                             please call build_model() before train()")
        self.history = defaultdict(list)
        loader = self.load_data(name, batch_size, target)
        lambda_gp = self.lambda_gp
        for epoch in range(epochs):
            local_history = self.train_on_epoch(loader)
            self.update_history(local_history)
            # self.lambda_gp = lambda_gp + 2 * ((epoch+1) // 2000)
            # self.lambda_gp = lambda_gp - 2 * ((epoch+1) // 2000)
            if (epoch + 1) % sample_cycle == 0 or (epoch + 1) == epochs:
                # self.sample(epoch + 1)
                # self.save_checkpoint(name=target)
                print(self.lambda_gp)
                self.print_local_history(
                    epoch+1, local_history, max_epoch=epochs)
        self.save_checkpoint(name=target)

    def train_on_epoch(self, loader):
        local_history = dict()
        tmp_history = defaultdict(list)
        for i, (x_batch, _)in enumerate(loader):
            x_batch = x_batch.to(self.device)
            batch_size = x_batch.size(0)
            x_batch = x_batch.view(batch_size, 1, -1)
            z = self.gen_noise(0, 1, (batch_size, self.latent_dim))
            x_gen = self.generator(z)
            self.optimizer_d.zero_grad()
            d_loss_real = self.criterion_itself(self.discriminator(x_batch))
            d_loss_fake = self.criterion_itself(
                            self.discriminator(x_gen), False)
            gradients_penalty = self.compute_gradient_penalty(x_batch, x_gen)
            d_loss = d_loss_real + d_loss_fake\
                + self.lambda_gp * gradients_penalty
            d_loss.backward(retain_graph=True)
            self.optimizer_d.step()
            tmp_history['d_loss'].append(d_loss.item())
            if i % self.n_critic == 0:
                self.optimizer_g.zero_grad()
                g_loss = self.criterion_itself(self.discriminator(x_gen))
                g_loss.backward()
                self.optimizer_g.step()
                tmp_history['g_loss'].append(g_loss.item())
        local_history['d_loss'] = np.mean(tmp_history['d_loss'])
        local_history['g_loss'] = np.mean(tmp_history['g_loss'])
        return local_history
