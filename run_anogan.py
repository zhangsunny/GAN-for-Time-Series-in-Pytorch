from utils.manage_import import *
from DCGAN.dcgan import *
from D2GAN.d2gan import *
from WGAN.wgan import *
from BiGAN.bigan import *
from D2BiGAN.d2bigan import *
from utils.data_process import load_npz
from utils.clf_analysis import load_data, classify_analysis, KL_Wasserstein
from sklearn.neighbors import KNeighborsClassifier
import time
from utils.visualize import plot_errorband
from utils.zutils import set_seed
from D2WGAN.d2wgan import *
from utils.spectral_normalization import *
from AnoGAN.anogan import AnoGAN

set_seed(0)


def build(model_cls, gen_cls, dis_cls, model_args={}, build_args={}):
    model = model_cls(
        INPUT_SHAPE, LATENT_DIM, LR,
        optimizer=torch.optim.Adam,
        opt_args={'betas': (0.5, 0.999)},
        noise_type='normal',
        **model_args,
        )
    model.build_model(gen_cls, dis_cls, **build_args)
    return model


INPUT_SHAPE = (1, 96)
NAME = 'ECG200'
# INPUT_SHAPE = (1, 500)
# NAME = 'FordA'
LATENT_DIM = 100
LR = 1e-4
EPOCHS = 1000
BATCH_SIZE = 512
SAMPLE_CYCLE = 1000
LOAD = 0
TRAIN = 1
LABELS = [-1, 1]
TARGET = 1

model_cls = DCGAN
gen_cls = DCGenerator
dis_cls = DCDiscriminator
enc_cls = BiDCEncoder
model_args = {
    # 'clip_val': 0.02,
    # 'lambda_gp': 10,
    # 'alpha': 0.2,
    # 'beta': 0.1,
}
build_args = {
    # 'enc_cls': enc_cls,
}

model = build(model_cls, gen_cls, dis_cls, model_args, build_args)
if TRAIN:
    tic = time.time()
    if LOAD:
        path =\
            './ckpt/{:s}_{:d}.pkl'.format(model.__class__.__name__, TARGET)
        model.load_model(path)
    print('-'*20 + 'TARGET=%s' % TARGET + '-'*20)
    model.train(
        name=NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        sample_cycle=SAMPLE_CYCLE,
        target=TARGET,
    )
    model.plot_history(name=TARGET)
    toc = time.time()
    print('Training costs {:.2f}s'.format(toc-tic))
else:
    path =\
        './ckpt/{:s}_{:d}.pkl'.format(model.__class__.__name__, TARGET)
    model.load_model(path)

(x_train, y_train), (x_test, y_test) = load_data(NAME)
print(
    x_train[y_train == TARGET].shape,
    x_train[y_train != TARGET].shape,
    x_test[y_test == TARGET].shape,
    x_test[y_test != TARGET].shape)

detector = AnoGAN(model, LATENT_DIM)
score = detector.anomaly_score(x_test, lambda_score=0.1)
score = score.cpu().data.numpy()
pred = score > 10
print(pred)
pred = (pred - 0.5) * 2
print(pred)
classify_analysis(y_test, pred, labels=LABELS)

plt.figure(figsize=(15, 5))
plt.scatter(range(len(pred)), pred, label='anomaly score', marker='o')
plt.scatter(range(len(y_test)), y_test, label='label', marker='x')
plt.legend(loc='best')
plt.show()
