from utils.manage_import import *
from DCGAN.dcgan import *
from WGAN.wgan import *
from utils.data_process import load_npz
from utils.clf_analysis import load_data, classify_analysis
from sklearn.neighbors import KNeighborsClassifier
from utils.visualize import plot_errorband


INPUT_SHAPE = (1, 96)
NAME = 'ECG200'
# INPUT_SHAPE = (1, 500)
# NAME = 'FordA'
LATENT_DIM = 100
LR = 1e-6
BATCH_SIZE = 32
EPOCHS = 5000
SAMPLE_CYCLE = 100
LOAD = 1
TRAIN = 0
LABELS = [-1, 1]

model_cls = WGANGP
gen_cls = DCGenerator
dis_cls = DCCritic


def build(model_cls, gen_cls, dis_cls):
    model = model_cls(
        INPUT_SHAPE, LATENT_DIM, LR,
        optimizer=torch.optim.Adam,
        opt_args={'betas': (0.5, 0.999)},
        noise_type='normal',
        )
    model.build_model(gen_cls, dis_cls)
    return model

model = build(model_cls, gen_cls, dis_cls)

if TRAIN:
    for TARGET in [-1, 1]:
        model = build(model_cls, gen_cls, dis_cls)
        if LOAD:
            model.load_model('./ckpt/model_{:d}.pkl'.format(TARGET))
        print('-'*20 + 'TARGET=%s' % TARGET + '-'*20)
        model.train(
            name=NAME,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            sample_cycle=SAMPLE_CYCLE,
            target=TARGET,
        )
        model.plot_history(name=TARGET)
else:
    NUM = 1000
    x_gen = []
    y_gen = []
    [m.eval() for m in model.models.values()]
    for i in LABELS:
        path = './ckpt/model_{:d}.pkl'.format(i)
        model.load_model(path)
        noise = model.gen_noise(0, 1, (NUM, LATENT_DIM))
        x = model.generator(noise)
        x = x.data.cpu().numpy().reshape(x.size(0), -1)
        y = np.zeros(NUM) + i
        x_gen.append(x)
        y_gen.append(y)
    x_gen = np.concatenate(x_gen)
    y_gen = np.concatenate(y_gen)

    data, label = load_npz()
    (x_train, y_train), (x_test, y_test) = load_data()

    clf_model = KNeighborsClassifier(n_neighbors=1)

    clf_model.fit(x_train, y_train)
    pred_data = clf_model.predict(data)
    pred_test = clf_model.predict(x_test)
    pred_gen = clf_model.predict(x_gen)
    classify_analysis(label, pred_data, labels=LABELS)
    classify_analysis(y_test, pred_test, labels=LABELS)
    classify_analysis(y_gen, pred_gen, labels=LABELS)
    print('-'*20 + 'TSTR' + '-'*20)
    clf_model.fit(x_gen, y_gen)
    pred_data = clf_model.predict(data)
    pred_test = clf_model.predict(x_test)
    pred_gen = clf_model.predict(x_gen)
    classify_analysis(label, pred_data, labels=LABELS)
    classify_analysis(y_test, pred_test, labels=LABELS)

    print(model.__class__.__name__)
    print(model_cls.__name__)
    plot_errorband({'fake': x_gen, 'real': data})
