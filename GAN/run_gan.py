from utils.manage_import import *
from utils.meta_model import *
from GAN.gan import RNNGAN
from utils.data_process import load_npz
from utils.clf_analysis import load_data, classify_analysis
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    INPUT_SHAPE = (96, 1)
    NAME = 'ECG200'
    # INPUT_SHAPE = (500, 1)
    # NAME = 'FordA'
    TIME_STEP = INPUT_SHAPE[0]
    INPUT_SIZE = INPUT_SHAPE[1]
    LATENT_DIM = 1
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 1000
    SAMPLE_CYCLE = 1
    LOAD = False
    TRAIN = 1
    LABELS = [-1, 1]

    model = RNNGAN(
        INPUT_SHAPE, LATENT_DIM,
        HIDDEN_SIZE,  NUM_LAYERS, LR,
        optimizer=torch.optim.Adam,
        opt_args={'betas': (0.5, 0.999)},
        noise_type='normal',
    )
    model.build_model(gen_cls=LSTMGenerator2, dis_cls=LSTMDiscriminator)

    if TRAIN:
        for TARGET in [-1, 1]:
            model = RNNGAN(
                INPUT_SHAPE, LATENT_DIM,
                HIDDEN_SIZE,  NUM_LAYERS, LR,
                optimizer=torch.optim.Adam,
                opt_args={'betas': (0.5, 0.999)},
                noise_type='normal',
            )
            model.build_model(
                gen_cls=LSTMGenerator2, dis_cls=LSTMDiscriminator)
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
        NUM = 10000
        x_gen = []
        y_gen = []
        [m.eval() for m in model.models.values()]
        for i in LABELS:
            path = './ckpt/model_{:d}.pkl'.format(i)
            model.load_model(path)
            noise = model.gen_noise(0, 1, (NUM, TIME_STEP, LATENT_DIM))
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
