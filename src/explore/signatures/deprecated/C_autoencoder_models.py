import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras import backend as K
from keras import layers, callbacks, models
from keras.initializers import RandomNormal

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
- transform the data, otherwise betweenness exaggerates a small error / loss
- unconstrained - should be similar to PCA
- add regularisation - not helping on shallow
- add dropout - seems to hurt rather than help
- batch normalisation seems to help a bit
- relu helps prevent vanishing gradients (vs. elu)
- deep - helps, a bit

On hyperparameter tuning for beta vae with capacity increase
https://github.com/1Konny/Beta-VAE/issues/8#issuecomment-445126239
"Gamma sets the strength of the penalty for deviating from the target KL, C. Here you want to tune this such that the (batch) average KL stays close to C (say within < 1 nat) across the range of C that you use. This exact value doesn't usually matter much, but just avoid it being too high such that it destabilises the optimisation. C itself should start from low (e.g. 0 or 1) and gradually increase to a value high enough such that reconstructions end up good quality. A good way to estimate Cmax is to train B-VAE on your dataset with a beta low enough such that reconstructions end up good quality and look at the trained model's average KL. That KL can be your Cmax because it gives you a rough guide as to the average amount of representational capacity needed to encode your dataset."

6 latent dimensions at beta=1 with capacity=0: about 7.5 average KL...?

'''

'''
the two following sources seem to agree in principle...
https://keras.io/examples/variational_autoencoder/
https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
see below for more detailed comments on vae loss

On writing own layers and models, see:
- https://keras.io/layers/writing-your-own-keras-layers/
- https://keras.io/models/about-keras-models/

nice example on using classes to structure layers:
https://github.com/scespinoza/sc-autoencoders/blob/master/models.py
'''

'''
The classes use multiple inheritance to reduce duplication:
- Call parent class explicitly -- instead of super() -- to avoid conflict via MRO
- MRO causes unexpected errors unless forcing keyword arguments... (siblings are checked for methods before parents)

To resolve:
1 - Class inheritance order: pass-in the template first otherwise keras Model throws a symbolic tensor error
    (due to empty _encode / _decode functions)
2 - Initialise parent classes directly (instead of super) to avoid unexpected argument errors
3 - Initialise Autoencoder before SimpleMixin otherwise template layers trigger complaint of missing Model parent
4 - Do not subclass Templates from Model otherwise diamond pattern triggers sibling MRO and missing arg errors
'''


class AEC:

    def _encoder(self, X):
        pass

    def _decoder(self, X):
        pass


class SimpleTemplate(AEC):

    def _encoder(self, X):
        X = layers.Dense(256, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dense(256, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dense(256, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        return X

    def _decoder(self, X):
        X = layers.Dense(256, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dense(256, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Dense(256, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        return X


class ConvTemplate(AEC):

    def __init__(self, raw_dim: int, n_features: int, n_distances: int, data_format: str = 'channels_last'):
        # variables
        self.raw_dim = raw_dim
        self.n_features = n_features
        self.n_distances = n_distances
        self.data_format = data_format
        self.y_dim = int(n_distances / 2)
        self.x_dim = int(raw_dim / 2 / self.y_dim)

    def _encoder(self, X):
        X = layers.Reshape((self.n_features, self.n_distances, 1))(X)
        X = layers.Conv2D(16, (1, 5), activation='relu', padding='same', data_format=self.data_format)(X)
        X = layers.BatchNormalization()(X)
        X = layers.MaxPooling2D((1, 2), padding='same', data_format=self.data_format)(X)
        X = layers.Conv2D(8, (1, 3), activation='relu', padding='same', data_format=self.data_format)(X)
        X = layers.BatchNormalization()(X)
        X = layers.Flatten()(X)
        return X

    def _decoder(self, X):
        X = layers.Dense(self.x_dim * self.y_dim, activation='linear')(X)
        X = layers.Reshape((self.x_dim, self.y_dim, 1))(X)
        X = layers.Conv2DTranspose(16, (1, 3), activation='relu', padding='same', data_format=self.data_format)(X)
        X = layers.BatchNormalization()(X)
        X = layers.UpSampling2D((1, 2), data_format=self.data_format)(X)
        X = layers.Conv2DTranspose(1, (1, 5), activation='relu', padding='same', data_format=self.data_format)(X)
        X = layers.Reshape((self.raw_dim,))(X)
        return X


class LSTMTemplate:

    def __init__(self, n_features: int, n_distances: int):
        self.n_features = n_features
        self.n_distances = n_distances

    def _encoder(self, X):
        # encoder layers
        X = layers.Reshape((self.n_features, self.n_distances))(X)
        # LSTM requires steps (distances) by features order
        X = layers.Permute((2, 1), input_shape=(self.n_features, self.n_distances))(X)
        X = layers.LSTM(64, activation='relu')(X)
        X = layers.BatchNormalization()(X)
        return X

    def _decoder(self, X):
        # decoder layers
        X = layers.RepeatVector(self.n_distances)(X)
        X = layers.LSTM(64, return_sequences=True, activation='relu')(X)
        # permute back to features by distances order
        X = layers.Permute((2, 1), input_shape=(self.n_distances, self.n_features))(X)
        # flatten to original
        X = layers.Flatten()(X)
        return X


class AutoEncoder(AEC):

    def __init__(self, raw_dim: int = None, latent_dim: int = None, theme_base='aec', seed=0, uid=0):
        super().__init__()
        np.random.seed(seed)
        # variables
        self.theme = f'{theme_base}_d{latent_dim}_s{seed}'
        self.raw_dim = raw_dim
        self.latent_dim = latent_dim
        self.seed = seed
        self.uid = uid
        # layers
        self.enc_input = layers.Input(shape=(self.raw_dim,), name=f'encoder_input_{self.uid}')
        self.enc_output = None
        self.dec_input = layers.Input(shape=(self.latent_dim,), name=f'decoder_input_{self.uid}')
        self.dec_output = None
        # models
        self.encoder_model = None
        self.decoder_model = None
        self.autoencoder = None
        # data
        self.hist_data = None

    def encode_steps(self):
        X = self._encoder(self.enc_input)
        self.enc_output = layers.Dense(self.latent_dim, activation='linear', name=f'LatentLayer_{self.uid}')(X)

    def decode_steps(self):
        X = self._decoder(self.dec_input)
        self.dec_output = layers.Dense(self.raw_dim, activation='linear', name=f'OutputLayer_{self.uid}')(X)

    def prep_model(self):
        # build encoder
        self.encode_steps()
        self.encoder_model = models.Model(self.enc_input, self.enc_output, name=f'Encoder_{self.uid}')
        X = self.encoder_model(self.enc_input)
        # build decoder
        self.decode_steps()
        self.decoder_model = models.Model(self.dec_input, self.dec_output, name=f'Decoder_{self.uid}')
        self.X_hat = self.decoder_model(X)
        # build combined model
        self.autoencoder = models.Model(self.enc_input, self.X_hat, name=f'Autoencoder_{self.uid}')

    def compile_model(self, optimizer, **kwargs):
        self.autoencoder.compile(optimizer, **kwargs)

    def fit_model(self, X: np.ndarray, logs_path: pathlib.Path = '', **kwargs):
        if 'y' in kwargs.keys():
            raise ValueError('"y" must not be provided ("X" will automatically be passed as the "y" parameter.)')

        log_key = f'{datetime.now().strftime("%Hh%Mm")}_{self.theme}'
        log_dir = logs_path / f'{log_key}'

        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3),
              callbacks.TensorBoard(log_dir=log_dir, write_graph=False)]

        if 'callbacks' in kwargs:
            kwargs['callbacks'] += cb
        else:
            kwargs['callbacks'] = cb

        self.autoencoder.fit(X, y=X, **kwargs)

    def save_weights(self, base_path):
        self.encoder_model.save_weights(str(base_path / f'seed_{self.seed}/model_{self.theme}_encoder.h5'))
        self.decoder_model.save_weights(str(base_path / f'seed_{self.seed}/model_{self.theme}_decoder.h5'))
        self.autoencoder.save_weights(str(base_path / f'seed_{self.seed}/model_{self.theme}_complete.h5'))

        hist_df = pd.DataFrame(self.autoencoder.history.history)
        with open(base_path / f'seed_{self.seed}/model_{self.theme}_history.json', mode='w') as write_file:
            hist_df.to_json(write_file)

    def load_weights(self, base_path):
        self.encoder_model.load_weights(base_path / f'seed_{self.seed}/model_{self.theme}_encoder.h5')
        self.decoder_model.load_weights(base_path / f'seed_{self.seed}/model_{self.theme}_decoder.h5')
        self.autoencoder.load_weights(base_path / f'seed_{self.seed}/model_{self.theme}_complete.h5')
        with open(base_path / f'seed_{self.seed}/model_{self.theme}_history.json') as open_file:
            self.hist_data = pd.read_json(open_file)


class SimpleAutoencoder(SimpleTemplate, AutoEncoder):

    def __init__(self, raw_dim: int, latent_dim: int, theme_base='aec_simple', seed=0, uid=0):
        AutoEncoder.__init__(self, raw_dim, latent_dim, theme_base=theme_base, seed=seed, uid=uid)
        SimpleTemplate.__init__(self)


class ConvAutoencoder(ConvTemplate, AutoEncoder):

    def __init__(self, raw_dim: int, latent_dim: int, n_features: int, n_distances: int,
                 data_format: str = 'channels_last', theme_base='aec_conv', seed=0, uid=0):
        AutoEncoder.__init__(self, raw_dim, latent_dim, theme_base=theme_base, seed=seed, uid=uid)
        ConvTemplate.__init__(self, raw_dim, n_features, n_distances, data_format=data_format)


class LSTMAutoencoder(LSTMTemplate, AutoEncoder):

    def __init__(self, raw_dim: int, latent_dim: int, n_features: int, n_distances: int, theme_base='aec_lstm', seed=0,
                 uid=0):
        AutoEncoder.__init__(self, raw_dim, latent_dim, theme_base=theme_base, seed=seed, uid=uid)
        LSTMTemplate.__init__(self, n_features, n_distances)


class VAE(AutoEncoder):

    def __init__(self, raw_dim: int, latent_dim: int, beta: float, capacity: float, theme_base='vae', seed=0, **kwargs):
        super().__init__(raw_dim, latent_dim, theme_base=theme_base, seed=seed)
        # variables
        self.theme = f'{theme_base}_d{latent_dim}_b{beta}_c{capacity}_s{seed}'
        self.beta = beta
        self.max_capacity = capacity
        self.capacity = K.variable(0.0)

    def sampling(self, inputs):
        Z_mu, Z_log_var = inputs
        batch = K.shape(Z_mu)[0]
        dim = K.int_shape(Z_mu)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        # see page 5 in auto-encoding variational bayes
        return Z_mu + K.exp(0.5 * Z_log_var) * epsilon

    def encode_steps(self):
        # enc_input prepared by parent class
        X = self._encoder(self.enc_input)
        self.Z_mu = layers.Dense(self.latent_dim, name='Z_mu', kernel_initializer=RandomNormal(stddev=0.001))(X)
        self.Z_log_var = layers.Dense(self.latent_dim, name='Z_log_var', kernel_initializer=RandomNormal(stddev=0.001))(
            X)
        self.Z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.Z_mu, self.Z_log_var])

    def decode_steps(self):
        # dec_input prepared by parent class
        X = self._decoder(self.dec_input)
        self.dec_output = layers.Dense(self.raw_dim, activation='linear', name=f'OutputLayer_{self.uid}')(X)

    def prep_model(self):
        # build encoder
        self.encode_steps()
        self.encoder_model = models.Model(self.enc_input, [self.Z, self.Z_mu, self.Z_log_var],
                                          name=f'Encoder_{self.uid}')
        Z, Z_mu, Z_log_var = self.encoder_model(self.enc_input)
        # build decoder
        self.decode_steps()
        self.decoder_model = models.Model(self.dec_input, self.dec_output, name=f'Decoder_{self.uid}')
        self.X_hat = self.decoder_model(Z)
        # build combined model
        self.autoencoder = models.Model(self.enc_input, self.X_hat, name=f'Autoencoder_{self.uid}')

    # custom loss
    # https://stackoverflow.com/questions/52034983/how-is-total-loss-calculated-over-multiple-classes-in-keras
    # https://stackoverflow.com/questions/52172859/loss-calculation-over-different-batch-sizes-in-keras/52173844#52173844
    # y_true and y_pred are batches of N samples x N features
    # the keras loss function archetype outputs the mean loss per sample
    # i.e. N samples x 1d loss per sample
    # e.g. MSE = K.mean(K.square(y_pred - y_true), axis=-1)
    # the wrapping "weighted_masked_objective" function then returns the final batch-wise mean
    # https://github.com/keras-team/keras/blob/2.2.4/keras/engine/training_utils.py#L374
    # note that it is possible to perform all batchwise weighting inside the function...
    # and to simply return a scalar (the wrapping function's mean operation then has no effect)...
    # examples of implementation:
    # A -> https://github.com/YannDubs/disentangling-vae
    # rec loss: disentangling-vae uses sum of reconstruction array divided by batch size
    # kl loss: takes mean kl per latent dimension, then sums
    # B -> https://github.com/google-research/disentanglement_lib
    # rec loss: takes sum of reconstruction per sample and then the mean
    # kl loss: takes the sum of kl per sample, then the mean
    def mse_sum(self, y_true, y_pred):
        rl = K.sum(K.square(y_pred - y_true), axis=-1)
        return K.mean(rl)

    def kl(self, y_true, y_pred):
        kl = 1 + self.Z_log_var - K.square(self.Z_mu) - K.exp(self.Z_log_var)
        kl = K.sum(kl, axis=-1)
        kl *= -0.5
        kl = K.mean(kl)
        return kl

    def kl_beta(self, y_true, y_pred):
        kl = self.kl(y_true, y_pred)
        return self.beta * kl

    def kl_cap(self, y_true, y_pred):
        kl = self.kl(y_true, y_pred)
        return self.beta * K.abs(kl - self.capacity)

    def vae_loss(self, y_true, y_pred):
        rl = self.mse_sum(y_true, y_pred)
        kl = self.kl_cap(y_true, y_pred)
        return rl + kl

    def compile_model(self, optimizer, **kwargs):
        if 'metrics' in kwargs:
            raise ValueError('Metrics are defined internally.')
        if 'loss' in kwargs:
            raise ValueError('VAE loss is defined internally.')
        kwargs['metrics'] = [self.mse_sum, self.kl, self.kl_beta, self.kl_cap, self.vae_loss]
        kwargs['loss'] = self.vae_loss
        self.autoencoder.compile(optimizer, **kwargs)

    def fit_model(self, X: np.ndarray, logs_path: pathlib.Path = '', **kwargs):
        if 'y' in kwargs.keys():
            raise ValueError('"y" must not be provided ("X" will automatically be passed as the "y" parameter.)')

        # prepare callbacks
        if 'epochs' not in kwargs.keys():
            raise ValueError('"epochs" must be defined explicitly for gamma update function.')
        max_capacity = self.max_capacity
        capacity = self.capacity

        def gammaUpdate(epoch):
            # epochs seem to start at index 0
            max_epoch = kwargs['epochs'] - 1
            if epoch >= max_epoch:
                new_capacity = self.max_capacity
            else:
                new_capacity = epoch / max_epoch * max_capacity
            K.set_value(capacity, new_capacity)
            logger.info(f'updated capacity to {K.get_value(capacity)}')

        log_key = f'{datetime.now().strftime("%Hh%Mm")}_{self.theme}'
        log_dir = logs_path / f'{log_key}'

        cb = [callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: gammaUpdate(epoch)),
              callbacks.EarlyStopping(monitor='val_loss', patience=3),
              callbacks.TensorBoard(log_dir=log_dir, write_graph=False)]

        if 'callbacks' in kwargs:
            kwargs['callbacks'] += cb
        else:
            kwargs['callbacks'] = cb

        self.autoencoder.fit(X, y=X, **kwargs)


class SimpleVAE(SimpleTemplate, VAE):

    def __init__(self, raw_dim: int, latent_dim: int, beta: float, capacity: float, theme_base='vae_simple', seed=0,
                 uid=0,
                 **kwargs):
        VAE.__init__(self, raw_dim, latent_dim, beta, capacity, theme_base=theme_base, seed=seed, uid=uid, **kwargs)
        SimpleTemplate.__init__(self)


class ConvVAE(ConvTemplate, VAE):

    def __init__(self, raw_dim: int, latent_dim: int, n_features: int, n_distances: int, beta: float, capacity: float,
                 data_format: str = 'channels_last', theme_base='vae_conv', seed=0, uid=0, **kwargs):
        VAE.__init__(self, raw_dim, latent_dim, beta, capacity, theme_base=theme_base, seed=seed, uid=uid, **kwargs)
        ConvTemplate.__init__(self, raw_dim, n_features, n_distances, data_format=data_format)


class LSTMVAE(LSTMTemplate, VAE):

    def __init__(self, raw_dim: int, latent_dim: int, n_features: int, n_distances: int, beta: float, capacity: float,
                 theme_base='vae_lstm', seed=0, uid=0, **kwargs):
        VAE.__init__(self, raw_dim, latent_dim, beta, capacity, theme_base=theme_base, seed=seed, uid=uid, **kwargs)
        LSTMTemplate.__init__(self, n_features, n_distances)


class SplitVAE(SimpleVAE):

    def __init__(self, raw_dim: int, latent_dim: int, split_input_dims: list, split_latent_dims: list, beta: float,
                 capacity: float, theme_base='merged', seed=0, uid=0, **kwargs):
        if len(split_input_dims) != 3 or len(split_latent_dims) != 3:
            raise ValueError('Merged model currently based on three splits for inputs and latents.')
        if sum(split_input_dims) != raw_dim:
            raise ValueError('Split input dimensions should sum to raw_dim')
        super().__init__(raw_dim, latent_dim, beta, capacity, theme_base=theme_base, seed=seed, uid=uid, **kwargs)
        self.split_input_dims = split_input_dims
        self.split_latent_dims = split_latent_dims
        self.model_a = SimpleAutoencoder(split_input_dims[0], split_latent_dims[0], seed=seed, uid=1)
        self.model_b = SimpleAutoencoder(split_input_dims[1], split_latent_dims[1], seed=seed, uid=2)
        self.model_c = SimpleAutoencoder(split_input_dims[2], split_latent_dims[2], seed=seed, uid=3)

    def slicer(self, input, start, end):
        return input[:, start:end]

    def concatenator(self, inputs):
        X_a, X_b, X_c = inputs
        return K.concatenate([X_a, X_b, X_c], axis=-1)

    # override encode steps for split workflow
    def encode_steps(self):
        # enc_input prepared by parent class
        a_end = self.split_input_dims[0]
        b_end = a_end + self.split_input_dims[1]
        c_end = b_end + self.split_input_dims[2]
        # split the inputs
        X_a = layers.Lambda(self.slicer, arguments={'start': 0, 'end': a_end},
                            output_shape=(self.split_input_dims[0],))(self.enc_input)
        X_b = layers.Lambda(self.slicer, arguments={'start': a_end, 'end': b_end},
                            output_shape=(self.split_input_dims[1],))(self.enc_input)
        X_c = layers.Lambda(self.slicer, arguments={'start': b_end, 'end': c_end},
                            output_shape=(self.split_input_dims[2],))(self.enc_input)
        X_a = self.model_a.encoder_model(X_a)
        X_b = self.model_b.encoder_model(X_b)
        X_c = self.model_c.encoder_model(X_c)
        # merge
        X = layers.Lambda(self.concatenator, output_shape=(sum(self.split_latent_dims),))([X_a, X_b, X_c])
        # now back to regular VAE
        X = self._encoder(X)
        self.Z_mu = layers.Dense(self.latent_dim, name='Z_mu', kernel_initializer=RandomNormal(stddev=0.001))(X)
        self.Z_log_var = layers.Dense(self.latent_dim, name='Z_log_var', kernel_initializer=RandomNormal(stddev=0.001))(
            X)
        self.Z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.Z_mu, self.Z_log_var])

    # decode steps don't change from parent

    def prep_model(self):
        # prepare sub models
        self.model_a.prep_model()
        self.model_b.prep_model()
        self.model_c.prep_model()
        # rest is the same as parent
        super().prep_model()
