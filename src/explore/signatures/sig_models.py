import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, metrics, initializers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Convolutional and LSTM variants no longer maintained, see deprecated folder

- transform the data, otherwise betweenness exaggerates a small error / loss
- unconstrained - should be similar to PCA
- add regularisation - not helping on shallow
- add dropout - seems to hurt rather than help
- batch normalisation seems to help a bit
- relu / swish helps prevent vanishing gradients (vs. elu)
- deep - helps, a bit

On hyperparameter tuning for beta vae with capacity increase
https://github.com/1Konny/Beta-VAE/issues/8#issuecomment-445126239
"Gamma sets the strength of the penalty for deviating from the target KL, C.
Here you want to tune this such that the (batch) average KL stays close to C (say within < 1 nat) across the range of C that you use.
This exact value doesn't usually matter much, but just avoid it being too high such that it destabilises the optimisation.
C itself should start from low (e.g. 0 or 1) and gradually increase to a value high enough such that reconstructions end up good quality.
A good way to estimate Cmax is to train B-VAE on your dataset with a beta low enough such that reconstructions end up good quality and look at the trained model's average KL.
That KL can be your Cmax because it gives you a rough guide as to the average amount of representational capacity needed to encode your dataset."

6 latent dimensions at beta=1 with capacity=0: about 7.5 average KL...?

'''

'''
the two following sources seem to agree in principle...
https://keras.io/examples/variational_autoencoder/
https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
see below for more detailed comments on vae loss

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


def mse_sum(y_true, y_pred):
    # sum over dimensions
    rl = K.sum(K.square(y_pred - y_true), axis=-1)
    # take mean over samples
    return K.mean(rl)


# default bias initialiser is zeros
# default weights initialiser is glorot uniform (Xavier)
# He is supposedly better for He but Xavier seems to untangle better
# glorot normal also doesn't work as well as glorot uniform
class DeepLayer(layers.Layer):

    def __init__(self,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 activity_reg: str = None,
                 dropout: float = None,
                 seed: int = 0,
                 batch_norm: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_dims = hidden_layer_dims
        self.deep_layers = []
        for idx, dim in enumerate(self.hidden_layer_dims):
            self.deep_layers.append(layers.Dense(dim,
                                                 activation=activation_func,
                                                 kernel_initializer=kernel_init,
                                                 activity_regularizer=activity_reg))
            if batch_norm:
                self.deep_layers.append(layers.BatchNormalization())
            # add dropout for first layer only - akin to simulating variegated data
            # leave remaining layers to form abstractions
            if dropout is not None:
                self.deep_layers.append(layers.Dropout(rate=dropout, seed=seed))

    def call(self, inputs, training=True, **kwargs):
        X = inputs
        for deep_layer in self.deep_layers:
            X = deep_layer(X, training=training)
        return X

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_layer_dims': self.hidden_layer_dims,
                       'dLayers': self.deep_layers,
                       'bnLayers': self.bnLayers})
        return config


class SamplingLayer(layers.Layer):

    def __init__(self, latent_dim, seed, mu_kernel_init, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.latent_dim = latent_dim
        # Z_log_var is used instead of standard deviation to speed up convergence
        # Standard deviation can be recovered per exp(0.5 * Z_log_var)
        # see page 435 in Hands On Machine Learning
        self.Z_mu_layer = layers.Dense(self.latent_dim,
                                       name='Z_mu',
                                       kernel_initializer=mu_kernel_init,
                                       activation='linear')
        self.Z_logvar_layer = layers.Dense(self.latent_dim,
                                           name='Z_log_var',
                                           kernel_initializer=initializers.Zeros(),
                                           activation='linear')

    def call(self, inputs, training=True, **kwargs):
        batch = K.shape(inputs)[0]
        Z_mu = self.Z_mu_layer(inputs, training=training)
        Z_log_var = self.Z_logvar_layer(inputs, training=training)
        # epsilon variable removes stochastic process from chain of differentiation
        epsilon = K.random_normal(shape=(batch, self.latent_dim), mean=0.0, stddev=1.0,
                                  seed=self.seed)
        # see page 5 in auto-encoding variational bayes
        Z = Z_mu + K.exp(0.5 * Z_log_var) * epsilon
        return [Z_mu, Z_log_var, Z]

    def get_config(self):
        config = super().get_config()
        config.update({
            'seed': self.seed,
            'latent_dim': self.latent_dim,
            'Z_mu_layer': self.Z_mu_layer,
            'Z_logvar_layer': self.Z_logvar_layer
        })
        return config


# custom losses
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

class KLDivergenceLayer(layers.Layer):

    def __init__(self, epochs, beta=1.0, capacity=0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.epochs = epochs
        self.max_capacity = capacity
        self.capacity = tf.Variable(0.0, trainable=False)

    def call(self, inputs, **kwargs):
        Z_mu, Z_log_var = inputs
        # add loss and metrics
        # see page 5 in auto-encoding variational bayes paper
        kl = 1 + Z_log_var - K.square(Z_mu) - K.exp(Z_log_var)
        kl = K.sum(kl, axis=-1)  # sum across latents
        kl *= -0.5
        kl = K.mean(kl)  # take mean across batch
        kl_beta = self.beta * kl
        kl_cap = self.beta * K.abs(kl - self.capacity)
        return kl, kl_beta, kl_cap

    # callback for capacity update
    def capacity_update(self, epoch_step):
        # epochs seem to start at index 0
        if epoch_step == self.epochs - 1:
            new_capacity = self.max_capacity
        else:
            new_capacity = epoch_step / (self.epochs - 1) * self.max_capacity
        K.set_value(self.capacity, new_capacity)
        logger.info(f'updated capacity to {K.get_value(self.capacity)}')

    def get_config(self):
        config = super().get_config()
        config.update({
            'beta': self.beta,
            'max_capacity': self.max_capacity,
            'capacity': 0.0,
            'capacityUpdate': self.capacity_update
        })
        return config


class VAE(models.Model):

    def __init__(self,
                 raw_dim: int,
                 latent_dim: int,
                 beta: float,
                 capacity: float,
                 epochs: int,
                 theme_base='vae',
                 seed=0,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 **kwargs):
        tf.random.set_seed(seed)
        logger.info('initialising VAE...')
        super().__init__(**kwargs)
        self.theme = f'{theme_base}_d{latent_dim}_b{beta}_c{capacity}_s{seed}'
        self.encoder = DeepLayer(name='DeepEncoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init)
        self.sampling = SamplingLayer(latent_dim,
                                      seed,
                                      initializers.TruncatedNormal(stddev=0.001, seed=seed),
                                      name='Sampling')
        self.kl_divergence = KLDivergenceLayer(epochs, beta=beta, capacity=capacity,
                                               name='KLDivergence')
        self.decoder = DeepLayer(name='DeepDecoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init)
        self.x_hat = layers.Dense(raw_dim, activation='linear', name=f'OutputLayer')
        # setup metrics
        self.summary_metrics = {}

    def call(self, inputs, training=True, **kwargs):
        Z_mu, Z_log_var, Z = self.encode(inputs, training=training)
        kl, kl_beta, kl_cap = self.kl_divergence([Z_mu, Z_log_var])
        self.add_loss(kl_cap)
        X_hat = self.decode(Z, training=training)
        rec_loss = mse_sum(inputs, X_hat)
        self.add_loss(rec_loss)
        # update summary metrics
        for key, metric in zip(['capacity_term', 'kl', 'kl_beta', 'kl_beta_cap', 'rec_loss'],
                               [self.kl_divergence.capacity, kl, kl_beta, kl_cap, rec_loss]):
            if key not in self.summary_metrics:
                self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
            self.summary_metrics[key](metric)
        return X_hat

    def encode(self, X, training=False):
        X = self.encoder(X, training=training)
        Z_mu, Z_log_var, Z = self.sampling(X, training=training)
        return Z_mu, Z_log_var, Z

    def decode(self, Z, training=False):
        X = self.decoder(Z, training=training)
        X_hat = self.x_hat(X, training=training)
        return X_hat

    def get_config(self):
        config = super().get_config()
        config.update({
            'theme': self.theme,
            'encoder': self.encoder,
            'sampling': self.sampling,
            'kl_divergence': self.kl_divergence,
            'decoder': self.decoder,
            'x_hat': self.x_hat,
            'encode': self.encode,
            'decode': self.decode
        })
        return config


class Slicer(layers.Layer):

    def __init__(self, start_idx, end_idx, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = start_idx
        self.end_idx = end_idx

    def call(self, X, **kwargs):
        return X[:, self.start_idx:self.end_idx]

    def get_config(self):
        config = super().get_config()
        config.update({
            'start_idx': self.start_idx,
            'end_idx': self.end_idx
        })
        return config


class SplitVaeEncoder(layers.Layer):

    def __init__(self,
                 split_input_dims: tuple,
                 split_latent_dims: tuple,  # for split layers in encoder
                 split_hidden_layer_dims: tuple,  # for split layers in encoder
                 **kwargs):
        super().__init__(**kwargs)
        a_end = split_input_dims[0]
        b_end = a_end + split_input_dims[1]
        c_end = b_end + split_input_dims[2]
        self.X_a = Slicer(0, a_end, name='SlicerA')
        self.X_b = Slicer(a_end, b_end, name='SlicerB')
        self.X_c = Slicer(b_end, c_end, name='SlicerC')
        self.model_a = DeepLayer(hidden_layer_dims=split_hidden_layer_dims[0], name='DeepA')
        self.model_b = DeepLayer(hidden_layer_dims=split_hidden_layer_dims[1], name='DeepB')
        self.model_c = DeepLayer(hidden_layer_dims=split_hidden_layer_dims[2], name='DeepC')
        self.latent_a = layers.Dense(split_latent_dims[0], activation='linear', name='LatentA')
        self.latent_b = layers.Dense(split_latent_dims[1], activation='linear', name='LatentB')
        self.latent_c = layers.Dense(split_latent_dims[2], activation='linear', name='LatentC')

    def call(self, inputs, training=True, **kwargs):
        # a
        X_a = self.X_a(inputs)
        X_a = self.model_a(X_a, training=training)
        X_a = self.latent_a(X_a, training=training)
        # b
        X_b = self.X_b(inputs)
        X_b = self.model_b(X_b, training=training)
        X_b = self.latent_b(X_b, training=training)
        # c
        X_c = self.X_c(inputs)
        X_c = self.model_c(X_c, training=training)
        X_c = self.latent_c(X_c, training=training)
        # merge
        return [X_a, X_b, X_c]

    def get_config(self):
        config = super().get_config()
        config.update({
            'X_a': self.X_a,
            'X_b': self.X_b,
            'X_c': self.X_c,
            'model_a': self.model_a,
            'model_b': self.model_b,
            'model_c': self.model_c,
            'latent_a': self.latent_a,
            'latent_b': self.latent_b,
            'latent_c': self.latent_c
        })
        return config


class SplitVAE(VAE):
    def __init__(self,
                 raw_dim: int,
                 latent_dim: int,
                 beta: float,
                 capacity: float,
                 epochs: int,
                 split_input_dims: tuple = None,  # how to apportion raw dims to split layers
                 split_latent_dims: tuple = None,  # for split layers in encoder
                 split_hidden_layer_dims: tuple = None,  # for split layers in encoder
                 theme_base='vae',
                 seed=0,
                 hidden_layer_dims: tuple = (256, 256, 256),  # decoder
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 **kwargs):
        logger.info('intialising SplitVAE...')
        if len(split_input_dims) != 3 or len(split_latent_dims) != 3 or len(
                split_hidden_layer_dims) != 3:
            raise ValueError('Merged model currently based on three splits.')
        if sum(split_input_dims) != raw_dim:
            raise ValueError('Split input dimensions should sum to raw_dim')
        if split_input_dims is None or split_latent_dims is None or split_hidden_layer_dims is None:
            raise ValueError(
                'Split input dims, split latent dims, and split hidden layer dims are required')
        super().__init__(raw_dim,
                         latent_dim,
                         beta,
                         capacity,
                         epochs,
                         theme_base,
                         seed,
                         hidden_layer_dims,
                         activation_func,
                         kernel_init,
                         **kwargs)
        # split the inputs
        self.encoder = SplitVaeEncoder(split_input_dims,
                                       split_latent_dims,
                                       split_hidden_layer_dims,
                                       name='SplitVaeEncoder')
        self.merged = layers.Concatenate(axis=-1, name='Xmerged')

    def call(self, inputs, training=True, **kwargs):
        Z_mu, Z_log_var, Z = self.encode(inputs, training=training)
        kl, kl_beta, kl_cap = self.kl_divergence([Z_mu, Z_log_var])
        self.add_loss(kl_cap)
        X = self.decoder(Z, training=training)
        X_hat = self.x_hat(X, training=training)
        rec_loss = mse_sum(inputs, X_hat)
        self.add_loss(rec_loss)
        # update summary metrics
        for key, metric in zip(['capacity_term', 'kl', 'kl_beta', 'kl_beta_cap', 'rec_loss'],
                               [self.kl_divergence.capacity, kl, kl_beta, kl_cap, rec_loss]):
            if key not in self.summary_metrics:
                self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
            self.summary_metrics[key](metric)
        return X_hat

    def encode(self, X, training=False):
        X_a, X_b, X_c = self.encoder(X, training=training)
        merged = self.merged([X_a, X_b, X_c])
        Z_mu, Z_log_var, Z = self.sampling(merged, training=training)
        return Z_mu, Z_log_var, Z

    def encode_sub_latents(self, X):
        X_a, X_b, X_c = self.encoder(X, training=False)
        return X_a, X_b, X_c

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': self.encoder,
            'merged': self.merged,
            'encode': self.encode,
            'encode_sub_latents': self.encode_sub_latents
        })
        return config


class Gamma(layers.Layer):

    def __init__(self, n_components, latent_dim, seed, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.latent_dim = latent_dim
        # use constraint to keep cat_pi normalised vs softmax?
        self.cat_pi = tf.Variable(name='cat_pi',
                                  initial_value=np.full(n_components, 1 / n_components),
                                  dtype='float32',
                                  trainable=True,
                                  constraint=lambda x: x / K.sum(x))
        self.cat_pi_softmax = layers.Softmax()
        # note that mu and log var initialisation is overwritten by priming...
        self.gmm_mu = self.add_weight(name='gmm_mu',
                                      shape=(self.latent_dim, self.n_components),
                                      initializer=initializers.GlorotUniform(seed=seed),
                                      trainable=True)
        self.gmm_log_var = tf.Variable(name='gmm_log_var',
                                       initial_value=np.full(
                                           (self.latent_dim, self.n_components), 0.01),
                                       dtype='float32',
                                       trainable=True,
                                       constraint=lambda x: K.abs(x))

    def call(self, Z, **kwargs):
        cat_pi = self.cat_pi_softmax(self.cat_pi)
        # gamma - see equation 19 in Appendix B of original paper
        # tensors are batch x dimension x components
        # insert components dimension for Z vars -> i.e. K.expand_dims(self.Z_mu, 2) (broadcasting may also be option)
        # then repeat the elements along the new axis
        # current strategy can also be switched for K.permute_dimensions(K.repeat(z, self.n_components), (0, 2, 1))
        Z_expand = K.repeat_elements(K.expand_dims(Z, -1), self.n_components, 2)
        # different repos differ here
        # some place K.log(cat_pi) term inside sum vs. others (outside seems correct)
        # i.e. cat_pi is batch x components so first sum across latent dimensions, then subtract...
        # sum over latent dimensions, i.e. batch x latent x components -> batch x components
        gamma = K.sum(
            0.5 * K.log(2 * np.pi * self.gmm_log_var) + K.square(Z_expand - self.gmm_mu) / (
                    2 * self.gmm_log_var),
            axis=1)
        # then subtract batch x components from batch x components
        gamma = K.exp(K.log(cat_pi) - gamma) + 1e-10
        # normalise
        gamma /= K.sum(gamma, axis=1, keepdims=True)
        # return batch x components
        return gamma

    def loss(self, Z_mu, Z_log_var, gamma):
        cat_pi = self.cat_pi_softmax(self.cat_pi)
        Z_mu_expand = K.repeat_elements(K.expand_dims(Z_mu, -1), self.n_components, 2)
        Z_log_var_expand = K.repeat_elements(K.expand_dims(Z_log_var, -1), self.n_components,
                                             2)
        # VaDE loss
        # ELBO - see equation 20 in Appendix C of original paper
        # For signs - see signs in equation 20 then signs in individual terms then flip for loss
        # i.e. all become positive except log_pc and q_entropy
        # log p(z|c)
        # first part - gives float - forgoes J / 2 per summing later
        h = K.log(np.pi * 2)
        # second part - gives batch x latent x components - foregoes J/2 per summing later
        # Z_mu vs. Z allows backpropogation
        h += K.log(self.gmm_log_var) + K.exp(Z_log_var_expand) / self.gmm_log_var + K.square(
            Z_mu_expand - self.gmm_mu) / self.gmm_log_var
        # sum over the latent dimensions
        h = K.sum(h, axis=1)  # batch x latent x components -> batch x components
        # Sum over the components
        log_pzc = K.sum(gamma * 0.5 * h, axis=1)  # batch x components -> batch
        # log q(z|x) -> sum across latent dimensions
        # The VaDE paper includes the pi term but the VaDE code repo omits the pi term - mistake mentioned by others?
        # Note use of self.Z_log_var, i.e. not version with added dimension for components
        q_entropy = -0.5 * K.sum(K.log(2 * np.pi) + 1 + Z_log_var,
                                 axis=1)  # batch x latent -> batch
        # log p(c) -> sum across components
        # author repo shows gamma inside log vs.paper and other implementations which do not
        log_pc = -K.sum(gamma * K.log(cat_pi + 1e-30), axis=1)  # batch x components -> batch
        # log q(c|x) -> sum across components
        log_qcx = K.sum(gamma * K.log(gamma + 1e-30), axis=1)  # batch x components -> batch
        return h, log_pzc, q_entropy, log_pc, log_qcx

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_components': self.n_components,
            'latent_dim': self.latent_dim,
            'cat_pi': self.cat_pi,
            'gmm_mu': self.gmm_mu,
            'gmm_log_var': self.gmm_log_var,
            'loss': self.loss
        })
        return config


'''
Paper:
https://arxiv.org/abs/1611.05148
Code repo for VaDE paper (keras - but not very canonical...?)
https://github.com/slim1017/VaDE/blob/master/VaDE.py
Claim mistakes were made in above (torch)
https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py
Similar but clearer implementation (torch)
https://github.com/IoannisStournaras/Deep-Learning-for-Deconvolution-of-scRNA-seq-Data/blob/master/Networks/VADE.py
can't follow changes:
https://github.com/scespinoza/sc-autoencoders/blob/fe62606f96c3890875c41dd4867da7db7469304f/models.py#L577
Keras implementation
https://github.com/OwalnutO/deep-generative-models/blob/master/VaDE/run_VaDE.py
'''


class VaDE(models.Model):
    def __init__(self,
                 raw_dim: int,
                 latent_dim: int,
                 n_components: int,
                 theme_base='VaDE',
                 seed=0,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 dropout: float = 0.0,
                 **kwargs):
        tf.random.set_seed(seed)
        logger.info('initialising VaDE...')
        super().__init__(**kwargs)
        dropout_text = str(dropout).replace('.', '_')
        self.theme = f'{theme_base}_s{seed}_d{latent_dim}_comp{n_components}_do{dropout_text}'
        self.n_components = n_components
        self.encoder = DeepLayer(name='DeepEncoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init,
                                 dropout=dropout)
        self.sampling = SamplingLayer(latent_dim,
                                      seed,
                                      initializers.GlorotNormal(seed=seed),
                                      name='Sampling')
        self.gamma = Gamma(n_components, latent_dim, seed)
        self.decoder = DeepLayer(name='DeepDecoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init,
                                 dropout=dropout)
        self.x_hat = layers.Dense(raw_dim, activation='linear', name=f'OutputLayer')
        # setup metrics
        self.summary_metrics = {}

    def call(self, inputs, training=True, **kwargs):
        Z_mu, Z_log_var, Z = self.encode(inputs, training=training)
        gamma = self.gamma(Z)
        h, log_pzc, q_entropy, log_pc, log_qcx = self.gamma.loss(Z_mu, Z_log_var, gamma)
        # sum losses, then take mean over batches
        self.add_loss(K.mean(log_pzc + q_entropy + log_pc + log_qcx))
        X_hat = self.decode(Z, training=training)
        rec_loss = mse_sum(inputs, X_hat)
        self.add_loss(rec_loss)
        # update summary metrics
        for key, metric in zip(
                ['gamma', 'h', 'log_pzc', 'q_entropy', 'log_pc', 'log_qcx', 'rec_loss'],
                [gamma, h, log_pzc, q_entropy, log_pc, log_qcx, rec_loss]):
            if key not in self.summary_metrics:
                self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
            self.summary_metrics[key](metric)
        return X_hat

    def encode(self, X, training=False):
        X = self.encoder(X, training=training)
        Z_mu, Z_log_var, Z = self.sampling(X, training=training)
        return Z_mu, Z_log_var, Z

    def decode(self, Z, training=False):
        X = self.decoder(Z, training=training)
        X_hat = self.x_hat(X, training=training)
        return X_hat

    def classify(self, inputs):
        Z_mu, Z_log_var, Z = self.encode(inputs, training=False)
        # no sampling required so return Z_mu
        return self.gamma(Z_mu, training=False)

    def prime_GMM(self, cat_prior, mu_prior, cov_prior):
        K.set_value(self.gamma.cat_pi, cat_prior)  # debatable
        K.set_value(self.gamma.gmm_mu, mu_prior.T)
        K.set_value(self.gamma.gmm_log_var, cov_prior.T)

    def get_config(self):
        config = super().get_config()
        config.update({
            'theme': self.theme,
            'n_components': self.n_components,
            'encoder': self.encoder,
            'gamma': self.gamma,
            'sampling': self.sampling,
            'decoder': self.decoder,
            'x_hat': self.x_hat,
            'encode': self.encode,
            'decode': self.decode,
            'classify': self.classify,
            'prime_GMM': self.prime_GMM
        })
        return config
