'''
Based on paper: https://arxiv.org/pdf/1611.02648.pdf
And accompanying torch code: https://github.com/Nat-D/GMVAE/compare/version2
NB - Note version 2 on github repo!!
'''
import logging

import numpy as np
import tensorflow as tf
from src.explore.signatures.sig_models import DeepLayer, SplitVaeEncoder
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, metrics, initializers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianSampler(layers.Layer):

    def __init__(self, n_MC, seed):
        super().__init__()
        self.n_MC = n_MC
        self.seed = seed

    def call(self, inputs, **kwargs):
        mu, log_var = inputs
        batch = K.int_shape(mu)[0]
        dim = K.int_shape(mu)[1]
        mu_rep = K.repeat_elements(K.expand_dims(mu, 0), self.n_MC, 0)
        epsilon = K.random_normal(shape=(self.n_MC, batch, dim), mean=0.0, stddev=1.0, seed=self.seed)
        std = K.exp(0.5 * log_var)
        std_rep = K.repeat_elements(K.expand_dims(std, 0), self.n_MC, 0)
        sampled = mu_rep + std_rep * epsilon
        return sampled

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_MC': self.n_MC,
            'seed': self.seed
        })
        return config


class QXLayer(layers.Layer):

    def __init__(self, x_dim, seed, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = x_dim
        self.qx_mu = layers.Dense(x_dim,
                                  activation='linear',
                                  kernel_initializer=initializers.GlorotNormal(seed=seed),
                                  name='QX_mu')
        self.qx_log_var = layers.Dense(x_dim,
                                       activation='linear',
                                       kernel_initializer=initializers.Zeros(),
                                       name='QX_log_var')

    def call(self, inputs, training=True, **kwargs):
        qx_mu = self.qx_mu(inputs, training=training)
        qx_log_var = self.qx_log_var(inputs, training=training)
        return [qx_mu, qx_log_var]

    def get_config(self):
        config = super().get_config()
        config.update({
            'x_dim': self.x_dim,
            'qx_mu': self.qx_mu,
            'qx_log_var': self.qx_log_var
        })
        return config


class QWLayer(layers.Layer):

    def __init__(self, w_dim, seed, **kwargs):
        super().__init__(**kwargs)
        self.w_dim = w_dim
        self.qw_mu = layers.Dense(w_dim,
                                  activation='linear',
                                  kernel_initializer=initializers.GlorotNormal(seed=seed),
                                  name='QW_mu')
        self.qw_log_var = layers.Dense(w_dim,
                                       activation='linear',
                                       kernel_initializer=initializers.Zeros(),
                                       name='QW_log_var')

    def call(self, inputs, training=True, **kwargs):
        qw_mu = self.qw_mu(inputs, training=training)
        qw_log_var = self.qw_log_var(inputs, training=training)
        return [qw_mu, qw_log_var]

    def get_config(self):
        config = super().get_config()
        config.update({
            'w_dim': self.w_dim,
            'qw_mu': self.qw_mu,
            'qw_log_var': self.qw_log_var
        })
        return config


class YGenerator(layers.Layer):

    def __init__(self,
                 raw_dim,
                 seed,
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 hidden_layer_dims: tuple = (256, 256, 256),
                 **kwargs):
        super().__init__(**kwargs)
        self.raw_dim = raw_dim
        self.decoder = DeepLayer(name='DeepEncoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init)
        self.y_mu = layers.Dense(raw_dim,
                                 activation='linear',
                                 kernel_initializer=initializers.GlorotNormal(seed=seed),
                                 name='YMean')
        self.y_log_var = layers.Dense(raw_dim,
                                      activation='softplus',  # explodes if linear
                                      kernel_initializer=initializers.Zeros(),
                                      name='YLogVar')

    def call(self, inputs, training=True, **kwargs):
        y = self.decoder(inputs, training=training)
        y_mu = self.y_mu(y, training=training)
        y_log_var = self.y_log_var(y, training=training)
        return y_mu, y_log_var

    def get_config(self):
        config = super().get_config()
        config.update({
            'raw_dim': self.raw_dim,
            'decoder': self.decoder,
            'y_mu': self.y_mu,
            'y_log_var': self.y_log_var
        })
        return config


class PXZLayer(layers.Layer):
    '''
    Per tensorflow docs on Dense layers:
    If the input to the layer has a rank greater than 2, then Dense computes the dot product between the inputs and the
    kernel along the last axis of the inputs and axis 1 of the kernel (using tf.tensordot). For example, if input has
    dimensions (batch_size, d0, d1), then we create a kernel with shape (d1, units), and the kernel operates along axis
    2 of the input, on every sub-tensor of shape (1, 1, d1) (there are batch_size * d0 such sub-tensors). The output in
    this case will have shape (batch_size, d0, units).
    '''

    def __init__(self, n_MC, k_components, x_dim, n_hidden, seed, kernel_init='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.n_MC = n_MC
        self.k_components = k_components
        self.x_dim = x_dim
        self.pxz_feed = layers.Dense(n_hidden,
                                     kernel_initializer=kernel_init,
                                     activation='swish',  # interesting
                                     name='PriorFeed')
        self.mu_dense = layers.Dense(x_dim * k_components,
                                     activation='linear',
                                     kernel_initializer=initializers.GlorotNormal(seed=seed),
                                     name='PriorMu')
        self.log_var_dense = layers.Dense(x_dim * k_components,
                                          activation='softplus',
                                          kernel_initializer=initializers.Zeros(),
                                          name='PriorLogVar')

    def call(self, inputs, training=True, **kwargs):
        pxz = self.pxz_feed(inputs, training=training)  # M x N x hidden
        # mu
        pxz_mu = self.mu_dense(pxz, training=training)  # M x N x (X x K)
        pxz_mu = K.reshape(pxz_mu, (self.n_MC, -1, self.x_dim, self.k_components))  # M x N x X x K
        pxz_mu = K.permute_dimensions(pxz_mu, (3, 0, 1, 2))  # K x M x N x X
        # log var
        pxz_log_var = self.log_var_dense(pxz, training=training)  # N x (X x K)
        pxz_log_var = K.reshape(pxz_log_var, (self.n_MC, -1, self.x_dim, self.k_components))  # M x N x X x K
        pxz_log_var = K.permute_dimensions(pxz_log_var, (3, 0, 1, 2))  # K x M x N x X
        return [pxz_mu, pxz_log_var]

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_MC': self.n_MC,
            'k_components': self.k_components,
            'x_dim': self.x_dim,
            'pxz_feed': self.pxz_feed,
            'mu_dense': self.mu_dense,
            'log_var_dense': self.log_var_dense,
        })
        return config


class GaussianCriterion(layers.Layer):
    '''
    log(sigma) + 0.5 * (2pi)) + 0.5 * (x - mu)^2/sigma^2
    '''

    def __init__(self, n_MC, **kwargs):
        super().__init__(**kwargs)
        self.n_MC = n_MC

    def call(self, inputs, **kwargs):
        y, y_mu, y_log_var = inputs
        # replicates by n_MC to match y_mu / y_log_var dimensionality
        y_rep = K.repeat_elements(K.expand_dims(y, 0), self.n_MC, 0)
        # gaussian criterion
        gc = 0.5 * y_log_var + 0.5 * K.log(2 * np.pi) + 0.5 * K.square(y_rep - y_mu) / K.exp(y_log_var)
        gc = K.sum(gc, axis=2)  # sum over raw dimensions: M x N x Raw -> M x N
        gc = K.mean(gc, axis=1)  # take mean over batch -> M
        return K.mean(gc)  # take mean over monte carlo samples

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_MC': self.n_MC
        })
        return config


class ExpectedKLDivergence(layers.Layer):
    '''
    E_z_w[KL(q(x)|| p(x|z,w))]
    KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2) / var2  - 1 )
    '''

    def __init__(self, n_MC, x_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_MC = n_MC
        self.x_dim = x_dim

    def call(self, inputs, **kwargs):
        # pz: N x K
        # qx_mu / qx_log_var: N x X
        # pxz_mu / pxz_log_var: K x M x N x X
        pz, qx_mu, qx_log_var, pxz_mu, pxz_log_var = inputs
        # replicates by n_MC to match pxz_mu / pxz_log_var dimensionality
        qx_mu_rep = K.repeat_elements(K.expand_dims(qx_mu, 0), self.n_MC, 0)  # M x N x X
        qx_log_var_rep = K.repeat_elements(K.expand_dims(qx_log_var, 0), self.n_MC, 0)  # M x N x X
        kld = K.square(qx_mu_rep - pxz_mu) + K.exp(qx_log_var_rep)
        kld /= K.exp(pxz_log_var)
        kld -= 1
        kld += pxz_log_var - qx_log_var_rep
        kld *= 0.5
        pz_rep = K.repeat_elements(K.expand_dims(pz, -1), self.x_dim, axis=-1)  # M x N x K
        kld = pz_rep * kld  # K x M x N x X
        kld = K.sum(kld, axis=0)  # sum over K -> M x N x X
        kld = K.sum(kld, axis=2)  # sum over X -> M x N
        kld = K.mean(kld, axis=1)  # take mean over batch -> M
        return K.mean(kld)  # take mean over monte carlo samples

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_MC': self.n_MC,
            'x_dim': self.x_dim
        })
        return config


class KLDCriterion(layers.Layer):
    '''
    KL( q(w) || P(w) )
    Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        qw_mu, qw_log_var = inputs  # N x W
        kld = 1 + qw_log_var - K.square(qw_mu) - K.exp(qw_log_var)
        kld = -0.5 * K.sum(kld, axis=1)  # sum over W
        return K.mean(kld)  # take mean over batch


class Likelihood(layers.Layer):
    '''
    -0.5 sum_d { (x_i - mu_i)^2/var_i } - 1/2 sum_d (logVar_i) - D/2 ln(2pi) [N]
    '''

    def __init__(self, x_dim, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = x_dim

    def call(self, inputs, **kwargs):
        # px: M x N x X
        # pxz_mu / pxz_log_var: K x M x N x X
        qx, pxz_mu, pxz_log_var = inputs
        # gaussian log likelihood
        llh = K.square(qx - pxz_mu) / K.exp(pxz_log_var)  # K x M x N x X
        llh = K.sum(llh, axis=3)  # sum over X -> K x M x N
        llh += K.sum(pxz_log_var, axis=3)  # sum over X -> K x M x N
        llh += self.x_dim * K.log(2 * np.pi)
        llh *= -0.5
        return llh  # K x M x N

    def get_config(self):
        config = super().get_config()
        config.update({
            'x_dim': self.x_dim
        })
        return config


class EntropyCriterion(layers.Layer):
    '''
    H(Z|X) = E_q(x)E_p(z|x)[- log P(z|x)]
    '''

    def __init__(self, n_MC, **kwargs):
        super().__init__(**kwargs)
        self.n_MC = n_MC

    def call(self, pz, **kwargs):
        cv = K.log(pz + 1e-10) * pz  # K x M x N
        cv = -K.sum(cv, axis=0)  # sum over K -> M x N
        cv = K.mean(cv, axis=1)  # take mean over batch
        return K.mean(cv)  # take mean over monte carlo samples

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_MC': self.n_MC
        })
        return config


class GMVAE(models.Model):

    def __init__(self,
                 raw_dim: int,
                 x_dim: int,
                 w_dim: int,
                 k_components: int,
                 split_input_dims: tuple,
                 split_latent_dims: tuple,
                 split_hiddenen_layer_dims: tuple,
                 n_MC: int = 1,
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 hidden_layer_dims: tuple = (256, 256, 256),
                 theme_base='GMVAE',
                 seed=0,
                 **kwargs):
        logger.info('initialising VAE...')
        self.seed = seed
        tf.random.set_seed(self.seed)
        super().__init__(**kwargs)
        self.theme = f'{theme_base}_x{x_dim}_w{w_dim}_k{k_components}_m{n_MC}'
        self.k_components = k_components
        self.encoder = SplitVaeEncoder(name='DeepEncoder',
                                       split_input_dims=split_input_dims,
                                       split_latent_dims=split_latent_dims,
                                       split_hidden_layer_dims=split_hiddenen_layer_dims)
        self.merged = layers.Concatenate(axis=-1, name='Xmerged')
        self.qx = QXLayer(x_dim, seed, name='QX')
        self.qx_sampler = GaussianSampler(n_MC, self.seed)
        self.qw = QWLayer(w_dim, seed, name='QW')
        self.qw_sampler = GaussianSampler(n_MC, self.seed)
        self.decoder = YGenerator(raw_dim, seed, activation_func, kernel_init, hidden_layer_dims)
        self.pxz = PXZLayer(n_MC, k_components, x_dim, hidden_layer_dims[-1], seed, kernel_init)
        self.recon_criterion = GaussianCriterion(n_MC)
        self.expected_kld = ExpectedKLDivergence(n_MC, x_dim)
        self.kld_criterion = KLDCriterion()
        self.likelihood = Likelihood(x_dim)
        self.entropy_criterion = EntropyCriterion(n_MC)
        # setup metrics
        self.summary_metrics = {}

    def call(self, X, training=True, **kwargs):
        X_a, X_b, X_c = self.encoder(X, training=training)
        X_merged = self.merged([X_a, X_b, X_c])
        qx_mu, qx_log_var = self.qx(X_merged, training=training)  # N x X
        qx_sample = self.qx_sampler([qx_mu, qx_log_var])  # M x N x X
        qw_mu, qw_log_var = self.qw(X_merged, training=training)  # N x W
        qw_sample = self.qw_sampler([qw_mu, qw_log_var])  # M x N x W
        y_mu, y_log_var = self.decoder(qx_sample, training=training)  # M x N x Raw
        pxz_mu, pxz_log_var = self.pxz(qw_sample, training=training)  # K x M x N x X
        pz = self.likelihood([qx_sample, pxz_mu, pxz_log_var])  # K x M x N
        pz = K.softmax(pz, axis=0)
        # reconstruction term
        # -E[logP(y|x)]
        recon_loss = self.recon_criterion([X, y_mu, y_log_var])
        self.add_loss(recon_loss)
        # conditional prior term - equation 5
        # E_z_w[KL(q(x)|| p(x|z,w))]
        exp_kld_loss = self.expected_kld([pz, qx_mu, qx_log_var, pxz_mu, pxz_log_var])
        self.add_loss(exp_kld_loss)
        # w-prior term
        # KL( q(w) || P(w) )
        kld_loss = self.kld_criterion([qw_mu, qw_log_var])
        self.add_loss(kld_loss)
        # z-prior term
        # E[KL(P(z|x,w)||P(z))] = E[ E[logP(z|x,w)] - E[logP(z)] ] = - CV + constant
        pz_h = self.entropy_criterion(pz)
        self.add_loss(pz_h)
        # update summary metrics
        for key, metric in zip(['rec_loss', 'exp_kld_loss', 'kld_loss', 'CV'],
                               [recon_loss, exp_kld_loss, kld_loss, pz_h]):
            if key not in self.summary_metrics:
                self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
            self.summary_metrics[key](metric)
        # average over the monte carlo samples
        return K.mean(y_mu, axis=0)

    def encode(self, X, training=False):
        X_a, X_b, X_c = self.encoder(X, training=training)
        X_merged = self.merged([X_a, X_b, X_c])
        qx_mu, qx_log_var = self.qx(X_merged)
        return qx_mu.numpy()

    def decode(self, X, training=False):
        y_mu, y_log_var = self.decoder(X, training=training)
        return K.mean(y_mu, axis=0).numpy()  # take mean over monte carlo samples

    def classify(self, X, training=False):
        X_a, X_b, X_c = self.encoder(X, training=training)
        X_merged = self.merged([X_a, X_b, X_c])
        qx_mu, qx_log_var = self.qx(X_merged, training=training)  # N x X
        qx_sample = self.qx_sampler([qx_mu, qx_log_var])  # M x N x X
        qw_mu, qw_log_var = self.qw(X_merged, training=training)  # N x W
        qw_sample = self.qw_sampler([qw_mu, qw_log_var])  # M x N x W
        pxz_mu, pxz_log_var = self.pxz(qw_sample, training=training)  # K x M x N x X
        pz = self.likelihood([qx_sample, pxz_mu, pxz_log_var])  # K x M x N
        pz = K.softmax(pz, axis=0)
        pz = K.mean(pz, axis=1)
        pz = K.permute_dimensions(pz, (1, 0))
        return pz.numpy()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.w_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'theme': self.theme,
            'k_components': self.k_components,
            'encoder': self.encoder,
            'merged': self.merged,
            'qx': self.qx,
            'qx_sampler': self.qx_sampler,
            'qw': self.qw,
            'qw_sampler': self.qw_sampler,
            'decoder': self.decoder,
            'pxz': self.pxz,
            'recon_criterion': self.recon_criterion,
            'expected_kld': self.expected_kld,
            'kld_criterion': self.kld_criterion,
            'likelihood': self.likelihood,
            'entropy_criterion': self.entropy_criterion,
            'encode': self.encode,
            'decode': self.decode,
            'classify': self.classify
        })
        return config
