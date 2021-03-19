"""
Based on blog post: http://ruishu.io/2016/12/25/gmvae/
And code: https://github.com/RuiShu/vae-clustering
"""
import logging

import numpy as np
import tensorflow as tf
from src.explore.signatures.sig_models import DeepLayer
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianSampler(layers.Layer):

    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def call(self, inputs, **kwargs):
        mu, log_var = inputs
        k_components = K.int_shape(mu)[0]
        batch = K.int_shape(mu)[1]
        dim = K.int_shape(mu)[2]
        epsilon = K.random_normal(shape=(k_components, batch, dim), mean=0.0, stddev=1.0, seed=self.seed)
        std = K.exp(0.5 * log_var)
        sampled = mu + std * epsilon
        return sampled

    def get_config(self):
        config = super().get_config()
        config.update({
            'seed': self.seed
        })
        return config


class QY(layers.Layer):

    def __init__(self,
                 k_components,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.qy_deep = DeepLayer(name='DeepClass',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func)
        self.logit = layers.Dense(k_components, name='Logit')
        self.softmax = layers.Softmax(axis=-1, name='Softmax')

    def call(self, inputs, training=True, **kwargs):
        qy = self.qy_deep(inputs, training=training)
        qy_logit = self.logit(qy, training=training)
        qy_softmax = self.softmax(qy_logit)
        return qy_logit, qy_softmax

    def get_config(self):
        config = super().get_config()
        config.update({
            'qy_deep': self.qy_deep,
            'logit': self.logit,
            'softmax': self.softmax
        })
        return config


class QZ(layers.Layer):

    def __init__(self,
                 k_components: int,
                 latent_dim: int,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'relu',
                 seed: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.k_components = k_components
        self.concat = layers.Concatenate(axis=-1, name='Concat')
        self.qz_deep = DeepLayer(name='DeepEncoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func)
        self.z_mu = layers.Dense(latent_dim, name='QZMu')
        self.z_log_var = layers.Dense(latent_dim, activation='softplus', name='QZLogVar')
        self.sample = GaussianSampler(seed=seed)

    def call(self, inputs, training=True, **kwargs):
        X, y = inputs
        X = K.repeat_elements(K.expand_dims(X, axis=0), self.k_components, axis=0)
        xy = self.concat([X, y])
        xy = self.qz_deep(xy, training=training)
        qz_mus = self.z_mu(xy, training=training)
        qz_log_vars = self.z_log_var(xy, training=training)
        qz_samples = self.sample([qz_mus, qz_log_vars])
        return qz_samples, qz_mus, qz_log_vars

    def get_config(self):
        config = super().get_config()
        config.update({
            'k_components': self.k_components,
            'concat': self.x_dim,
            'qz_deep': self.qz_deep,
            'zm': self.qy_deep,
            'zv': self.logit,
            'sample': self.softmax
        })
        return config


class PX(layers.Layer):

    def __init__(self,
                 k_components: int,
                 raw_dim: int,
                 latent_dim: int,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.k_components = k_components
        self.px_deep = DeepLayer(name='DeepEncoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func)
        self.zm_prior = layers.Dense(latent_dim, name='PXMu')
        self.zv_prior = layers.Dense(latent_dim, activation='softplus', name='PXLogVar')
        self.px_logit = layers.Dense(raw_dim, activation='linear', name='PXLogit')

    def call(self, inputs, training=True, **kwargs):
        qz_sample, y = inputs
        zm_priors = self.zm_prior(y, training=training)
        zv_priors = self.zv_prior(y, training=training)
        px = self.px_deep(qz_sample, training=training)
        px_logits = self.px_logit(px)
        return zm_priors, zv_priors, px_logits

    def get_config(self):
        config = super().get_config()
        config.update({
            'k_components': self.k_components,
            'px_deep': self.px_deep,
            'px_mu': self.px_mu,
            'px_log_var': self.px_log_var,
            'px_logit': self.px_logit
        })
        return config


def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = K.log(1.0 - eps) - K.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val, name='clipped_logit')
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), axis)


def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = var + eps
    return -0.5 * tf.reduce_sum(K.log(2 * np.pi) + K.log(var) + K.square(x - mu) / var, axis)


class LabelledLoss(layers.Layer):

    def __init__(self, k_components: int, eps: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.k_components = k_components
        self.eps = eps

    def call(self, inputs, **kwargs):
        X, px_logits, qz_samples, qz_mus, qz_log_vars, zm_priors, zv_priors = inputs
        # losses = []
        # for i in range(self.k_components):
        X = K.repeat_elements(K.expand_dims(X, axis=0), self.k_components, axis=0)
        xy_loss = -log_bernoulli_with_logits(X, px_logits)
        xy_loss += log_normal(qz_samples, qz_mus, qz_log_vars)
        xy_loss -= log_normal(qz_samples, zm_priors, zv_priors)
        # losses.append(xy_loss)
        # losses = K.stack(losses, axis=1)
        return xy_loss - np.log(0.1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'k_components': self.k_components,
            'eps': self.eps
        })
        return config


class GMVAERUI(models.Model):

    def __init__(self,
                 raw_dim,
                 latent_dim,
                 k_components,
                 hidden_layer_dims: tuple = (256, 256, 256),
                 activation_func: str = 'relu',
                 seed: int = 0,
                 theme_base='GMVAERUI',
                 **kwargs):
        logger.info('initialising VAE...')
        self.k_components = k_components
        self.seed = seed
        tf.random.set_seed(self.seed)
        super().__init__(**kwargs)
        self.theme = f'{theme_base}_x{latent_dim}_k{k_components}'
        self.qy = QY(k_components, hidden_layer_dims, activation_func)
        self.qz = QZ(k_components, latent_dim, hidden_layer_dims, activation_func, seed=seed)
        self.px = PX(k_components, raw_dim, latent_dim, hidden_layer_dims, activation_func)
        self.labelled_loss = LabelledLoss(k_components)
        self.diag = tf.eye(k_components)

    def call(self, X, training=True, **kwargs):
        batch = K.int_shape(X)[0]
        y = tf.zeros(shape=(batch, self.k_components, self.k_components))
        y += self.diag  # broadcast diagonal into y placeholder
        y = K.permute_dimensions(y, (1, 0, 2))
        # propose distribution over y
        qy_logit, qy_softmax = self.qy(X, training=training)
        # for each proposed y, infer z and reconstruct x
        qz_samples, qz_mus, qz_log_vars = self.qz([X, y], training=training)
        zm_priors, zv_priors, px_logits = self.px([qz_samples, y], training=training)
        if training:
            # losses
            neg_xh_loss = tf.nn.softmax_cross_entropy_with_logits(qy_softmax, qy_logit)
            self.add_metric(K.mean(neg_xh_loss), aggregation='mean', name='neg_xh_loss')
            gm_loss = -self.labelled_loss([X, px_logits, qz_samples, qz_mus, qz_log_vars, zm_priors, zv_priors])
            gm_loss = K.permute_dimensions(gm_loss, (1, 0))
            gm_loss *= qy_softmax
            gm_loss = K.mean(gm_loss, axis=1)  # mean over K
            self.add_metric(K.mean(gm_loss), aggregation='mean', name='gm_loss')
            combined_loss = neg_xh_loss + gm_loss
            combined_loss = K.mean(combined_loss)  # mean over batch
            self.add_loss(combined_loss)
            self.add_metric(combined_loss, aggregation='mean', name='combined_loss')
        return qy_softmax

    def get_config(self):
        config = super().get_config()
        config.update({
            'k_components': self.k_components,
            'seed': self.seed,
            'theme': self.theme,
            'qy': self.qy,
            'qz': self.qz,
            'px': self.px,
            'labelled_loss': self.labelled_loss
        })
        return config
