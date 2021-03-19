'''
Models used by other scripts in this module
'''

import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, metrics, losses, initializers

from src.explore.signatures.sig_models import DeepLayer, SamplingLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandUsePredictor(models.Model):
    '''
    Predicts landuses from centralities and census data
    '''

    def __init__(self,
                 theme_base='mixed_pred',
                 hidden_layer_dims=(256, 256, 256),
                 activation_func='swish',
                 seed=0,
                 **kwargs):
        tf.random.set_seed(seed)
        logger.info('intialising regressor...')
        super().__init__(**kwargs)
        self.theme = f'{theme_base}_s{seed}'
        for hidden_dims in hidden_layer_dims:
            self.theme += f'_x{hidden_dims}'
        self.predictor = DeepLayer(hidden_layer_dims=hidden_layer_dims,
                                   activation_func=activation_func,
                                   kernel_init='glorot_uniform',
                                   dropout=0.1,
                                   seed=seed)
        self.x_hat = layers.Dense(1, activation='linear', name='XHat')

    def call(self, inputs, training=True, **kwargs):
        X = self.predictor(inputs, training=training)
        X_hat = self.x_hat(X, training=training)
        return X_hat

    def get_config(self):
        config = super().get_config()
        config.update({
            'theme': self.theme,
            'predictor': self.predictor,
            'x_hat': self.x_hat
        })
        return config


class NewTownClassifier(models.Model):
    '''
    Stand-alone predictive classifier based on outputs from step 1
    Aim is to create a predictive model that classifies new vs. historic towns at higher res
    '''

    def __init__(self,
                 theme_base='classifier',
                 rep_dims=8,
                 seed=0,
                 hidden_layer_dims=(256, 256, 256),
                 activation_func='swish',
                 dropout=0.5,
                 **kwargs):
        tf.random.set_seed(seed)
        logger.info('intialising classifier...')
        super().__init__(**kwargs)
        self.theme = f'{theme_base}_s{seed}_rd{rep_dims}_do{dropout}'
        for hidden_dims in hidden_layer_dims:
            self.theme += f'_x{hidden_dims}'
        self.predictor = DeepLayer(hidden_layer_dims=hidden_layer_dims,
                                   activation_func=activation_func,
                                   kernel_init='glorot_uniform',
                                   dropout=dropout,
                                   seed=seed)
        self.consolidator = layers.Dense(rep_dims, activation='linear', name='Consolidator')
        self.sigmoid = layers.Dense(1, activation='sigmoid', name='Squashed')

    def call(self, inputs, training=True, **kwargs):
        X = self.predictor(inputs, training=training)
        X = self.consolidator(X, training=training)
        X_hat = self.sigmoid(X, training=training)
        return X_hat

    def pre_hat(self, inputs):
        X = self.predictor(inputs, training=False)
        X = self.consolidator(X, training=False)
        return X

    def get_config(self):
        config = super().get_config()
        config.update({
            'theme': self.theme,
            'predictor': self.predictor,
            'consolidator': self.consolidator,
            'sigmoid': self.sigmoid,
            'pre_hat': self.pre_hat
        })
        return config


class M2Likelihood(layers.Layer):

    def __init__(self, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.bce = losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=True, **kwargs):
        X, Z_mu, Z_log_var, y_lab_oh, X_hat_mu, X_hat_log_var = inputs
        # logpy - OK
        y_prior = 1 / self.n_classes * tf.ones(shape=tf.shape(y_lab_oh))  # batch x classes
        logpy = - self.bce(y_prior, y_lab_oh)
        # logpx - OK
        logpx = -0.5 * np.log(2 * np.pi)
        # some cases where excessively small or large X_hat_log_var causes problematic values
        # in float 32 overflow:
        # taking the exponential at less than -100 gives zero
        # whereas values greater than +80 give infinity
        logpx = logpx - 0.5 * (X_hat_log_var + K.square(X - X_hat_mu) / K.exp(X_hat_log_var))
        logpx = K.sum(logpx, axis=1)  # batch x dimensions -> sum across dimensions
        # kingma
        # logpz = -0.5 * (np.log(2 * np.pi) + (K.square(Z_mu) + K.exp(Z_log_var)))
        # return (logpx + logpy) + logpz
        # bjlkeng (see above) and response 777 use factored KLD version
        kld = -0.5 * (1 + Z_log_var - K.square(Z_mu) - K.exp(Z_log_var))  # batch x latents
        kld = K.sum(kld, axis=1)  # sum across latents
        return logpx + logpy - kld


class M2(models.Model):
    '''
    Semi-supervised M2 classifier per Kingma paper.

    Inputs must include a y classification column.
    Where samples have y=-1 these will be treated as unlabelled.

    See the following implementations for examples of code implementations:
    - https://github.com/dpkingma/nips14-ssl
    - http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    - https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-semi_supervised_learning
    - https://github.com/Response777/Semi-supervised-VAE
    - https://github.com/saemundsson/semisupervised_vae

    bjlkeng blog: objective is:
    log p(x) >= E[log p(x|y, z) + log p(y) + log p(z) - log q(y, z|x)]
        the log q(y, z|x) term is factored into y and z parts
        with log p(z) and log q(z|x) then rearranged into KL divergence between p and q of z
        the log p(y) term is effectively a constant because of multinomial prior
              = log p(x|y, z) + log p(y) - KL[q(z|x)||p(z)] - log q(y|x)

    Author's repo states:
    lower bound L(x) = logpx + logpz
    - z is computed with g(.) from eps and variational parameters
    - let logpx be the generative model density: log p(y) + log p(x|y,z) where z=g(.)
    - let logpz be the prior of Z plus the entropy of q(z|x,y): logp(z) + H_q(z|x)
    - loq q(z|x) - entropy of z

    L = (logpx + logpy) + logpz - logqz + logqy

    # logpy = categorical cross entropy between y prior and y labels
    # logpx depends on distribution - see author's anglepy/logpdfs.py
    # guassian:
    logpx = -0.5 * np.log(2 * np.pi) - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))
    ### vs. Response777 which factors out /2
    logpx = -0.5 * np.log(2 * np.pi) - 0.5 * (logvar + tf.square(x - mu) / tf.exp(logvar) )
    # combine logpx with logpy
    logpx += logpy

    # logpz depends on distribution - see author's anglepy/logpdfs.py
    # gaussian marginal
    ### used in saemundsson implementation
    logpz = -0.5 * (np.log(2 * np.pi) + (K.square(q_mean) + K.exp(q_logvar)))
    # vs. gaussian:
    logpz = -0.5 * np.log(2 * np.pi) - K.square(q_mean) / 2
    ### vs. Response777 repo: uses KLD
    KLD = -0.5 * (1 + q_logvar - K.square(q_mean) - K.exp(q_logvar))

    # logqz depends on distribution - see author's anglepy/logpdfs.py
    # guassian marginal
    ### used in saemundsson implementation
    logqz = -0.5 * (np.log(2 * np.pi) + 1 + q_logvar)
    # gaussian
    logqz = -0.5 * np.log(2 * np.pi) - q_logvar / 2 - K.square(x - q_mean) / (2 * K.exp(q_logvar))
          = -0.5 * (np.log(2 * np.pi) - q_logvar - K.square(x - q_mean) / K.exp(q_logvar))

    # logqy is computed in the parent class and differs between labelled and unlabelled
    # neither implementation is entirely clear in author's repo...
    # Response777 uses:
    ## labelled: adds categorical cross entropy (between predicted and actual)
    ## unlabelled: uses H term (y_prob * (L_U - log(y_prob))

    # weighting is applied per number of samples, e.g.
    beta = alpha * (1 * n_samples / n_labels)
    logqy = beta * categorical cross entropy?

    '''

    def __init__(self,
                 raw_dim: int,
                 latent_dim: int,
                 n_samples: int,
                 theme_base: str,
                 seed=0,
                 hidden_layer_dims: tuple = (256, 256, 256),  # encoder / decoder
                 activation_func: str = 'swish',
                 kernel_init: str = 'glorot_uniform',
                 dropout=0.5,
                 **kwargs):
        tf.random.set_seed(seed)
        logger.info('initialising M2...')
        super().__init__(**kwargs)
        self.theme = f'{theme_base}_l{latent_dim}_s{seed}_do{dropout}'
        self.n_samples = n_samples
        self.n_classes = 2  # binary is a given
        self.encoder = DeepLayer(name='Encoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init,
                                 dropout=dropout,
                                 seed=seed)
        self.sampling = SamplingLayer(latent_dim,
                                      seed,
                                      initializers.TruncatedNormal(stddev=0.001, seed=seed),
                                      name='Sampling')
        self.decoder = DeepLayer(name='Decoder',
                                 hidden_layer_dims=hidden_layer_dims,
                                 activation_func=activation_func,
                                 kernel_init=kernel_init,
                                 dropout=dropout,
                                 seed=seed)
        self.x_hat_mu = layers.Dense(raw_dim,
                                     activation='linear',
                                     kernel_initializer=initializers.TruncatedNormal(
                                         stddev=0.001, seed=seed),
                                     name=f'OutputLayer')
        # use small initialisation to prevent overflow
        # can't use zeros here because no stochasticity term
        # which otherwise provides enough initial variance
        self.x_hat_log_var = layers.Dense(raw_dim,
                                          activation='linear',
                                          kernel_initializer=initializers.TruncatedNormal(
                                              stddev=0.001, seed=seed),
                                          name=f'OutputLayer')
        # y classifier
        self.y_classifier = DeepLayer(hidden_layer_dims=hidden_layer_dims,
                                      activation_func='swish',
                                      kernel_init='glorot_uniform',
                                      name='Classifier',
                                      dropout=dropout,
                                      seed=seed)
        self.y_logits = layers.Dense(2, activation='linear', name='Logits')
        self.y_softmax = layers.Softmax(axis=-1)
        self.y_bce = losses.BinaryCrossentropy(from_logits=False)
        # for merging layers on last axis
        self.merger = layers.Concatenate(axis=-1)
        # for calculating likelihood
        self.likelihood = M2Likelihood(n_classes=2)
        # setup metrics
        self.summary_metrics = {}

    def call(self, inputs, training=True, **kwargs):
        # storing classes in last column
        X = inputs[:, :-1]
        y = inputs[:, -1:]
        # M2 steps
        labelled_idx = (K.flatten(y) != -1)
        unlabelled_idx = (K.flatten(y) == -1)
        # prepare X and y labelled vs. non-labelled
        X_lab = tf.boolean_mask(X, labelled_idx, axis=0)
        X_ula = tf.boolean_mask(X, unlabelled_idx, axis=0)
        # placeholder for combined loss
        combined_loss = 0.0

        # handle labelled case
        def compute_labelled():
            # cast y labelled to int and then one-hot
            y_lab = tf.boolean_mask(y, labelled_idx, axis=0)
            y_lab = tf.cast(K.flatten(y_lab), dtype=tf.int32)
            y_lab_oh = tf.one_hot(y_lab, depth=self.n_classes)
            # predict y_lab from X_lab
            y_lab_pred = self.y_classifier(X_lab, training=training)
            y_lab_pred = self.y_logits(y_lab_pred, training=training)
            y_lab_pred = self.y_softmax(y_lab_pred)
            y_lab_loss = self.y_bce(y_lab_oh, y_lab_pred)
            # alpha is the relative weight between generative and discriminative learning
            # paper uses 0.1 * N
            alpha = 0.1 * X.shape[0]
            agg_loss = alpha * y_lab_loss
            # handle labelled data
            X_y_lab = self.merger([X_lab, y_lab_oh])
            Z_mu_lab, Z_log_var_lab, Z_lab = self.encode(X_y_lab, training=training)
            Z_y_lab = self.merger([Z_lab, y_lab_oh])
            X_hat_lab_mu, X_hat_lab_log_var = self.decode(Z_y_lab, training=training)
            # log likelihood for labelled
            like_lab = self.likelihood([X_lab,
                                        Z_mu_lab,
                                        Z_log_var_lab,
                                        y_lab_oh,
                                        X_hat_lab_mu,
                                        X_hat_lab_log_var])
            # can't apply to empty tensor - gives NaN
            like_lab = K.mean(like_lab, axis=0)  # take mean over batch
            # flip sign and add to loss
            like_lab *= -1
            agg_loss += like_lab
            # add related metrics
            for key, metric in zip(['likelihood_labelled', 'labelled_pred_bce'],
                                   [like_lab, y_lab_loss]):
                if key not in self.summary_metrics:
                    self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
                self.summary_metrics[key](metric)
            return agg_loss

        # don't run if no labelled instances
        combined_loss += tf.cond(tf.not_equal(tf.shape(X_lab)[0], 0),  # tf.not_equal(1, 0)
                                 true_fn=compute_labelled,
                                 false_fn=lambda: 0.0)

        # handle unlabelled data
        def compute_unlabelled():
            inner_likes = []
            for idx in range(self.n_classes):
                y_ula = idx * tf.ones(shape=tf.shape(X_ula)[0], dtype=tf.int32)
                y_ula_oh = tf.one_hot(y_ula, depth=self.n_classes)
                # encode and decode for each class
                X_y_ula = self.merger([X_ula, y_ula_oh])
                Z_mu_ula, Z_log_var_ula, Z_ula = self.encode(X_y_ula, training=training)
                Z_y_ula = self.merger([Z_ula, y_ula_oh])
                X_hat_ula_mu, X_hat_ula_log_var = self.decode(Z_y_ula, training=training)
                # weld together likelihoods from each class
                inner_like = self.likelihood([X_ula,
                                              Z_mu_ula,
                                              Z_log_var_ula,
                                              y_ula_oh,
                                              X_hat_ula_mu,
                                              X_hat_ula_log_var])
                inner_likes.append(K.expand_dims(inner_like, axis=-1))
            like_ula = self.merger(inner_likes)
            # score unlabelled
            y_ula_pred = self.y_classifier(X_ula, training=training)
            y_ula_pred = self.y_logits(y_ula_pred, training=training)
            y_ula_pred = self.y_softmax(y_ula_pred)
            # H(q(y|x))
            like_ula = y_ula_pred * (like_ula - K.log(y_ula_pred + K.epsilon()))
            like_ula = K.sum(like_ula, axis=1)  # sum over classes
            like_ula = K.mean(like_ula)  # take mean over batch
            # flip sign
            like_ula *= -1
            agg_loss = like_ula
            # add related metrics
            for key, metric in zip(['likelihood_unlabelled'], [like_ula]):
                if key not in self.summary_metrics:
                    self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
                self.summary_metrics[key](metric)
            return agg_loss

        # don't run if no unlabelled instances
        combined_loss += tf.cond(tf.not_equal(tf.shape(X_ula)[0], 0),
                                 true_fn=compute_unlabelled,
                                 false_fn=lambda: 0.0)
        # add prior loss
        prior_loss = 0
        for var in self.trainable_variables:
            if self.name not in var.name:
                continue
            C = - 0.5 * np.log(2 * np.pi)
            std_gaussian = C - K.square(var) / 2
            prior_loss += K.sum(std_gaussian)
        # cast to negative
        prior_loss *= -1
        # weight by samples
        prior_wt = 1 / self.n_samples
        prior_loss *= prior_wt
        # add to combined loss
        combined_loss += prior_wt * prior_loss
        # add loss to model
        self.add_loss(combined_loss)
        # update summary metrics
        for key, metric in zip(['prior_loss', 'combined_loss'], [prior_loss, combined_loss]):
            if key not in self.summary_metrics:
                self.summary_metrics[key] = metrics.Mean(name=key, dtype='float32')
            self.summary_metrics[key](metric)

        # due to sometimes empty tensors - can't return y unlabelled predictions...
        return inputs

    def encode(self, X_y, training=False):
        X_y = self.encoder(X_y, training=training)
        Z_mu, Z_log_var, Z = self.sampling(X_y, training=training)
        return Z_mu, Z_log_var, Z

    def decode(self, Z_y, training=False):
        X = self.decoder(Z_y, training=training)
        X_hat_mu = self.x_hat_mu(X, training=training)
        X_hat_log_var = self.x_hat_log_var(X, training=training)
        return X_hat_mu, X_hat_log_var

    def classify(self, X_ula):
        y_ula_pred = self.y_classifier(X_ula, training=False)
        y_ula_pred = self.y_logits(y_ula_pred, training=False)
        return self.y_softmax(y_ula_pred)

    def get_config(self):
        config = super().get_config()
        config.update({
            'theme': self.theme,
            'n_samples': self.n_samples,
            'n_classes': self.n_classes,
            'encoder': self.encoder,
            'sampling': self.sampling,
            'decoder': self.decoder,
            'x_hat_mu': self.x_hat_mu,
            'x_hat_log_var': self.x_hat_log_var,
            'y_classifier': self.y_classifier,
            'y_logits': self.y_logits,
            'y_softmax': self.y_softmax,
            'y_bce': self.y_bce,
            'merger': self.merger,
            'likelihood': self.likelihood,
            'summary_metrics': self.summary_metrics,
            'encode': self.encode,
            'decode': self.decode,
            'classify': self.classify
        })
        return config
