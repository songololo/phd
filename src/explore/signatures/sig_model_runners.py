import io
import json
import logging
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics

from src import util_funcs
from src.explore import plot_funcs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# https://www.tensorflow.org/tensorboard/image_summaries#visualizing_multiple_images
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def mse_sum(y_true, y_pred):
    # sum over dimensions
    rl = K.sum(K.square(y_pred - y_true), axis=-1)
    # take mean over samples
    return K.mean(rl)


class Trainer():

    def __init__(self,
                 model=None,
                 X_samples=None,
                 labels=None,
                 distances=None,
                 logs_path=None,
                 epochs=1,
                 batch=256,
                 lr=1e-3,
                 clip_norm=1.0,
                 best_loss=False,
                 save_path: pathlib.Path = None,
                 test_indices: np.ndarray = None,
                 shuffle_seed: int = 0):
        self.model = model
        self.labels = labels
        self.distances = distances
        self.epochs = epochs
        self.batch = batch
        self.best_loss = best_loss
        self.train_loss = metrics.Mean(name='train_loss', dtype='float32')
        self.val_loss = metrics.Mean(name='val_loss', dtype='float32')
        self.history = {}
        # optimizer
        self.learning_rate = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clip_norm)
        # logging
        self.writer = None
        if logs_path is not None:
            path = f'{datetime.now().strftime("%Hh%Mm%Ss")}_{model.theme}_e{epochs}_b{batch}_lr{lr}'
            path = str(logs_path / f'{path}')
            logger.info(f'Tensorboard log directory: {path}')
            self.writer = tf.summary.create_file_writer(path)
        # saving
        self.save_path = None
        self.save_path_history = None
        if save_path is not None:
            if not save_path.exists():
                save_path.mkdir(exist_ok=True, parents=True)
            self.save_path = str(pathlib.Path(save_path / 'weights'))
            self.save_path_history = str(pathlib.Path(save_path / 'history.json'))
        # setup datasets
        self.X_train = X_samples[~test_indices]
        self.X_val = X_samples[test_indices]
        # training dataset
        training_dataset = tf.data.Dataset.from_tensor_slices(self.X_train)
        training_dataset = training_dataset.shuffle(buffer_size=self.X_train.shape[0],
                                                    reshuffle_each_iteration=True,
                                                    seed=shuffle_seed)
        self.training_dataset = training_dataset.batch(batch, drop_remainder=False)
        # validation dataset
        validation_dataset = tf.data.Dataset.from_tensor_slices(self.X_val)
        validation_dataset = validation_dataset.shuffle(buffer_size=self.X_val.shape[0],
                                                        reshuffle_each_iteration=False,
                                                        seed=shuffle_seed)
        self.validation_dataset = validation_dataset.batch(batch, drop_remainder=False)

    @tf.function
    def training_step(self, batch_x, training):
        with tf.GradientTape() as tape:
            self.model(batch_x, training=training)
            loss = sum(self.model.losses)
        # Update the weights of the VAE.
        grads = tape.gradient(loss, self.model.trainable_weights)
        # guards against exploding gradients
        # grads, global_norm = tf.clip_by_global_norm(grads, self.max_global_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        if training:
            self.train_loss(loss)
        else:
            self.val_loss(loss)

    def train(self):
        # process epochs
        least_loss = np.inf
        best_epoch = 0
        batch_counter = 0  # track aggregate counter over different batches
        for epoch_step in range(self.epochs):
            logger.info(f'Epoch: {epoch_step + 1}')
            # setup progress bar
            progress_bar = tf.keras.utils.Progbar(self.X_train.shape[0])
            # run custom epoch setup steps
            self.epoch_setup(epoch_step)
            # reset metrics states
            self.train_loss.reset_states()
            self.reset_summary_metrics()
            # run batches
            for batch_step, training_batch in enumerate(
                    self.training_dataset.as_numpy_iterator()):
                # iterate
                self.training_step(training_batch, training=True)
                if not np.isfinite(self.train_loss.result()):
                    logger.warning(f'Invalid loss encountered on batch step {batch_step}')
                    break
                # custom batch operations
                if batch_step % 25 == 0:
                    # process batch level tensorboard
                    self.batch_writes(batch_counter)
                    self.reset_summary_metrics()
                # progress
                progress_bar.add(self.batch, values=[('loss', self.train_loss.result())])
                batch_counter += 1
            # epoch steps
            if not np.isfinite(self.train_loss.result()):
                break
            # reset metrics states
            self.val_loss.reset_states()
            self.reset_summary_metrics()
            # compute validation loss
            for validation_batch in self.validation_dataset.as_numpy_iterator():
                self.training_step(validation_batch, training=False)
            if not np.isfinite(self.val_loss.result()):
                logger.warning(
                    f'Step: {epoch_step + 1}: non finite validation loss encountered: '
                    f'{self.val_loss.result()}')
            else:
                logger.info(
                    f'Step: {epoch_step + 1}: validation loss: {self.val_loss.result()}')
                # process epoch level tensorboard
                self.epoch_writes(epoch_step)
            # write validation history
            self.update_history()
            # save if best weights
            if self.save_path is not None and self.best_loss and self.val_loss.result() < least_loss:
                logger.info(f'Updating least loss - saving to {self.save_path}')
                least_loss = self.val_loss.result()
                best_epoch = epoch_step + 1
                self.model.save_weights(self.save_path, overwrite=True)
        # finalise weights
        if not np.isfinite(self.train_loss.result()):
            logger.error(f'Invalid loss encountered: {self.train_loss.result()}')
        elif self.save_path is not None:
            if self.best_loss:
                logger.info(f'Best loss {least_loss:.2f} from epoch step: {best_epoch}.')
            else:
                logger.info('Saving last weights')
                self.model.save_weights(self.save_path, overwrite=True)
            # save history
            with open(self.save_path_history, 'w') as json_out:
                json.dump(self.history, json_out)

    def reset_summary_metrics(self):
        if hasattr(self.model, 'summary_metrics'):
            for metric_val in self.model.summary_metrics.values():
                metric_val.reset_states()

    def update_history(self):
        if 'val_loss' not in self.history:
            self.history['train_loss'] = [float(self.train_loss.result())]
            self.history['val_loss'] = [float(self.val_loss.result())]
        else:
            self.history['train_loss'].append(float(self.train_loss.result()))
            self.history['val_loss'].append(float(self.val_loss.result()))
        # add metrics - this should be based on validation step
        if hasattr(self.model, 'summary_metrics'):
            for metric_name, metric_val in self.model.summary_metrics.items():
                target_name = f'val_{metric_name}'
                if target_name not in self.history:
                    self.history[target_name] = [float(metric_val.result().numpy())]
                else:
                    self.history[target_name].append(float(metric_val.result().numpy()))

    def batch_writes(self, batch_step):
        # log scalars
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar('training loss', self.train_loss.result(), step=batch_step)
                if hasattr(self.model, 'summary_metrics'):
                    for metric_name, metric_val in self.model.summary_metrics.items():
                        tf.summary.scalar(metric_name, metric_val.result(), step=batch_step)

    def epoch_setup(self, epoch_step):
        pass

    def epoch_writes(self, epoch_step):
        if self.writer is not None:
            with self.writer.as_default():
                # write validation loss
                tf.summary.scalar('validation loss', self.val_loss.result(), step=epoch_step)
                if hasattr(self.model, 'summary_metrics'):
                    for metric_name, metric_val in self.model.summary_metrics.items():
                        target_name = f'val_{metric_name}'
                        tf.summary.scalar(target_name, metric_val.result(), step=epoch_step)


class VAE_trainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def epoch_setup(self, epoch_step):
        # update capacity term
        self.model.kl_divergence.capacity_update(epoch_step)

    def epoch_writes(self, epoch_step):
        if self.writer is not None:
            with self.writer.as_default():
                super().epoch_writes(epoch_step)
                # extract images
                Z_mu, Z_log_var, Z = self.model.encode(self.X_val, training=False)
                x_hat = self.model.decode(Z_mu, training=False)
                # images of mean x vs x_hat vs diff
                x_img = np.mean(self.X_val, axis=0).reshape(len(self.labels),
                                                            len(self.distances))
                x_hat_img = np.mean(x_hat, axis=0).reshape(len(self.labels),
                                                           len(self.distances))
                # stack if passing data directly
                # stacked_img = np.vstack([x_img, x_hat_img])
                # stacked_img = np.reshape(stacked_img, (-1, len(labels), len(distances), 1))
                util_funcs.plt_setup()
                fig, axes = plt.subplots(1, 3, figsize=(6, 8))
                plot_funcs.plot_heatmap(axes[0], x_img, row_labels=self.labels,
                                        col_labels=self.distances)
                plot_funcs.plot_heatmap(axes[1], x_hat_img, set_row_labels=False,
                                        col_labels=self.distances)
                plot_funcs.plot_heatmap(axes[2], x_img - x_hat_img, set_row_labels=False,
                                        col_labels=self.distances)
                tf.summary.image('x | x hat | diff', plot_to_image(fig), step=epoch_step)
                # images of latents
                latent_dim = Z_mu.shape[1]
                util_funcs.plt_setup()
                fig, axes = plt.subplots(1, latent_dim, figsize=(12, 8))
                for l_idx in range(latent_dim):
                    corr = plot_funcs.correlate_heatmap(len(self.labels),
                                                        len(self.distances),
                                                        self.X_val,
                                                        Z_mu[:, l_idx])
                    plot_funcs.plot_heatmap(axes[l_idx],
                                            corr,
                                            row_labels=self.labels,
                                            set_row_labels=l_idx == 0,
                                            col_labels=self.distances)
                tf.summary.image('latents', plot_to_image(fig), step=epoch_step)
                # histograms
                if hasattr(self.model, 'sampling'):
                    tf.summary.histogram('Z mu biases',
                                         self.model.sampling.Z_mu_layer.weights[1],
                                         step=epoch_step)
                    tf.summary.histogram('Z mu weights',
                                         self.model.sampling.Z_mu_layer.weights[0],
                                         step=epoch_step)
                    tf.summary.histogram('Z logvar biases',
                                         self.model.sampling.Z_logvar_layer.weights[1],
                                         step=epoch_step)
                    tf.summary.histogram('Z logvar weights',
                                         self.model.sampling.Z_logvar_layer.weights[0],
                                         step=epoch_step)


class VaDE_trainer(VAE_trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def epoch_setup(self, epoch_step):
        pass

    def batch_writes(self, batch_step):
        super().batch_writes(batch_step)
        # log scalars
        if self.writer is not None:
            with self.writer.as_default():
                # image of cluster priors
                cat_pi_img = self.model.classify(self.X_val)
                cat_pi_img = np.reshape(cat_pi_img,
                                        (-1, cat_pi_img.shape[0], cat_pi_img.shape[1], 1))
                tf.summary.image('Batch cluster probabilities', cat_pi_img, step=batch_step)

    def epoch_writes(self, epoch_step):
        super().epoch_writes(epoch_step)
        # tensorboard
        if self.writer is not None:
            with self.writer.as_default():
                # histograms
                tf.summary.histogram(f'GMM cat p', self.model.gamma.cat_pi, step=epoch_step)
                tf.summary.histogram(f'GMM mu', self.model.gamma.gmm_mu, step=epoch_step)
                tf.summary.histogram(f'GMM logvar', self.model.gamma.gmm_log_var,
                                     step=epoch_step)


class GMVAE_trainer(Trainer):

    def __init__(self, lambda_threshold: float = 0.0, cv_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_threshold = lambda_threshold  # "Free bits threshold"
        self.cv_weight = cv_weight  # "Weight of the information theoretic cost term" **kwargs):

    @tf.function
    def training_step(self, batch_x, training):
        with tf.GradientTape() as tape:
            self.model(batch_x, training=training)
            recon_loss, exp_kld_loss, kld_loss, pz_h = self.model.losses
            loss = recon_loss + exp_kld_loss + kld_loss
            # max_h = -K.log(1.0 / self.k_components)
            if pz_h < self.lambda_threshold:
                pz_h = tf.stop_gradient(pz_h)
            loss += (pz_h - self.lambda_threshold) * self.cv_weight
        # Update the weights of the VAE.
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        if training:
            self.train_loss(loss)
        else:
            self.val_loss(loss)

    def epoch_writes(self, epoch_step):
        super().epoch_writes(epoch_step)
        if self.writer is not None:
            with self.writer.as_default():
                '''
                # images of latents
                QX_mu = self.model.encode(self.X_val, training=False)
                latent_dim = QX_mu.shape[1]
                phd_util.plt_setup()
                fig, axes = plt.subplots(1, latent_dim, figsize=(12, 8))
                for l_idx in range(latent_dim):
                    d = QX_mu[:, l_idx]
                    # catch NaN
                    # TODO: temporary workaround for NaN
                    if np.any(~np.isfinite(d)):
                        continue
                    corr = plot_funcs.correlate_heatmap(len(self.labels),
                                                        len(self.distances),
                                                        self.X_val,
                                                        d)
                    plot_funcs.plot_heatmap(axes[l_idx],
                                            corr,
                                            row_labels=self.labels,
                                            set_row_labels=l_idx == 0,
                                            col_labels=self.distances)
                '''
                # histograms
                tf.summary.histogram('QX mean',
                                     self.model.qx.qx_mu.weights[1],
                                     step=epoch_step)
                tf.summary.histogram('QX log var',
                                     self.model.qx.qx_log_var.weights[0],
                                     step=epoch_step)
                tf.summary.histogram('QW mean',
                                     self.model.qw.qw_mu.weights[1],
                                     step=epoch_step)
                tf.summary.histogram('QW log var',
                                     self.model.qw.qw_log_var.weights[0],
                                     step=epoch_step)
