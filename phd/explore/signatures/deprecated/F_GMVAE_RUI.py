# %%
from importlib import reload
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

import phd_util
from explore import plot_funcs
from explore.signatures import sig_model_runners
from explore.signatures.deprecated import D_GMVAE_rui
from explore.theme_setup import data_path, logs_path
from explore.theme_setup import generate_theme

reload(phd_util)
reload(plot_funcs)
reload(D_GMVAE_rui)
reload(sig_model_runners)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  %% load from disk
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
# PREPARE
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)

latent_dim = 8
seed = 0
batch = 256
learning_rate = 1e-6
description = ''
n_components = 8
for epochs in [1, 2]:
    theme_base = f'GMVAE_e{epochs}_lr{learning_rate}_s{seed}_{description}'
    gmVAE = D_GMVAE_rui.GMVAERUI(raw_dim=X_trans.shape[1],
                                 latent_dim=latent_dim,
                                 k_components=n_components,
                                 theme_base=theme_base,
                                 seed=seed)
    print('')
    print(gmVAE.theme)
    sig_model_runners.GMVAERUI_trainer(gmVAE,
                                       X_trans,
                                       labels,
                                       distances,
                                       lr=learning_rate,
                                       logs_path=logs_path,
                                       epochs=epochs,
                                       batch=batch)

    #  %%
    cluster_probs_gmVAE = gmVAE(X_trans, training=False)
    cluster_assign_gmVAE = np.argmax(cluster_probs_gmVAE, axis=1)  # N
    plot_funcs.plot_prob_clusters(cluster_probs_gmVAE, f'{gmVAE.theme}_mu', table.x, table.y)

    phd_util.plt_setup()
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    cmap = plt.cm.get_cmap('hsv')
    c = []
    for i in cluster_assign_gmVAE:
        c.append(cmap(i / n_components))
    plot_funcs.plot_scatter(ax,
                            table.x,
                            table.y,
                            vals=cluster_assign_gmVAE,
                            c=c,
                            s=1,
                            x_extents=(-12000, 12000))
    path = f'../../phd-admin/PhD/part_3/images/signatures/{gmVAE.theme}_cluster.png'
    plt.savefig(path, dpi=300)

# %%
# PLOT COMPONENTS FROM LATENTS
QX_mu = gmVAE.encode(X_trans)
plot_funcs.plot_components(list(range(latent_dim)),
                           labels,
                           distances,
                           X_trans,
                           QX_mu,
                           table.x,
                           table.y,
                           label_all=False,
                           s_min=0,
                           s_max=0.8,
                           c_exp=3,
                           s_exp=5,
                           cbar=True)
plt.suptitle(
    r"Pearson's $\rho$ correlations to source variables and geographic mappings for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/{gmVAE.theme}_latents.png'
plt.savefig(path, dpi=300)

def GMVAERUI_trainer(gmVAE_model,
                     X_raw,
                     labels,
                     distances,
                     logs_path=None,
                     epochs=1,
                     batch=256,
                     lr=1e-3):
    X_raw = X_raw.astype('float32')

    writer = None
    if logs_path is not None:
        path = f'{datetime.now().strftime("%Hh%Mm%Ss")}_{gmVAE_model.theme}'
        path = str(logs_path / f'{path}')
        logger.info(f'Tensorboard log directory: {path}')
        writer = tf.summary.create_file_writer(path)

    # optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # dataset.
    dataset = tf.data.Dataset.from_tensor_slices(X_raw)
    dataset = dataset.shuffle(buffer_size=X_raw.shape[0]).batch(batch)

    @tf.function
    def training_step(x):
        with tf.GradientTape() as tape:
            x_hat = gmVAE_model(x)  # Compute input reconstruction.
            loss = sum(gmVAE_model.losses)
        # Update the weights of the VAE.
        grads = tape.gradient(loss, gmVAE_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, gmVAE_model.trainable_weights))  # np.any(np.isnan(grads[0].numpy()))a
        bundle = {
            'x': x,
            'x_hat': x_hat,
            'loss': loss
        }
        return bundle

    # process epochs
    batch_counter = 0
    epoch_counter = 0
    for epoch_step in range(epochs):
        logger.info(f'Epoch: {epoch_step + 1}')
        progbar = tf.keras.utils.Progbar(X_raw.shape[0])
        # track losses
        losses = []
        # non local variables for epoch tensorboard step
        bundle = None
        x = None
        for step, x in enumerate(dataset):
            bundle = training_step(x)
            loss = bundle['loss']
            losses.append(float(loss))
            # log scalars
            if step % 100 == 0:
                if writer is not None:
                    with writer.as_default():
                        tf.summary.scalar('loss', loss, step=batch_counter)
                        for idx, metric_name in enumerate(gmVAE_model.metrics_names):
                            metric_result = gmVAE_model.metrics[idx].result()
                            gmVAE_model.metrics[idx].reset_states()
                            tf.summary.scalar(metric_name, metric_result, step=batch_counter)
                batch_counter += 1
            progbar.add(batch, values=[('loss', loss)])
        '''
        # log to console
        print("Step:", epoch_step + 1, "Loss:", sum(losses) / len(losses))
        # tensorboard
        if writer is not None:
            with writer.as_default():
                # images of mean x vs x_hat vs diff
                x_img = np.mean(bundle['x'], axis=0).reshape(len(labels), len(distances))
                x_hat_img = np.mean(bundle['x_hat'], axis=0).reshape(len(labels), len(distances))
                # stack if passing data directly
                # stacked_img = np.vstack([x_img, x_hat_img])
                # stacked_img = np.reshape(stacked_img, (-1, len(labels), len(distances), 1))
                phd_util.plt_setup()
                fig, axes = plt.subplots(1, 3, figsize=(6, 8))
                plot_funcs.plot_heatmap(axes[0], x_img, row_labels=labels, col_labels=distances)
                plot_funcs.plot_heatmap(axes[1], x_hat_img, set_row_labels=False, col_labels=distances)
                plot_funcs.plot_heatmap(axes[2], x_img - x_hat_img, set_row_labels=False, col_labels=distances)
                tf.summary.image('x | x hat | diff', plot_to_image(fig), step=epoch_counter)
                # images of latents
                Z_mu = gmVAE_model.encode(x)
                latent_dim = Z_mu.shape[1]
                phd_util.plt_setup()
                fig, axes = plt.subplots(1, latent_dim, figsize=(12, 8))
                for l_idx in range(latent_dim):
                    corr = plot_funcs.correlate_heatmap(len(labels),
                                                        len(distances),
                                                        x,
                                                        Z_mu[:, l_idx])
                    plot_funcs.plot_heatmap(axes[l_idx],
                                            corr,
                                            row_labels=labels,
                                            set_row_labels=l_idx == 0,
                                            col_labels=distances)
                tf.summary.image('latents', plot_to_image(fig), step=epoch_counter)
                # histograms
                tf.summary.histogram('QX mean',
                                     gmVAE_model.qx.qx_mu.weights[1],
                                     step=epoch_counter)
                tf.summary.histogram('QX log var',
                                     gmVAE_model.qx.qx_log_var.weights[0],
                                     step=epoch_counter)
                tf.summary.histogram('QW mean',
                                     gmVAE_model.qw.qw_mu.weights[1],
                                     step=epoch_counter)
                tf.summary.histogram('QW log var',
                                     gmVAE_model.qw.qw_log_var.weights[0],
                                     step=epoch_counter)
        '''
        epoch_counter += 1
