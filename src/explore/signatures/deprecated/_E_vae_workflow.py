# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import phd_util
from src.explore import plot_funcs
from src.explore.theme_setup import plot_path, logs_path, weights_path
from src.explore.theme_setup import process_selected_columns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# %%
n_components = 8
abbrev_dist = [100, 200, 300, 400, 500, 600, 700, 800, 1200, 1600]

# %%
'''
2A - Choose level of network decomposition and plot heatmaps:
Show amount of correlation and respective ordering vs. strongest PCA components
'''
# PCA centers the data but doesn't scale
# optionally use whiten param to scale to unit variance
n_components = 4
base = 'vae_workflow_PCA'

for outer_theme in ['all', 'cent', 'lu', 'cens']:  # 'all', 'cent', 'lu', 'cens'

    print('outer', outer_theme)
    if outer_theme == 'all':
        target_columns = columns_cent + columns_lu + columns_cens
    elif outer_theme == 'cent':
        target_columns = columns_cent
    elif outer_theme == 'lu':
        target_columns = columns_lu
    elif outer_theme == 'cens':
        target_columns = columns_cens

    selected_columns, selected_columns_labels = process_selected_columns(target_columns)
    X_raw = table[selected_columns]
    X_trans = StandardScaler().fit_transform(X_raw)

    model = PCA(n_components=n_components, whiten=False)
    X_reduced = model.fit_transform(X_trans)

    # plot scatter of dim reduction vs. selected locations
    phd_util.plt_setup(dark=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=1, alpha=0.05)
    for loc, vals in location_keys.items():
        v1 = X_reduced[:, 0][vals['iloc']]
        v2 = X_reduced[:, 1][vals['iloc']]
        ax.scatter(v1, v2, s=20, alpha=1, label=loc)
    ax.legend(fontsize='medium')
    ax.set_xlabel('PCA dimension 0')
    ax.set_ylabel('PCA dimension 1')
    ax.set_xlim(left=np.nanpercentile(X_reduced[:, 0], 0.005), right=np.nanpercentile(X_reduced[:, 0], 99.995))
    ax.set_ylim(bottom=np.nanpercentile(X_reduced[:, 1], 0.005), top=np.nanpercentile(X_reduced[:, 1], 99.995))

    scatter_title = f'{base}_{outer_theme}_scatter'
    plt.suptitle(scatter_title)
    plt.savefig(plot_path / f'{scatter_title}.png', dpi=300)

    exp_var = model.explained_variance_
    exp_var_ratio = model.explained_variance_ratio_
    # eigenvector by eigenvalue - i.e. correlation to original
    loadings = model.components_.T * np.sqrt(exp_var)
    loadings = loadings.T  # transform for slicing

    phd_util.plt_setup()
    plot_funcs.plot_components([0, 1, 2, 3],
                               selected_columns_labels,
                               abbrev_dist,
                               X_trans,
                               X_reduced,
                               table.x,
                               table.y,
                               explained_variances=exp_var_ratio,
                               label_all=True,
                               loadings=loadings,
                               dark=False,
                               allow_inverse=False,
                               s_exp=4,
                               cbar=True)

    # plot map of dim reduction
    plt_title = f'{base}_{outer_theme}_components'
    plt.suptitle(f'{plt_title}')
    plt.savefig(plot_path / f'{plt_title}.png', dpi=300)

# %%
'''
DATA PREPROCESSING!!
- minmax doesn't work well - long-tailed outliers cause too much bunching - poor results
- StandardScaler and RobustScaler are problematic for ml methods, but work well for PCA
- relu is more robust to vanishing gradients
- added batch normalisation to help guard against vanishing gradients
- Tried preparing data per feature, per horizontal (feature theme) axis, and per vertical (distance theme) axis
  this didn't work well for ML.
- PowerTransformer works best at the end of the day... which makes sense because VAE is casting data to normal dist.
'''

# PREPARE

from datetime import datetime
from src.explore.signatures import D_autoencoder_models

reload(D_autoencoder_models)
from keras import callbacks

reload(callbacks)

for outer_theme in ['all']:  # 'all', 'cent', 'lu', 'cens'

    if outer_theme == 'all':
        target_columns = columns_cent + columns_lu + columns_cens
    elif outer_theme == 'cent':
        target_columns = columns_cent
    elif outer_theme == 'lu':
        target_columns = columns_lu
    elif outer_theme == 'cens':
        target_columns = columns_cens

    selected_columns, selected_columns_labels = process_selected_columns(target_columns)
    X_raw = table[selected_columns]
    X_trans = StandardScaler().fit_transform(X_raw)

    for latent_dim in [2, 4]:
        for inner_theme in ['simple', 'conv', 'lstm']:
            print('outer', outer_theme, 'inner', inner_theme)

            encoder, decoder, vae, model_key = D_autoencoder_models.prepare_autoencoder(
                len(target_columns),
                len(abbrev_dist),
                inner_theme,
                latent_dim)

            # TRAIN
            early_stopping = callbacks.EarlyStopping(monitor='mse', patience=2)

            log_key = f'{datetime.now().strftime("%Hh%Mm")}_{model_key}_{outer_theme}'
            log_dir = logs_path / f'{log_key}'
            tensor_board = callbacks.TensorBoard(log_dir=log_dir, write_graph=False)

            # train the autoencoder
            vae.fit(X_trans,
                    X_trans,  # required for custom vae_loss function
                    batch_size=1024,
                    epochs=10,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[tensor_board, early_stopping],
                    verbose=True)

            # save model
            encoder.save_weights(str(weights_path / f'ae/model_{model_key}_{outer_theme}_encoder.h5'))
            decoder.save_weights(str(weights_path / f'ae/model_{model_key}_{outer_theme}_decoder.h5'))
            vae.save_weights(str(weights_path / f'ae/model_{model_key}_{outer_theme}.h5'))

            hist_df = pd.DataFrame(vae.history.history)
            with open(weights_path / f'ae/model_{model_key}_{outer_theme}_history.json', mode='w') as write_file:
                hist_df.to_json(write_file)

# %%
# plot
for outer_theme in ['all']:  # 'all', 'cent', 'lu', 'cens'

    if outer_theme == 'all':
        target_columns = columns_cent + columns_lu + columns_cens
    elif outer_theme == 'cent':
        target_columns = columns_cent
    elif outer_theme == 'lu':
        target_columns = columns_lu
    elif outer_theme == 'cens':
        target_columns = columns_cens

    selected_columns, selected_columns_labels = process_selected_columns(target_columns)
    X_raw = table[selected_columns]
    X_trans = StandardScaler().fit_transform(X_raw)

    maxes = np.max(np.abs(X_trans), axis=0)

    for latent_dim in [2]:
        for inner_theme in ['simple', 'conv', 'lstm']:

            encoder, decoder, vae, model_key = D_autoencoder_models.prepare_autoencoder(
                len(target_columns),
                len(abbrev_dist),
                inner_theme,
                latent_dim)

            encoder.load_weights(str(weights_path / f'ae/model_{model_key}_{outer_theme}_encoder.h5'))
            decoder.load_weights(str(weights_path / f'ae/model_{model_key}_{outer_theme}_decoder.h5'))
            vae.load_weights(str(weights_path / f'ae/model_{model_key}_{outer_theme}.h5'))

            with open(weights_path / f'ae/model_{model_key}_{outer_theme}_history.json') as open_file:
                hist_data = pd.read_json(open_file)
            vae_loss = hist_data['val_loss'].iloc[-1].round(4)
            mse_loss = hist_data['val_mean_squared_error'].iloc[-1].round(4)
            mae_loss = hist_data['val_mean_absolute_error'].iloc[-1].round(4)

            #  %% encode
            encoded_vae = encoder.predict(X_trans)
            # zs = encoded_vae[2]
            zs = encoded_vae

            # prepare titles
            title = f'{inner_theme} vae_loss: {vae_loss} mse_loss: {mse_loss} dim: {latent_dim}'
            path_title = f'{inner_theme}_dim_{latent_dim}'

            #  %%
            if latent_dim == 2:
                # plot scatter of vae vs. selected locations for comparison to PCA
                phd_util.plt_setup(dark=True)
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.scatter(zs[:, 0], zs[:, 1], s=1, alpha=0.05)
                for loc, vals in location_keys.items():
                    v1 = zs[:, 0][vals['iloc']]
                    v2 = zs[:, 1][vals['iloc']]
                    ax.scatter(v1, v2, s=20, alpha=1, label=loc)
                ax.legend(fontsize='medium')
                ax.set_xlabel(f'{inner_theme} AEC dimension 0')
                ax.set_ylabel(f'{inner_theme} AEC dimension 1')
                ax.set_xlim(left=np.nanpercentile(zs[:, 0], 0.005), right=np.nanpercentile(zs[:, 0], 99.995))
                ax.set_ylim(bottom=np.nanpercentile(zs[:, 1], 0.005), top=np.nanpercentile(zs[:, 1], 99.995))
                path = plot_path / f'vae_loc_{path_title}.png'
                plt.suptitle(title)
                plt.savefig(path, dpi=300)
                plt.show()

                #  %% plot gridwise decodings
                phd_util.plt_setup(dark=True)
                fig, ax_rows = plt.subplots(7, 7, figsize=(10, 15))
                for i_idx, i_key in enumerate(list(range(-3, 4))):
                    for j_idx, j_key in enumerate(list(range(-3, 4))):
                        ax = ax_rows[i_idx][j_idx]
                        z_key = np.array([[i_key, j_key]])  # expects 2d
                        arr = decoder.predict(z_key)
                        arr /= maxes
                        arr = np.reshape(arr, (len(selected_columns_labels), len(distances)))
                        plot_funcs.plot_heatmap(ax, arr, selected_columns_labels, distances,
                                                set_labels=False, dark=True)
                        ax.set_xlabel(f'{i_key} | {j_key}')
                        if j_idx == 0:
                            ax.set_yticks(list(range(len(selected_columns_labels))))
                            ax.set_yticklabels(selected_columns_labels, rotation='horizontal')

                path = plot_path / f'vae_grid_{path_title}.png'
                plt.suptitle(title)
                plt.savefig(path, dpi=300)
                plt.show()

            #  %%
            phd_util.plt_setup(dark=True)
            x_dims = int(latent_dim / 2)
            if latent_dim <= 4:
                fig, ax_rows = plt.subplots(1, latent_dim, figsize=(latent_dim * 4, 5))
            else:
                fig, ax_rows = plt.subplots(2, x_dims, figsize=(x_dims * 4, 10))
            counter = 0
            for ax_row in ax_rows:
                if latent_dim <= 4:
                    ax_row = [ax_row]
                for ax in ax_row:
                    heatmap = np.full((len(selected_columns_labels), len(distances)), 0.0)
                    inner_counter = 0
                    for i in range(len(selected_columns_labels)):
                        for j in range(len(distances)):
                            heatmap[i][j] = spearmanr(X_raw.iloc[:, inner_counter], zs[:, counter])[0]
                            inner_counter += 1
                    plot_funcs.plot_heatmap(ax, heatmap, selected_columns_labels, distances, set_labels=True, dark=True)
                    ax.set_title(f'latent dim {counter}')
                    counter += 1
            plt.suptitle(title)
            path = plot_path / f'ae_latent_dims_{path_title}.png'
            plt.savefig(path, dpi=300)
            plt.show()

            #  %%
            # plots decoded exemplar landuses
            phd_util.plt_setup(dark=True)
            fig, ax_rows = plt.subplots(2, 5, figsize=(15, 10))
            names = [n for n in location_keys.keys()]
            indices = [i['iloc'] for i in location_keys.values()]
            counter = 0
            for ax_row in ax_rows:
                for ax_n, ax in enumerate(ax_row):
                    name = names[counter]
                    idx = indices[counter]
                    z_state = zs[idx]
                    z_key = np.array([[*z_state]])  # expects 2d
                    arr = decoder.predict(z_key)
                    arr /= maxes
                    arr = np.reshape(arr, (len(selected_columns_labels), len(distances)))
                    plot_funcs.plot_heatmap(ax_row[ax_n], arr, selected_columns_labels, distances, set_labels=True,
                                            dark=True)
                    ax_row[ax_n].set_title(name)
                    # prepare codes
                    z_txt = [str(z) for z in z_key.round(2)[0]]
                    z_arr = []
                    z_str_part = ''
                    for z_n, z_t in enumerate(z_txt):
                        if z_n != 0 and z_n % 4 == 0:
                            z_arr.append(z_str_part)
                            z_str_part = ''
                        z_str_part += f'| {z_t} |'
                        if z_n + 1 == len(z_txt):
                            z_arr.append(z_str_part)
                    z_str = str.join('\n', z_arr)
                    ax_row[ax_n].set_xlabel(z_str, fontsize='medium')
                    counter += 1
            plt.suptitle(title)
            path = plot_path / f'ae_proj_flat_{path_title}.png'
            plt.savefig(path, dpi=300)
            plt.show()

# %%
