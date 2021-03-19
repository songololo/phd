# %%
import json
import pathlib


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import phd_util
from src.explore import plot_funcs
from src.explore.signatures import sig_models
from src.explore.signatures import step_B2_vae_latents_UDR
from src.explore.theme_setup import data_path, weights_path
from src.explore.theme_setup import generate_theme
from sklearn.preprocessing import StandardScaler, quantile_transform


reload(plot_funcs)
reload(sig_models)
reload(step_B2_vae_latents_UDR)

from keras import models

# %% load from disk
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')


# %%
def gather_loss(seeds, epochs, latent_dims, betas, caps):
    vae_losses = ['val_mse_sum',
                  'val_kl_cap',
                  'val_vae_loss']
    vade_losses = ['val_mse_sum',
                   'val_gamma_loss',
                   'val_vade_loss']
    vae_history_data = {}
    vade_history_data = {}
    # iterate dims, betas, caps
    for latent_dim in latent_dims:
        for beta in betas:
            for cap in caps:
                # no point iterating capacity if beta is zero...
                if beta == 0 and cap != 0:
                    continue
                key = f'e{epochs}_d{latent_dim}_b{beta}_c{cap}'
                # each model gets a key in the dictionary - all epochs and seeds will be stored here
                vae_history_data[key] = {}
                for loss in vae_losses:
                    # each loss gets a nested key of epochs and seeds
                    vae_history_data[key][loss] = np.full((epochs, len(seeds)), np.nan)
                # same for vade data
                vade_history_data[key] = {}
                for loss in vade_losses:
                    vade_history_data[key][loss] = np.full((epochs, len(seeds)), np.nan)
                # iterate the seeds
                for seed_idx, seed in enumerate(seeds):
                    # in case not all models have completed running
                    fp = f'temp_weights/seed_{seed}/model_VAE_split_{key}_s{seed}_history.json'
                    if pathlib.Path(fp).is_file():
                        with open(fp) as f:
                            vae_data = json.load(f)
                            # fetch each of the losses
                            for loss in vae_losses:
                                for epoch_idx in range(epochs):
                                    # loss dictionary epoch keys are strings...
                                    epoch_idx_str = str(epoch_idx)
                                    # because of early stopping, check whether epoch key exists:
                                    if (epoch_idx_str) in vae_data[loss]:
                                        vae_history_data[key][loss][epoch_idx, seed_idx] = vae_data[loss][epoch_idx_str]
                    # repeat for vade
                    fp = f'temp_weights/seed_{seed}/model_VaDE_{key}_s{seed}_history.json'
                    if pathlib.Path(fp).is_file():
                        with open(fp) as f:
                            vade_data = json.load(f)
                            for loss in vade_losses:
                                for epoch_idx in range(epochs):
                                    epoch_idx_str = str(epoch_idx)
                                    if (epoch_idx_str) in vade_data[loss]:
                                        vade_history_data[key][loss][epoch_idx, seed_idx] = vade_data[loss][
                                            epoch_idx_str]

    return vae_history_data, vade_history_data


def generate_heatmap(ax, cat, theme, epochs, latent_dim, betas, caps, theme_data):
    # empty numpy array for containing
    arr = np.full((len(betas), len(caps)), np.nan)
    for row_idx, beta in enumerate(betas):
        for col_idx, cap in enumerate(caps):
            if beta == 0 and cap != 0:
                continue
            # key for referencing history data
            key = f'e{epochs}_d{latent_dim}_b{beta}_c{cap}'
            loss_dict = theme_data[key]
            # iterate the ax rows and ax cols
            # each loss_dict contains the loss keys with numpy values of epochs x seeds
            # take the argmin across the epoch axis then take the mean of the seeds
            if theme == 'mse':
                arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict['val_mse_sum'], axis=0))
            elif theme == 'kl_cap':
                arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict['val_kl_cap'], axis=0))
            elif theme == 'gamma':
                arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict['val_gamma_loss'], axis=0))
            elif theme == 'vae':
                arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict['val_vae_loss'], axis=0))
            elif theme == 'vade':
                arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict['val_vade_loss'], axis=0))
    # remove outliers
    scaled = arr.copy()
    # sklearn preprocessing works only per axis dimension
    scaled = scaled.reshape((-1, 1))
    scaled = quantile_transform(scaled, n_quantiles=9, output_distribution='uniform')
    scaled /= 3
    scaled = scaled.reshape((len(betas), len(caps)))
    # remove text longer than four characters
    text = arr.astype('str')
    for i in range(text.shape[0]):
        for j in range(text.shape[1]):
            text[i][j] = f'{arr[i][j]:^4.2f}'
    # plot
    plot_funcs.plot_heatmap(ax,
                            heatmap=scaled,
                            row_labels=[r'$\beta=' + f'{b}$' for b in betas],
                            col_labels=[r'$C=' + f'{c}$' for c in caps],
                            text=text)
    if theme == 'mse':
        ax_title = f'{cat}: MSE reconstruction loss: {latent_dim} latents'
    elif theme == 'kl':
        ax_title = f'{cat}: Internal ' + r'$D_{KL}$ loss:' + f' {latent_dim} latents'
    elif theme == 'kl_cap':
        ax_title = f'{cat}: ' + r'$\beta \cdot |D_{KL} - C|$ loss:' + f' {latent_dim} latents'
    elif theme == 'gamma':
        ax_title = f'{cat}: ' + r'$\gamma$ loss:' + f' {latent_dim} latents'
    elif theme == 'vae':
        ax_title = f'{cat}: Combined loss:' + f' {latent_dim} latents'
    elif theme == 'vade':
        ax_title = f'{cat}: Combined loss:' + f' {latent_dim} latents'
    ax.set_title(ax_title)


def generate_arr_heatmap(ax, arr, row_vals, col_vals, cmap=None, constrain=(-1, 1), row_label=r'\beta=',
                         col_label='C='):
    # remove outliers
    scaled = arr.copy()
    # sklearn preprocessing works only per axis dimension
    scaled = scaled.reshape((-1, 1))
    scaled = quantile_transform(scaled, n_quantiles=9, output_distribution='uniform')
    scaled /= 3
    scaled = scaled.reshape((len(row_vals), len(col_vals)))
    # remove text longer than four characters
    text = arr.astype('str')
    for i in range(text.shape[0]):
        for j in range(text.shape[1]):
            # remove nan
            text[i][j] = f'{arr[i][j]:^.2g}'
    # plot
    plot_funcs.plot_heatmap(ax,
                            heatmap=scaled,
                            row_labels=[f'${row_label}{b}$' for b in row_vals],
                            col_labels=[f'${col_label}{c}$' for c in col_vals],
                            text=text,
                            constrain=constrain,
                            cmap=cmap)


# %%
'''
Course grid plots:
Two pages - one each for VAE / VaDE
Each row divided into dimensions: 4 / 6 / 8 / 12
Each block is beta x capacity
'''
# gather the data
epochs = 10
table = df_20
seeds = [0]
latent_dims = [4, 6, 8, 12]
betas = [0, 0.5, 1, 2, 4, 8, 16, 32, 64]
caps = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
vae_history_data, vade_history_data = gather_loss(seeds, epochs, latent_dims, betas, caps)
# plot
# TODO: remove VaDE??
for cat, theme_set, theme_data in zip(
        ['VAE', 'VaDE'],
        [['mse', 'kl_cap', 'vae'], ['mse', 'gamma', 'vade']],
        [vae_history_data, vade_history_data]):
    phd_util.plt_setup()
    fig, axes = plt.subplots(len(theme_set), len(latent_dims), figsize=(12, 10))
    for ax_row, theme in zip(axes, theme_set):
        for ax, dim in zip(ax_row, latent_dims):
            generate_heatmap(ax, cat, theme, epochs, dim, betas, caps, theme_data)
    plt.suptitle(f'Gridplots comparing model losses for {cat} hyperparameters (20 epochs)')
    path = f'../../phd-admin/PhD/part_3/images/signatures/model_scores_{cat}.png'
    plt.savefig(path, dpi=300)
    break

# %%
'''
Fine grid plots:
One page: one column VAE and one for VaDE
Each block is beta x capacity
'''
epochs = 20
table = df_20
seeds = [1, 2, 3, 4, 5, 6]
latent_dims = [8]
betas = [1, 2, 4, 8, 16]
caps = [0, 5, 10, 20, 40, 80, 160]
vae_history_data, vade_history_data = gather_loss(seeds, epochs, latent_dims, betas, caps)

# %%
# returns 3d array - beta x caps x seeds
# load files if already calculated, otherwise compute, then save
VAE_udr_path = pathlib.Path(weights_path / f'data/udr_VAE.npy')
if not VAE_udr_path.exists():
    VAE_udr_arr = step_B2_vae_latents_UDR.generate_udr_grid(latent_dims[0],
                                                            epochs,
                                                            seeds,
                                                            betas,
                                                            caps,
                                                            kl_threshold=0.1)
    np.save(VAE_udr_path, VAE_udr_arr)
else:
    VAE_udr_arr = np.load(VAE_udr_path)

# %%
# TODO: remove??
# same for VaDE
VaDE_udr_path = pathlib.Path(weights_path / f'data/udr_VaDE.npy')
if not VaDE_udr_path.exists():
    VaDE_udr_arr = step_B2_vae_latents_UDR.generate_udr_grid(latent_dims[0],
                                                             epochs,
                                                             seeds,
                                                             betas,
                                                             caps,
                                                             kl_threshold=0.1,
                                                             vade=True)
    np.save(VaDE_udr_path, VaDE_udr_arr)
else:
    VaDE_udr_arr = np.load(VaDE_udr_path)

# %%
'''
VAE and VaDE losses
'''
phd_util.plt_setup()
fig, axes = plt.subplots(4, 2, figsize=(8, 12))
for idx, (ax_row, theme_set) in enumerate(zip(
        axes,
        [['mse', 'mse'],
         ['kl_cap', 'gamma'],
         ['vae', 'vade'],
         ['udr', 'udr']])):
    for ax, cat, theme, theme_data, udr_arr in zip(
            ax_row,
            ['VAE', 'VaDE'],
            theme_set,
            [vae_history_data, vade_history_data],
            [VAE_udr_arr, VaDE_udr_arr]):
        if idx != 3:
            generate_heatmap(ax, cat, theme, epochs, latent_dims[0], betas, caps, theme_data)
        else:
            green_cmap = plt.get_cmap('Greens')
            # take mean across seeds on dimension 2
            udr_arr_mu = np.nanmean(udr_arr, axis=2)
            generate_arr_heatmap(ax, udr_arr_mu, betas, caps, cmap=green_cmap, constrain=(0, 1))
            ax.set_title(f'{cat}: UDR score')
plt.suptitle(
    f'Gridplots comparing averaged (9 seeds) model losses for hyperparameters based on 6 latents and 50 epochs')
path = f'../../phd-admin/PhD/part_3/images/signatures/model_scores_fine.png'
plt.savefig(path, dpi=300)

# %%
# returns 4d array - beta x caps x components x seeds
# load files if already calculated, otherwise compute, then save
VAE_comp_path = pathlib.Path(weights_path / f'data/components_VAE.npy')
if not VAE_comp_path.exists():
    VAE_comp_dict = step_B2_vae_latents_UDR.generate_comp_grid(latent_dims[0],
                                                               epochs,
                                                               seeds,
                                                               betas,
                                                               caps)
    np.save(VAE_comp_path, VAE_comp_dict)
else:
    VAE_comp_dict = np.load(VAE_comp_path, allow_pickle=True)
    VAE_comp_dict = VAE_comp_dict.item()

# same for VaDE
VaDE_comp_path = pathlib.Path(weights_path / f'data/components_VaDE.npy')
if not VaDE_comp_path.exists():
    VaDE_comp_dict = step_B2_vae_latents_UDR.generate_comp_grid(latent_dims[0],
                                                                epochs,
                                                                seeds,
                                                                betas,
                                                                caps,
                                                                vade=True)
    np.save(VaDE_comp_path, VaDE_comp_dict)
else:
    VaDE_comp_dict = np.load(VaDE_comp_path, allow_pickle=True)
    VaDE_comp_dict = VaDE_comp_dict.item()

# %%
'''
VAE and VaDE n component scores
'''
components = tuple(range(2, 13))
for comp_idx, components in enumerate(components):
    phd_util.plt_setup()
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    for ax_row, theme in zip(axes, ['bic', 'aic', 'ch', 'js']):
        for ax, cat, comp_arr in zip(ax_row, ['VAE', 'VaDE'], [VAE_comp_dict, VaDE_comp_dict]):
            # take the averages across components = 0
            # beta / cap / components / seeds
            d_arr = comp_arr[theme]
            # normalise across all of the axes
            cmap = None
            constrain = (-1, 1)
            if theme == 'ch':
                cmap = plt.get_cmap('Greens')
                constrain = (0, 1)
            # index array for given components, i.e. shape from (7, 8, 11, 9) to (7, 8, 9) - i.e. beta, caps, seeds
            # take mean over last axis, i.e. 9 seeds
            v_mu_arr = np.nanmean(d_arr[:, :, comp_idx], axis=-1)
            generate_arr_heatmap(ax, v_mu_arr, betas, caps, cmap=cmap, constrain=constrain)
            ax.set_title(f'{cat}: {theme} score')
    plt.suptitle(
        f'Gridplots comparing averaged (9 seeds) model losses for hyperparameters based on 6 latents and 50 epochs')
    path = f'../../phd-admin/PhD/part_3/images/signatures/model_component_scores_{components}.png'
    plt.savefig(path, dpi=300)

# %%
'''
Prepare split model
'''
seed = 1
epochs = 250
latent_dim = 6
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)
theme_base = f'VAE_split_e{epochs}'
# prepare the splits
n_d = len(distances)
split_input_dims = [int(5 * n_d), int(18 * n_d), int(4 * n_d)]
split_latent_dims = (4, 6, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
beta = 4
cap = 10
vae = models.SplitVAE(raw_dim=X_trans.shape[1],
                      latent_dim=latent_dim,
                      split_input_dims=split_input_dims,
                      split_latent_dims=split_latent_dims,
                      beta=beta,
                      capacity=cap,
                      split_hidden_layer_dims=split_hidden_layer_dims,
                      theme_base=theme_base,
                      seed=seed)
vae.prep_model()
vae.compile_model(optimizer='adam')
vae.load_weights(weights_path)
# get main latent dimensions (Z)
Z_mu, Z_log_var, Z = vae.encoder_model.predict(X_trans)
# single latent kl divergence, i.e. don't sum over latents
kl_loss = -0.5 * (1 + Z_log_var - np.square(Z_mu) - np.exp(Z_log_var))
# average kl divergence per latent
kl_vec = np.mean(kl_loss, axis=0)
# create sub-models for inner latent dimensions
model_A = models.Model(vae.X_input, vae.X_a)
model_B = models.Model(vae.X_input, vae.X_b)
model_C = models.Model(vae.X_input, vae.X_c)
# get thematic latent dimensions
latent_A = model_A.predict(X_trans)
latent_B = model_B.predict(X_trans)
latent_C = model_C.predict(X_trans)

#  %%
'''
plot latent maps
'''
# plot
phd_util.plt_setup()
fig, axes = plt.subplots(2, 6, figsize=(15, 8))
for ax_row, inverse in zip(axes, [False, True]):
    L = np.copy(Z)
    if inverse:
        L *= -1
    for latent_idx in range(latent_dim):
        ax = ax_row[latent_idx]
        plot_funcs.plot_scatter(ax,
                                table.x,
                                table.y,
                                L[:, latent_idx],
                                s_min=0,
                                s_max=0.8,
                                c_exp=3,
                                s_exp=5)
        ax.set_title(f'Latent {latent_idx + 1} - KL {kl_vec[latent_idx]:.2}')
        if inverse:
            ax.set_title(f'Inverse {latent_idx + 1}')
plt.suptitle(r"Geographic mappings (and inverse) for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/latents_map_inv_{beta}_{cap}.png'
plt.savefig(path, dpi=300)

#  %%
'''
plot latent correlations
'''
phd_util.plt_setup()
fig, axes = plt.subplots(2, 6, figsize=(15, 8))
for ax_row, inverse in zip(axes, [False, True]):
    L = np.copy(Z)
    if inverse:
        L *= -1
    for latent_idx in range(latent_dim):
        ax = ax_row[latent_idx]
        heatmap_corrs = plot_funcs.correlate_heatmap(len(labels), len(distances), X_trans, L[:, latent_idx])
        plot_funcs.plot_heatmap(ax,
                                heatmap=heatmap_corrs,
                                row_labels=labels,
                                col_labels=distances,
                                set_row_labels=True,
                                cbar=True)
        ax.set_title(f'Latent {latent_idx + 1} - KL {kl_vec[latent_idx]:.2}')
        if inverse:
            ax.set_title(f'Inverse {latent_idx + 1}')
plt.suptitle(r"Pearson's $\rho$ correlations (and inverse) to source variables for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/latents_corr_inv_{beta}_{cap}.png'
plt.savefig(path, dpi=300)

#  %%
'''
Plot split model latents - correlations
'''
phd_util.plt_setup()
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(3, 6, height_ratios=[5, 18, 4], width_ratios=[1, 1, 1, 1, 1, 1])
for title, ax_row, label_indices, sub_latent_X in zip(
        ('Centrality: C-{n}', 'Land-use: LU-{n}', 'Population: P-{n}'),
        ((fig.add_subplot(gs[0, i]) for i in range(1, 5)),
         (fig.add_subplot(gs[1, i]) for i in range(6)),
         (fig.add_subplot(gs[2, i]) for i in range(2, 4))),
        ((0, 5), (5, 24), (24, 27)),
        (latent_A, latent_B, latent_C)):
    lab = labels[label_indices[0]:label_indices[1]]  # five row labels
    start_src_idx, end_src_idx = label_indices
    start_src_idx *= len(distances)
    end_src_idx *= len(distances)
    for latent_idx, ax in enumerate(ax_row):
        heatmap_corrs = plot_funcs.correlate_heatmap(len(lab),
                                                     len(distances),
                                                     X_trans[:, start_src_idx:end_src_idx],
                                                     sub_latent_X[:, latent_idx])
        plot_funcs.plot_heatmap(ax,
                                heatmap=heatmap_corrs,
                                row_labels=lab,
                                col_labels=distances,
                                set_row_labels=True,
                                cbar=True)
        ax.set_title(title.format(n=latent_idx + 1))
plt.suptitle(r"Pearson's $\rho$ correlations to source variables for the split sub-latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/sub_latent_correlations_{beta}_{cap}.png'
plt.savefig(path, dpi=300)

#  %%
'''
Plot split model latents - maps
'''
phd_util.plt_setup()
fig, axes = plt.subplots(2, 6, figsize=(15, 8), gridspec_kw={'height_ratios': [1] * 2, 'width_ratios': [1] * 6})
for title, ax_indices, label_indices, sub_latent_X in zip(
        ('Centrality: C-{n}',
         'Land-use: LU-{n}',
         'Population: P-{n}'),
        ((0, 0, 4), (1, 0, 6), (0, 4, 2)),
        ((0, 5), (5, 24), (24, 27)),
        (latent_A, latent_B, latent_C)):
    lab = labels[label_indices[0]:label_indices[1]]  # five row labels
    start_src_idx, end_src_idx = label_indices
    start_src_idx *= len(distances)
    end_src_idx *= len(distances)
    row_idx, col_start_idx, n_latents = ax_indices
    for latent_idx in range(n_latents):
        # pick colour map based on strongest correlations - i.e. positive vs negative = red vs blue
        heatmap_corrs = plot_funcs.correlate_heatmap(len(lab),
                                                     len(distances),
                                                     X_trans[:, start_src_idx:end_src_idx],
                                                     sub_latent_X[:, latent_idx])
        # find largest magnitude
        t = title.format(n=latent_idx + 1)
        if np.max(heatmap_corrs) > abs(np.min(heatmap_corrs)):
            v = sub_latent_X[:, latent_idx]
        else:
            v = -sub_latent_X[:, latent_idx]
            t += ' (inverse)'
        plot_funcs.plot_scatter(axes[row_idx][col_start_idx + latent_idx],
                                table.x,
                                table.y,
                                v,
                                s_min=0,
                                s_max=0.8,
                                c_exp=3,
                                s_exp=5)
        axes[row_idx][col_start_idx + latent_idx].set_title(t)
plt.suptitle(r"Geographic mapping of the split sub-latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/sub_latent_maps_{beta}_{cap}.png'
plt.savefig(path, dpi=300)

#  %%
'''
Generate tables of the weight mappings from Z mu and Z log var to main latents
'''
Z_mu_weights = vae.autoencoder.get_layer(name='Encoder_0').get_layer(name='Z_mu').get_weights()
# rows are latent dimensions, columns are split latent dimensions
table_start = r'''
\begin{tabular}{ r | c c c c c c }
    & Latent 1 & Latent 2 & Latent 3 & Latent 4 & Latent 5 & Latent 6 \\
    \midrule'''
table_end = r'''
\end{tabular}
'''
table_header = None
table_rows = ''
# weights is a tuple of the weights 12x6 and biases x 6
z_mu_wts = Z_mu_weights[0]
z_mu_totals = np.nansum(np.abs(z_mu_wts), axis=0)
z_mu_pcs = z_mu_wts / z_mu_totals * 100
for prefix, start_idx, n_sub_latents in zip(['C', 'LU', 'P'], [0, 4, 10], [4, 6, 2]):
    # iterate the sub latents to build the row
    for i in range(n_sub_latents):
        idx = start_idx + i
        # start a new row
        table_rows += f'''
        {prefix}-{i + 1}''' + r' $Z_{\mu}$'
        for latent_idx in range(latent_dim):
            z_mu = z_mu_wts[idx, latent_idx]
            z_pc = z_mu_pcs[idx, latent_idx]
            z_colour = round(abs(z_pc) / 10) * 10  # round to 10
            if z_pc < 0:
                z_colour = '\cellcolor{blue_' + f'{z_colour:.0f}' + '}'
            else:
                z_colour = '\cellcolor{red_' + f'{z_colour:.0f}' + '}'
            table_rows += f' &{z_colour} ${z_mu:.2f}$'
        table_rows += r' \\'
    table_rows += '''
    \midrule'''
table_complete = table_start + table_rows + table_end
print(table_complete)

Z_log_var_weights = vae.autoencoder.get_layer(name='Encoder_0').get_layer(name='Z_log_var').get_weights()
# rows are latent dimensions, columns are split latent dimensions
table_start = r'''
\begin{tabular}{ r | c c c c c c }
    & Latent 1 & Latent 2 & Latent 3 & Latent 4 & Latent 5 & Latent 6 \\
    \midrule'''
table_end = r'''
\end{tabular}
'''
table_header = None
table_rows = ''
# weights is a tuple of the weights 12x6 and biases x 6
z_lv_wts = Z_log_var_weights[0]
z_lv_totals = np.nansum(np.abs(z_lv_wts), axis=0)
z_lv_pcs = z_lv_wts / z_lv_totals * 100
for prefix, start_idx, n_sub_latents in zip(['C', 'LU', 'P'], [0, 4, 10], [4, 6, 2]):
    # iterate the sub latents to build the row
    for i in range(n_sub_latents):
        idx = start_idx + i
        # start a new row
        table_rows += f'''
        {prefix}-{i + 1}''' + r' $Z_{\sigma}$'
        for latent_idx in range(latent_dim):
            # weights is a tuple of the weights 12x6 and biases x 6
            z_lv = z_lv_wts[idx, latent_idx]
            z_pc = z_lv_pcs[idx, latent_idx]
            z_colour = round(abs(z_pc) / 10) * 10  # round to 10
            if z_pc < 0:
                z_colour = '\cellcolor{blue_' + f'{z_colour:.0f}' + '}'
            else:
                z_colour = '\cellcolor{red_' + f'{z_colour:.0f}' + '}'
            # add to row
            table_rows += f' &{z_colour} ${z_lv:.2f}$'
        # add the row
        table_rows += r' \\'
    table_rows += '''
    \midrule'''
table_complete = table_start + table_rows + table_end
print(table_complete)

#  %%
'''
NOT USED spider plots of weights from sub-latents to main latents

# weights is a tuple of the weights 12x6 and biases x 6
# mu
Z_mu_weights = vae.autoencoder.get_layer(name='Encoder_0').get_layer(name='Z_mu').get_weights()
z_mu_wts = Z_mu_weights[0]
z_mu_totals = np.nansum(np.abs(z_mu_wts), axis=0)
z_mu_pcs = z_mu_wts / z_mu_totals * 100
# log var
Z_log_var_weights = vae.autoencoder.get_layer(name='Encoder_0').get_layer(name='Z_log_var').get_weights()
z_lv_wts = Z_log_var_weights[0]
z_lv_totals = np.nansum(np.abs(z_lv_wts), axis=0)
z_lv_pcs = np.abs(z_lv_wts) / z_lv_totals * 100
# plot
phd_util.plt_setup()
fig = plt.figure(figsize=(8, 12))
# labels
labels = ['C-1', 'C-2', 'C-3', 'C-4', 'LU-1', 'LU-2', 'LU-3', 'LU-4', 'LU-5', 'LU-6', 'P-1', 'P-2']
n_labels = len(labels)
theta = list(np.linspace(0.0, 2 * np.pi, n_labels, endpoint=False))
theta.append(0.0)
for ax_idx in range(6):
    ax = plt.subplot(3, 2, ax_idx + 1, projection='polar')
    ax.margins(0.6, tight=False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'Latent {ax_idx + 1}')
    ax.set_ylim(bottom=0, top=30)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    for data_arr, data_label in zip([z_mu_pcs, z_lv_pcs], [r'$Z_{\mu}$', r'$Z_{\sigma}$']):
        values = list(data_arr[:, ax_idx])
        values += values[:1]
        ax.plot(theta, values, linewidth=1, linestyle='solid', label=f'latent {data_label}')
        ax.fill(theta, values, 'b', alpha=0.1)
plt.suptitle('Spider plots showing distribution of sub-latent weights relative to latents.')
path = f'../../phd-admin/PhD/part_3/images/signatures/latent_spiders.png'
plt.savefig(path, dpi=300)
'''

# %%
'''
Sweep plots - x by y grids
'''
for l1, l2 in [[0, 3], [3, 5]]:
    # extract latents into a placeholder array
    arrs = np.full((9, 9, X_trans.shape[1]), np.nan)
    mins = np.inf
    maxs = -np.inf
    sweeps = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    for i_idx, i_key in enumerate(sweeps):
        for j_idx, j_key in enumerate(sweeps):
            # expects 2d - remember that Z is sorted by KL so extrapolate to original order
            # in this case, indices 3 and 5
            z_key = np.array([[0, 0, 0, 0, 0, 0]])
            z_key[0][l1] = i_key
            z_key[0][l2] = j_key
            arr = vae.decoder_model.predict(z_key)
            if arr.min() < mins:
                mins = arr.min()
            if arr.max() > maxs:
                maxs = arr.max()
            arrs[i_idx][j_idx] = arr
    max_val = np.array([mins, maxs])
    max_val = np.abs(max_val)
    max_val = np.min(max_val)
    arrs /= max_val
    # select indices for display
    display_idx = [1, 4, 5, 7, 9, 14, 15, 16, 23]

    phd_util.plt_setup()
    fig, ax_rows = plt.subplots(9, 9, figsize=(10, 12))
    for i_idx, i_key in enumerate(sweeps):
        for j_idx, j_key in enumerate(sweeps):
            arr = arrs[i_idx][j_idx]
            arr = np.reshape(arr, (len(labels), len(distances)))
            ax = ax_rows[i_idx][j_idx]
            rl = False
            if j_idx == 0:
                rl = True
            cl = False
            if i_idx == len(sweeps) - 1:
                cl = True
            plot_funcs.plot_heatmap(ax,
                                    arr[display_idx],
                                    np.array(labels)[display_idx],
                                    distances,
                                    set_row_labels=rl,
                                    set_col_labels=cl)
            ax.set_xlabel(f'{i_key} | {j_key}')
    plt.suptitle(f'VAE latent variable sweeps across latent dimensions {l1 + 1} & {l2 + 1}.')  # 5 and 3 in unsorted Z
    path = f'../../phd-admin/PhD/part_3/images/signatures/vae_grid_{beta}_{cap}_{l1}_{l2}.png'
    plt.savefig(path, dpi=300)

# %%
'''
Sweep plots - each latent
'''
arrs = np.full((6, 9, X_trans.shape[1]), np.nan)
mins = np.inf
maxs = -np.inf
sweeps = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
for latent_idx in list(range(6)):
    for sweep_idx, sweep in enumerate(sweeps):
        z_key = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        z_key[0][latent_idx] = sweep
        arr = vae.decoder_model.predict(z_key)
        if arr.min() < mins:
            mins = arr.min()
        if arr.max() > maxs:
            maxs = arr.max()
        arrs[latent_idx][sweep_idx] = arr
max_val = np.array([mins, maxs])
max_val = np.abs(max_val)
max_val = np.min(max_val)
arrs /= max_val

for i, (latent_start, latent_end) in enumerate(zip([1, 3, 5], [3, 5, 7])):
    phd_util.plt_setup()
    fig, ax_rows = plt.subplots(2, 9, figsize=(15, 10))
    for j, latent_idx in enumerate(list(range(latent_start, latent_end))):
        for sweep_idx, sweep in enumerate(sweeps):
            arr = arrs[latent_idx - 1][sweep_idx]
            arr = np.reshape(arr, (len(labels), len(distances)))
            ax = ax_rows[j][sweep_idx]
            rl = False
            if sweep_idx == 0:
                rl = True
            cl = False
            if j == 1:
                cl = True
            plot_funcs.plot_heatmap(ax,
                                    arr,
                                    labels,
                                    distances,
                                    set_row_labels=rl,
                                    set_col_labels=cl)
            ax.set_xlabel(f'latent: {latent_idx} param: {sweep}')
    plt.suptitle(f'VAE latent variable sweeps across individual latent dimensions {latent_start} & {latent_end - 1}.')
    path = f'../../phd-admin/PhD/part_3/images/signatures/vae_sweep_{i}_{beta}_{cap}.png'
    plt.savefig(path, dpi=300)
