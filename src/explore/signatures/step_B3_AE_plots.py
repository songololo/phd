# %%
'''
A variety of plots pertaining to the VAE models
e.g.
- grids of model losses for various hyperparameters
- various visualisations of latents (correlations to source variables, geographic plots, etc)
'''
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, quantile_transform

from src import util_funcs
from src.explore import plot_funcs
from src.explore.signatures import sig_models
from src.explore.signatures import step_B2_vae_latents_UDR
from src.explore.theme_setup import data_path, weights_path
from src.explore.theme_setup import generate_theme

#  %% load from disk
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')


# %%
def gather_loss(seeds, epochs, latent_dims, batch=256):
    vae_losses = ['train_loss',
                  'val_loss',
                  'val_capacity_term',
                  'val_kl',
                  'val_kl_beta',
                  'val_kl_beta_cap',
                  'val_rec_loss']
    vae_history_data = {}
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    # iterate dims, betas, caps
    for latent_dim in latent_dims:
        for beta in betas:
            caps = [0, 4, 8, 12, 16, 20]
            if beta == 0:
                caps = [0]
            for cap in caps:
                key = f'e{epochs}_d{latent_dim}_b{beta}_c{cap}'
                # each model gets a key in the dictionary - all epochs and seeds will be stored here
                vae_history_data[key] = {}
                for loss in vae_losses:
                    # each loss gets a nested key of epochs and seeds
                    vae_history_data[key][loss] = np.full((epochs, len(seeds)), np.nan)
                # iterate the seeds
                for seed_idx, seed in enumerate(seeds):
                    # in case not all models have completed running
                    fp = weights_path / f'signatures_weights/seed_{seed}/VAE_{key}_s{seed}_epochs_{epochs}_batch_{batch}_train/history.json'
                    if pathlib.Path(fp).is_file():
                        with open(fp) as f:
                            vae_data = json.load(f)
                            # fetch each of the losses
                            for loss in vae_losses:
                                for epoch_idx in range(epochs):
                                    try:
                                        vae_history_data[key][loss][epoch_idx, seed_idx] = float(
                                            vae_data[loss][epoch_idx])
                                    except IndexError as e:
                                        print(f'Key: {key}, epoch idx: {epoch_idx} unvailable for loss: {loss}')
                    else:
                        print(f'File not found: {fp}')
    return vae_history_data


def format_text(val):
    if val < 1:
        return f'{val:^.2f}'
    elif val < 10:
        return f'{val:^.1f}'
    else:
        return f'{val:^.0f}'


def generate_heatmap(ax,
                     theme,
                     epochs,
                     latent_dim,
                     theme_data,
                     set_row_labels,
                     set_col_labels):
    # empty numpy array for containing
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    arr = np.full((len(betas), 6), np.nan)
    # iterate dims, betas, caps
    for row_idx, beta in enumerate(betas):
        caps = [0, 4, 8, 12, 16, 20]
        if beta == 0:
            caps = [0]
        for col_idx, cap in enumerate(caps):
            # key for referencing history data
            key = f'e{epochs}_d{latent_dim}_b{beta}_c{cap}'
            loss_dict = theme_data[key]
            # iterate the ax rows and ax cols
            # each loss_dict contains the loss keys with numpy values of epochs x seeds
            # take the argmin across the epoch axis then take the mean of the seeds
            # arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict[theme], axis=0))
            arr[row_idx][col_idx] = np.nanmean(loss_dict[theme][-1])
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
            if i == 0 and j > 0:
                text[i][j] = ''
            else:
                text[i][j] = format_text(arr[i][j])
    # plot
    plot_funcs.plot_heatmap(ax,
                            heatmap=scaled,
                            row_labels=[r'$\beta=' + f'{b}$' for b in betas],
                            set_row_labels=set_row_labels,
                            col_labels=[f'$C={c}$' for c in [0, 4, 8, 12, 16, 20]],
                            set_col_labels=set_col_labels,
                            text=text,
                            grid_fontsize='xx-small')
    if theme == 'val_loss':
        ax_title = 'Combined loss'
    elif theme == 'val_rec_loss':
        ax_title = 'MSE reconstruction'
    elif theme == 'val_kl':
        ax_title = r'$D_{KL}$'
    elif theme == 'val_kl_beta':
        ax_title = r'$\beta \cdot D_{KL}$'
    elif theme == 'val_kl_beta_cap':
        ax_title = r'$\beta \cdot |D_{KL} - C|$'
    elif theme == 'val_capacity_term':
        ax_title = 'Capacity term $C$'
    ax.set_xlabel(ax_title)


def generate_arr_heatmap(ax,
                         arr,
                         cmap=None,
                         constrain=(-1, 1)):
    # remove outliers
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    caps = [0, 4, 8, 12, 16, 20]
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
            # beta == 0
            if i == 0 and j > 0:
                text[i][j] = ''
            else:
                text[i][j] = format_text(arr[i][j])
    # plot
    plot_funcs.plot_heatmap(ax,
                            heatmap=scaled,
                            row_labels=None,
                            set_row_labels=False,
                            col_labels=[f'$C={c}$' for c in caps],
                            set_col_labels=True,
                            text=text,
                            constrain=constrain,
                            cmap=cmap,
                            grid_fontsize='xx-small')


# %%
'''
Fine grid plots:
Each block is beta x capacity
'''
epochs = 5
table = df_20
seeds = list(range(10))
latent_dims = [6]
history_data = gather_loss(seeds, epochs, latent_dims)
# returns 3d array - beta x caps x seeds
# load files if already calculated, otherwise compute, then save
udr_path = pathlib.Path(weights_path / f'signatures_weights/data/udr_scores.npy')
udr_mask_path = pathlib.Path(weights_path / f'signatures_weights/data/udr_mask_counts.npy')
if not udr_path.exists():
    print('UDR data does not exist, creating:')
    udr_arr, udr_mask_count_arr = step_B2_vae_latents_UDR.generate_udr_grid(latent_dims[0],
                                                                            epochs,
                                                                            seeds,
                                                                            kl_threshold=0.01)
    np.save(udr_path, udr_arr)
    np.save(udr_mask_path, udr_mask_count_arr)
else:
    print('Loading precomputed UDR data:')
    udr_arr = np.load(udr_path)
    udr_mask_count_arr = np.load(udr_mask_path)

# %%
'''
VAE losses
'''
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 4,
                         figsize=(7, 5.5))
theme_sets = [['val_loss',
               'val_rec_loss',
               'udr',
               'mask'],
              ['val_capacity_term',
               'val_kl',
               'val_kl_beta',
               'val_kl_beta_cap']]
for ax_row_n, (ax_row, theme_row) in enumerate(zip(axes, theme_sets)):
    for ax_col_n, (ax, theme) in enumerate(zip(ax_row, theme_row)):
        if theme == 'udr':
            green_cmap = plt.get_cmap('Greens')
            # take mean across seeds on dimension 2
            udr_arr_mu = np.nanmean(udr_arr, axis=2)
            generate_arr_heatmap(ax,
                                 udr_arr_mu,
                                 cmap=green_cmap,
                                 constrain=(0, 1))
            ax.set_xlabel('UDR score')
        elif theme == 'mask':
            green_cmap = plt.get_cmap('Greens')
            # take mean across seeds on dimension 2
            udr_mask_count_arr_mu = np.nanmean(udr_mask_count_arr, axis=2)
            generate_arr_heatmap(ax,
                                 udr_mask_count_arr_mu,
                                 cmap=green_cmap,
                                 constrain=(0, 1))
            ax.set_xlabel('Active axes')
        else:
            generate_heatmap(ax,
                             theme,
                             epochs,
                             latent_dims[0],
                             history_data,
                             set_row_labels=ax_col_n == 0,
                             set_col_labels=ax_row_n == 0)
plt.suptitle(r'10 seed avg. losses for $\beta$ and $C$ hyperparameters (5 epochs)')
path = f'../phd-doc/doc/images/signatures/model_scores_fine.pdf'
plt.savefig(path, dpi=300)

# %%
'''
Prepare split model
'''
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all_towns', bandwise=True, max_dist=800)
X_trans = StandardScaler().fit_transform(X_raw)
test_idx = util_funcs.train_test_idxs(df_20, 200)  # 200 gives about 25%

# setup parameters
epochs = 5
batch = 256
theme_base = f'VAE_e{epochs}'
n_d = len(distances)
split_input_dims = (int(2 * n_d), int(9 * n_d), int(2 * n_d))
split_latent_dims = (4, 6, 2)
split_hidden_layer_dims = ([24, 24, 24],
                           [32, 32, 32],
                           [8, 8, 8])
latent_dim = 6
seed = 0
beta = 1
cap = 8
lr = 1e-3

# setup model
vae = sig_models.SplitVAE(raw_dim=X_trans.shape[1],
                          latent_dim=latent_dim,
                          beta=beta,
                          capacity=cap,
                          epochs=epochs,
                          split_input_dims=split_input_dims,
                          split_latent_dims=split_latent_dims,
                          split_hidden_layer_dims=split_hidden_layer_dims,
                          theme_base=theme_base,
                          seed=seed,
                          name='VAE')
dir_path = pathlib.Path(
    weights_path / f'signatures_weights/seed_{seed}/{vae.theme}_epochs_{epochs}_batch_{batch}_train')
# visualise
vae.load_weights(str(dir_path / 'weights'))
# get main latent dimensions (Z)
Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)
# single latent kl divergence, i.e. don't sum over latents
kl_loss = -0.5 * (1 + Z_log_var - np.square(Z_mu) - np.exp(Z_log_var.numpy().astype('float64')))
# average kl divergence per latent
kl_vec = np.mean(kl_loss, axis=0)
# get sub latents
latent_A, latent_B, latent_C = vae.encode_sub_latents(X_trans)

# %%
'''
plot latent maps
'''
# plot
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 6, figsize=(7, 9))
for ax_row, inverse in zip(axes, [False, True]):
    L = np.copy(Z_mu)
    if inverse:
        L *= -1
    for latent_idx in range(latent_dim):
        ax = ax_row[latent_idx]
        plot_funcs.plot_scatter(fig,
                                ax,
                                table.x,
                                table.y,
                                vals=L[:, latent_idx],
                                s_min=0,
                                s_max=1,
                                c_exp=2,
                                s_exp=3,
                                rasterized=True)
        if not inverse:
            ax.set_xlabel(f'#{latent_idx + 1} - ' + r'$D_{KL}=' + f'{kl_vec[latent_idx]:.2f}$')
        else:
            ax.set_xlabel(f'#{latent_idx + 1} - Inverse')
plt.suptitle(r"Geographic mappings (and inverses) for the latents of the VAE model")
path = f'../phd-doc/doc/images/signatures/latents_map_inv_{beta}_{str(cap).replace(".", "_")}.pdf'
plt.savefig(path, dpi=300)

# %%
'''
plot latent correlations
'''
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 6, figsize=(7, 5.5))
for row_n, (ax_row, inverse) in enumerate(zip(axes, [False, True])):
    L = np.copy(Z_mu)
    if inverse:
        L *= -1
    for latent_idx in range(latent_dim):
        ax = ax_row[latent_idx]
        heatmap_corrs = plot_funcs.correlate_heatmap(len(labels),
                                                     len(distances),
                                                     X_trans,
                                                     L[:, latent_idx])
        hm = plot_funcs.plot_heatmap(ax,
                                     heatmap=heatmap_corrs,
                                     row_labels=labels,
                                     set_row_labels=latent_idx == 0,
                                     col_labels=distances,
                                     set_col_labels=row_n == 0)
        if not inverse:
            ax.set_xlabel(f'#{latent_idx + 1} - ' + r'$D_{KL}=' + f'{kl_vec[latent_idx]:.2f}$')
        else:
            ax.set_xlabel(f'#{latent_idx + 1} - Inverse')
cbar = fig.colorbar(hm,
                    ax=axes,
                    aspect=50,
                    pad=0.01,
                    orientation='horizontal',
                    shrink=0.5)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('bottom')
cbar.set_label('Spearman $\\rho$ correlations against source variables')
plt.suptitle(r"Correlations to source variables for the VAE model latents")
path = f'../phd-doc/doc/images/signatures/latents_corr_inv_{beta}_{str(cap).replace(".", "_")}.pdf'
plt.savefig(path, dpi=300)

# %%
'''
Plot split model latents - correlations
'''
for inverse in [False, True]:
    util_funcs.plt_setup()
    fig = plt.figure(figsize=(7, 3.35))
    gs = fig.add_gridspec(3, 6, height_ratios=[2, 9, 2], width_ratios=[1, 1, 1, 1, 1, 1])
    counter = 0
    for title, ax_row, label_indices, sub_latent_X in zip(
            ('Centrality: C-{n}', 'Land-use: LU-{n}', 'Population: P-{n}'),
            ((fig.add_subplot(gs[0, i]) for i in range(0, 4)),
             (fig.add_subplot(gs[1, i]) for i in range(6)),
             (fig.add_subplot(gs[2, i]) for i in range(0, 2))),
            ((0, 2), (2, 11), (11, 13)),
            (latent_A, latent_B, latent_C)):
        lab = labels[label_indices[0]:label_indices[1]]  # five row labels
        start_src_idx, end_src_idx = label_indices
        start_src_idx *= len(distances)
        end_src_idx *= len(distances)
        for latent_idx, ax in enumerate(ax_row):
            X_latent = sub_latent_X[:, latent_idx]
            if inverse:
                X_latent *= -1
            heatmap_corrs = plot_funcs.correlate_heatmap(len(lab),
                                                         len(distances),
                                                         X_trans[:, start_src_idx:end_src_idx],
                                                         X_latent)
            plot_funcs.plot_heatmap(ax,
                                    heatmap=heatmap_corrs,
                                    row_labels=lab,
                                    set_row_labels=latent_idx == 0,
                                    col_labels=distances,
                                    set_col_labels=counter == 0)
            ax.set_xlabel(title.format(n=latent_idx + 1))
        counter += 1
    if not inverse:
        plt.suptitle(r"Spearman $\rho$ correlations for the VAE model sub-latents")
    else:
        plt.suptitle(r"Spearman $\rho$ inverse correlations for the VAE model sub-latents")
    path = f'../phd-doc/doc/images/signatures/sub_latent_correlations_{beta}_{str(cap).replace(".", "_")}'
    if inverse:
        path += '_inverse'
    path += '.pdf'
    plt.savefig(path, dpi=300)

# %%
'''
Plot split model latents - maps
'''
set_A = [('Cent. #{n}', (0, 4), (0, 2), latent_A),
         ('Pop. #{n}', (4, 2), (2, 11), latent_C)]
set_B = [('Land-use #{n}', (0, 6), (11, 13), latent_B)]

for lb, set in zip(['A', 'B'], [set_A, set_B]):
    util_funcs.plt_setup()
    fig, axes = plt.subplots(2, 6, figsize=(7, 9))
    for title, ax_indices, label_indices, sub_latent_X in set:
        lab = labels[label_indices[0]:label_indices[1]]  # five row labels
        start_src_idx, end_src_idx = label_indices
        start_src_idx *= len(distances)
        end_src_idx *= len(distances)
        col_start_idx, n_latents = ax_indices
        for latent_idx in range(n_latents):
            # find largest magnitude
            t = title.format(n=latent_idx + 1)
            v = sub_latent_X[:, latent_idx]
            if title == 'Cent: C-{n}' and latent_idx == 0:
                # remove outliers for centrality
                v = np.clip(v, np.percentile(v, 0.05), np.percentile(v, 99.5))
            ax = axes[0][col_start_idx + latent_idx]
            plot_funcs.plot_scatter(fig,
                                    ax,
                                    table.x,
                                    table.y,
                                    vals=v,
                                    s_min=0,
                                    s_max=1,
                                    c_exp=2,
                                    s_exp=3,
                                    rasterized=True)
            ax.set_xlabel(t)
            # plot inverses
            ax = axes[1][col_start_idx + latent_idx]
            plot_funcs.plot_scatter(fig,
                                    ax,
                                    table.x,
                                    table.y,
                                    vals=-v,
                                    s_min=0,
                                    s_max=1,
                                    c_exp=2,
                                    s_exp=3,
                                    rasterized=True)
            ax.set_xlabel(f'{t} inv.')
    plt.suptitle(r"Geographic mappings (and inverses) for the sub-latents of the VAE model")
    path = f'../phd-doc/doc/images/signatures/sub_latent_maps_{lb}_{beta}_{str(cap).replace(".", "_")}.pdf'
    plt.savefig(path, dpi=300)

# %%
"""
'''
Generate tables of the weight mappings from Z mu and Z log var to main latents
'''
Z_mu_weights = vae.sampling.Z_mu_layer.get_weights()
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

Z_log_var_weights = vae.sampling.Z_logvar_layer.get_weights()
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
path = f'../phd-doc/doc/images/signatures/latent_spiders.pdf'
plt.savefig(path, dpi=300)
"""

# %%
'''
Sweep plots - x by y grids
'''
for l1, l2 in [[1, 0]]:
    # extract latents into a placeholder array
    arrs = np.full((5, 5, X_trans.shape[1]), np.nan)
    sweeps = [-2, -1, 0, 1, 2]
    for i_idx, i_key in enumerate(sweeps):
        for j_idx, j_key in enumerate(sweeps):
            # expects 2d - remember that Z is sorted by KL so extrapolate to original order
            # in this case, indices 3 and 5
            z_key = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            z_key[0][l1] = i_key
            z_key[0][l2] = j_key
            arr = vae.decode(z_key).numpy()
            arrs[i_idx][j_idx] = arr
    # plot
    util_funcs.plt_setup()
    fig, ax_rows = plt.subplots(5, 5, figsize=(4, 8))
    for i_idx, i_key in enumerate(sweeps):
        for j_idx, j_key in enumerate(sweeps):
            arr = arrs[i_idx][j_idx]
            arr = np.reshape(arr, (len(labels), len(distances)))
            ax = ax_rows[i_idx][j_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            hm = plot_funcs.plot_heatmap(ax,
                                         arr,
                                         labels,
                                         distances,
                                         constrain=(-3, 3),
                                         set_row_labels=False,
                                         set_col_labels=False)
            ax.set_xlabel(f'{i_key} $\sigma$ / {j_key} $\sigma$')
    cbar = fig.colorbar(hm,
                        ax=ax_rows,
                        aspect=50,
                        pad=0.01,
                        orientation='horizontal',
                        shrink=0.75)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label('Standard deviations from the mean for decoded parameter sweeps',
                   fontsize='x-small')
    plt.suptitle(f'2D decoded "sweep" across latents {l1 + 1} & {l2 + 1}')
    path = f'../phd-doc/doc/images/signatures/vae_grid_{beta}_{str(cap).replace(".", "_")}_{l1}_{l2}.pdf'
    plt.savefig(path, dpi=300)

# %%
'''
Sweep plots - each latent
'''
arrs = np.full((6, 7, X_trans.shape[1]), np.nan)
sweeps = [-3, -2, -1, 0, 1, 2, 3]
for latent_idx in range(6):
    for sweep_idx, sweep in enumerate(sweeps):
        z_key = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        z_key[0][latent_idx] = sweep
        arr = vae.decode(z_key).numpy()
        arrs[latent_idx][sweep_idx] = arr

util_funcs.plt_setup()
fig, ax_rows = plt.subplots(6, 7, figsize=(4, 8))
for latent_idx in range(6):
    for sweep_idx, sweep in enumerate(sweeps):
        arr = arrs[latent_idx][sweep_idx]
        arr = np.reshape(arr, (len(labels), len(distances)))
        ax = ax_rows[latent_idx][sweep_idx]
        hm = plot_funcs.plot_heatmap(ax,
                                     arr,
                                     labels,
                                     distances,
                                     constrain=(-3, 3),
                                     set_row_labels=False,
                                     set_col_labels=False)
        if sweep_idx == 0:
            ax.set_ylabel(f'Latent #{latent_idx + 1}')
        ax.set_xlabel(f'${sweep}\sigma$')
cbar = fig.colorbar(hm,
                    ax=ax_rows,
                    aspect=50,
                    pad=0.01,
                    orientation='horizontal',
                    shrink=0.75)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('bottom')
cbar.set_label('Standard deviations from the mean for decoded parameter sweeps',
               fontsize='x-small')
plt.suptitle(f'Decoded "sweeps" across each latent')
path = f'../phd-doc/doc/images/signatures/vae_sweep_{beta}_{str(cap).replace(".", "_")}.pdf'
plt.savefig(path, dpi=300)
