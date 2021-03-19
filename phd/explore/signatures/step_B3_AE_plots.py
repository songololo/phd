# %%
'''
A variety of plots pertaining to the VAE models
e.g.
- grids of model losses for various hyperparameters
- various visualisations of latents (correlations to source variables, geographic plots, etc)
'''
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, quantile_transform

import phd_util
from explore import plot_funcs
from explore.signatures import sig_models
from explore.signatures import step_B2_vae_latents_UDR
from explore.theme_setup import data_path, weights_path
from explore.theme_setup import generate_theme

from importlib import reload

reload(phd_util)
reload(plot_funcs)
reload(sig_models)
reload(step_B2_vae_latents_UDR)

# %% load from disk
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
                    fp = f'temp_weights/seed_{seed}/VAE_{key}_s{seed}_epochs_{epochs}_batch_{batch}_train/history.json'
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


def generate_heatmap(ax, theme, epochs, latent_dim, theme_data):
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
            # beta == 0
            if i == 0 and j > 0:
                text[i][j] = ''
            else:
                text[i][j] = f'{arr[i][j]:^.2g}'
    # plot
    plot_funcs.plot_heatmap(ax,
                            heatmap=scaled,
                            row_labels=[r'$\beta=' + f'{b}$' for b in betas],
                            col_labels=[f'$C={c}$' for c in [0, 4, 8, 12, 16, 20]],
                            text=text)
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
    ax.set_title(ax_title)


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
                text[i][j] = f'{arr[i][j]:^.2g}'
    # plot
    plot_funcs.plot_heatmap(ax,
                            heatmap=scaled,
                            row_labels=[r'$\beta=' + f'{b}$' for b in betas],
                            col_labels=[f'$C={c}$' for c in caps],
                            text=text,
                            constrain=constrain,
                            cmap=cmap)


# %%
'''
Fine grid plots:
Each block is beta x capacity
'''
epochs = 10
table = df_20
seeds = list(range(1, 11))
latent_dims = [8]
history_data = gather_loss(seeds, epochs, latent_dims)
# returns 3d array - beta x caps x seeds
# load files if already calculated, otherwise compute, then save
udr_path = pathlib.Path(weights_path / f'data/udr_scores.npy')
udr_mask_path = pathlib.Path(weights_path / f'data/udr_mask_counts.npy')
if not udr_path.exists():
    udr_arr, udr_mask_count_arr = step_B2_vae_latents_UDR.generate_udr_grid(latent_dims[0],
                                                                            epochs,
                                                                            seeds,
                                                                            kl_threshold=0.01)
    np.save(udr_path, udr_arr)
    np.save(udr_mask_path, udr_mask_count_arr)
else:
    udr_arr = np.load(udr_path)
    udr_mask_count_arr = np.load(udr_mask_path)


# %%
'''
VAE losses
'''
phd_util.plt_setup()
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
theme_sets = [['val_loss', 'val_rec_loss', 'udr', 'mask'],
              ['val_capacity_term', 'val_kl', 'val_kl_beta', 'val_kl_beta_cap']]
ax_idx = 1
for ax_row, theme_row in zip(axes, theme_sets):
    for ax, theme in zip(ax_row, theme_row):
        if theme == 'udr':
            green_cmap = plt.get_cmap('Greens')
            # take mean across seeds on dimension 2
            udr_arr_mu = np.nanmean(udr_arr, axis=2)
            generate_arr_heatmap(ax, udr_arr_mu, cmap=green_cmap, constrain=(0, 1))
            ax.set_title('UDR score')
        elif theme == 'mask':
            green_cmap = plt.get_cmap('Greens')
            # take mean across seeds on dimension 2
            udr_mask_count_arr_mu = np.nanmean(udr_mask_count_arr, axis=2)
            generate_arr_heatmap(ax, udr_mask_count_arr_mu, cmap=green_cmap, constrain=(0, 1))
            ax.set_title('Active axes')
        else:
            generate_heatmap(ax, theme, epochs, latent_dims[0], history_data)

plt.suptitle(r'10 seed avg. losses for $\beta$ and $C$ hyperparameters (10 epochs)')
path = f'../../phd-admin/PhD/part_3/images/signatures/model_scores_fine.png'
plt.savefig(path, dpi=300)

# %%
'''
Prepare split model
'''
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)
test_idx = phd_util.train_test_idxs(df_20, 200)  # 200 gives about 25%

# setup paramaters
epochs = 25
batch = 256
theme_base = f'VAE_e{epochs}'
n_d = len(distances)
split_input_dims = (int(5 * n_d), int(18 * n_d), int(4 * n_d))
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
latent_dim = 8
seed = 0
beta = 4
cap = 12
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
dir_path = pathlib.Path(weights_path / f'seed_{seed}/{vae.theme}_epochs_{epochs}_batch_{batch}_train')
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

#  %%
'''
plot latent maps
'''
# plot
phd_util.plt_setup()
fig, axes = plt.subplots(2, 8, figsize=(12, 8))
for ax_row, inverse in zip(axes, [False, True]):
    L = np.copy(Z_mu)
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
                                c_exp=5,
                                s_exp=3.5)
        ax.set_title(f'#{latent_idx + 1} - ' + r'$D_{KL}=' + f'{kl_vec[latent_idx]:.4f}$')
        if inverse:
            ax.set_title(f'Dim. {latent_idx + 1} - Inverse')
plt.suptitle(r"Geographic mappings (and inverses) for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/latents_map_inv_{beta}_{str(cap).replace(".", "_")}.png'
plt.savefig(path, dpi=300)

#  %%
'''
plot latent correlations
'''
phd_util.plt_setup()
fig, axes = plt.subplots(2, 8, figsize=(12, 8))
for ax_row, inverse in zip(axes, [False, True]):
    L = np.copy(Z_mu)
    if inverse:
        L *= -1
    for latent_idx in range(latent_dim):
        ax = ax_row[latent_idx]
        heatmap_corrs = plot_funcs.correlate_heatmap(len(labels), len(distances), X_trans, L[:, latent_idx])
        plot_funcs.plot_heatmap(ax,
                                heatmap=heatmap_corrs,
                                row_labels=labels,
                                col_labels=distances,
                                set_row_labels=latent_idx == 0,
                                cbar=True)
        ax.set_title(f'#{latent_idx + 1} - ' + r'$D_{KL}=' + f'{kl_vec[latent_idx]:.4f}$')
        if inverse:
            ax.set_title(f'Dim. {latent_idx + 1} - Inverse')
plt.suptitle(r"Pearson's $\rho$ correlations (and inverses) to source variables for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/latents_corr_inv_{beta}_{str(cap).replace(".", "_")}.png'
plt.savefig(path, dpi=300)

#  %%
'''
Plot split model latents - correlations
'''
for inverse in [False, True]:
    phd_util.plt_setup()
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 8, height_ratios=[5, 18, 4], width_ratios=[1, 1, 1, 1, 1, 1, 1, 1])
    for title, ax_row, label_indices, sub_latent_X in zip(
            ('Centrality: C-{n}', 'Land-use: LU-{n}', 'Population: P-{n}'),
            ((fig.add_subplot(gs[0, i]) for i in range(0, 6)),
             (fig.add_subplot(gs[1, i]) for i in range(8)),
             (fig.add_subplot(gs[2, i]) for i in range(0, 2))),
            ((0, 5), (5, 24), (24, 27)),
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
                                    col_labels=distances,
                                    set_row_labels=latent_idx == 0)
            ax.set_title(title.format(n=latent_idx + 1))
    st = r"Pearson's $\rho$ correlations to source variables for the split sub-latents of the VAE model"
    if inverse:
        st += ' - Inverse'
    plt.suptitle(st)
    path = f'../../phd-admin/PhD/part_3/images/signatures/sub_latent_correlations_{beta}_{str(cap).replace(".", "_")}'
    if inverse:
        path += '_inverse'
    path += '.png'
    plt.savefig(path, dpi=300)

#  %%
'''
Plot split model latents - maps
'''
set_A = [('Cent: C-{n}', (0, 6), (0, 5), latent_A),
         ('Pop: P-{n}', (6, 2), (24, 27), latent_C)]
set_B = [('Uses: LU-{n}', (0, 8), (5, 24), latent_B)]

for lb, set in zip(['A', 'B'], [set_A, set_B]):
    phd_util.plt_setup()
    fig, axes = plt.subplots(2, 8, figsize=(12, 8), gridspec_kw={'height_ratios': [1] * 2, 'width_ratios': [1] * 8})
    for title, ax_indices, label_indices, sub_latent_X in set:
        lab = labels[label_indices[0]:label_indices[1]]  # five row labels
        start_src_idx, end_src_idx = label_indices
        start_src_idx *= len(distances)
        end_src_idx *= len(distances)
        col_start_idx, n_latents = ax_indices
        for latent_idx in range(n_latents):
            # pick colour map based on strongest correlations - i.e. positive vs negative = red vs blue
            heatmap_corrs = plot_funcs.correlate_heatmap(len(lab),
                                                         len(distances),
                                                         X_trans[:, start_src_idx:end_src_idx],
                                                         sub_latent_X[:, latent_idx])
            # find largest magnitude
            t = title.format(n=latent_idx + 1)
            v = sub_latent_X[:, latent_idx]
            if title == 'Cent: C-{n}' and latent_idx == 0:
                # remove outliers for dimension 1 of centrality?
                v = np.clip(v, np.percentile(v, 0.05), np.percentile(v, 99.5))
            plot_funcs.plot_scatter(axes[0][col_start_idx + latent_idx],
                                    table.x,
                                    table.y,
                                    v,
                                    s_min=0,
                                    s_max=0.8,
                                    c_exp=5,
                                    s_exp=3.5)
            axes[0][col_start_idx + latent_idx].set_title(t)
            # plot inverses
            plot_funcs.plot_scatter(axes[1][col_start_idx + latent_idx],
                                    table.x,
                                    table.y,
                                    -v,
                                    s_min=0,
                                    s_max=0.8,
                                    c_exp=5,
                                    s_exp=3.5)
            axes[1][col_start_idx + latent_idx].set_title(t + ' - Inverse')
    plt.suptitle(r"Geographic mapping of the split sub-latents of the VAE model")
    path = f'../../phd-admin/PhD/part_3/images/signatures/sub_latent_maps_{lb}_{beta}_{str(cap).replace(".", "_")}.png'
    plt.savefig(path, dpi=300)

#  %%
'''
Generate tables of the weight mappings from Z mu and Z log var to main latents
'''
Z_mu_weights = vae.sampling.Z_mu_layer.get_weights()
# rows are latent dimensions, columns are split latent dimensions
table_start = r'''
\begin{tabular}{ r | c c c c c c c c }
    & Latent 1 & Latent 2 & Latent 3 & Latent 4 & Latent 5 & Latent 6 & Latent 7 & Latent 8 \\
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
for prefix, start_idx, n_sub_latents in zip(['C', 'LU', 'P'], [0, 6, 14], [6, 8, 2]):
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
\begin{tabular}{ r | c c c c c c c c}
    & Latent 1 & Latent 2 & Latent 3 & Latent 4 & Latent 5 & Latent 6 & Latent 7 & Latent 8 \\
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
for prefix, start_idx, n_sub_latents in zip(['C', 'LU', 'P'], [0, 6, 14], [6, 8, 2]):
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

#  %%
'''
Sweep plots - x by y grids
'''
for l1, l2 in [[2, 3], [3, 7]]:
    # extract latents into a placeholder array
    arrs = np.full((7, 7, X_trans.shape[1]), np.nan)
    mins = np.inf
    maxs = -np.inf
    sweeps = [-2.5, -2, -1, 0, 1, 2, 2.5]
    for i_idx, i_key in enumerate(sweeps):
        for j_idx, j_key in enumerate(sweeps):
            # expects 2d - remember that Z is sorted by KL so extrapolate to original order
            # in this case, indices 3 and 5
            z_key = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
            z_key[0][l1] = i_key
            z_key[0][l2] = j_key
            arr = vae.decode(z_key).numpy()
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
    # plot
    phd_util.plt_setup()
    fig, ax_rows = plt.subplots(7, 7, figsize=(10, 12))
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
            ax.set_xlabel(f'{i_key} $\sigma$ | {j_key} $\sigma$')
    plt.suptitle(f'VAE latent variable sweeps across latent dimensions {l1 + 1} & {l2 + 1}.')  # 5 and 3 in unsorted Z
    path = f'../../phd-admin/PhD/part_3/images/signatures/vae_grid_{beta}_{str(cap).replace(".", "_")}_{l1}_{l2}.png'
    plt.savefig(path, dpi=300)

#  %%
'''
Sweep plots - each latent
'''
arrs = np.full((12, 12, X_trans.shape[1]), np.nan)
mins = np.inf
maxs = -np.inf
sweeps = [-2.5, -2, -1, 0.5, 0, 0.5, 1, 2, 2.5]
for latent_idx in list(range(8)):
    for sweep_idx, sweep in enumerate(sweeps):
        z_key = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        z_key[0][latent_idx] = sweep
        arr = vae.decode(z_key).numpy()
        if arr.min() < mins:
            mins = arr.min()
        if arr.max() > maxs:
            maxs = arr.max()
        arrs[latent_idx][sweep_idx] = arr
max_val = np.array([mins, maxs])
max_val = np.abs(max_val)
max_val = np.min(max_val)
arrs /= max_val

for i, (latent_start, latent_end) in enumerate(zip([1, 3, 5, 7], [3, 5, 7, 9])):
    phd_util.plt_setup()
    fig, ax_rows = plt.subplots(2, 9, figsize=(12, 8))
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
            ax.set_xlabel(f'latent {latent_idx}: {sweep} $\sigma$')
    plt.suptitle(f'VAE latent variable sweeps across individual latent dimensions {latent_start} & {latent_end - 1}.')
    path = f'../../phd-admin/PhD/part_3/images/signatures/vae_sweep_{i}_{beta}_{str(cap).replace(".", "_")}.png'
    plt.savefig(path, dpi=300)
