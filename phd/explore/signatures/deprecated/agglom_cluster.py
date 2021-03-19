#%%
import pathlib
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
# IMPORT FIRST OTHERWISE KERAS IMPORTS OVERRIDE PLAIDML!!
from explore.theme_setup import data_path, logs_path, weights_path
from explore.theme_setup import generate_theme

import phd_util
from explore import plot_funcs
from explore.signatures import sig_models

reload(phd_util)
reload(plot_funcs)
reload(sig_models)

#%%
appendage = 'agglom_prime'

tb_key = 'df_20'
if tb_key == 'df_100':
    df_100 = pd.read_feather(data_path / 'df_100.feather')
    df_100 = df_100.set_index('id')
    table = df_100
elif tb_key == 'df_50':
    df_50 = pd.read_feather(data_path / 'df_50.feather')
    df_50 = df_50.set_index('id')
    table = df_50
elif tb_key == 'df_20':
    df_20 = pd.read_feather(data_path / 'df_20.feather')
    df_20 = df_20.set_index('id')
    table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)

pre_epochs = 100
pre_theme_base = f'VAE_split_e{pre_epochs}'

n_d = len(distances)
split_input_dims = [int(5 * n_d), int(18 * n_d), int(4 * n_d)]
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
early_stopping = 50
latent_dim = 12
seed = 0
beta = 4
cap = 10

#%%
VAE_Z_mu_path = pathlib.Path(weights_path / f'temp/vae_Z_mu_temp_{tb_key}_e{pre_epochs}_d{latent_dim}_s{seed}_b{beta}_c{cap}.npy')
if VAE_Z_mu_path.exists():
    VAE_Z_mu = np.load(VAE_Z_mu_path)
else:
    # INITIAL PRETRAIN
    vae = sig_models.SplitVAE(raw_dim=X_trans.shape[1],
                              latent_dim=latent_dim,
                              split_input_dims=split_input_dims,
                              split_latent_dims=split_latent_dims,
                              beta=beta,
                              capacity=cap,
                              split_hidden_layer_dims=split_hidden_layer_dims,
                              theme_base=pre_theme_base,
                              seed=seed)
    print(vae.theme)
    vae.prep_model()
    vae.compile_model(optimizer='adam')
    vae.fit_model(X_trans,
                  logs_path=logs_path,
                  batch_size=1024,
                  epochs=pre_epochs,
                  shuffle=True,
                  validation_split=0.2,
                  verbose=False,
                  early_stopping=early_stopping)
    vae.save_weights(weights_path)

    vae.load_weights(weights_path)
    VAE_Z_mu, VAE_Z_log_var, VAE_Z = vae.encoder_model.predict(X_trans)
    np.save(VAE_Z_mu_path, VAE_Z_mu)

    plot_funcs.plot_components(list(range(latent_dim)), labels, distances, X_trans, VAE_Z_mu, table.x, table.y,
                                label_all=False, s_min=0, s_max=0.8, c_exp=3, s_exp=5, cbar=True)
    plt.suptitle(r"Pearson's $\rho$ correlations to source variables and geographic mappings for the latents of the VAE model")
    path = f'../../phd-admin/PhD/part_3/images/signatures/{vae.theme}_pre{pre_epochs}_VAE_latents_{tb_key}_{appendage}.png'
    plt.savefig(path, dpi=300)


#%%
def sample(n):
    view_idx, x_left, x_right, y_bottom, y_top = plot_funcs.view_idx(table.x, table.y, (-4000, 6000), (-5000, 7000))
    view_idx = np.array(view_idx)
    if len(view_idx) > n:
        lim_idx = np.random.choice(len(view_idx), n, replace=False)
        sample_idx = view_idx[lim_idx]
    else:
        sample_idx = view_idx
    return sample_idx

#%%
sample_idx = sample(50000)
sample_data = VAE_Z_mu[sample_idx]

#%%
# silhouette scores plots
sil_scores = []
ch_scores = []
component_range = list(range(3, 50))
for comps in component_range:  # range(3, 50):
    print(comps)
    model = AgglomerativeClustering(n_clusters=comps)
    agglom_assign = model.fit_predict(sample_data)

    phd_util.plt_setup()
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    cmap = plt.cm.get_cmap('hsv')
    c = []
    for i in agglom_assign:
        c.append(cmap(i / comps))
    plot_funcs.plot_scatter(ax,
                            table.x[sample_idx],
                            table.y[sample_idx],
                            vals=agglom_assign,
                            c=c,
                            s=2,
                            x_extents=(-4000, 6000))
    path = f'../../phd-admin/PhD/part_3/images/signatures/agglom_clust_{tb_key}_e{pre_epochs}_d{latent_dim}_s{seed}_b{beta}_c{cap}_comp{comps}.png'
    plt.savefig(path, dpi=300)

    ss = silhouette_score(sample_data, agglom_assign, random_state=0)
    sil_scores.append(ss)
    print('silhouette', ss)

    ch = calinski_harabasz_score(sample_data, agglom_assign)
    ch_scores.append(ch)
    print('calinski harabasz', ch)

#  %%
phd_util.plt_setup()
fig, axes = plt.subplots(2, 1, figsize=(6, 8))
for ax, theme, scores in zip(axes, ['silhouette', 'calinski-harabaz'], [sil_scores, ch_scores]):
    ax.plot(component_range, scores)
    ax.set_xlim(left=component_range[0], right=component_range[-1])
    ax.set_xlabel('n components')
    ax.set_xticks(component_range)
    ax.set_xticklabels(component_range)
    ax.set_ylabel(f'{theme} score')
path = f'../../phd-admin/PhD/part_3/images/signatures/agglom_cluster_scores.png'
plt.savefig(path, dpi=300)

#%%
comps = 27
model = AgglomerativeClustering(n_clusters=comps)
agglom_assign = model.fit_predict(sample_data)

#%%
# extract means and covariances from components
counts = []
gather_mus = []
gather_covs = []
for cl_idx in range(comps):
    # index by class
    idx = agglom_assign == cl_idx
    # add assigned count
    counts.append(idx.sum())
    # index into data
    d = sample_data[idx, :]
    # take the mean values (for all latents) for the selected component
    gather_mus.append(np.mean(d, axis=0))
    # take the covariances for the selected component
    # row is variable unless using rowvar
    cov = np.cov(d, rowvar=False)
    # takes the diagonals
    diag_cov = cov[np.diag_indices(latent_dim)]
    gather_covs.append(diag_cov)
cats = np.array(counts) / np.sum(counts)
mus = np.array(gather_mus)  # GMM.means_.T - shape: (8, 16)
covs = np.array(gather_covs)  # GMM.covariances_.T - shape: (8, 16)

#  %%
epochs = 20
theme_base = f'VaDE_e{epochs}'
n_components = 27
vade = sig_models.VaDE(n_components=n_components,
                       theme_base=theme_base,
                       pre_raw_dim=X_trans.shape[1],
                       pre_latent_dim=latent_dim,
                       pre_split_input_dims=split_input_dims,
                       pre_beta=beta,
                       pre_capacity=cap,
                       pre_split_latent_dims=split_latent_dims,
                       pre_split_hidden_layer_dims=split_hidden_layer_dims,
                       pre_theme_base=pre_theme_base,
                       pre_seed=seed)
print(vade.theme, 'based on pretrained:', vade.pretrained_theme)
vade.prep_model()
vade.compile_model(optimizer='adam')
vade.load_weights_pretrained(weights_path)
vade.external_prime_GMM(cats, mus, covs)
# vade.prime_GMM(X_trans)

#  %%
# INITIAL TRAIN
vade.fit_model(X_trans,
               logs_path=logs_path,
               batch_size=1024,
               epochs=epochs,
               shuffle=True,
               validation_split=0.2,
               verbose=False,
               early_stopping=early_stopping)
vade.save_weights(weights_path)

#  %%
vade.load_weights_gamma(weights_path)
VaDE_Z_mu, VaDE_Z_log_var, VaDE_Z = vade.encoder_model.predict(X_trans)
plot_funcs.plot_components(list(range(latent_dim)), labels, distances, X_trans, VaDE_Z_mu, table.x, table.y,
                            label_all=False, s_min=0, s_max=0.8, c_exp=3, s_exp=5, cbar=True)
plt.suptitle(r"Pearson's $\rho$ correlations to source variables and geographic mappings for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/{vade.theme}_pre{pre_epochs}_VaDE_latents_{tb_key}_{appendage}.png'
plt.savefig(path, dpi=300)

#  %%
cluster_probs_VaDE = vade.predict_classifications(X_trans)
cluster_assign_VaDE = np.argmax(cluster_probs_VaDE, axis=1)
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
cmap = plt.cm.get_cmap('hsv')
c = []
for i in cluster_assign_VaDE:
    c.append(cmap(i / n_components))
plot_funcs.plot_scatter(ax,
                        table.x,
                        table.y,
                        vals=cluster_assign_VaDE,
                        c=c,
                        s=1,
                        x_extents=(-12000, 12000))
path = f'../../phd-admin/PhD/part_3/images/signatures/{vade.theme}_pre{pre_epochs}_cluster_{tb_key}_{appendage}.png'
plt.savefig(path, dpi=300)
