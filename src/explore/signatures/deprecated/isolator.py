# %%
import pathlib


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import phd_util
from src.explore import plot_funcs
from src.explore.signatures import sig_models
# IMPORT FIRST OTHERWISE KERAS IMPORTS OVERRIDE PLAIDML!!
from src.explore.theme_setup import data_path, logs_path, weights_path
from src.explore.theme_setup import generate_theme
from sklearn.preprocessing import StandardScaler


reload(plot_funcs)
reload(sig_models)

from tensorflow.keras import backend as K

#  %%
tb_key = 'df_20'
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)

pre_epochs = 50
pre_theme_base = f'VAE_split_e{pre_epochs}'

n_d = len(distances)
split_input_dims = [int(5 * n_d), int(18 * n_d), int(4 * n_d)]
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
latent_dim = 8
seed = 0
beta = 4
cap = 10
gamma_strength = 0.0001
n_components = 21
epochs = 10
theme_base = f'VAE_e{epochs}'
appendage = f'moving_on'
vae = sig_models.SplitVAE(raw_dim=X_trans.shape[1],
                          latent_dim=latent_dim,
                          beta=beta,
                          capacity=cap,
                          split_input_dims=split_input_dims,
                          split_latent_dims=split_latent_dims,
                          split_hidden_layer_dims=split_hidden_layer_dims,
                          theme_base=pre_theme_base,
                          seed=seed,
                          name='VAE')
sig_models.VAE_trainer(vae,
                       X_trans,
                       labels,
                       distances,
                       logs_path=logs_path,
                       epochs=pre_epochs,
                       batch=256)
dir_path = pathlib.Path(weights_path / f'seed_{seed}/{vae.theme}')
dir_path.mkdir(exist_ok=True)
vae.save_weights(str(dir_path), overwrite=True)

# visualise
VAE_Z_mu, VAE_Z_log_var, VAE_Z = vae.encode(X_trans)
plot_funcs.plot_components(list(range(latent_dim)), labels, distances, X_trans, VAE_Z_mu, table.x, table.y,
                           label_all=False, s_min=0, s_max=0.8, c_exp=3, s_exp=5, cbar=True)
plt.suptitle(
    r"Pearson's $\rho$ correlations to source variables and geographic mappings for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/{vae.theme}_pre{pre_epochs}_VAE_latents_{tb_key}_{appendage}.png'
plt.savefig(path, dpi=300)

# %%
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
vade.load_weights_pretrained()
vade.prime_GMM(X_trans)
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

print(vade.GMM.means_.T[0])
print(K.eval(vade.GammaLayer.gmm_mu[0]))
print(vade.GMM.covariances_.T[0])
print(K.eval(vade.GammaLayer.gmm_log_var[0]))
# vade.load_weights_gamma(weights_path)
#  %%
# MAXES
cluster_probs_VaDE = vade.predict_classifications(X_trans)
plot_funcs.plot_prob_clusters(cluster_probs_VaDE, f'{vade.theme}_{gamma_strength}_{appendage}', table.x, table.y)
plot_funcs.plot_prob_clusters(cluster_probs_VaDE, f'{vade.theme}_{gamma_strength}_{appendage}', table.x, table.y,
                              max_only=True)

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

# %%
# PLOT COMPONENTS FROM LATENTS
VaDE_Z = vade.encoder_model.predict(X_trans)
plot_funcs.plot_components(list(range(latent_dim)), labels, distances, X_trans, VaDE_Z_mu, table.x, table.y,
                           label_all=False, s_min=0, s_max=0.8, c_exp=3, s_exp=5, cbar=True)
plt.suptitle(
    r"Pearson's $\rho$ correlations to source variables and geographic mappings for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/signatures/{vade.theme}_pre{pre_epochs}_VaDE_latents_{tb_key}_{appendage}.png'
plt.savefig(path, dpi=300)

# %%
# PLOT GMM FROM LATENTS
from sklearn.mixture import GaussianMixture

gmm_VaDE_Z = GaussianMixture(n_components=16, covariance_type='diag')
cluster_probs_VaDE_Z_gmm = gmm_VaDE_Z.predict_proba(VaDE_Z_mu)
cluster_assign_VaDE_Z = np.argmax(cluster_probs_VaDE_Z_gmm, axis=1)

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
path = f'../../phd-admin/PhD/part_3/images/signatures/{vade.theme}_pre{pre_epochs}_cluster_latents_{tb_key}_{appendage}.png'
plt.savefig(path, dpi=300)
