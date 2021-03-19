# %%
import pathlib
from importlib import reload
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/gareth/dev/github/songololo/phd-generators/phd')
import phd_util
from explore import plot_funcs
from explore.signatures import sig_model_runners
from explore.signatures.deprecated import D_GMVAE_Model
from explore.theme_setup import data_path, logs_path, weights_path
from explore.theme_setup import generate_theme

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reload(phd_util)
reload(plot_funcs)
reload(D_GMVAE_Model)

#  %% load from disk
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
# PREPARE
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)

x_dim = 16
w_dim = 5
n_MC = 3
seed = 0
lambda_threshold = 0  # max entropy for 16 components is -np.log(1/16) = 2.772
cv_weight = 1
n_components = 21
epochs = 5
n_d = 8
split_input_dims = (int(5 * n_d), int(18 * n_d), int(4 * n_d))
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])

theme_base = f'GMVAE_e{epochs}_cvw{cv_weight}_lb{lambda_threshold}_s{seed}'
gmVAE = D_GMVAE_Model.GMVAE(X_trans.shape[1],
                            x_dim,
                            w_dim,
                            n_components,
                            split_input_dims,
                            split_latent_dims,
                            split_hidden_layer_dims,
                            n_MC=n_MC,
                            theme_base=theme_base,
                            seed=seed)
print(gmVAE.theme)

dir_path = pathlib.Path(weights_path / f'seed_{seed}/{gmVAE.theme}_epochs_{epochs}')
if not dir_path.exists():
    trainer = sig_model_runners.GMVAE_trainer(model=gmVAE,
                                              X_samples=X_trans,
                                              labels=labels,
                                              distances=distances,
                                              logs_path=logs_path,
                                              epochs=epochs,
                                              val_split=0.2,
                                              best_loss=True,
                                              save_path=dir_path,
                                              lambda_threshold=lambda_threshold,
                                              cv_weight=cv_weight)
    trainer.train()
else:
    gmVAE.load_weights(str(dir_path / 'weights'))

print('plotting')
cluster_probs_gmVAE = gmVAE.classify(X_trans)
cluster_assign_gmVAE = np.argmax(cluster_probs_gmVAE, axis=1)
plot_funcs.plot_prob_clusters(cluster_probs_gmVAE, f'{gmVAE.theme}_mu', table.x, table.y)

#  %%
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
colours, sizes = plot_funcs.map_diversity_colour(X_raw, cluster_assign_gmVAE)
plot_funcs.plot_scatter(ax,
                        table.x,
                        table.y,
                        c=colours,
                        s=sizes,
                        x_extents=(-12000, 12000))
path = f'../../phd-admin/PhD/part_3/images/signatures/{gmVAE.theme}_cluster.png'
plt.savefig(path, dpi=300)

#  %%
x_dim = 16
w_dim = 5
n_MC = 3
seed = 0
lambda_threshold = 0  # max entropy for 16 components is -np.log(1/16) = 2.772
cv_weight = 1
description = 'checkpoint_closeup'
n_components = 21
epochs = 20

theme_base_closeup = f'GMVAE_e{epochs}_cvw{cv_weight}_lb{lambda_threshold}_s{seed}_closeup'
gmVAE_closeup = D_GMVAE_Model.GMVAE(X_trans.shape[1],
                                    x_dim,
                                    w_dim,
                                    n_components,
                                    split_input_dims,
                                    split_latent_dims,
                                    split_hidden_layer_dims,
                                    n_MC=n_MC,
                                    theme_base=theme_base,
                                    seed=seed)
#  %% select by index and redo
idx = cluster_assign_gmVAE == 4
idx = np.logical_or(idx, cluster_assign_gmVAE == 20)
dir_path_closeup = pathlib.Path(weights_path / f'seed_{seed}/{gmVAE.theme}_epochs_{epochs}_closeup')
if not dir_path_closeup.exists():
    trainer_closeup = sig_model_runners.GMVAE_trainer(model=gmVAE_closeup,
                                                      X_samples=X_trans[idx],
                                                      labels=labels,
                                                      distances=distances,
                                                      logs_path=logs_path,
                                                      epochs=epochs,
                                                      val_split=0.2,
                                                      best_loss=True,
                                                      save_path=dir_path_closeup,
                                                      lambda_threshold=lambda_threshold,
                                                      cv_weight=cv_weight)
    trainer_closeup.train()
else:
    gmVAE_closeup.load_weights(str(dir_path_closeup / 'weights'))

print('plotting closeups')
#  %%
cluster_probs_gmVAE_closeup = gmVAE_closeup.classify(X_trans[idx])
cluster_assign_gmVAE_closeup = np.argmax(cluster_probs_gmVAE_closeup, axis=1)
plot_funcs.plot_prob_clusters(cluster_probs_gmVAE_closeup, f'{gmVAE_closeup.theme}_mu_closeup', table.x[idx], table.y[idx])
#  %%
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
colours, sizes = plot_funcs.map_diversity_colour(X_raw, cluster_assign_gmVAE)
plot_funcs.plot_scatter(ax,
                        table.x[idx],
                        table.y[idx],
                        c=colours,
                        s=sizes,
                        x_extents=(-12000, 12000))
path = f'../../phd-admin/PhD/part_3/images/signatures/{gmVAE.theme}_cluster_closeup.png'
plt.savefig(path, dpi=300)
