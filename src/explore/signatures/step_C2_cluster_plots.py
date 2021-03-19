# %%
'''
A series of plots related to clustering
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj, transform
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import phd_util
from src.explore import plot_funcs
from src.explore.signatures import sig_models
from src.explore.theme_setup import data_path
from src.explore.theme_setup import generate_theme

#  %%
'''
VaDE
'''
# PREPARE
#  %% load data and prep
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)
# setup paramaters
seed = 0
latent_dim = 16
n_d = len(distances)
split_input_dims = (int(5 * n_d), int(18 * n_d), int(4 * n_d))
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
# setup VaDE
epochs = 50
n_components = 21
theme_base = f'VaDE'
dropout = 0.05
#
vade = sig_models.VaDE(raw_dim=X_trans.shape[1],
                       latent_dim=latent_dim,
                       n_components=n_components,
                       theme_base=theme_base,
                       seed=seed,
                       dropout=dropout)
dir_path = pathlib.Path(f'./temp_weights/signatures_weights/seed_{seed}/{vade.theme}_d{latent_dim}_e{epochs}')
vade.load_weights(str(dir_path / 'weights'))
# predictions
cluster_probs_VaDE = vade.classify(X_trans).numpy()
cluster_assign_VaDE = np.argmax(cluster_probs_VaDE, axis=1)

#  %%
'''
Overall cluster max plot
'''
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colours, sizes = plot_funcs.map_diversity_colour(X_raw, cluster_assign_VaDE, n_components=n_components)
plot_funcs.plot_scatter(ax,
                        table.x,
                        table.y,
                        c=colours,
                        s=sizes,
                        x_extents=(-7000, 11500),
                        y_extents=(-4500, 7500))
plt.suptitle('Individual probabilistic components - VaDE GMM')
path = f'../phd-admin/PhD/part_3/images/signatures/{vade.theme}_cluster.png'
plt.savefig(path, dpi=300)

# %%
'''
mu of each cluster
simultaneously find locations closest to the means of the respective clusters (least norm?)

Kensington Palace Gardens
Cluster 3 centre at df iloc idx: 558090
Sample id: 1_3927D740-2BA3-4284-9192-C7C465D52CF0_2_1_9B1A2738-28F3-4C72-BCFB-EEB7E681D63F
Lng: -0.19206422999646736 Lat: 51.508717527113824

Broadway Street: West Ealing
Cluster 9 centre at df iloc idx: 364797
Sample id: 1_82F3A81F-5074-440C-8EE4-CCA545C87579_2_1_B9BC6ACA-6677-405D-B788-BA23D5E8156F
Lng: -0.32695193411871576 Lat: 51.50971434150192

Soho Square
Cluster 10 centre at df iloc idx: 392759
Sample id: 1_82DA840E-EB67-4C14-8BFC-049CF6406917
Lng: -0.13275366611025008 Lat: 51.51514873770344

Parkway Street: Camden
Cluster 12 centre at df iloc idx: 979498
Sample id: 1_545ECB3D-082E-466B-86FC-FD57F36E1E8D
Lng: -0.1460060141274238 Lat: 51.53697689237661

Eastcastle Street: Fitzrovia
Cluster 16 centre at df iloc idx: 828237
Sample id: 1_E176A6A6-3D3A-4C96-86A7-0D8006E891B7_4_1_BB45E200-4005-473D-91A9-1089A28AC3B2
Lng: -0.1376411493160056 Lat: 51.516705394800475

Hornchurch:
Cluster 20 centre at df iloc idx: 1087833
Sample id: 1_B529E5B0-05FA-4704-B03E-0B1AF85BD6E3_0_1_9D633CE9-2C4A-4D2C-878B-449AFDEF2010
Lng: 0.2142670213065938 Lat: 51.57348446030934
'''
# for getting lng lat of targeted examples
bng = Proj('epsg:27700')
lng_lat = Proj('epsg:4326')
#
phd_util.plt_setup()
fig, axes = plt.subplots(3, 7, figsize=(8, 11))
cluster_idx = 0
for row_idx, ax_row in enumerate(axes):
    for ax_idx, ax in enumerate(ax_row):
        X_cluster = X_trans[cluster_assign_VaDE == cluster_idx]
        print(f'cluster {cluster_idx + 1}: n: {X_cluster.shape[0]}')
        if X_cluster.shape[0] != 0:
            X_mu = np.mean(X_cluster, axis=0)
            if cluster_idx in [2, 15, 9, 11, 8, 19]:  # subtract "1" from plots
                # find least norm
                X_diff = X_trans - X_mu  # do against X_trans so that index corresponds to df_20
                X_norm = np.linalg.norm(X_diff, axis=1)
                X_smallest_idx = np.argmin(X_norm)
                sample_id = df_20.index[X_smallest_idx]
                lat, lng = transform(bng,
                                     lng_lat,
                                     df_20.loc[sample_id, 'x'],
                                     df_20.loc[sample_id, 'y'])
                print(f'''
                    Cluster {cluster_idx + 1} centre at df iloc idx: {X_smallest_idx}
                    Sample id: {sample_id}
                    Lng: {lng} Lat: {lat}
                ''')
            X_mu = X_mu.reshape((len(labels), -1))
            plot_funcs.plot_heatmap(ax,
                                    heatmap=X_mu,
                                    row_labels=labels,
                                    col_labels=distances,
                                    constrain=(-5, 5),
                                    set_row_labels=ax_idx == 0,
                                    set_col_labels=row_idx == 2,
                                    cbar=False)
            ax.set_title(f'Cluster {cluster_idx + 1}')
        else:
            ax.axis('off')
        cluster_idx += 1
plt.suptitle('Mean composition of each VaDE - GMM component')
path = f'../phd-admin/PhD/part_3/images/signatures/{vade.theme}_cluster_mus.png'
plt.savefig(path, dpi=300)

# %%
'''
Individual probabilistic clusters
'''
plot_funcs.plot_prob_clusters(X_raw,
                              cluster_probs_VaDE,
                              n_components,
                              f'{vade.theme}_clust_ind',
                              table.x,
                              table.y,
                              x_extents=(0, 4500),
                              y_extents=(500, 7000),
                              suptitle='Individual probabilistic components - VaDE GMM')

# %%
'''
Plots maps comparing X + GMM, Z + GMM, X + PCA + GMM, and VaDE  
Can't remember original experiments: seem to recall: beta doesn't necessarily help clustering 
'''
# prepare VAE latents
vae = sig_models.SplitVAE(raw_dim=X_trans.shape[1],
                          latent_dim=16,
                          beta=0,
                          capacity=0,
                          epochs=15,
                          split_input_dims=split_input_dims,
                          split_latent_dims=split_latent_dims,
                          split_hidden_layer_dims=split_hidden_layer_dims,
                          theme_base='VAE_e15',
                          seed=0,
                          name='VAE')
dir_path = pathlib.Path(f'./temp_weights/signatures_weights/seed_0/VAE_e15_d16_b0_c0_s0_epochs_15_batch_256_train')
vae.load_weights(str(dir_path / 'weights'))
VAE_Z_mu, VAE_Z_log_var, VAE_Z = vae.encode(X_trans, training=False)

#  %%
# prepare GMM models
# run GMM on full data otherwise discrepancy from VaDE training set
# setup X + GMM
# very slow per cycle and takes 180 iters until convergence
gmm_X = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=2, max_iter=250, verbose=2)
gmm_X.fit(X_trans)
cluster_probs_X = gmm_X.predict_proba(X_trans)
# setup Z + GMM
gmm_Z = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=2, verbose=2)
gmm_Z.fit(VAE_Z_mu)
cluster_probs_VAE_Z_gmm = gmm_Z.predict_proba(VAE_Z_mu)
# setup PCA + GMM
pca = PCA(n_components=latent_dim)
X_pca = pca.fit_transform(X_trans)
gmm_PCA = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=2, verbose=2)
gmm_PCA.fit(X_pca)
cluster_probs_PCA = gmm_PCA.predict_proba(X_pca)
# cluster max
cluster_assign_X = np.argmax(cluster_probs_X, axis=1)
cluster_assign_VAE_Z = np.argmax(cluster_probs_VAE_Z_gmm, axis=1)
cluster_assign_PCA = np.argmax(cluster_probs_PCA, axis=1)

#  %%
# Map plots
themes = ['GMM - Input variables', 'GMM - autoencoder latents', 'GMM - PCA latents', 'GMM - VaDE']
assignments = [cluster_assign_X, cluster_assign_VAE_Z, cluster_assign_PCA, cluster_assign_VaDE]
phd_util.plt_setup()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
theme_idx = 0
for ax_row in axes:
    for ax in ax_row:
        theme = themes[theme_idx]
        assigned = assignments[theme_idx]
        cmap = plt.cm.get_cmap('hsv')
        colours, sizes = plot_funcs.map_diversity_colour(X_raw, assigned, n_components)
        plot_funcs.plot_scatter(ax,
                                table.x,
                                table.y,
                                c=colours,
                                s=sizes / 2,
                                x_extents=(-5500, 8500),
                                y_extents=(-1500, 7000))
        ax.set_title(theme)
        theme_idx += 1
plt.suptitle('Comparative GMM clustering scenarios')
path = f'../phd-admin/PhD/part_3/images/signatures/{vade.theme}_gmm_comparisons_maps.png'
plt.savefig(path, dpi=300)
