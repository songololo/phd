# %%
'''
A series of plots related to clustering
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pyproj import Proj, transform
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import util_funcs
from src.explore import plot_funcs
from src.explore.signatures import sig_models, sig_model_runners
from src.explore.theme_setup import data_path, weights_path
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
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True, max_dist=800)
X_trans = StandardScaler().fit_transform(X_raw)
# setup paramaters
seed = 0
latent_dim = 32
dropout = 0
epochs = 25
n_components = 21
theme_base = f'VaDE'
#
vade = sig_models.VaDE(raw_dim=X_trans.shape[1],
                       latent_dim=latent_dim,
                       n_components=n_components,
                       theme_base=theme_base,
                       seed=seed,
                       dropout=dropout)
dir_path = pathlib.Path(weights_path / f'signatures_weights/seed_{seed}/{vade.theme}_d{latent_dim}_e{epochs}')
vade.load_weights(str(dir_path / 'weights'))
# predictions
cluster_probs_VaDE = vade.classify(X_trans).numpy()
cluster_assign_VaDE = np.argmax(cluster_probs_VaDE, axis=1)

# %%
'''
Overall cluster max plot
'''
util_funcs.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
sample_colours, sample_sizes, mu_means, mu_means_cols, sorted_cluster_idx = \
    plot_funcs.map_diversity_colour(X_raw,
                                    cluster_assign_VaDE,
                                    n_components=n_components)
plot_funcs.plot_scatter(fig,
                        ax,
                        table.x,
                        table.y,
                        c=sample_colours,
                        s=sample_sizes,
                        km_per_inch=3.25,
                        rasterized=True)
# custom legend
mu_means_sorted = mu_means[sorted_cluster_idx]
mu_means_cols_sorted = mu_means_cols[sorted_cluster_idx]
legend_elements = []
for cluster_idx, mu_mean, mu_mean_col in zip(sorted_cluster_idx, mu_means_sorted, mu_means_cols_sorted):
    legend_elements.append(
        Patch(facecolor=mu_mean_col,
              edgecolor='w',
              label=f'#{cluster_idx + 1}: {mu_mean:.2f}')
    )
leg = ax.legend(handles=legend_elements,
                title=r'100m mixed-uses (cluster means)',
                fontsize='x-small',
                title_fontsize='small',
                ncol=7,
                loc=9)
plt.suptitle('Individual probabilistically assigned components - VaDE GMM')
path = f'../phd-doc/doc/images/signatures/{vade.theme}_cluster.pdf'
plt.savefig(path, dpi=300)

# %%
'''
mu of each cluster
simultaneously find locations closest to the means of the respective clusters (least norm?)

High Rd. Leytonstone - stronger main streets and environs
Cluster 1 centre at df iloc idx: 753586
Sample id: 1_B00368DD-EFE1-4321-B2EA-22D6A11E8999_0_1_DC4BCCB0-000E-4520-ACE5-282C129B0D5D
Lng: 0.011043025105092134 Lat: 51.56846022125958

Highgate: Posh residential
Cluster 3 centre at df iloc idx: 803698
Sample id: 1_97C32A27-385A-49F3-8FEC-B78C341A0204_7_1_86441039-05F0-4E38-A55F-72CECCDE2516
Lng: -0.14702181665694422 Lat: 51.563204928991816

Roehampton: Estates
Cluster 5 centre at df iloc idx: 160128
Sample id: 1_42182A5A-6CEC-410B-8B96-DBE27E83E7BE_5_1_AC15E57D-4FF2-4685-B4A7-ECCFDD880C8D
Lng: -0.24438732558678886 Lat: 51.44966483494396

Near London Bridge / City Hall - back-street office areas
Cluster 6 centre at df iloc idx: 205456
Sample id: 1_3AE6E91C-7A66-459C-8C8D-33AF21CD53E3_1_1_8E324766-B525-4277-ADDE-A0A2A159CD90
Lng: -0.08024486551064719 Lat: 51.503204547844014

Soho - vibrant
Cluster 8 centre at df iloc idx: 1095300
Sample id: 1_691C51F0-FADA-4607-ADD6-AF29AAF51613_2_1_39222C74-D7C9-47D8-8B29-45DCAD559324
Lng: -0.13574983887746525 Lat: 51.51084555099987

Illford - suburbia-like
Cluster 9 centre at df iloc idx: 109625
Sample id: 1_F51A9BAB-29C9-4483-9E3E-82DDFDF85F33_2_1_33941A52-D531-46C1-ADB1-40DA2687919B
Lng: 0.07137008744864858 Lat: 51.555855126769465

Clerkenwell - charismatic
Cluster 12 centre at df iloc idx: 123994
Sample id: 1_1B0A83CB-A6D3-48EE-8B74-445010AB397A_2_1_490B8D09-C9D7-4429-B7FE-FEF1B9B0F51B
Lng: -0.11031135027826866 Lat: 51.52215644190008

Residential area near Edmonton
Cluster 16 centre at df iloc idx: 919940
Sample id: 1_98E712C3-9EAE-46E4-A282-70CAD9AFB10D_7_1_60A42612-77CA-4A1A-84FF-83A463219D8D
Lng: -0.10234238117032159 Lat: 51.63070046494594
'''
# for getting lng lat of targeted examples
bng = Proj('epsg:27700')
lng_lat = Proj('epsg:4326')
util_funcs.plt_setup()
fig, axes = plt.subplots(3, 7, figsize=(7, 11))
counter = 0
for row_idx, ax_row in enumerate(axes):
    for ax_idx, ax in enumerate(ax_row):
        # get sorted cluster index as mapped from current iter
        cluster_sorted_idx = sorted_cluster_idx[counter]
        counter += 1
        #
        X_cluster = X_trans[cluster_assign_VaDE == cluster_sorted_idx]
        X_cluster = X_cluster.clip(-3, 3)
        print(f'cluster {cluster_sorted_idx + 1}: n: {X_cluster.shape[0]}')
        if X_cluster.shape[0] != 0:
            X_mu = np.mean(X_cluster, axis=0)
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
                Cluster {cluster_sorted_idx + 1} centre at df iloc idx: {X_smallest_idx}
                Sample id: {sample_id}
                Lng: {lng} Lat: {lat}
            ''')
            X_mu = X_mu.reshape((len(labels), -1))
            im = plot_funcs.plot_heatmap(ax,
                                         heatmap=X_mu,
                                         row_labels=labels,
                                         col_labels=distances,
                                         constrain=(-3, 3),
                                         set_row_labels=ax_idx == 0,
                                         set_col_labels=row_idx == 0)
            ax.set_xlabel(f'#{cluster_sorted_idx + 1}')
        else:
            ax.axis('off')
cbar = fig.colorbar(im,
                    ax=axes,
                    aspect=80,
                    pad=0.02,
                    orientation='horizontal',
                    ticks=[-3, -2, -1, 0, 1, 2, 3],
                    shrink=0.5)
cbar.ax.set_xticklabels([-3, -2, -1, r'$\sigma$', 1, 2, 3])
plt.suptitle('Mean composition of each VaDE - GMM cluster')
path = f'../phd-doc/doc/images/signatures/{vade.theme}_cluster_mus.pdf'
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
                              suptitle='Individual probabilistic components - VaDE GMM')

# %%
'''
Plots maps comparing X + GMM, Z + GMM, X + PCA + GMM, and VaDE  
'''
# prepare VAE latents
epochs = 5
vae = sig_models.VAE(raw_dim=X_trans.shape[1],
                     latent_dim=8,
                     beta=0,
                     capacity=0,
                     epochs=epochs,
                     name='VAE')
test_idx = util_funcs.train_test_idxs(df_20, 200)  # 200 gives about 25%
lr = 1e-3
batch = 256
trainer = sig_model_runners.VAE_trainer(model=vae,
                                        X_samples=X_trans,
                                        labels=labels,
                                        distances=distances,
                                        epochs=epochs,
                                        batch=batch,
                                        best_loss=True,
                                        lr=lr,
                                        save_path=dir_path,
                                        test_indices=test_idx)
trainer.train()
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

# %%
# Map plots
themes = ['GMM - Input variables', 'GMM - autoencoder latents', 'GMM - PCA latents', 'GMM - VaDE']
assignments = [cluster_assign_X, cluster_assign_VAE_Z, cluster_assign_PCA, cluster_assign_VaDE]
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
theme_idx = 0
for ax_row in axes:
    for ax in ax_row:
        theme = themes[theme_idx]
        assigned = assignments[theme_idx]

        sample_colours, sample_sizes, mu_means, mu_means_cols, sorted_cluster_idx = \
            plot_funcs.map_diversity_colour(X_raw,
                                            assigned,
                                            n_components=n_components)
        plot_funcs.plot_scatter(fig,
                                ax,
                                table.x,
                                table.y,
                                c=sample_colours,
                                s=sample_sizes,
                                km_per_inch=3.25,
                                rasterized=True)
        ax.set_xlabel(theme)
        theme_idx += 1
        mu_means_sorted = mu_means[sorted_cluster_idx]
        mu_means_cols_sorted = mu_means_cols[sorted_cluster_idx]
        legend_elements = []
        for cluster_idx, mu_mean, mu_mean_col in zip(sorted_cluster_idx, mu_means_sorted, mu_means_cols_sorted):
            legend_elements.append(
                Patch(facecolor=mu_mean_col,
                      edgecolor='w',
                      label=f'#{mu_mean:.2f}')
            )
        leg = ax.legend(handles=legend_elements,
                        title=r'100m mixed-uses (cluster means)',
                        fontsize='xx-small',
                        title_fontsize='x-small',
                        ncol=6,
                        loc=9)
plt.suptitle('Comparative GMM clustering scenarios')
path = f'../phd-doc/doc/images/signatures/{vade.theme}_gmm_comparisons_maps.pdf'
plt.savefig(path, dpi=300)
