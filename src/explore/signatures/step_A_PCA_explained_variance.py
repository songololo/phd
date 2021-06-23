#%%
'''
PCA and accompanying explained variance plots
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src import util_funcs
from src.explore import plot_funcs
from src.explore.theme_setup import data_path
from src.explore.theme_setup import generate_theme

#  %% load from disk
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')

#  %%
'''
Plot PCA

- normalising lu by segment lengths doesn't seem to help
- PCA gives the most consistent and interpretable behaviour
- Sensitive to scaling:
- no scaling means that large numbers (betweenness) will overwhelm
- standard scaling scales to unit variance - balances out contributions
- bandwise increases definition - unclutters bands
- robust scaling - exacerbates internal variances (scales to 1st and 3rd quartiles)
'''
table = df_20
X_raw, distances, labels = generate_theme(table,
                                          'all_towns',
                                          bandwise=True,
                                          max_dist=800)
n_components = 6
model = PCA()
X_trans = StandardScaler().fit_transform(X_raw)
X_latent = model.fit_transform(X_trans)

#%%
# explained variance
exp_var = model.explained_variance_
exp_var_ratio = model.explained_variance_ratio_
# eigenvector by eigenvalue - i.e. correlation to original
loadings = model.components_.T * np.sqrt(exp_var)
loadings = loadings.T  # transform for slicing
# plot
X_latent_clipped = np.clip(X_latent,
                           np.percentile(X_latent, 2),
                           np.percentile(X_latent, 100))
im = plot_funcs.plot_components([0, 1, 2, 3],
                                labels,
                                [f'{d}m' for d in distances],
                                None,  # X ignored if loadings is not None
                                X_latent_clipped,
                                df_20.x,
                                df_20.y,
                                corr_tags=[f'Loadings #{n}' for n in range(1, 5)],
                                map_tags=[r'explained $\sigma^{2}$' + f' {e_v:.1%}' for e_v in exp_var_ratio],
                                loadings=loadings,
                                label_all=False,
                                s_min=0,
                                 s_max=1,
                                c_exp=1,
                                s_exp=2,
                                cbar=True,
                                figsize=(7, 7.5))
plt.suptitle('Principal Component Analysis')
path = f'../phd-doc/doc/images/signatures/pca.pdf'
plt.savefig(path, dpi=300)

# %%
'''
Generate the data used for the components plot
'''
table = df_20
X_raw, distances, labels = generate_theme(table, 'all_towns', bandwise=True, max_dist=800)
X_trans = StandardScaler().fit_transform(X_raw)
#
explained_total_variance = []
explained_indiv_variance = None
reconstruction_loss = []
#
components = list(range(1, 21))
# for efficiency, compute PCA only once per component, then assign to respective dictionary keys
for n_components in components:
    print(f'processing component: {n_components}')
    model = PCA(n_components=n_components)
    X_latent = model.fit_transform(X_trans)
    X_restored = model.inverse_transform(X_latent)
    #
    explained_total_variance.append(model.explained_variance_ratio_.sum() * 100)
    reconstruction_loss.append(mean_squared_error(X_restored, X_trans) * 100)
    # if the last model, then save individual variances
    if n_components == components[-1]:
        explained_indiv_variance = model.explained_variance_ratio_ * 100

#  %%
'''
Plot explained variance and error loss
'''
cityseer_cmap = util_funcs.cityseer_cmap()
util_funcs.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
# plot the explained variance row
# explained variance is plotted in the first row in respective table order
ax.plot(components, explained_total_variance, c='grey', lw=1, label='Total explained variance')
ax.plot(components, reconstruction_loss, c=cityseer_cmap(1.0), ls='--', label='MSE')
ax.bar(components, explained_indiv_variance, color=cityseer_cmap(0.2), align='center', label='Explained variance')
# setup axes
ax.set_ylabel('explained $\sigma^2$ / $MSE$')
ax.set_ylim(bottom=0, top=100)
ax.set_xlim(left=components[0], right=components[-1])
ax.set_xticks(components)
ax.set_xticklabels(components)
ax.set_xlabel('$n$ components')
ax.legend(loc='center right')
ax.legend()

plt.suptitle(f'PCA explained $\sigma^2$ / $MSE$ reconstruction losses for ${components[-1]}$ components')
path = f'../phd-doc/doc/images/signatures/pca_components.pdf'
plt.savefig(path)
