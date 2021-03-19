'''
PCA and accompanying explained variance plots
'''

# %%
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA, NMF, MiniBatchSparsePCA, FastICA
from sklearn.mixture import GaussianMixture
from palettable.cartocolors.qualitative import Prism_10
import phd_util
reload(phd_util)
from explore import plot_funcs
reload(plot_funcs)
from explore.theme_setup import data_path, plot_path
from explore.theme_setup import generate_theme


# %% load from disk
# df_full = pd.read_feather(data_path / 'df_full.feather')
# df_100 = pd.read_feather(data_path / 'df_100.feather')
# df_50 = pd.read_feather(data_path / 'df_50.feather')
df_20 = pd.read_feather(data_path / 'df_20.feather')

# df_full = df_full.set_index('id')
# df_100 = df_100.set_index('id')
# df_50 = df_50.set_index('id')
df_20 = df_20.set_index('id')


# %%
# the number of samples
def _get_sparse_explained_variance(X, components):
    # the number of samples
    n_samples = X.shape[0]
    n_components = components.shape[0]
    unit_vecs = components.copy()
    components_norm = np.linalg.norm(components, axis=1)[:, np.newaxis]
    components_norm[components_norm == 0] = 1
    unit_vecs /= components_norm

    # Algorithm, as we compute the adjustd variance for each component, we
    # subtract the variance from components in the direction of previous axes
    proj_corrected_vecs = np.zeros_like(components)
    for i in range(n_components):
        vec = components[i].copy()
        # subtract the previous projections
        for j in range(i):
            vec -= np.dot(unit_vecs[j], vec) * unit_vecs[j]

        proj_corrected_vecs[i] = vec

    # get estimated variance of Y which is matrix product of feature vector
    # and the adjusted components
    Y = np.tensordot(X, proj_corrected_vecs.T, axes=(1, 0))
    YYT = np.tensordot(Y.T, Y, axes=(1, 0))
    explained_variance = np.diag(YYT) / (n_samples - 1)

    return explained_variance


# %%
'''
CONVENTIONAL DIMENSIONALITY REDUCTION METHODS IN SCIKIT LEARN
- normalising lu by segment lengths doesn't seem to help
- PCA gives the most consistent and interpretable behaviour
- Sensitive to scaling:
- no scaling means that large numbers (betweenness) will overwhelm
- standard scaling scales to unit variance - balances out contributions
- bandwise increases definition - unclutters bands
- robust scaling - exacerbates internal variances (scales to 1st and 3rd quartiles)
'''
n_components = 4
agg_df_cols = []
for outer_theme in ['cent', 'lu', 'cens']:
    for i in range(n_components):
        agg_df_cols.append(f'{outer_theme}_{i}')
agg_df = pd.DataFrame(columns=agg_df_cols)

for outer_theme in ['all', 'cent', 'lu', 'cens']:  # , 'cent', 'lu', 'cens'
    for inner_theme in ['PCA']:  # 'PCA', 'NMF', 'sparse_PCA', 'fast_ICA'
        for bandwise in [True]:  # False,

            print('outer', outer_theme, 'inner', inner_theme)
            table = df_20
            X_raw, distances, labels = generate_theme(table, outer_theme, bandwise=bandwise)

            if inner_theme == 'PCA':
                model = PCA(n_components=n_components, whiten=False)
                X_trans = StandardScaler().fit_transform(X_raw)
                X_latent = model.fit_transform(X_trans)
            elif inner_theme == 'NMF':
                X_trans = QuantileTransformer().fit_transform(X_raw)
                model = NMF(n_components=n_components)
                X_latent = model.fit_transform(X_trans)
            elif inner_theme == 'sparse_PCA':
                model = MiniBatchSparsePCA(n_components=n_components)
                X_trans = StandardScaler().fit_transform(X_raw)
                X_latent = model.fit_transform(X_trans)
            elif inner_theme == 'fast_ICA':
                model = FastICA(n_components=n_components)
                X_trans = StandardScaler().fit_transform(X_raw)
                X_latent = model.fit_transform(X_trans)

            # get explained variance if possible
            if inner_theme == 'sparse_PCA':
                exp_var = _get_sparse_explained_variance(X_trans, model.components_)
                exp_var_ratio = None
                # eigenvector by eigenvalue - i.e. correlation to original
                loadings = model.components_.T * np.sqrt(exp_var)
                loadings = loadings.T  # transform for slicing
            elif inner_theme in ['NMF', 'fast_ICA']:
                exp_var = None
                loadings = None
            else:
                exp_var = model.explained_variance_
                exp_var_ratio = model.explained_variance_ratio_
                # eigenvector by eigenvalue - i.e. correlation to original
                loadings = model.components_.T * np.sqrt(exp_var)
                loadings = loadings.T  # transform for slicing

            component_idx = list(range(n_components))
            plot_funcs.plot_components(component_idx,
                                       labels,
                                       distances,
                                       X_trans,
                                       X_latent,
                                       table.x,
                                       table.y,
                                       explained_variances=exp_var_ratio,
                                       label_all=True,
                                       loadings=loadings,
                                       dark=False,
                                       allow_inverse=False,
                                       s_exp=4,
                                       cbar=True)

            t = f'dim_reduct_{outer_theme}_{inner_theme}'
            if bandwise:
                t += '_bandwise'
            plt.suptitle(t)
            path = plot_path / f'{t}.png'
            plt.savefig(path, dpi=300)

            cluster = GaussianMixture(n_components=10,
                                      n_init=1,
                                      max_iter=100)
            clusters = cluster.fit_predict(X_latent)
            path = plot_path / f'{t}_clusters.png'
            plot_clusters(table, clusters, Prism_10.mpl_colors, path)

            # pool components
            if outer_theme != 'all':
                for i in range(n_components):
                    agg_df[f'{outer_theme}_{i}'] = X_latent[:, i]

    # post cluster
    '''
    post pooled is dubious because clusters in lower dimensional subspaces may have been lost
    '''
    X_pooled = agg_df[agg_df_cols]
    cluster = GaussianMixture(n_components=10,
                              n_init=1,
                              max_iter=100)
    clusters = cluster.fit_predict(X_latent)
    path = plot_path / f'{t}_clusters_pooled_data.png'
    plot_funcs.plot_clusters(table, clusters, Prism_10.mpl_colors, path)




# %%
'''
0 - Plot spearman correlation matrix to demonstrate main structures
'''

these_cols = [
    # 'mu_hill_branch_wt_0_{dist}',
    # 'mu_hill_branch_wt_1_{dist}',
    # 'mu_hill_branch_wt_2_{dist}',
    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    # 'ac_eating_nw_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    # 'ac_commercial_nw_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_government_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    # 'ac_retail_food_nw_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    # 'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}'
    # 'ac_total_{dist}'
]
cols_theme = 'test'
distances_select = [50, 100, 200, 400, 800, 1600]

cols = []
for c in these_cols:
    for d in distances_select:
        cols.append(c.format(dist=d))

table = df_data_20

X = table[cols]
X = PowerTransformer().fit_transform(X)
model = PCA(svd_solver='full')
model.fit(X)
cov = model.get_covariance()

for i in range(len(cov)):
    for j in range(len(cov)):
        if j > i:
            cov[i][j] = 0

util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.suptitle(f'Covariance matrix for centralities on full (undecomposed) network')

cityseer_div_cmap = util.cityseer_diverging_cmap(dark=True)
cov_matrix = ax.imshow(cov,
                       vmin=-1.0,
                       vmax=1.0,
                       cmap=cityseer_div_cmap,
                       extent=(-0.5, len(cov) - 0.5, len(cov) - 0.5, -0.5))
ax.grid(color='white')
labels = []
for c in these_cols:
    for d in distances_select:
        if d == 50:
            labels.append(c.format(dist=d).replace('ac_', ''))
        else:
            labels.append(f'{d}')
labels.append('')
ax.set_xticks(list(range(len(labels) - 1)))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_yticks(list(range(len(labels) - 1)))
ax.set_yticklabels(labels, rotation='horizontal')
plt.colorbar(cov_matrix, ax=ax, fraction=0.05, shrink=0.5, aspect=40)

plt.savefig(f'./explore/F_feature_extraction/plots/lu_covariance.png', dpi=300)
plt.show()


# %%
'''
1A - Generate the data used for the following plot
'''

these_cols = [
    # 'mu_hill_branch_wt_0_{dist}',
    # 'mu_hill_branch_wt_1_{dist}',
    # 'mu_hill_branch_wt_2_{dist}',
    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    # 'ac_eating_nw_{dist}',
    # 'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    # 'ac_commercial_nw_{dist}',
    # 'ac_tourism_{dist}',
    # 'ac_entertainment_{dist}',
    # 'ac_government_{dist}',
    # 'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    # 'ac_retail_food_nw_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    # 'ac_health_{dist}',
    # 'ac_education_{dist}',
    # 'ac_parks_{dist}',
    # 'ac_cultural_{dist}',
    # 'ac_sports_{dist}'
    # 'ac_total_{dist}'
]

tables = [df_data_full, df_data_100, df_data_50, df_data_20]
table_names = ['full', '100m', '50m', '20m']

# columns are unravelled by column then by distance
cols = []
for c in these_cols:
    for d in distances:
        cols.append(c.format(dist=d))

# dictionary of tables containing key:tables -> list:explained variance at n components
explained_total_variance = {}
explained_indiv_variance = {}
# a nested dictionary containing key:tables -> key:cent_columns -> key:cent_dist -> list:loss at n_components
projection_loss = {}

num_components = list(range(1, 11))
for ax_col, table, table_name in zip(list(range(4)), tables, table_names):

    print(f'processing table: {table_name}')

    X = table[cols]
    X = PowerTransformer().fit_transform(X)

    # setup keys for explained variance
    explained_total_variance[table_name] = []
    explained_indiv_variance[table_name] = None
    # setup keys for projection losses
    projection_loss[table_name] = {}
    # add nested keys for centrality types
    for col in these_cols:
        theme = col.replace('_{dist}', '')
        projection_loss[table_name][theme] = {}
        for d in distances:
            projection_loss[table_name][theme][d] = []

    # for efficiency, compute PCA only once per component, then assign to respective dictionary keys
    for n_components in num_components:
        print(f'processing component: {n_components}')
        model = PCA(n_components=n_components)
        # reduce
        X_reduced = model.fit_transform(X)
        assert X_reduced.shape[1] == n_components
        # project back
        X_restored = model.inverse_transform(X_reduced)
        # now assign explained variance
        explained_total_variance[table_name].append(model.explained_variance_ratio_.sum() * 100)
        # if the last model, then save individual variances
        if n_components == num_components[-1]:
            explained_indiv_variance[table_name] = model.explained_variance_ratio_ * 100
        # and unpack projection losses to the respective columns
        counter = 0  # use a counter because of nested iteration
        for theme in projection_loss[table_name].keys():
            for d in projection_loss[table_name][theme].keys():
                err = mean_squared_error(X_restored[counter], X[counter])
                projection_loss[table_name][theme][d].append(err)
                counter += 1

# %%
'''
1B - Plot explained variance and error loss for different levels of decomposition at all network decompositions
'''

util.plt_setup(dark=True)
fig = plt.figure(figsize=(8, 12))
# use grid spec to allow merging legend columns
ht_ratios = [2, 0.02]
ht_ratios += [1 for n in these_cols]
n_rows = len(these_cols) + 2
gs = fig.add_gridspec(n_rows, 4,
                      height_ratios=ht_ratios)
# create upper row ax
ax_first_row = [fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]),
                fig.add_subplot(gs[0, 3])]
# create legend ax
ax_legend = fig.add_subplot(gs[1, :])  # legend spans both columns
ax_legend.axis('off')
# create content axes
axes_content = []
for i in range(2, n_rows):
    axes_content.append([fig.add_subplot(gs[i, 0]),
                         fig.add_subplot(gs[i, 1]),
                         fig.add_subplot(gs[i, 2]),
                         fig.add_subplot(gs[i, 3])])

# plot the explained variance row
for ax_col, (table_name, exp_var) in enumerate(explained_total_variance.items()):
    # explained variance is plotted in the first row in respective table order
    ax_first_row[ax_col].plot(num_components,
                              exp_var,
                              c=cityseer_cmap(1.0),
                              lw=1,
                              label='total')
    # 95% explained variance line
    ax_first_row[ax_col].axhline(90, c='grey', lw=0.5, ls='--', alpha=0.8)
    ax_first_row[ax_col].text(1.5, 92, '90%', fontsize=5)
    # y label
    if ax_col == 0:
        ax_first_row[ax_col].set_ylabel('explained variance')
    else:
        ax_first_row[ax_col].set_yticklabels([])
    # plot bars for individual explained variances
    ax_first_row[ax_col].bar(num_components,
                             explained_indiv_variance[table_name],
                             color=cityseer_cmap(0.2),
                             align='edge',
                             label='individual')
    # setup axes
    ax_first_row[ax_col].set_ylim(bottom=0, top=100)
    ax_first_row[ax_col].set_xlim(left=num_components[0], right=num_components[-1])
    ax_first_row[ax_col].set_xticks(num_components)
    ax_first_row[ax_col].set_xticklabels(num_components)
    ax_first_row[ax_col].set_xlabel(f'$n$ components - {table_name}')
    ax_first_row[ax_col].legend(loc='center right')

# first row setup for shared y axis
for ax_col, (table_name, cent_data) in enumerate(projection_loss.items()):
    for ax_row_n, (cent_key, dist_data) in enumerate(cent_data.items()):
        ax_row = axes_content[ax_row_n]
        # plot projected loss
        for n, (dist_key, proj_loss) in enumerate(dist_data.items()):
            ax_row[ax_col].plot(num_components,
                                np.array(proj_loss) * 100,
                                c=cityseer_cmap(n / (len(distances) - 1)),
                                alpha=0.9,
                                lw=0.75,
                                label=dist_key)
        # 5% loss line
        ax_row[ax_col].axhline(10, c='grey', lw=0.5, ls='--', alpha=0.8)
        ax_row[ax_col].text(14.5, 12, '10%', fontsize=5, horizontalalignment='right')
        # y label
        if ax_col == 0:
            cent_key = cent_key.replace('ac_', '')
            cent_key = cent_key.replace('mu_hill_branch_wt_0', 'mixed_uses')
            cent_key = cent_key.replace('_{dist}', '')
            cent_key = cent_key.replace('accommodation', 'accommod.')
            cent_key = cent_key.replace('total', 'all_landuses')
            ax_row[0].set_ylabel(f'$loss$: {cent_key}')
        else:
            ax_row[ax_col].set_yticklabels([])
        # setup axes
        ax_row[ax_col].set_ylim(bottom=0, top=100)
        ax_row[ax_col].set_xlim(left=num_components[0], right=num_components[-1])
        ax_row[ax_col].set_xticks(num_components)
        if ax_row_n == 19:
            ax_row[ax_col].set_xticklabels(num_components)
            ax_row[ax_col].set_xlabel(f'$n$ components - {table_name}')
        else:
            ax_row[ax_col].set_xticklabels([])

handles, labels = axes_content[-1][0].get_legend_handles_labels()
ax_legend.legend(
    handles,
    labels,
    title='distance thresholds for respective landuse measures',
    title_fontsize=7,
    ncol=10,
    loc='upper center')

plt.suptitle(f'PCA Explained Variance and Reconstruction Losses for $n$ components')

plt.savefig(f'./explore/F_feature_extraction/plots/lu_exp_var_reconst_losses.png', dpi=300)
plt.show()
