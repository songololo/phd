# %%

import asyncio
import logging


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import phd_util
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import power_transform, minmax_scale



import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cityseer cmap
cityseer_cmap = phd_util.cityseer_cmap()
cityseer_div_cmap = phd_util.cityseer_diverging_cmap()

# db connection params
db_config = {
    'host': 'localhost',
    # 'host': 'host.docker.internal',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

# weighted
centrality_columns = [
    'id',
    'city_pop_id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y'
]

centrality_columns_dist = [
    'c_node_dens_simpl_{dist}',
    'c_node_dens_short_{dist}',
    'c_between_simpl_{dist}',
    'c_between_short_{dist}',
    'c_farness_simpl_{dist}',
    'c_farness_short_{dist}'
]

landuse_columns = [
    'id'
]

# non weighted
landuse_columns_dist = [
    'mu_hill_0_simpl_{dist}',
    'mu_hill_0_short_{dist}',
    'ac_commercial_nw_simpl_{dist}',
    'ac_commercial_nw_short_{dist}',
    'ac_retail_food_nw_simpl_{dist}',
    'ac_retail_food_nw_short_{dist}',
    'ac_eating_nw_simpl_{dist}',
    'ac_eating_nw_short_{dist}'
]

distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

for d in distances:
    centrality_columns += [c.format(dist=d) for c in centrality_columns_dist]
    landuse_columns += [c.format(dist=d) for c in landuse_columns_dist]

columns = centrality_columns + landuse_columns

# %%
print(f'loading columns: {columns}')
df_data_cent_full_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_full_dual', 'WHERE city_pop_id = 1'))
df_data_cent_full_dual = df_data_cent_full_dual.set_index('id')
# TODO: accidentally wrote the landuse columns to normal database - update if ever running again
df_data_landuse_full_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_full_dual', 'WHERE city_pop_id = 1'))
df_data_landuse_full_dual = df_data_landuse_full_dual.set_index('id')
df_data_full_dual = pd.concat([df_data_cent_full_dual, df_data_landuse_full_dual], axis=1, verify_integrity=True)

df_data_cent_100_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_100_dual', 'WHERE city_pop_id = 1'))
df_data_cent_100_dual = df_data_cent_100_dual.set_index('id')
# TODO: accidentally wrote the landuse columns to normal database - update if ever running again
df_data_landuse_100_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_100_dual', 'WHERE city_pop_id = 1'))
df_data_landuse_100_dual = df_data_landuse_100_dual.set_index('id')
df_data_100_dual = pd.concat([df_data_cent_100_dual, df_data_landuse_100_dual], axis=1, verify_integrity=True)

df_data_cent_50_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_50_dual', 'WHERE city_pop_id = 1'))
df_data_cent_50_dual = df_data_cent_50_dual.set_index('id')
# TODO: accidentally wrote the landuse columns to normal database - update if ever running again
df_data_landuse_50_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_50_dual', 'WHERE city_pop_id = 1'))
df_data_landuse_50_dual = df_data_landuse_50_dual.set_index('id')
df_data_50_dual = pd.concat([df_data_cent_50_dual, df_data_landuse_50_dual], axis=1, verify_integrity=True)

df_data_cent_20_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_20_dual', 'WHERE city_pop_id = 1'))
df_data_cent_20_dual = df_data_cent_20_dual.set_index('id')
# TODO: accidentally wrote the landuse columns to normal database - update if ever running again
df_data_landuse_20_dual = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_20_dual', 'WHERE city_pop_id = 1'))
df_data_landuse_20_dual = df_data_landuse_20_dual.set_index('id')
df_data_20_dual = pd.concat([df_data_cent_20_dual, df_data_landuse_20_dual], axis=1, verify_integrity=True)

print('cleaning data')
# be careful with cleanup...
df_dual_full_clean = phd_util.clean_pd(df_data_full_dual, drop_na='any', fill_inf=np.nan)
df_dual_100_clean = phd_util.clean_pd(df_data_100_dual, drop_na='any', fill_inf=np.nan)
df_dual_50_clean = phd_util.clean_pd(df_data_50_dual, drop_na='any', fill_inf=np.nan)
df_dual_20_clean = phd_util.clean_pd(df_data_20_dual, drop_na='any', fill_inf=np.nan)

# %%
'''
0 - Plot spearman correlation matrix to demonstrate main structures
'''
cols = []
for c in centrality_columns_dist:
    for d in distances:
        cols.append(c.format(dist=d))

table = df_dual_full_clean

X = table[cols]
X = power_transform(X, method='yeo-johnson')
model = PCA(svd_solver='full')
model.fit(X)
cov = model.get_covariance()

for i in range(len(cov)):
    for j in range(len(cov)):
        if j > i:
            cov[i][j] = 0

phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.suptitle(f'Covariance matrix for centralities on full (undecomposed) network')

cov_matrix = ax.imshow(cov,
                       vmin=-1.0,
                       vmax=1.0,
                       cmap=cityseer_div_cmap,
                       extent=(-0.5, len(cov) - 0.5, len(cov) - 0.5, -0.5))
ax.grid(color='white')
labels = []
for c in centrality_columns_dist:
    for d in distances:
        if d == 50:
            labels.append(c.format(dist=d).replace('c_', ''))
        else:
            labels.append(f'{d}')
labels.append('')
ax.set_xticks(list(range(len(labels) - 1)))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_yticks(list(range(len(labels) - 1)))
ax.set_yticklabels(labels, rotation='horizontal')
plt.colorbar(cov_matrix, ax=ax, fraction=0.05, shrink=0.5, aspect=40)

plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/centrality_covariance_dual.png', dpi=300)
plt.show()

# %%
'''
1A - Generate the data used for the following plot
'''
tables = [df_dual_full_clean, df_dual_100_clean, df_dual_50_clean, df_dual_20_clean]
table_names = ['full', '100m', '50m', '20m']

# columns are unravelled by column then by distance
cols = []
for c in centrality_columns_dist:
    for d in distances:
        cols.append(c.format(dist=d))

# dictionary of tables containing key:tables -> list:explained variance at n components
explained_total_variance = {}
explained_indiv_variance = {}
# a nested dictionary containing key:tables -> key:cent_columns -> key:cent_dist -> list:loss at n_components
projection_loss = {}

num_components = list(range(1, 16))
for ax_col, table, table_name in zip(list(range(4)), tables, table_names):

    print(f'processing table: {table_name}')

    X = table[cols]
    X = power_transform(X, method='yeo-johnson', standardize=True)

    # setup keys for explained variance
    explained_total_variance[table_name] = []
    explained_indiv_variance[table_name] = None
    # setup keys for projection losses
    projection_loss[table_name] = {}
    # add nested keys for centrality types
    for col in centrality_columns_dist:
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

phd_util.plt_setup()
fig = plt.figure(figsize=(8, 12))
# use grid spec to allow merging legend columns
gs = fig.add_gridspec(8, 4, height_ratios=[2, 0.02, 1, 1, 1, 1, 1, 1])
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
for i in range(2, 8):
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
            cent_key = cent_key.replace('c_', '')
            cent_key = cent_key.replace('_met_', '')
            ax_row[0].set_ylabel(f'$loss$: {cent_key}')
        else:
            ax_row[ax_col].set_yticklabels([])
        # setup axes
        ax_row[ax_col].set_ylim(bottom=0, top=100)
        ax_row[ax_col].set_xlim(left=num_components[0], right=num_components[-1])
        ax_row[ax_col].set_xticks(num_components)
        if ax_row_n == 5:
            ax_row[ax_col].set_xticklabels(num_components)
            ax_row[ax_col].set_xlabel(f'$n$ components - {table_name}')
        else:
            ax_row[ax_col].set_xticklabels([])

handles, labels = axes_content[-1][0].get_legend_handles_labels()
ax_legend.legend(
    handles,
    labels,
    title='distance thresholds for respective centrality measures',
    title_fontsize=7,
    ncol=10,
    loc='upper center')

plt.suptitle(f'PCA Explained Variance and Reconstruction Losses for $n$ components')

plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/cent_exp_var_reconst_losses_dual.png', dpi=300)
plt.show()

# %%
'''
2 - Choose level of network decomposition and plot heatmaps:
Show amount of correlation and respective ordering vs. strongest PCA components
'''

cols = []
for c in centrality_columns_dist:
    for d in distances:
        cols.append(c.format(dist=d))

table = df_dual_20_clean
X = table[cols]
# PCA centers the data but doesn't scale - use whiten param to scale to unit variance
X = power_transform(X, method='yeo-johnson', standardize=True)

n_components = 6

# model = PCA(n_components=n_components, whiten=True)
# theme = 'PCA'

X = minmax_scale(X)
from sklearn.decomposition import NMF

model = NMF(n_components=n_components)
theme = 'NMF'

# from sklearn.decomposition import SparsePCA
# model = SparsePCA(n_components=n_components)
# theme = 'sparse_PCA'

# from sklearn.decomposition import MiniBatchSparsePCA
# model = MiniBatchSparsePCA(n_components=n_components)
# theme = 'minibatch_sparse_PCA'

# from sklearn.decomposition import FastICA
# model = FastICA(n_components=n_components)
# theme = 'fast_ICA'

# CRASHES
# from sklearn.decomposition import KernelPCA
# model = KernelPCA(n_components=n_components, kernel='poly')
# theme = 'kernel_PCA'

# reduce
X_reduced = model.fit_transform(X)
assert X_reduced.shape[1] == n_components
# project back
# X_restored = model.inverse_transform(X_reduced)


phd_util.plt_setup()
fig = plt.figure(figsize=(12, 6))
# use grid spec to allow merging legend columns
gs = fig.add_gridspec(2, n_components + 1,
                      width_ratios=[1, 1, 1, 1, 1, 1, 0.02],
                      height_ratios=[1, 2])
# create content axes
axes_content = []
for ax_row in range(2):
    ax_elem = []
    for ax_col in range(n_components):
        ax_elem.append(fig.add_subplot(gs[ax_row, ax_col]))
    axes_content.append(ax_elem)

# create legend axes
axes_legends = [fig.add_subplot(gs[0, n_components]), fig.add_subplot(gs[1, 6]), fig.add_subplot(gs[1, 6])]
for ax in axes_legends:
    ax.axis('off')

# create heatmaps for original vectors plotted against the top PCA components
for n in range(n_components):
    print(f'processing component {n}')
    heatmap = np.full((len(centrality_columns_dist), 10), 0.0)
    counter = 0
    for i in range(len(centrality_columns_dist)):
        for j in range(len(distances)):
            heatmap[i][j] = spearmanr(X[:, counter], X_reduced[:, n])[0]
            counter += 1
    # loadings = (model.components_.T * np.sqrt(model.explained_variance_)).T
    # for i in range(len(these_cols)):
    #    heatmap[i] = loadings[n][i * 10:i * 10 + 10]
    # plot
    ax = axes_content[0][n]
    ax.imshow(heatmap,
              cmap=cityseer_div_cmap,
              vmin=-1,
              vmax=1,
              origin='upper')
    # set axes
    if n == 0:
        labels = ['']
        for c in centrality_columns_dist:
            lb = c.replace('_{dist}', '')
            lb = lb.replace('c_', '')
            labels.append(lb)
        ax.set_yticklabels(labels, rotation='horizontal')
    else:
        ax.set_yticklabels([])
    ax.set_xticks(list(range(len(distances))))
    ax.set_xticklabels([])

    c = X_reduced[:, n]
    ax_map = axes_content[1][n]
    c = power_transform(c.reshape(-1, 1), method='yeo-johnson')
    c = c.T[0]
    c = np.clip(c, np.percentile(c, 1), np.percentile(c, 99))
    c = minmax_scale(c)
    ax_map.scatter(table.x,
                   table.y,
                   c=-c,
                   s=.3,
                   # alpha=1,
                   # marker='.',
                   linewidths=0,
                   edgecolors='none',
                   cmap='Spectral')
    x_center = np.nanmedian(table.x)
    y_center = np.nanmedian(table.y)
    ax_map.set_xlim(left=x_center - 2500, right=x_center + 4000)
    ax_map.set_ylim(bottom=y_center - 4000, top=y_center + 6000)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_aspect(1)

    # axes_legends[0].colorbar(aspect=50, pad=0.05, ticks=[])

plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/cent_dim_reduct_{theme}_dual.png', dpi=450)
plt.show()
