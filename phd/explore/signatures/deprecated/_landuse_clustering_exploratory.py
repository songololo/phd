'''
Various attempts at separating / clustering via manual three-fold and other methods
'''

#%% for jupyter notebooks
import logging
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import linalg
from importlib import reload
from sklearn.preprocessing import robust_scale, PowerTransformer, PolynomialFeatures, minmax_scale
from sklearn.cluster import KMeans, MeanShift

from palettable.colorbrewer.qualitative import Set1_4

import phd_util
reload(phd_util)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cityseer cmap
cityseer_cmap = phd_util.cityseer_cmap()

# db connection params
db_config = {
    'host': 'localhost',
    #'host': 'host.docker.internal',
    'port': 5432,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

###
# landuse columns
landuse_columns = [
    # for table - only need specific distances (but only where not already covered in columns below)
    'id'
]

landuse_columns_dist = [
    'mu_hill_branch_wt_0_{dist}'
]

# selected landuse distances are distinct from the selected centrality columns
landuse_distances = [50, 100, 200, 400, 800, 1600]
for d in landuse_distances:
    landuse_columns += [c.format(dist=d) for c in landuse_columns_dist]


#%% load data
df_data_full = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_full_lu',
    'WHERE city_pop_id = 1'))
df_data_full = df_data_full.set_index('id')
df_full_clean = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)

df_data_100 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_100_lu',
    'WHERE city_pop_id = 1'))
df_data_100 = df_data_100.set_index('id')
df_100_clean = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)

df_data_50 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_50_lu',
    'WHERE city_pop_id = 1'))
df_data_50 = df_data_50.set_index('id')
df_50_clean = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)

df_data_20 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_20_lu',
    'WHERE city_pop_id = 1'))
df_data_20 = df_data_20.set_index('id')
df_20_clean = phd_util.clean_pd(df_data_20, drop_na='all', fill_inf=np.nan)


#%%
def set_colours(_y):
    colours = np.full(len(_y), '#0000000D')
    colours[_y == 1] = Set1_4.hex_colors[2] + 'CC'
    colours[_y == 2] = Set1_4.hex_colors[1] + 'CC'
    colours[_y == 3] = Set1_4.hex_colors[0] + 'CC'
    return colours


#%%
def X_smushed(_X):
    X = np.copy(_X)
    X = minmax_scale(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    assert X.ndim == 2
    dims = X.shape[1]
    for d in range(dims, 0, -1):
        X[:, d - 1] = np.product(X[:, :d], axis=1)
    X = robust_scale(X)
    return X

def X_diffed(_X):
    X = np.copy(_X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    assert X.ndim == 2
    dims = X.shape[1]
    for d in range(1, dims):
        X[:, d] = X[:, d] - np.sum(X[:, :d], axis=1)
    X = robust_scale(X)
    return X


#%% SVD and PCA intuition
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD, PCA

cols = []
for d in landuse_distances:
    cols.append(f'mu_hill_branch_wt_0_{d}')
X = df_full_clean[cols].values

n_comp = 2

U, Sigma, V = randomized_svd(X,
                              n_components=n_comp,
                              n_iter='auto')
# sigmas: [22079.63082254, 2555.61541932, 624.52426945, 203.34498523, 75.02773799, 26.30270526]

X_smush = X_smushed(X).round(5)
U_smush, Sigma_smush, VT_smush= randomized_svd(X_smush,
                              n_components=n_comp,
                              n_iter='auto')
# sigmas: [1362.75337409, 467.64142801, 286.77558805, 144.19760746, 63.29131599, 26.01008058]

X_diff = X_diffed(X).round(5)
U_diff, Sigma_diff, VT_diff = randomized_svd(X_diff,
                              n_components=n_comp,
                              n_iter='auto')
# sigmas: [2.00099743e+07, 8.06043560e+05, 4.44853779e+04, 7.76018403e+03, 7.74703749e+02, 1.45802539e+02]

#%%
svd_diff = TruncatedSVD(n_components=2).fit_transform(X_diff)
pca_diff = PCA(n_components=2).fit_transform(X_diff)


#%%
# see plot folder and see landuses_classification_kmeans.qgz for visual comparisons
# do scale (if multiple) otherwise larger scale of larger thresholds will overwhelm the smaller thresholds - e.g. 50m
# do not transform - transforming shifts detail to miniscule quantities where you don't want it - clouds important areas
# don't use polynomial features - shifts too much detail into the largest quantities
# going with scale=true, transform=False, poly=1
# comparison of individual distance classification yields best detail at 100m

base_theme = 'kmeans'

cols = []
for d in landuse_distances:
    cols.append(f'mu_hill_branch_wt_0_{d}')

X = df_full_clean[cols].values
X = X_diffed(X)

theme = f'{base_theme}_diffed'
print('theme:', theme)


labels = cluster.labels_
print(set(labels))

phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
pca_diff = PCA(n_components=2).fit_transform(X_diff)
ax.scatter(pca_diff[:, 0], pca_diff[:, 1], c=labels)
plt.suptitle(theme)
plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/lu_{theme}.png')
plt.show()

#asyncio.run(util.write_col_data(
#    db_config,
#    'analysis.roadnodes_50_lu',
#    labels,
#    'kmeans_4_shaped',
#    'int',
#    df_50_clean.index,
#    'id'))


#%% single distances

for d in [50, 100, 200, 400]:
    theme = f'{base_theme}_scale_True_trans_False_polyn_1_dist_{d}'
    print('theme:', theme)
    # select columns
    X = df_full_clean[[f'mu_hill_branch_wt_0_{d}']]
    X = RobustScaler().fit_transform(X)
    cluster = KMeans(n_clusters=5).fit(X)
    labels = cluster.labels_
    phd_util.plt_setup()
    df_full_clean.plot.scatter('mu_hill_branch_wt_0_50', 'mu_hill_branch_wt_0_1600', c=set_colours(labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/lu_{theme}.png')
    plt.show()


#%%
# for different n layers
cols = []
for d in landuse_distances:
    cols.append(f'mu_hill_branch_wt_0_{d}')
X = df_full_clean[cols]
X = RobustScaler().fit_transform(X)

for i in range(3, 8):
    theme = f'{base_theme}_scale_True_trans_False_polyn_1_layers_{i}'
    print('theme:', theme)
    cluster = KMeans(n_clusters=i).fit(X)
    labels = cluster.labels_
    phd_util.plt_setup()
    df_full_clean.plot.scatter('mu_hill_branch_wt_0_50', 'mu_hill_branch_wt_0_1600', c=set_colours(labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/lu_{theme}.png')
    plt.show()

    # asyncio.run(util.write_col_data(
    #   db_config,
    #   'analysis.roadnodes_full_lu',
    #   labels,
    #   theme,
    #   'int',
    #   df_temp.index,
    #   'id'))


#%%
# distance specific across all tables
tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_names = ['analysis.roadnodes_full_lu', 'analysis.roadnodes_100_lu', 'analysis.roadnodes_50_lu', 'analysis.roadnodes_20_lu']

for table, table_name in zip(tables, table_names):
    for d in [50, 100, 200]:
        theme = f'kmeans_cluster_5_{table_name.split(".")[-1]}_d_{d}'
        X = table[[f'mu_hill_branch_wt_0_{d}']]
        X = RobustScaler().fit_transform(X)
        cluster = KMeans(n_clusters=5).fit(X)
        labels = cluster.labels_
        phd_util.plt_setup()
        table.plot.scatter('mu_hill_branch_wt_0_50', 'mu_hill_branch_wt_0_1600', c=set_colours(labels))
        plt.suptitle(theme)
        plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/lu_{theme}.png')
        plt.show()

        #asyncio.run(util.write_col_data(
        #  db_config,
        #  table_name,
        #  labels,
        #  f'kmeans_cluster_5_d_{d}',
        #  'int',
        #  table.index,
        #  'id'))


#%%
# all distances
tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_names = ['analysis.roadnodes_full_lu', 'analysis.roadnodes_100_lu', 'analysis.roadnodes_50_lu', 'analysis.roadnodes_20_lu']

cols = []
for d in landuse_distances:
    cols.append(f'mu_hill_branch_wt_0_{d}')

for table, table_name in zip(tables, table_names):
    theme = f'kmeans_cluster_5_{table_name.split(".")[-1]}_d_all'
    X = table[cols]
    X = RobustScaler().fit_transform(X)
    cluster = KMeans(n_clusters=5).fit(X)
    labels = cluster.labels_
    phd_util.plt_setup()
    table.plot.scatter('mu_hill_branch_wt_0_50', 'mu_hill_branch_wt_0_1600', c=set_colours(labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/lu_{theme}.png')
    plt.show()

    #asyncio.run(util.write_col_data(
    #  db_config,
    #  table_name,
    #  labels,
    #  'kmeans_cluster_5_d_all',
    #  'int',
    #  table.index,
    #  'id'))


#%%
# manual threefold binary method - see kmeans version for starting point on thresholds

# all distances
tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_names = ['analysis.roadnodes_full_lu', 'analysis.roadnodes_100_lu', 'analysis.roadnodes_50_lu', 'analysis.roadnodes_20_lu']

for table, table_name in zip(tables, table_names):

    near_split = 2.5
    mid_split = 25
    far_split = 115

    table['lu_class'] = 0

    table.loc[(table.mu_hill_branch_wt_0_100 >= near_split) &
          (table.mu_hill_branch_wt_0_400 >= mid_split) &
          (table.mu_hill_branch_wt_0_1600 >= far_split), 'lu_class'] = 3

    table.loc[(table.mu_hill_branch_wt_0_100 >= near_split) &
              (table.mu_hill_branch_wt_0_400 >= mid_split) &
              (table.mu_hill_branch_wt_0_1600 < far_split), 'lu_class'] = 2

    table.loc[(table.mu_hill_branch_wt_0_100 >= near_split) &
              (table.mu_hill_branch_wt_0_400 < mid_split), 'lu_class'] = 1

    theme = f'lu_cluster_manual'
    table_name_short = table_name.split('.')[-1]

    phd_util.plt_setup()
    table.plot.scatter('mu_hill_branch_wt_0_50', 'mu_hill_branch_wt_0_1600', c=set_colours(table.lu_class))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/{theme}_{table_name_short}.png')
    plt.show()

    #asyncio.run(util.write_col_data(
    #  db_config,
    #  table_name,
    #  table.lu_class.values,
    #  theme,
    #  'int',
    #  table.index,
    #  'id'))


#%% STEPPED
# threefold binary method based on kmeans

tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_names = ['analysis.roadnodes_full_lu', 'analysis.roadnodes_100_lu', 'analysis.roadnodes_50_lu', 'analysis.roadnodes_20_lu']

for table, table_name in zip(tables, table_names):

    theme = f'lu_cluster_stepped_kmeans'
    table_name_short = table_name.split('.')[-1]
    print('theme:', theme)

    df_stepped = table.copy(deep=True)

    X_a = RobustScaler().fit_transform(df_stepped[['mu_hill_branch_wt_0_100']])
    cluster = KMeans(n_clusters=2).fit(X_a)
    labels = cluster.labels_
    df_stepped['step_a_labels'] = labels
    phd_util.plt_setup()
    df_stepped.plot.scatter('mu_hill_branch_wt_0_100', 'mu_hill_branch_wt_0_1600', c=set_colours(labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/{theme}_{table_name_short}_step_A.png')
    plt.show()

    k_a = 0
    if np.nanmean(df_stepped[df_stepped.step_a_labels == 1]) > np.nanmean(df_stepped[df_stepped.step_a_labels == 0]):
        k_a = 1

    idx = df_stepped.index[df_stepped.step_a_labels == k_a]
    X_b = RobustScaler().fit_transform(df_stepped.loc[idx, ['mu_hill_branch_wt_0_400']])
    cluster = KMeans(n_clusters=2).fit(X_b)
    labels = cluster.labels_
    df_stepped.loc[idx, 'step_b_labels'] = labels
    phd_util.plt_setup()
    df_stepped.plot.scatter('mu_hill_branch_wt_0_100', 'mu_hill_branch_wt_0_1600', c=set_colours(df_stepped.step_b_labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/{theme}_{table_name_short}_step_B.png')
    plt.show()
    k_b = 0
    if np.nanmean(df_stepped[df_stepped.step_b_labels == 1]) > np.nanmean(df_stepped[df_stepped.step_b_labels == 0]):
        k_b = 1

    idx = df_stepped.index[df_stepped.step_b_labels == k_b]
    X_c = RobustScaler().fit_transform(df_stepped.loc[idx, ['mu_hill_branch_wt_0_1600']])
    cluster = KMeans(n_clusters=2).fit(X_c)
    labels = cluster.labels_
    df_stepped.loc[idx, 'step_c_labels'] = labels
    phd_util.plt_setup()
    df_stepped.plot.scatter('mu_hill_branch_wt_0_100', 'mu_hill_branch_wt_0_1600', c=set_colours(df_stepped.step_c_labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/{theme}_{table_name_short}_step_C.png')
    plt.show()
    k_c = 0
    if np.nanmean(df_stepped[df_stepped.step_c_labels == 1]) > np.nanmean(df_stepped[df_stepped.step_c_labels == 0]):
        k_c = 1

    # compound
    df_stepped['lu_class'] = 0
    df_stepped.loc[(df_stepped.step_a_labels == k_a) &
                   (df_stepped.step_b_labels == k_b) &
                   (df_stepped.step_c_labels == k_c), 'lu_class'] = 3
    df_stepped.loc[(df_stepped.step_a_labels == k_a) &
                   (df_stepped.step_b_labels == k_b) &
                   (df_stepped.step_c_labels != k_c), 'lu_class'] = 2
    df_stepped.loc[(df_stepped.step_a_labels == k_a) &
                   (df_stepped.step_b_labels != k_b), 'lu_class'] = 1

    phd_util.plt_setup()
    df_stepped.plot.scatter('mu_hill_branch_wt_0_100', 'mu_hill_branch_wt_0_1600', c=set_colours(df_stepped.lu_class))
    plt.suptitle(f'{theme}_{table_name_short}')
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/{theme}_{table_name_short}_step_combined.png')
    plt.show()

    #asyncio.run(util.write_col_data(
    #  db_config,
    #  table_name,
    #  df_stepped.lu_class.values,
    #  theme,
    #  'int',
    #  df_stepped.index,
    #  'id'))


#%%
# MEANSHIFT IS SLOW!! USE sklearn.mixture.BayesianGaussianMixture instead
tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_names = ['analysis.roadnodes_full_lu', 'analysis.roadnodes_100_lu', 'analysis.roadnodes_50_lu', 'analysis.roadnodes_20_lu']

cols = []
for d in [100, 400, 1600]:
    cols.append(f'mu_hill_branch_wt_0_{d}')

for table, table_name in zip(tables, table_names):

    theme = f'meanshift_cluster_bw_0_5_bf_90'
    table_name_short = table_name.split('.')[-1]
    print('theme:', theme)

    # select columns
    X = table[cols]
    X = RobustScaler().fit_transform(X)

    #bw = estimate_bandwidth(X, n_jobs=-1)
    #print(theme, d, 'bandwidth:', bw)
    cluster = MeanShift(bandwidth=0.5, min_bin_freq=90, bin_seeding=True, cluster_all=True, n_jobs=-1).fit(X)
    labels = cluster.labels_
    phd_util.plt_setup()
    table.plot.scatter('mu_hill_branch_wt_0_50', 'mu_hill_branch_wt_0_1600', c=set_colours(labels))
    plt.suptitle(theme)
    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/lu_{theme}_{table_name_short}.png')
    plt.show()

    #asyncio.run(util.write_col_data(
    #      db_config,
    #      table_name,
    #      labels,
    #      theme,
    #      'int',
    #      table.index,
    #      'id'))

