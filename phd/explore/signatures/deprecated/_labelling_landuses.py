'''
Earlier attempts to find ways to separate urban areas into definable areas / clusters
Ended up being a manual affair using cutoff thresholds...
'''

#%% for jupyter notebooks
import logging
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MeanShift

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
    'port': 5433,
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
landuse_distances = [100, 400, 1600]
for d in landuse_distances:
    landuse_columns += [c.format(dist=d) for c in landuse_columns_dist]


#%% load data
df_data_full = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_full_lu',
    'WHERE city_pop_id <= 150'))
df_data_full = df_data_full.set_index('id')
df_full_clean = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)

df_data_100 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_100_lu',
    'WHERE city_pop_id <= 150'))
df_data_100 = df_data_100.set_index('id')
df_100_clean = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)

df_data_50 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_50_lu',
    'WHERE city_pop_id <= 150'))
df_data_50 = df_data_50.set_index('id')
df_50_clean = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)

df_data_20 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_20_lu',
    'WHERE city_pop_id <= 150'))
df_data_20 = df_data_20.set_index('id')
df_20_clean = phd_util.clean_pd(df_data_20, drop_na='all', fill_inf=np.nan)


#%%
cols = []
for d in [100, 400, 1600]:
    cols.append(f'mu_hill_branch_wt_0_{d}')

tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_names = ['analysis.roadnodes_full_lu', 'analysis.roadnodes_100_lu', 'analysis.roadnodes_50_lu', 'analysis.roadnodes_20_lu']
decomp_levels = ['full', '100', '50', '20']
meanshift_bin_freq = [30, 45, 50, 120]

x_label = r'Mixed Uses $H_{0\ \beta=-0.04\ d_{max}=100m}}$'
y_label = r'Mixed Uses $H_{0\ \beta=-0.0025\ d_{max}=1600m}}$'

from palettable.colorbrewer.qualitative import Set1_4

def label_sorter(_X, _labels):

    # calculate the means across all dimensions of the X matrix
    idx_0 = np.where(_labels == 0)
    mean_0 = np.nanmean(_X[idx_0])
    idx_1 = np.where(_labels == 1)
    mean_1 = np.nanmean(_X[idx_1])
    idx_2 = np.where(_labels == 2)
    mean_2 = np.nanmean(_X[idx_2])
    idx_3 = np.where(_labels == 3)
    mean_3 = np.nanmean(_X[idx_3])
    # get the index order of the sorted means
    order = np.argsort([mean_0, mean_1, mean_2, mean_3])
    # create a new labels array based on the new ordering
    new_labels = np.full(len(_labels), 0)
    new_labels[_labels == order[1]] = 1
    new_labels[_labels == order[2]] = 2
    new_labels[_labels == order[3]] = 3
    # create a color array with colors matching the above sort order
    # colors based on Set1_4 colorbrewer palette
    colors = np.full(len(_labels), '#2e2e2e26')
    colors[_labels == order[1]] = Set1_4.hex_colors[2] + '66'
    colors[_labels == order[2]] = Set1_4.hex_colors[1] + '66'
    colors[_labels == order[3]] = Set1_4.hex_colors[0] + '66'
    return new_labels, colors

for table, table_name, decomp, ms_bf in zip(tables, table_names, decomp_levels, meanshift_bin_freq):

    print('table', table_name)
    phd_util.plt_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle('Comparison of landuse intensity clustering methods')

    intensity_categories = ['LLL', 'HLL', 'HHL', 'HHH']

    #### KMEANS METHOD
    theme = f'lu_cluster_kmeans'
    print('theme:', theme)

    X = table[cols]
    X = RobustScaler().fit_transform(X)

    cluster = KMeans(n_clusters=4).fit(X)
    labels = cluster.labels_

    labels_shuffled, colors = label_sorter(X, labels)

    for n in range(4):
        idx = np.where(labels_shuffled == n)
        x = table.mu_hill_branch_wt_0_100.iloc[idx]
        y = table.mu_hill_branch_wt_0_1600.iloc[idx]
        c = colors[idx]
        axes[0][0].scatter(x,
                           y,
                           s=8,
                           c=c,
                           edgecolors='none')
    axes[0][0].set_xlabel(x_label)
    axes[0][0].set_ylabel(y_label)
    axes[0][0].set_xlim(left=0, right=15)
    axes[0][0].set_ylim(bottom=0)
    axes[0][0].title.set_text('KMeans clustering')

    # asyncio.run(util.write_col_data(
    #    db_config,
    #    table_name,
    #    labels_shuffled,
    #    theme,
    #    'int',
    #    table.index,
    #    'id'))

    #### MEANSHIFT METHOD
    theme = f'lu_cluster_meanshift'
    print('theme:', theme)

    # select columns
    X = table[cols]
    X = RobustScaler().fit_transform(X)

    cluster = MeanShift(bandwidth=0.75, min_bin_freq=ms_bf, bin_seeding=True, cluster_all=True, n_jobs=-1).fit(X)
    labels = cluster.labels_

    axes[0][1].scatter(table.mu_hill_branch_wt_0_100,
                       table.mu_hill_branch_wt_0_1600,
                       s=8,
                       c=labels,
                       cmap='Set1',
                       edgecolors='none')
    axes[0][1].set_xlabel(x_label)
    axes[0][1].set_ylabel(y_label)
    axes[0][1].set_xlim(left=0, right=15)
    axes[0][1].set_ylim(bottom=0)
    axes[0][1].title.set_text('MeanShift clustering')

    # asyncio.run(util.write_col_data(
    #       db_config,
    #       table_name,
    #       labels_shuffled,
    #       theme,
    #       'int',
    #       table.index,
    #       'id'))

    #### STEPPED CLUSTER METHOD
    theme = f'lu_cluster_stepped_kmeans'
    print('theme:', theme)

    df_stepped = table.copy(deep=True)

    X_a = RobustScaler().fit_transform(df_stepped[['mu_hill_branch_wt_0_100']])
    cluster = KMeans(n_clusters=2).fit(X_a)
    labels = cluster.labels_
    df_stepped['step_a_labels'] = labels

    k_a = 0
    if np.nanmean(df_stepped[df_stepped.step_a_labels == 1]) > np.nanmean(df_stepped[df_stepped.step_a_labels == 0]):
        k_a = 1

    idx = df_stepped.index[df_stepped.step_a_labels == k_a]
    X_b = RobustScaler().fit_transform(df_stepped.loc[idx, ['mu_hill_branch_wt_0_400']])
    cluster = KMeans(n_clusters=2).fit(X_b)
    labels = cluster.labels_
    df_stepped.loc[idx, 'step_b_labels'] = labels

    k_b = 0
    if np.nanmean(df_stepped[df_stepped.step_b_labels == 1]) > np.nanmean(df_stepped[df_stepped.step_b_labels == 0]):
        k_b = 1

    idx = df_stepped.index[df_stepped.step_b_labels == k_b]
    X_c = RobustScaler().fit_transform(df_stepped.loc[idx, ['mu_hill_branch_wt_0_1600']])
    cluster = KMeans(n_clusters=2).fit(X_c)
    labels = cluster.labels_
    df_stepped.loc[idx, 'step_c_labels'] = labels

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

    # doesn't need sorting but do need colours:
    X = table[cols]
    X = RobustScaler().fit_transform(X)
    labels_shuffled, colors = label_sorter(X, df_stepped.lu_class.values)

    for n, lb in enumerate(intensity_categories):
        idx = np.where(labels_shuffled == n)
        x = table.mu_hill_branch_wt_0_100.iloc[idx]
        y = table.mu_hill_branch_wt_0_1600.iloc[idx]
        c = colors[idx]
        axes[1][0].scatter(x,
                           y,
                           s=8,
                           c=c,
                           edgecolors='none',
                           label=lb)
    axes[1][0].set_xlabel(x_label)
    axes[1][0].set_ylabel(y_label)
    axes[1][0].set_xlim(left=0, right=15)
    axes[1][0].set_ylim(bottom=0)
    axes[1][0].legend(title='intensity', loc='lower right')
    axes[1][0].title.set_text('Stepped KMeans clustering')

    # asyncio.run(util.write_col_data(
    #   db_config,
    #   table_name,
    #   labels_shuffled,
    #   theme,
    #   'int',
    #   df_stepped.index,
    #   'id'))

    #### MANUAL METHOD

    theme = f'lu_cluster_manual'
    print('theme:', theme)

    near_split = 2
    mid_split = 20
    far_split = 100

    table['lu_class'] = 0
    table.loc[(table.mu_hill_branch_wt_0_100 >= near_split) &
              (table.mu_hill_branch_wt_0_400 >= mid_split) &
              (table.mu_hill_branch_wt_0_1600 >= far_split), 'lu_class'] = 3

    table.loc[(table.mu_hill_branch_wt_0_100 >= near_split) &
              (table.mu_hill_branch_wt_0_400 >= mid_split) &
              (table.mu_hill_branch_wt_0_1600 < far_split), 'lu_class'] = 2

    table.loc[(table.mu_hill_branch_wt_0_100 >= near_split) &
              (table.mu_hill_branch_wt_0_400 < mid_split), 'lu_class'] = 1

    # doesn't need sorting but do need colours:
    X = table[cols]
    X = RobustScaler().fit_transform(X)
    labels_shuffled, colors = label_sorter(X, table.lu_class.values)

    for n, lb in enumerate(intensity_categories):
        idx = np.where(labels_shuffled == n)
        x = table.mu_hill_branch_wt_0_100.iloc[idx]
        y = table.mu_hill_branch_wt_0_1600.iloc[idx]
        c = colors[idx]
        axes[1][1].scatter(x,
                           y,
                           s=8,
                           c=c,
                           edgecolors='none',
                           label=lb)
    axes[1][1].set_xlabel(x_label)
    axes[1][1].set_ylabel(y_label)
    axes[1][1].set_xlim(left=0, right=15)
    axes[1][1].set_ylim(bottom=0)
    axes[1][1].legend(title='intensity', loc='lower right')
    axes[1][1].title.set_text('Manually stepped clustering')

    #asyncio.run(util.write_col_data(
    #  db_config,
    #  table_name,
    #  labels_shuffled,
    #  theme,
    #  'int',
    #  table.index,
    #  'id'))

    plt.savefig(f'./explore/5-feature_extraction/exploratory_plots/cluster_comparisons_{decomp}.png', dpi=300)
    plt.show()
