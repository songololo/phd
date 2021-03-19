# %%
import asyncio
import logging
import pandas as pd
import numpy as np
from importlib import reload
from scipy.stats import spearmanr
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from explore import plot_funcs
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
    # 'host': 'host.docker.internal',
    'port': 5432,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

columns = [
    # for table - only need specific distances (but only where not already covered in columns below)
    'id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y'
]

columns_base = [
    'c_node_harmonic',
    'c_segment_harmonic',
    'mu_hill_branch_wt_0',
    'mu_hill_branch_wt_0_rdm'
]

col_template = '{col}[{i}] as {col}_{dist}'
distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
for col_base in columns_base:
    for i, d in enumerate(distances):
        columns.append(col_template.format(col=col_base, i=i + 1, dist=d))
# bandwise distances do not include 50m
distances_bandwise = distances[1:]


#%%
print('loading columns')
df_full = await phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_full',
    'WHERE city_pop_id = 1 and within = true')
df_full = df_full.set_index('id')
df_full = phd_util.clean_pd(df_full, drop_na='all', fill_inf=np.nan)

df_100 = await phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_100',
    'WHERE city_pop_id = 1 and within = true')
df_100 = df_100.set_index('id')
df_100 = phd_util.clean_pd(df_100, drop_na='all', fill_inf=np.nan)

df_50 = await phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_50',
    'WHERE city_pop_id = 1 and within = true')
df_50 = df_50.set_index('id')
df_50 = phd_util.clean_pd(df_50, drop_na='all', fill_inf=np.nan)

df_20 = await phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_20',
    'WHERE city_pop_id = 1 and within = true')
df_20 = df_20.set_index('id')
df_20 = phd_util.clean_pd(df_20, drop_na='all', fill_inf=np.nan)

def get_band(df, dist, target, seg_beta_norm=False):
    # don't want to use 50 - otherwise distance steps are inconsistent
    if dist == 50:
        print('Dont use 50')
        return None
    # the first band does not subtract anything
    elif dist == 100:
        t = df[target.format(dist=100)]
        s = df[f'c_segment_beta_{dist}']
    # subsequent bands subtract the prior band
    else:
        lag_dist = dist - 100
        t_cur = df[target.format(dist=dist)]
        t_lag = df[target.format(dist=lag_dist)]
        t = t_cur - t_lag
        s_cur = df[f'c_segment_beta_{dist}']
        s_lag = df[f'c_segment_beta_{lag_dist}']
        s = s_cur - s_lag
    # whether or not to normalise by segments at equivalent betas
    if not seg_beta_norm:
        return t
    else:
        v = t.values / s.values
        v[np.isnan(v)] = 0
        return v


# %%
'''
distance thresholds (including large distances) for correlations vs. Spearman mixed-use
Compares correlations as measured, vs. randomised, vs. difference between the two
'''

phd_util.plt_setup()

fig = plt.figure(figsize=(12, 8))
# use grid spec to allow merging legend columns
gs = fig.add_gridspec(5, 3, height_ratios=[0.02, 1, 1, 1, 1])
# create legend ax
ax_legend = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2])]
# create content axes
axes_content = []
for i in range(1, 5):
    _ax_left = fig.add_subplot(gs[i, 0])
    _ax_middle = fig.add_subplot(gs[i, 1])
    _ax_right = fig.add_subplot(gs[i, 2])
    axes_content.append([_ax_left, _ax_middle, _ax_right])

x_themes = [
    'c_imp_close_{d}',
    'c_gravity_{d}',
    'c_between_{d}',
    'c_between_wt_{d}'
]
x_labels = [
    'improved closeness',
    'gravity index',
    'betweeness',
    'weighted betweenness'
]
# weighted vs unweighted y
y_themes = [
    'mu_hill_0_{d}',
    'mu_hill_branch_wt_0_{d}',
    'mu_hill_0_{d}',
    'mu_hill_branch_wt_0_{d}'
]
# weighted vs unweighted y
y_themes_rdm = [
    'mu_hill_0_rdm_{d}',
    'mu_hill_branch_wt_0_rdm_{d}',
    'mu_hill_0_rdm_{d}',
    'mu_hill_branch_wt_0_rdm_{d}'
]

selected_cent_distances = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
plot_grid = list(range(len(selected_cent_distances)))
landuse_betas = [r'$\beta=-0.08$', r'$\beta=-0.04$', r'$\beta=-0.02$', r'$\beta=-0.01$', r'$\beta=-0.005$',
                 r'$\beta=-0.0025$']

for ax_row, x_theme, x_label, y_theme, y_theme_rdm in \
        zip(axes_content, x_themes, x_labels, y_themes, y_themes_rdm):

    print(f'calculating for theme: {x_label}')

    # iterate the landuse distances
    for n, (dist_landuse, beta_landuse) in enumerate(zip(
            reversed(landuse_distances), reversed(landuse_betas))):

        corrs = []
        corrs_rdm = []
        corrs_diff = []

        # nested iterate the centrality distances and compute correlations
        for dist_centrality in selected_cent_distances:
            x = df_full_clean[x_theme.format(d=dist_centrality)]

            y = df_full_clean[y_theme.format(d=dist_landuse)]
            y_corr = spearmanr(x, y)[0]
            corrs.append(y_corr)

            y_rdm = df_full_clean[y_theme_rdm.format(d=dist_landuse)]

            y_corr_rdm = spearmanr(x, y_rdm)[0]
            corrs_rdm.append(y_corr_rdm)

            y_diff = y_corr - y_corr_rdm
            corrs_diff.append(y_diff)

        # print the correlation line (across centrality distances) for current landuse distance
        dist_label = f'${dist_landuse}m$'

        ax_row[0].plot(plot_grid, corrs,
                       color=cityseer_cmap(n / 5), lw=1, alpha=1, linestyle='-', marker='.', label=dist_label)

        ax_row[1].plot(plot_grid, corrs_rdm,
                       color=cityseer_cmap(n / 5), lw=1, alpha=1, linestyle='-', marker='.', label=dist_label)

        ax_row[2].plot(plot_grid, corrs_diff,
                       color=cityseer_cmap(n / 5), lw=1, alpha=1, linestyle='-', marker='.', label=dist_label)

    # configure the axes
    for n, col_n in enumerate(range(3)):
        min_x = plot_grid[0]
        max_x = plot_grid[-1]
        ax_row[col_n].set_xlim([min_x, max_x])
        ax_row[col_n].set_xticks(plot_grid)
        if x_label in ['gravity index', 'weighted betweenness']:
            ax_row[col_n].set_xticklabels(
                [r'$\beta=-0.08$', '', r'$\beta=-0.02$', '', r'$\beta=-0.005$', '', r'$\beta=-0.00125$', '', ''])
        else:
            ax_row[col_n].set_xticklabels([f'{d}m' for d in selected_cent_distances])
        ax_row[col_n].set_xlabel(x_label)
        # background polygon
        ax_row[col_n].fill([min_x, max_x, max_x, min_x],
                           [-0.1, -0.1, 0, 0],
                           c='grey',
                           lw=0,
                           alpha=0.1,
                           zorder=1)

for ax_row in axes_content:
    ax_row[0].set_ylim([-0.1, 0.8])
    ax_row[1].set_ylim([-0.1, 0.8])
    ax_row[2].set_ylim([-0.1, 0.8])
    ax_row[0].set_ylabel(r'spearman $\rho$')
    ax_row[1].set_ylabel(r'spearman $\rho$')
    ax_row[2].set_ylabel(r'spearman $\rho$')

column_themes = [
    'correlations for actual mixed-uses:',
    'correlations for randomised mixed-uses:',
    'correlations for differenced mixed-uses:'
]
for col_idx, col_theme in enumerate(column_themes):
    ax_legend[col_idx].title.set_text(col_theme)
    handles, labels = axes_content[0][col_idx].get_legend_handles_labels()
    ax_legend[col_idx].axis('off')
    ax_legend[col_idx].legend(
        handles,
        labels,
        title='mixed-use $H_{0}\ d_{max}$ thresholds:',
        title_fontsize=7,
        ncol=6,
        loc='upper center')

plt.suptitle('Spearman correlation coefficients for mixed-use richness compared to network centrality')

path = f'../../phd-admin/PhD/part_1/images/cityseer/primal_correlations_vs_randomised_mu.png'
plt.savefig(path, dpi=300)

plt.show()

# %%
'''
compare basic stats for measured vs. randomised landuses
Intent is to help explain why correlations strengthen e.g. for increased distances
'''

tables = [df_full_clean, df_100_clean, df_50_clean, df_20_clean]
table_labels = ['$full$', '$100m$', '$50m$', '$20m$']

cols = ['mu_hill_0_{d}', 'mu_hill_0_rdm_{d}', 'mu_hill_branch_wt_0_{d}', 'mu_hill_branch_wt_0_rdm_{d}']
weighted = [False, False, True, True]
randomised = [False, True, False, True]

merged_data = []
for t, l in zip(tables, table_labels):
    for col, wt, rdm in zip(cols, weighted, randomised):
        for d in landuse_distances:
            # prepare column names
            old_col = col.format(d=d)
            new_col = 'mixed_uses'
            # column indexing produces a copy
            df_sel = t[[old_col]]
            # rename all columns to mixed uses
            df_sel = df_sel.rename(columns={old_col: new_col})
            # add columns for distance, decomp, wt, rdm
            df_sel['distance'] = d
            df_sel['decomp'] = l
            df_sel['wt'] = wt
            df_sel['rdm'] = rdm
            # append to merge list
            merged_data.append(df_sel)
# generate the merged dataframe
df_merged = pd.concat(merged_data)

phd_util.plt_setup()

fig = plt.figure(figsize=(8, 12))
# use grid spec to allow merging legend columns
gs = fig.add_gridspec(7, 2, height_ratios=[0.02, 1, 1, 1, 1, 1, 1])
# create legend ax
ax_legend = fig.add_subplot(gs[0, :])  # legend spans both columns
ax_legend.axis('off')
# create content axes
axes_content = []
for i in range(1, 7):
    _ax_left = fig.add_subplot(gs[i, 0])
    _ax_right = fig.add_subplot(gs[i, 1])
    axes_content.append([_ax_left, _ax_right])

# prepare font dict
font = {'size': 5}

# populate
for ax_idx, (ax_row, d) in enumerate(zip(axes_content, landuse_distances)):
    for ax, wt in zip(ax_row, [False, True]):
        data = df_merged[(df_merged.distance == d) & (df_merged.wt == wt)]
        sns.violinplot(
            data=data,
            ax=ax,
            x='decomp',
            y='mixed_uses',
            hue='rdm',
            saturation=1,
            bw=1,
            split=True,
            inner='quart',
            palette={True: '#64c1ff', False: '#d32f2f'})
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.legend().set_visible(False)
        # add the text annotations
        trans = ax.get_xaxis_transform()
        dec_pts = 1
        if ax_idx < 2:
            dec_pts = 2
        for idx, l in enumerate(table_labels):
            decomp_data = data[data.decomp == l]
            decomp_data = decomp_data['mixed_uses']
            # x in data coords and y in axis coords via trans
            mu = round(np.nanmean(decomp_data), dec_pts)
            ax.text(idx - 0.025, 1, f'$\mu= {mu}$', ha='right', va='top', fontdict=font, transform=trans)
            v = round(np.nanvar(decomp_data), dec_pts)
            ax.text(idx - 0.025, 0.94, f'$\sigma^2= {v}$', ha='right', va='top', fontdict=font, transform=trans)

# set ax labels and extents
ht_factor = 17
wt_ht_factor = 4
landuse_betas = [r'\beta=-0.08', r'\beta=-0.04', r'\beta=-0.02', r'\beta=-0.01', r'\beta=-0.005', r'\beta=-0.0025']
for i, (ax_row, d, b) in enumerate(zip(axes_content, landuse_distances, landuse_betas)):
    ax_row[0].set_ylabel('mixed-use $H_{0}\ d_{max=' + str(d) + 'm}$')
    ax_row[0].set_ylim(bottom=0, top=ht_factor)
    ht_factor *= 2
    ax_row[1].set_ylabel('mixed use richness $_{' + b + '}$')
    ax_row[1].set_ylim(bottom=0, top=wt_ht_factor)
    wt_ht_factor *= 2

axes_content[-1][0].set_xlabel('mixed-uses at respective distances / decompositions')
axes_content[-1][1].set_xlabel('weighted mixed-uses at respective distances / decompositions')

handles, labels = axes_content[0][0].get_legend_handles_labels()
ax_legend.legend(
    handles,
    ['actual landuses', 'randomised landuses'],
    title='mixed-uses source data:',
    title_fontsize=7,
    ncol=2,
    loc='upper center')

plt.suptitle('Violin plots comparing actual landuses to randomised landuses')

path = f'../../phd-admin/PhD/part_1/images/cityseer/landuse_stats_comparisons_decomp.png'
plt.savefig(path, dpi=300)

plt.show()


# %%
from pysal.lib import weights


def generate_weights(eastings, northings, threshold):
    logger.info(f'generating spatial weights matrix for threshold: {threshold}')
    points = []
    for e, n in zip(eastings, northings):
        points.append((e, n))
    # alpha is ignored if binary is True
    W = weights.DistanceBand(points, threshold, binary=True)
    return W


# generate spatial weights matrices
W = generate_weights(df_full_clean['x'], df_full_clean['y'], threshold=200)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression(X, y):
    if X.ndim == 1:
        X = X.values.reshape(-1, 1)
        assert X.ndim == 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    # score on test data
    y_hat = reg.predict(X_test)
    r2 = r2_score(y_test, y_hat)
    rmse = np.sqrt(mean_squared_error(y_test, y_hat))
    nrmse = rmse / np.nanmean(y_test)
    logger.info(f'r2: {r2}, nrmse: {nrmse}')
    # the residuals are computed for the entirety of the dataset
    y_full = reg.predict(X)
    residuals = y_full - y
    return r2, nrmse, residuals


# %%
from pysal.explore.esda import moran


def autocorrelate(data, W):
    G = moran.Moran(data, W, permutations=5)
    logger.info(f'Morans I: {G.I}')
    return G.I


# %%

'''
distance thresholds for full, 100, 50, 20m decompositions vs. 
Spearman mixed-use correlations / r^2 / moran's I
'''

phd_util.plt_setup()

fig, axes = plt.subplots(5, 3,
                         figsize=(12, 8),
                         gridspec_kw={'height_ratios': [0.02, 1, 1, 1, 1]})

x_themes = [
    'c_imp_close_{d}',
    'c_gravity_{d}',
    'c_between_{d}',
    'c_between_wt_{d}'
]
x_labels = [
    'improved closeness',
    'gravity index',
    'betweeness',
    'weighted betweenness'
]
# weighted vs unweighted y
y_themes = [
    'mu_hill_0_100',
    'mu_hill_branch_wt_0_200',
    'mu_hill_0_100',
    'mu_hill_branch_wt_0_200'
]
# weighted vs unweighted y
y_themes_rdm = [
    'mu_hill_0_rdm_100',
    'mu_hill_branch_wt_0_rdm_200',
    'mu_hill_0_rdm_100',
    'mu_hill_branch_wt_0_rdm_200'
]

plot_grid = list(range(len(cent_distances)))
for ax_row, x_theme, x_label, y_theme, y_theme_rdm in \
        zip(axes[1:], x_themes, x_labels, y_themes, y_themes_rdm):

    logger.info(f'calculating for theme: {x_label}')

    corrs = np.full((4, len(cent_distances)), np.nan)
    corrs_rdm = np.full((4, len(cent_distances)), np.nan)
    r2 = np.full((4, len(cent_distances)), np.nan)
    morans = np.full((4, len(cent_distances)), np.nan)

    for d_idx, d in enumerate(cent_distances):
        logger.info(f'...distance: {d}')
        for arr_idx, table in enumerate([df_full_clean, df_100_clean, df_50_clean, df_20_clean]):
            # correlations
            x = table[x_theme.format(d=d)]
            y = table[y_theme]
            corrs[arr_idx][d_idx] = spearmanr(x, y)[0]

            # random data
            y_rdm = table[y_theme_rdm]
            s_r_rdm = spearmanr(x, y_rdm)[0]
            corrs_rdm[arr_idx][d_idx] = s_r_rdm

            # linear regression
            # tried x transformation via boxcox, but doesn't appear to make much difference
            # also tried y_diff, though gives about the same r2 and slightly weaker on the betweenness measures
            _r2, nrmse, residuals = linear_regression(x, y)
            r2[arr_idx][d_idx] = _r2

            # spatially autocorrelate the residuals
            # all decompositions are compared against the same full table id columns
            keys = table[table.x.isin(df_full_clean.x) & table.y.isin(df_full_clean.y)].index
            indexer = table.index.get_indexer_for(keys)
            assert len(indexer) == len(df_full_clean)
            selected_residuals = residuals.values.flatten()[indexer]
            morans_I = autocorrelate(selected_residuals, W)
            morans[arr_idx][d_idx] = morans_I

    lw_raw = 1.2
    alpha_raw = 0.8
    ls_raw = '-'
    marker_raw = '.'

    lw_rdm = 0.8
    alpha_rdm = 1
    ls_rdm = '--'
    marker_rdm = 'x'

    lw_diff = 1.2
    alpha_diff = 0.8
    ls_diff = '-'
    marker_diff = '.'

    for arr_idx, (table_label, table_color) in enumerate(zip(['full', '100', '50', '20'], ['C0', 'C1', 'C2', 'C3'])):
        ax_row[0].plot(plot_grid, corrs[arr_idx],
                       color=table_color, lw=lw_raw, alpha=alpha_raw, linestyle=ls_raw, marker=marker_raw,
                       label=f'${table_label}$')
        ax_row[0].plot(plot_grid, corrs_rdm[arr_idx],
                       color=table_color, lw=lw_rdm, alpha=alpha_rdm, linestyle=ls_rdm, marker=marker_rdm,
                       label=f'${table_label}$ random.')
        ax_row[1].plot(plot_grid, r2[arr_idx],
                       color=table_color, lw=lw_raw, alpha=alpha_raw, linestyle=ls_raw, marker=marker_raw,
                       label=f'${table_label}$')
        ax_row[2].plot(plot_grid, morans[arr_idx],
                       color=table_color, lw=lw_raw, alpha=alpha_raw, linestyle=ls_raw, marker=marker_raw,
                       label=f'${table_label}$')

    postfix = 'd_{max}=100m'
    for ax in ax_row:
        ax.set_xlim(left=plot_grid[0], right=plot_grid[-1])
        ax.set_xticks(plot_grid)
        if x_label in ['gravity index', 'weighted betweenness']:
            ax.set_xticklabels(
                ['', r'$\beta=-0.04$', '', r'$\beta=-0.02$', '', r'$\beta=-0.01$', '', r'$\beta=-0.005$', '',
                 r'$\beta=-0.0025$'])
            postfix = r'\beta=-0.02\ d_{max}=200m'
        else:
            ax.set_xticklabels([f'{d}m' for d in cent_distances])

    ax_row[0].set_xlabel(x_label + ' (mixed uses $H_{0\ ' + postfix + '}$)')
    ax_row[1].set_xlabel(x_label + ' (mixed uses $H_{0\ ' + postfix + '}$)')
    ax_row[2].set_xlabel(x_label + ' (mixed uses $H_{0\ ' + postfix + '}$)')

    ax_row[0].set_ylim([0, 0.8])
    ax_row[1].set_ylim([0, 0.6])
    ax_row[2].set_ylim([0, 1.0])

    ax_row[0].set_ylabel(r'spearman $\rho$')
    ax_row[1].set_ylabel('$r^{2}$')
    ax_row[2].set_ylabel('moran\'s $I$')

for n, ax in enumerate(axes[0]):
    ax.axis('off')
    handles, labels = axes[1][n].get_legend_handles_labels()
    if n == 0:
        title = 'mixed-use correlations at decomposition:'
    elif n == 1:
        title = 'linear regression $r^{2}$ at decomposition:'
    elif n == 2:
        title = 'autocorrelation of residuals at decomposition:'
    ax.legend(handles,
              labels,
              title=title,
              ncol=4,
              loc='upper center',
              borderaxespad=0.0,
              borderpad=0.0,
              handletextpad=.25,
              columnspacing=1)

plt.suptitle('Correlations, $r^{2}$, and Moran\'s I for centrality and mixed-use richness $H_{0}$ ')

path = f'../../phd-admin/PhD/part_1/images/cityseer/primal_correlations_cent_vs_mu.png'
plt.savefig(path, dpi=300)

plt.show()