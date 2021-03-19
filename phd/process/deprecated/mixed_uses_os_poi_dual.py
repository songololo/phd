#%%
import logging
import numpy as np
from importlib import reload
import phd_util
reload(phd_util)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#%% set variables
distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]


cols = [
    'id',
    'ac_transport_simpl_100',
    'ac_transport_short_100',
    'ac_eating_simpl_100',
    'ac_eating_short_100',
    'ac_commercial_simpl_100',
    'ac_commercial_short_100',
    'ac_retail_simpl_100',
    'ac_retail_short_100'
]

cols_wt = [
    'dual_met_betweenness_{d}',
    'dual_ang_betweenness_{d}',
    'dual_node_count_{d}',
    'dual_met_farness_{d}',
    'dual_ang_farness_{d}',
    'uses_simplest_hill_0_{d}',
    'uses_simplest_hill_1_{d}',
    'uses_simplest_hill_2_{d}',
    'uses_simplest_hill_funct_wt_0_{d}',
    'uses_simplest_hill_funct_wt_1_{d}',
    'uses_simplest_hill_funct_wt_2_{d}',
    'uses_shortest_hill_0_{d}',
    'uses_shortest_hill_1_{d}',
    'uses_shortest_hill_2_{d}',
    'uses_shortest_hill_funct_wt_0_{d}',
    'uses_shortest_hill_funct_wt_1_{d}',
    'uses_shortest_hill_funct_wt_2_{d}'
]

for c in cols_wt:
    for d in distances:
        cols.append(c.format(d=d))


#%% load data
df_data_full = phd_util.load_data_as_pd_df(cols, 'analysis.roadnodes_full_dual', 'WHERE city_pop_id = 1')
df_data_100 = phd_util.load_data_as_pd_df(cols, 'analysis.roadnodes_100_dual', 'WHERE city_pop_id = 1')
df_data_50 = phd_util.load_data_as_pd_df(cols, 'analysis.roadnodes_50_dual', 'WHERE city_pop_id = 1')
df_data_20 = phd_util.load_data_as_pd_df(cols, 'analysis.roadnodes_20_dual', 'WHERE city_pop_id = 1')


#%% clean data
df_full_clean = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)
df_100_clean = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)
df_50_clean = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)
df_20_clean = phd_util.clean_pd(df_data_20, drop_na='all', fill_inf=np.nan)


#%% create the closeness columns
for d in distances:
    df_full_clean[f'dual_imp_ang_far_{d}'] = df_full_clean[f'dual_ang_farness_{d}'] / df_full_clean[f'dual_node_count_{d}'] ** 2
    df_full_clean[f'dual_imp_met_far_{d}'] = df_full_clean[f'dual_met_farness_{d}'] / df_full_clean[f'dual_node_count_{d}'] ** 2
    df_100_clean[f'dual_imp_ang_far_{d}'] = df_100_clean[f'dual_ang_farness_{d}'] / df_100_clean[f'dual_node_count_{d}'] ** 2
    df_100_clean[f'dual_imp_met_far_{d}'] = df_100_clean[f'dual_met_farness_{d}'] / df_100_clean[f'dual_node_count_{d}'] ** 2
    df_50_clean[f'dual_imp_ang_far_{d}'] = df_50_clean[f'dual_ang_farness_{d}'] / df_50_clean[f'dual_node_count_{d}'] ** 2
    df_50_clean[f'dual_imp_met_far_{d}'] = df_50_clean[f'dual_met_farness_{d}'] / df_50_clean[f'dual_node_count_{d}'] ** 2
    df_20_clean[f'dual_imp_ang_far_{d}'] = df_20_clean[f'dual_ang_farness_{d}'] / df_20_clean[f'dual_node_count_{d}'] ** 2
    df_20_clean[f'dual_imp_met_far_{d}'] = df_20_clean[f'dual_met_farness_{d}'] / df_20_clean[f'dual_node_count_{d}'] ** 2


#%%
'''
1A - COMPUTE
Decomposition set against full, 100m, 50m, 20m tables
Each graph displays all variants against a selected centrality metric at the corresponding distances (see above plots to choose)
'''

data_3 = {
    'angular_closeness': {
        'x': r'uses_simplest_hill_0_{d}',
        'x_label': r'Hill $H_{0\ simplest}$ diversity',
        'y': r'dual_imp_ang_far_{d}',
        'y_label': r'$\measuredangle$ -farness',
        'invert': True
    },
    'metric_closeness': {
        'x': r'uses_shortest_hill_0_{d}',
        'x_label': r'Hill $H_{0\ centrality}$ diversity',
        'y': r'dual_imp_met_far_{d}',
        'y_label': r'-farness',
        'invert': True
    },
    'angular_betw': {
        'x': r'uses_simplest_hill_0_{d}',
        'x_label': r'Hill $H_{0\ simplest}$ diversity',
        'y': r'dual_ang_betweenness_{d}',
        'y_label': r'$\measuredangle$ betweenness centrality',
        'invert': False
    },
    'metric_betw': {
        'x': r'uses_shortest_hill_0_{d}',
        'x_label': r'Hill $H_{0\ centrality}$ diversity',
        'y': r'dual_met_betweenness_{d}',
        'y_label': r'betweenness centrality',
        'invert': False
    }
}

for k, v in data_3.items():

    # calculate spearman rank correlations
    logger.info(f'calculating for theme: {k}')

    data_3[k]['corrs_full_hill_0'] = []
    data_3[k]['corrs_100_hill_0'] = []
    data_3[k]['corrs_50_hill_0'] = []
    data_3[k]['corrs_20_hill_0'] = []

    for d in distances:

        x_theme = v['x'].format(d=d)
        y_theme = v['y'].format(d=d)

        # full graph
        y = df_full_clean[y_theme]
        data_3[k]['corrs_full_hill_0'].append(pearsonr(*phd_util.prep_xy(df_full_clean[x_theme], y, transform=True))[0])

        # 100 graph
        y = df_100_clean[y_theme]
        data_3[k]['corrs_100_hill_0'].append(pearsonr(*phd_util.prep_xy(df_100_clean[x_theme], y, transform=True))[0])

        # 50 graph
        y = df_50_clean[y_theme]
        data_3[k]['corrs_50_hill_0'].append(pearsonr(*phd_util.prep_xy(df_50_clean[x_theme], y, transform=True))[0])

        # 20 graph
        y = df_20_clean[y_theme]
        data_3[k]['corrs_20_hill_0'].append(pearsonr(*phd_util.prep_xy(df_20_clean[x_theme], y, transform=True))[0])


# %%
'''
1B - PLOT
'''

phd_util.plt_setup()

fig, axes = plt.subplots(4, 1, figsize=(8, 10))

lw = 1
a = 1
line = '-'
c_0 = 'C0'
c_1 = 'C1'
c_2 = 'C2'
c_3 = 'C3'

for ax, (k, v) in zip(axes, data_3.items()):

    d_full = data_3[k]['corrs_full_hill_0']
    d_100 = data_3[k]['corrs_100_hill_0']
    d_50 = data_3[k]['corrs_50_hill_0']
    d_20 = data_3[k]['corrs_20_hill_0']

    if v['invert']:
        d_full = -np.array(d_full)
        d_100 = -np.array(d_100)
        d_50 = -np.array(d_50)
        d_20 = -np.array(d_20)

    ax.plot(distances, d_full, lw=lw, alpha=a, linestyle=line, color=c_0, marker='.', label=r'full')

    ax.plot(distances, d_100, lw=lw, alpha=a, linestyle=line, color=c_1, marker='.', label=r'100m')

    ax.plot(distances, d_50, lw=lw, alpha=a, linestyle=line, color=c_2, marker='.', label=r'50m')

    ax.plot(distances, d_20, lw=lw, alpha=a, linestyle=line, color=c_3, marker='.', label=r'20m')

    ax.set_xlim([50, 1600])
    ax.set_xticks(distances)
    ax.set_xlabel(r'Dual network correlation coefficients for ' + f'{v["x_label"]} and {v["y_label"]}')
    ax.set_ylim([0.1, 0.9])
    ax.set_ylabel(r"pearson's $r$")

    ax.legend(loc='lower right', title='')

plt.show()



#%%
'''
2
Comparison against a range of dual graph centralities

IMPORTANT NOTE -> distance weighted variants are arguably nonsensical for simplest angular path routed variants...
'''

table = df_20_clean

phd_util.plt_setup()

fig, axes = plt.subplots(4, 1, figsize=(8, 10))

data_4 = {
    'angular_closeness': {
        'x': [
            ('uses_simplest_hill_0_{d}', r'$H_{0}$', '-', 'x', 1.5, 0.8),
            ('uses_simplest_hill_1_{d}', r'$H_{1}$', '-', 'x', 1.5, 0.8),
            ('uses_simplest_hill_2_{d}', r'$H_{2}$', '-', 'x', 1.5, 0.8),
            ('uses_simplest_hill_funct_wt_0_{d}', r'$H_{0\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_simplest_hill_funct_wt_1_{d}', r'$H_{1\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_simplest_hill_funct_wt_2_{d}', r'$H_{2\ pairwise\ wt.}$', '--', '.', 1, 1)
        ],
        'x_label': r'simplest path diversity indices',
        'y': r'dual_imp_ang_far_{d}',
        'y_label': r'$\measuredangle$ -farness',
        'invert': True
    },
    'metric_closeness': {
        'x': [
            ('uses_shortest_hill_0_{d}', r'$H_{0}$', '-', 'x', 1.5, 0.8),
            ('uses_shortest_hill_1_{d}', r'$H_{1}$', '-', 'x', 1.5, 0.8),
            ('uses_shortest_hill_2_{d}', r'$H_{2}$', '-', 'x', 1.5, 0.8),
            ('uses_shortest_hill_funct_wt_0_{d}', r'$H_{0\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_shortest_hill_funct_wt_1_{d}', r'$H_{1\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_shortest_hill_funct_wt_2_{d}', r'$H_{2\ pairwise\ wt.}$', '--', '.', 1, 1)
        ],
        'x_label': r'centrality path diversity indices',
        'y': r'dual_imp_met_far_{d}',
        'y_label': r'-farness',
        'invert': True
    },
    'angular_betw': {
        'x': [
            ('uses_simplest_hill_0_{d}', r'$H_{0}$', '-', 'x', 1.5, 0.8),
            ('uses_simplest_hill_1_{d}', r'$H_{1}$', '-', 'x', 1.5, 0.8),
            ('uses_simplest_hill_2_{d}', r'$H_{2}$', '-', 'x', 1.5, 0.8),
            ('uses_simplest_hill_funct_wt_0_{d}', r'$H_{0\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_simplest_hill_funct_wt_1_{d}', r'$H_{1\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_simplest_hill_funct_wt_2_{d}', r'$H_{2\ pairwise\ wt.}$', '--', '.', 1, 1)
        ],
        'x_label': r'simplest path diversity indices',
        'y': r'dual_ang_betweenness_{d}',
        'y_label': r'$\measuredangle$ betweenness centrality',
        'invert': False
    },
    'metric_betw': {
        'x': [
            ('uses_shortest_hill_0_{d}', r'$H_{0}$', '-', 'x', 1.5, 0.8),
            ('uses_shortest_hill_1_{d}', r'$H_{1}$', '-', 'x', 1.5, 0.8),
            ('uses_shortest_hill_2_{d}', r'$H_{2}$', '-', 'x', 1.5, 0.8),
            ('uses_shortest_hill_funct_wt_0_{d}', r'$H_{0\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_shortest_hill_funct_wt_1_{d}', r'$H_{1\ pairwise\ wt.}$', '--', '.', 1, 1),
            ('uses_shortest_hill_funct_wt_2_{d}', r'$H_{2\ pairwise\ wt.}$', '--', '.', 1, 1)
        ],
        'x_label': r'centrality path diversity indices',
        'y': r'dual_met_betweenness_{d}',
        'y_label': r'betweenness centrality',
        'invert': False
    }
}

for ax, (k, v) in zip(axes, data_4.items()):

    # calculate spearman rank correlations
    logger.info(f'calculating for theme: {k}')

    for x_key, x_label, x_line, x_marker, x_lw, x_a in v['x']:

        corrs = []

        for d in distances:

            x_theme = x_key.format(d=d)
            y_theme = v['y'].format(d=d)
            corrs.append(pearsonr(*phd_util.prep_xy(table[x_theme], table[y_theme], transform=True))[0])

        if v['invert']:
            corrs = -np.array(corrs)

        ax.plot(distances, corrs, lw=x_lw, alpha=x_a, linestyle=x_line, marker=x_marker, label=x_label)

    ax.set_xlim([50, 1600])
    ax.set_xticks(distances)
    ax.set_xlabel(f'Correlation coefficients for {v["x_label"]} to {v["y_label"]} on the $20m$ dual network')
    ax.set_ylim([0.1, 0.9])
    ax.set_ylabel(r"pearson's $r$")

    ax.legend(loc='lower right', title='')

plt.show()


#%%
'''
3
Comparisons between mixed-use measures and various land use categories on 20m network
'''

phd_util.plt_setup()

fig, axes = plt.subplots(4, 1, figsize=(8, 10))

table = df_20_clean

themes = [
    'access_simplest_eating_100',
    'access_simplest_commercial_100',
    'access_simplest_retail_100',
    'access_simplest_transport_100'
]

labels = [
    r'eat / drink venue accessibility $_{\beta=-0.04}$',
    r'commercial accessibility $_{\beta=-0.04}$',
    r'retail accessibility $_{\beta=-0.04}$',
    r'transit accessibility $_{\beta=-0.04}$',
]

for ax, theme, label in zip(axes, themes, labels):

    # calculate spearman rank correlations
    print(f'calculating for theme: {theme}')

    hill_0 = []
    hill_1 = []
    hill_2 = []
    hill_pw_wt_0 = []
    hill_pw_wt_1 = []
    hill_pw_wt_2 = []

    y = table[theme]

    for d in distances:

        x = table[f'uses_simplest_hill_0_{d}']
        hill_0.append(pearsonr(*phd_util.prep_xy(x, y, transform=True))[0])

        x = table[f'uses_simplest_hill_1_{d}']
        hill_1.append(pearsonr(*phd_util.prep_xy(x, y, transform=True))[0])

        x = table[f'uses_simplest_hill_2_{d}']
        hill_2.append(pearsonr(*phd_util.prep_xy(x, y, transform=True))[0])

        x = table[f'uses_simplest_hill_funct_wt_0_{d}']
        hill_pw_wt_0.append(pearsonr(*phd_util.prep_xy(x, y, transform=True))[0])

        x = table[f'uses_simplest_hill_funct_wt_1_{d}']
        hill_pw_wt_1.append(pearsonr(*phd_util.prep_xy(x, y, transform=True))[0])

        x = table[f'uses_simplest_hill_funct_wt_2_{d}']
        hill_pw_wt_2.append(pearsonr(*phd_util.prep_xy(x, y, transform=True))[0])

    lw = 1.25
    a = 0.7
    line_1 = '-'
    line_2 = '--'

    ax.plot(distances, hill_0, lw=lw, alpha=a, linestyle=line_1, marker='.', color='C0', label=r'$H_{0}$')
    ax.plot(distances, hill_1, lw=lw, alpha=a, linestyle=line_1, marker='.', color='C1', label=r'$H_{1}$')
    ax.plot(distances, hill_2, lw=lw, alpha=a, linestyle=line_1, marker='.', color='C2', label=r'$H_{2}$')
    ax.plot(distances, hill_pw_wt_0, lw=lw, alpha=a, linestyle=line_2, marker='x', color='C0', label=r'$H_{0\ pw.\ wt.}$')
    ax.plot(distances, hill_pw_wt_1, lw=lw, alpha=a, linestyle=line_2, marker='x', color='C1', label=r'$H_{1\ pw.\ wt.}$')
    ax.plot(distances, hill_pw_wt_2, lw=lw, alpha=a, linestyle=line_2, marker='x', color='C2', label=r'$H_{2\ pw.\ wt.}$')
    ax.set_xlim([50, 1600])
    ax.set_xticks(distances)
    ax.set_xlabel(r'Correlation coefficients for diversity indices $_{simplest}$ to local ' + label + r' on the $20m$ network')
    ax.set_ylim([0.1, 0.9])
    ax.set_ylabel(r"pearson's $r$")
    ax.legend(loc='lower right', title='')

plt.show()