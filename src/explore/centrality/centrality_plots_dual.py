# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src import util_funcs
from src.explore import plot_funcs

# db connection params 
db_config = {
    'host': 'localhost',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

# columns
columns = [
    'id',
    'city_pop_id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y',
    'mu_hill_branch_wt_0[3] as mu_hill_branch_wt_0_300',
    'ac_retail_food[3] as ac_retail_food_300',
    'mu_hill_branch_wt_0[10] as mu_hill_branch_wt_0_1000',
    'ac_commercial[10] as ac_commercial_1000'
]

columns_base = [
    'c_node_density',
    'c_node_harmonic',
    'c_node_beta',
    'c_node_betweenness',
    'c_node_betweenness_beta',
    'c_node_harmonic_angular',
    'c_node_betweenness_angular']
col_template = '{col}[{i}] as {col}_{dist}'
distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
for col_base in columns_base:
    for i, d in enumerate(distances):
        columns.append(col_template.format(col=col_base, i=i + 1, dist=d))
# bandwise distances do not include 50m
distances_bandwise = distances[1:]

#  %% load nodes data
print('loading columns')
df_full = util_funcs.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_full_dual',
    'WHERE city_pop_id = 1 and within = true')
df_full = df_full.set_index('id')
df_full = util_funcs.clean_pd(df_full, drop_na='all', fill_inf=np.nan)

#  %%
'''
correlation matrix plot
'''

themes = [
    'c_node_density_{dist}',
    'c_node_harmonic_{dist}',
    'c_node_beta_{dist}',
    'c_node_harmonic_angular_{dist}',
    'c_node_betweenness_{dist}',
    'c_node_betweenness_beta_{dist}',
    'c_node_betweenness_angular_{dist}']

labels = ['Node Density',
          'Node Harmonic',
          r'Node $\beta$',
          r'Nd. Harmonic Angular',
          r'Node Betw.',
          r'Node Betw. $\beta$',
          r'Node Betw. Angular']

y_themes = [
    'mu_hill_branch_wt_0_300',
    'ac_retail_food_300',
    'mu_hill_branch_wt_0_1000',
    'ac_commercial_1000'
]

y_labels = [
    r'weighted mixed-use richness $_{q=0\ \beta=0.01\bar{3}\ d_{max}=300m}$',
    r'food retail $_{\beta=0.01\bar{3}\ d_{max}=300m}$',
    r'weighted mixed-use richness $_{q=0\ \beta=0.004\ d_{max}=1000m}$',
    r'commercial $_{\beta=0.004\ d_{max}=1000m}$'
]

util_funcs.plt_setup()
fig, axes = plt.subplots(2, 2, figsize=(10, 4.6))
theme_dim = 0
for ax_row in range(2):
    for ax_col in range(2):
        y_theme = y_themes[theme_dim]
        y_label = y_labels[theme_dim]
        print('processing theme: ', y_label)
        ax = axes[ax_row][ax_col]
        # create an empty numpy array
        corrs = np.full((len(themes), len(distances_bandwise)), np.nan)
        # calculate the correlations for each type and respective distance of mixed-use measure
        for t_idx, (theme, label) in enumerate(zip(themes, labels)):
            for d_idx, dist in enumerate(distances_bandwise):
                # prepare theme and send to data prep function
                x_val = df_full[theme.format(dist=dist)]
                y_val = df_full[y_theme]
                # deduce correlation
                corrs[t_idx][d_idx] = spearmanr(x_val, y_val)[0]
        if ax_col == 0:
            display_row_labels = True
            cbar = False
        else:
            display_row_labels = False
            cbar = True
        if ax_row > 0:
            display_col_labels = True
        else:
            display_col_labels = False
        # plot
        im = plot_funcs.plot_heatmap(ax,
                                     heatmap=corrs,
                                     row_labels=labels,
                                     col_labels=distances_bandwise,
                                     set_col_labels=display_col_labels,
                                     set_row_labels=display_row_labels,
                                     cbar=cbar,
                                     text=corrs,
                                     fontsize=6)
        ax.set_title(y_label)
        theme_dim += 1
plt.suptitle(r'Spearman $\rho$ ' + f'correlations for centralities to mixed-uses & landuses on the dual graph')
path = f'../phd-doc/doc/part_2/centrality/images/dual_centralities_corr_grid.pdf'
plt.savefig(path)
