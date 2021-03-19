# %%
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from phd import phd_util
from phd.explore import plot_funcs

# db connection params 
db_config = {
    'host': 'localhost',
    'port': 5432,
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
    'mu_hill_branch_wt_0[1] as mu_hill_branch_wt_0_100',
    'mu_hill_branch_wt_0[2] as mu_hill_branch_wt_0_200',
    'ac_retail_food[1] as ac_retail_food_100',
    'ac_retail_food[2] as ac_retail_food_200',
    'mu_hill_branch_wt_0[6] as mu_hill_branch_wt_0_600',
    'mu_hill_branch_wt_0[8] as mu_hill_branch_wt_0_800',
    'ac_commercial[6] as ac_commercial_600',
    'ac_commercial[8] as ac_commercial_800'
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
df_full = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_full_dual',
    'WHERE city_pop_id = 1 and within = true')
df_full = df_full.set_index('id')
df_full = phd_util.clean_pd(df_full, drop_na='all', fill_inf=np.nan)


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
          'Node Beta',
          r'Node Harmonic Angular',
          r'Node Betweenness',
          r'Node Betweenness $\beta$',
          r'Node Betweenness Angular']

y_themes = [
    'mu_hill_branch_wt_0_100',
    'ac_retail_food_100',
    'mu_hill_branch_wt_0_800',
    'ac_commercial_800'
]

y_labels = [
    r'Hill wt. $_{q=0\ \beta=-0.04\ d_{max}=100m}$',
    r'Food Retail $_{\beta=-0.04\ d_{max}=100m}$',
    r'Hill wt. $_{q=0\ \beta=-0.005\ d_{max}=800m}$',
    r'Commercial $_{\beta=-0.005\ d_{max}=800m}$'
]

phd_util.plt_setup()
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
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
        # plot
        im = plot_funcs.plot_heatmap(ax,
                                     heatmap=corrs,
                                     row_labels=labels,
                                     col_labels=distances_bandwise,
                                     cbar=True,
                                     text=corrs)
        ax.set_title(y_label)
        theme_dim += 1
plt.suptitle('Correlation matrices for centrality measures to mixed-use & landuse themes')
path = f'../phd-admin/PhD/part_2/images/centrality/dual_centralities_corr_grid.png'
plt.savefig(path, dpi=300)

