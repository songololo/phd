# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn import preprocessing

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
columns_base = [
    'id',
    'city_pop_id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y',
    'mu_hill_branch_wt_0[3] as mu_hill_branch_wt_0_300',
    'ac_retail_food[3] as ac_retail_food_300',
    'mu_hill_branch_wt_0[10] as mu_hill_branch_wt_0_1000',
    'ac_commercial[10] as ac_commercial_1000'
]

columns_rdm = [
    'mu_hill_branch_wt_0_rdm[3] as mu_hill_branch_wt_0_300_rdm',
    'ac_retail_food_rdm[3] as ac_retail_food_300_rdm',
    'mu_hill_branch_wt_0_rdm[10] as mu_hill_branch_wt_0_1000_rdm',
    'ac_commercial_rdm[10] as ac_commercial_1000_rdm'
]

columns_interp = [
    'c_node_density',
    'c_node_farness',
    'c_node_cycles',
    'c_node_harmonic',
    'c_node_beta',
    'c_segment_density',
    'c_segment_harmonic',
    'c_segment_beta',
    'c_node_betweenness',
    'c_node_betweenness_beta',
    'c_segment_betweenness',
    'c_node_harmonic_angular',
    'c_segment_harmonic_hybrid',
    'c_node_betweenness_angular',
    'c_segment_betweeness_hybrid'
]

'''
# Decided not to do bandwise because bands are inconsistent across scales
# i.e. MAUP muddies relevancy of correlations at larger scales / larger bundles
# experiments do confirm, however, that correlations drop-off at larger scales
# in spite of MAUP boost
def get_band(df, dist, target, all_distances):
    if dist == 50:
        t = df[target.format(dist=50)]
    else:
        cur_idx = all_distances.index(dist)
        lag_idx = cur_idx - 1
        lag_dist = dist - all_distances[lag_idx]
        t_cur = df[target.format(dist=dist)]
        t_lag = df[target.format(dist=lag_dist)]
        t = t_cur - t_lag
    return t
    # subsequent bands subtract the prior band
'''

#  %% load nodes data
col_template = '{col}[{i}] as {col}_{dist}'
all_distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                 1100, 1200, 1300, 1400, 1500, 1600, 3200, 4800, 6400, 8000]
columns = [c for c in columns_base]
columns += [c for c in columns_rdm]
for col in columns_interp:
    for i, d in enumerate(all_distances):
        columns.append(col_template.format(col=col, i=i + 1, dist=d))

print('loading columns')
df_full = util_funcs.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_full',
    'WHERE city_pop_id = 1 and within = true')
df_full = df_full.set_index('id')
df_full = util_funcs.clean_pd(df_full, drop_na='all', fill_inf=np.nan)

for d in all_distances:
    df_full[f'node_improved_{d}'] = df_full[f'c_node_density_{d}'] ** 2 / df_full[f'c_node_farness_{d}']
    df_full[f'conv_close_{d}'] = 1 / df_full[f'c_node_farness_{d}']
    df_full[f'conv_close_norm_{d}'] = df_full[f'c_node_density_{d}'] / df_full[f'c_node_farness_{d}']
df_full[np.isnan(df_full)] = 0

distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

columns = [c for c in columns_base]
for col in columns_interp:
    for i, d in enumerate(distances):
        columns.append(col_template.format(col=col, i=i + 1, dist=d))

df_100 = util_funcs.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_100',
    'WHERE city_pop_id = 1 and within = true')
df_100 = df_100.set_index('id')
df_100 = util_funcs.clean_pd(df_100, drop_na='all', fill_inf=np.nan)

df_50 = util_funcs.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_50',
    'WHERE city_pop_id = 1 and within = true')
df_50 = df_50.set_index('id')
df_50 = util_funcs.clean_pd(df_50, drop_na='all', fill_inf=np.nan)

df_20 = util_funcs.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_20',
    'WHERE city_pop_id = 1 and within = true')
df_20 = df_20.set_index('id')
df_20 = util_funcs.clean_pd(df_20, drop_na='all', fill_inf=np.nan)

for d in distances:
    # improved closeness
    df_100[f'node_improved_{d}'] = df_100[f'c_node_density_{d}'] ** 2 / df_100[f'c_node_farness_{d}']
    df_50[f'node_improved_{d}'] = df_50[f'c_node_density_{d}'] ** 2 / df_50[f'c_node_farness_{d}']
    df_20[f'node_improved_{d}'] = df_20[f'c_node_density_{d}'] ** 2 / df_20[f'c_node_farness_{d}']
    # global closeness
    df_100[f'conv_close_{d}'] = 1 / df_100[f'c_node_farness_{d}']
    df_50[f'conv_close_{d}'] = 1 / df_50[f'c_node_farness_{d}']
    df_20[f'conv_close_{d}'] = 1 / df_20[f'c_node_farness_{d}']
    # normalised global closeness
    df_100[f'conv_close_norm_{d}'] = df_100[f'c_node_density_{d}'] / df_100[f'c_node_farness_{d}']
    df_50[f'conv_close_norm_{d}'] = df_50[f'c_node_density_{d}'] / df_50[f'c_node_farness_{d}']
    df_20[f'conv_close_norm_{d}'] = df_20[f'c_node_density_{d}'] / df_20[f'c_node_farness_{d}']

# %%
'''
correlation matrix plot
'''
themes = [
    'c_node_density_{dist}',
    'c_node_cycles_{dist}',
    'c_node_harmonic_{dist}',
    'c_node_harmonic_angular_{dist}',
    'c_node_beta_{dist}',
    'node_improved_{dist}',
    'conv_close_{dist}',
    'conv_close_norm_{dist}',
    'c_segment_density_{dist}',
    'c_segment_harmonic_{dist}',
    'c_segment_harmonic_hybrid_{dist}',
    'c_segment_beta_{dist}',
    'c_node_betweenness_{dist}',
    'c_node_betweenness_angular_{dist}',
    'c_node_betweenness_beta_{dist}',
    'c_segment_betweenness_{dist}',
    'c_segment_betweeness_hybrid_{dist}']
labels = ['Nd. Density',
          'Nd. Cycles',
          'Nd. Harmonic',
          'Nd. Harm. Ang.',
          r'Nd. $\beta$',
          'Nd. Improved',
          'Closeness',
          'Closen. Norm.',
          'Seg. Density',
          'Seg. Harmonic',
          'Seg. Harm. Ang.',
          r'Seg. $\beta$',
          'Nd. Betw.',
          'Nd. Betw. Ang.',
          r'Nd. Betw. $\beta$',
          r'Seg. Betw. $\beta$',
          'Seg. Betw. Ang.']
lu_themes = [
    'mu_hill_branch_wt_0_300',
    'ac_retail_food_300',
    'mu_hill_branch_wt_0_1000',
    'ac_commercial_1000']
lu_labels = [
    r'mixed-uses $d_{max}=300m$',
    r'food retail $d_{max}=300m$',
    r'mixed-uses $d_{max}=1000m$',
    r'commercial $d_{max}=1000m$']

slim_distances = [50, 100, 200, 400, 800, 1600, 3200, 6400, 8000]

# each plot theme plots to a new page
for rdm in [False, True]:
    for lu_theme_base, lu_label in zip(lu_themes, lu_labels):
        print(f'Processing landuse theme {lu_theme_base}')
        # setup the plot
        util_funcs.plt_setup()
        fig, axes = plt.subplots(1, 2, figsize=(7, 6.5))
        # iterate the permutations
        for ax_col, seg_norm in enumerate([False, True]):
            # create an empty numpy array
            corrs = np.full((len(themes), len(slim_distances)), np.nan)
            # if rdm, then add key
            lu_theme = lu_theme_base
            if rdm:
                lu_theme += '_rdm'
            # get the landuse theme data
            lu_val = df_full[lu_theme]
            # normalise by segment lengths
            if seg_norm:
                if '300' in lu_theme:
                    lu_val = df_full[lu_theme] / df_full['c_segment_density_300']
                elif '1000' in lu_theme:
                    lu_val = df_full[lu_theme] / df_full['c_segment_density_1000']
                else:
                    raise ValueError('Distance not contained in theme description?')
            # ax = axes[ax_row][ax_col]
            ax = axes[ax_col]
            # calculate the correlations for each type and respective distance of mixed-use measure
            for t_idx, (theme, label) in enumerate(zip(themes, labels)):
                for d_idx, dist in enumerate(slim_distances):
                    # prepare theme and send to data prep function
                    x_val = df_full[theme.format(dist=dist)]
                    # deduce correlation
                    corrs[t_idx][d_idx] = spearmanr(x_val, lu_val)[0]
            im = plot_funcs.plot_heatmap(ax,
                                         heatmap=corrs,
                                         row_labels=labels,
                                         col_labels=slim_distances,
                                         set_col_labels=True,
                                         set_row_labels=ax_col == 0,
                                         text=corrs.round(2))
            if rdm and seg_norm:
                ax_title = f'Randomised & length-normalised mixed-uses'
            elif rdm:
                ax_title = f'Randomised mixed-uses'
            elif seg_norm:
                ax_title = f'Mixed-uses normalised by street lengths'
            else:
                ax_title = r'Mixed-uses as measured'
            ax.set_xlabel(ax_title)
        if rdm:
            app_title = r'randomised '
            app_path = '_rdm'
        else:
            app_title = ''
            app_path = ''
        cbar = fig.colorbar(im,
                            ax=axes,
                            aspect=50,
                            pad=0.02,
                            orientation='horizontal',
                            ticks=[-1, 0, 1],
                            shrink=0.5)
        cbar.ax.set_xticklabels(['-1', r'Spearman $\rho$ correlations', '1'])
        if rdm:
            plt.suptitle(f'Centrality measures correlated to randomised {lu_label}')
        else:
            plt.suptitle(f'Centrality measures correlated to {lu_label}')
        path = f'../phd-doc/doc/images/centrality/primal_centralities_corr_grid_{lu_theme_base}{app_path}.pdf'
        plt.savefig(path)

# %%
'''
plot the maps showing the mapped centrality measures
'''
theme_sets = [
    ('c_node_density_{dist}',
     'c_segment_density_{dist}',
     'conv_close_norm_{dist}'),
    ('c_node_harmonic_{dist}',
     'c_node_harmonic_angular_{dist}',
     'c_node_beta_{dist}'),
    ('c_segment_harmonic_{dist}',
     'c_segment_harmonic_hybrid_{dist}',
     'c_segment_beta_{dist}'),
    ('c_node_betweenness_{dist}',
     'c_node_betweenness_angular_{dist}',
     'c_node_betweenness_beta_{dist}'),
    ('c_segment_betweenness_{dist}',
     'c_segment_betweeness_hybrid_{dist}',
     'c_node_cycles_{dist}')
]
theme_label_sets = [
    ('Node Density',
     'Segment Density',
     'Normalised Closeness'),
    ('Node Harmonic',
     'Node Harmonic Angular',
     r'Node $\beta$'),
    ('Segment Harmonic',
     'Segment Harmonic Hybrid',
     r'Segment $\beta$'),
    ('Node Betweenness',
     'Node Betweenness Angular',
     r'Node Betweenness $\beta$'),
    (r'Segment Betweenness $\beta$',
     'Segment Betweenness Hybrid',
     'Node Cycles'),
]
theme_metas = (
    'contrast',
    'closeness_node',
    'closeness_segment',
    'betweenness_node',
    'betweenness_segment')
theme_wts = (4, 4, 4, 2, 2)

for n, (theme_set, theme_labels, theme_meta, theme_wt) in enumerate(zip(theme_sets,
                                                                        theme_label_sets,
                                                                        theme_metas,
                                                                        theme_wts)):
    print(f'processing set: {n}')
    util_funcs.plt_setup()
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    for t_idx, (t, label) in enumerate(zip(theme_set, theme_labels)):
        for d_idx, (dist, beta) in enumerate(zip([400, 1600],
                                                 [r'0.01', r'0.0025'])):
            ax = axes[t_idx][d_idx]
            theme = t.format(dist=dist)
            data = df_20[theme]
            plot_funcs.plot_scatter(fig,
                                    ax,
                                    df_20.x,
                                    df_20.y,
                                    vals=data,
                                    km_per_inch=3.5,
                                    s_exp=theme_wt)
            ax.set_xlabel(f'{label} ' + '$d_{max}=' + f'{dist}m$')
    plt.suptitle(f'Plots of centrality measures for inner London (set {n + 1})')
    path = f'../phd-doc/doc/images/centrality/primal_centrality_comparisons_{theme_meta}.pdf'
    plt.savefig(path, dpi=300)

# %%
'''
demonstrate the typical distributions
the contention is that low values preclude mixed-uses but high values don't guarantee them
the spike in the distributions is an artefact of search distances vis-a-vis straight roads with no intersections,
in which case a certain number of adjacent nodes is reached, resulting in a stable preponderance of a certain value
this would not be present in gravity-weighted forms of centrality
'''
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 2,
                         figsize=(7, 3.5),
                         sharey='row',
                         sharex='col',
                         gridspec_kw={'height_ratios': [4, 1]})
# left panel - gravity index
x1 = df_full['c_segment_beta_1000']
x2 = df_full['c_segment_betweenness_1000']
y = df_full['mu_hill_branch_wt_0_1000']
s = preprocessing.minmax_scale(y, (0.1, 50))
lw = preprocessing.minmax_scale(y, (0.05, 0.2))
# hexbin
axes[0][0].hexbin(x1,
                  y,
                  gridsize=(80, 50),
                  mincnt=5,
                  bins='log',
                  xscale='linear',
                  yscale='linear',
                  lw=0.2,
                  edgecolors='w',
                  cmap='Blues')
# histogram
axes[1][0].hist(x1, 100, edgecolor='white', linewidth=0.3, log=False, color='#0064b7')
axes[1][0].set_xlabel(r'segment $\beta$ weighted closeness $d_{max}=1000m$')
axes[1][0].set_xlim(left=0, right=6000)
# right panel - hexbin
axes[0][1].hexbin(x2[x2 > 0],
                  y[x2 > 0],
                  gridsize=(180, 50),
                  mincnt=5,
                  bins='log',
                  xscale='log',
                  yscale='linear',
                  lw=0.2,
                  edgecolors='w',
                  cmap='Blues')
# histogram
b_w = np.logspace(np.log10(100), np.log10(100000), 100)
axes[1][1].hist(x2, b_w, edgecolor='white', linewidth=0.3, log=False, color='#0064b7')
axes[1][1].set_xlabel(r'$\log$ segment $\beta$ weighted betweenness $d_{max}=1000m$')
axes[1][1].set_xscale('log')
axes[1][1].set_xlim(left=300, right=100000)
# shared
axes[0][0].set_ylim(bottom=0, top=100)
axes[0][0].set_ylabel(r'mixed-use richness $d_{max}=1000m$')

plt.suptitle('Hexbin plots and distributions: centrality measures to mixed-use richness')
path = f'../phd-doc/doc/images/centrality/primal_hexbin_gravity_betweenness.pdf'
plt.savefig(path, dpi=300)

# %%
'''
plot betweenness vs closeness, coloring mixed-uses
'''
util_funcs.plt_setup()

fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))

# sort by mixed uses for print order
df_temp = df_full.copy(deep=True)
df_temp = df_temp.sort_values(by='mu_hill_branch_wt_0_1000')

x = df_temp['c_segment_beta_1000']
y = df_temp['c_segment_betweenness_1000']
c = df_temp[['mu_hill_branch_wt_0_1000']]

sp = ax.hexbin(x[y > 0],
               y[y > 0],
               C=c[y > 0],
               gridsize=(80, 100),
               mincnt=5,
               bins='log',
               xscale='linear',
               yscale='log',
               cmap='YlOrRd',
               lw=0.2,
               edgecolors='w',
               reduce_C_function=np.mean)
cbar_max = round(c.values.max())
cbar = fig.colorbar(sp,
                    ax=ax,
                    aspect=50,
                    pad=0.01,
                    orientation='vertical',
                    ticks=[0, 10, 100],
                    shrink=0.9)
cbar.set_label('mixed-use richness',
               labelpad=-10,
               loc='bottom',
               fontsize='xx-small')
ax.set_xlim(left=0, right=6000)
ax.set_ylim(bottom=300, top=100000)
ax.set_yscale('log')
ax.set_xlabel(r'segment $\beta$ weighted closeness $d_{max}=1000m$')
ax.set_ylabel(r'$\log$ segment $\beta$ wt. betweenness $d_{max}=1000m$')

plt.suptitle('Hexbin plot of closeness and betweenness')
path = f'../phd-doc/doc/images/centrality/primal_gravity_betweenness_mixed_hexbin.pdf'
plt.savefig(path, dpi=300)

# %%
"""
DEPRECATED
#  %%
'''
compare different variants of closeness
'''
util_funcs.plt_setup()
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
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
themes = [
    'c_node_density_{dist}',
    'c_node_harmonic_{dist}',
    'c_node_harmonic_angular_{dist}',
    'c_node_beta_{dist}',
    'c_segment_density_{dist}',
    'c_segment_harmonic_{dist}',
    'c_segment_harmonic_hybrid_{dist}',
    'c_segment_beta_{dist}'
]
labels = [
    'Node Density',
    r'Node Harmonic',
    r'Node Harmonic Angular',
    r'Node $\beta$',
    'Segment Density',
    'Segment Harmonic',
    'Segment Harmonic Hybrid',
    r'Segment $\beta$'
]
markers = ['x', 'x', 'x', 'x', '', '', '', '']
styles = ['-.', '-.', '-.', '-.', '-', '-', '-', '-']
widths = [1, 1, 1, 1, 2, 2, 2, 2]
colours = ['C0', 'C1', 'C2', 'C3', 'C0', 'C1', 'C2', 'C3']
alphas = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
for ax, y_theme, y_label in zip(axes, y_themes, y_labels):
    if '300' in y_label:
        y_val = df_full[y_theme] / df_full['c_segment_density_300']
    elif '1000' in y_label:
        y_val = df_full[y_theme] / df_full['c_segment_density_1000']
    else:
        raise ValueError('Distance not contained in theme description?')
    for x_theme, x_label, mk, ls, lw, lc, la in zip(themes, labels, markers, styles, widths, colours, alphas):
        vals = []
        for d in distances:
            x_th = x_theme.format(dist=d)
            x_val = df_full[x_th]
            vals.append(spearmanr(x_val, y_val)[0])

        ax.plot(distances, vals, lw=lw, alpha=la, color=lc, linestyle=ls, marker=mk, label=x_label)

    ax.set_xlim([50, 1600])
    ax.set_xticks(distances)
    ax.set_xticklabels([f'{d}m' for d in distances])
    ax.set_ylim([0, 0.5])
    ax.legend(loc='lower right', title='')
    ax.set_ylabel(r'spearman $\rho$')
    ax.set_xlabel(f'Correlations for closeness measures to length normalised {y_label}')
plt.suptitle('Correlations for closeness measures to length-normalised mixed-use and landuse themes')

path = f'../phd-doc/doc/images/centrality/primal_correlations_closeness.pdf'
plt.savefig(path)

#  %%
# compare variants of betweenness
util_funcs.plt_setup()
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
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
themes = [
    'c_node_betweenness_{dist}',
    'c_node_betweenness_angular_{dist}',
    'c_node_betweenness_beta_{dist}',
    'c_segment_betweenness_{dist}',
    'c_segment_betweeness_hybrid_{dist}'
]
labels = [
    r'Node Betweenness',
    r'Node Betweenness Angular',
    r'Node Betweenness $\beta$',
    r'Segment Betweenness $\beta$',
    'Segment Betweenness Hybrid'
]
markers = ['x', 'x', 'x', '', '', ]
styles = ['-.', '-.', '-.', '-', '-']
widths = [1, 1, 1, 2, 2]
alphas = [1, 1, 1, 0.5, 0.5]
for ax, y_theme, y_label in zip(axes, y_themes, y_labels):
    if '300' in y_label:
        y_val = df_full[y_theme] / df_full['c_segment_density_300']
    elif '1000' in y_label:
        y_val = df_full[y_theme] / df_full['c_segment_density_1000']
    else:
        raise ValueError('Distance not contained in theme description?')
    for x_theme, x_label, mk, ls, lw, lc, la in zip(themes, labels, markers, styles, widths, colours, alphas):
        vals = []
        for d in distances:
            x_th = x_theme.format(dist=d)
            x_val = df_full[x_th]
            vals.append(spearmanr(x_val, y_val)[0])
        ax.plot(distances, vals, lw=lw, alpha=la, color=lc, linestyle=ls, marker=mk, label=x_label)

    ax.set_xlim([50, 1600])
    ax.set_xticks(distances)
    ax.set_xticklabels([f'{d}m' for d in distances])
    ax.set_ylim([0, 0.5])
    ax.legend(loc='lower right', title='')
    ax.set_ylabel(r'spearman $\rho$')
    ax.set_xlabel(f'Correlations for betweenness measures to length normalised {y_label}')
plt.suptitle('Correlations for betweenness measures to length-normalised mixed-use and landuse themes')

path = f'../phd-doc/doc/images/centrality/primal_correlations_betweenness.pdf'
plt.savefig(path)
"""

# %%
'''
decomposition set against full, 100m, 50m, 20m tables
'''
outer_themes_angular = [
    'c_node_harmonic_angular_{dist}',
    'c_node_betweenness_angular_{dist}',
]
outer_labels_angular = [
    'Node Harmonic Angular',
    'Node Betweenness Angular',
]
outer_themes = [
    'c_node_beta_{dist}',
    'c_node_betweenness_beta_{dist}',
]
outer_labels = [
    r'Node $\beta$',
    r'Node Betweenness $\beta$',
]
inner_themes = ['mu_hill_branch_wt_0_300',
                'mu_hill_branch_wt_0_1000']
inner_labels = [r'mixed-use richness $d_{max}=300$',
                r'mixed-use richness $d_{max}=1000$']

util_funcs.plt_setup()
fig, axes = plt.subplots(2, 1, figsize=(7, 7))
for ax, outer_theme, outer_label in zip(axes,
                                        outer_themes,
                                        outer_labels):
    print(f'calculating for theme: {outer_label}')
    for inner_theme, inner_label, lt, mt in zip(inner_themes,
                                                inner_labels,
                                                ['-', '--'],
                                                ['x', '.']):
        corrs_full = []
        corrs_100 = []
        corrs_50 = []
        corrs_20 = []
        if '300' in inner_theme:
            inner_seg_key = 'c_segment_density_300'
        else:
            inner_seg_key = 'c_segment_density_1000'
        for d_idx, dist in enumerate(distances):
            x_theme = outer_theme.format(dist=dist)
            # full
            y_full = df_full[inner_theme]
            x_full = df_full[x_theme]
            corrs_full.append(spearmanr(x_full, y_full)[0])
            # 100 network
            y_100 = df_100[inner_theme]
            x_100 = df_100[x_theme]
            corrs_100.append(spearmanr(x_100, y_100)[0])
            # 50 network
            y_50 = df_50[inner_theme]
            x_50 = df_50[x_theme]
            corrs_50.append(spearmanr(x_50, y_50)[0])
            # 20 network
            y_20 = df_20[inner_theme]
            x_20 = df_20[x_theme]
            corrs_20.append(spearmanr(x_20, y_20)[0])
        a = 1
        lw = 1
        c_0 = 'C0'
        c_1 = 'C1'
        c_2 = 'C2'
        c_3 = 'C3'
        ax.plot(distances,
                corrs_full,
                lw=lw,
                alpha=a,
                linestyle=lt,
                color=c_0,
                marker=mt,
                label=f'full - {inner_label}')
        ax.plot(distances,
                corrs_100,
                lw=lw,
                alpha=a,
                linestyle=lt,
                color=c_1,
                marker=mt,
                label=f'100m - {inner_label}')
        ax.plot(distances,
                corrs_50,
                lw=lw,
                alpha=a,
                linestyle=lt,
                color=c_2,
                marker=mt,
                label=f'50m - {inner_label}')
        ax.plot(distances,
                corrs_20,
                lw=lw,
                alpha=a,
                linestyle=lt,
                color=c_3,
                marker=mt,
                label=f'20m - {inner_label}')
    ax.set_xlim([50, 1600])
    ax.set_xticks(distances)
    ax.set_xlabel(f'Correlations for length-normalised mixed-use richness to {outer_label} centrality')
    ax.set_ylim([0, 1])
    ax.set_ylabel(r"spearman $\rho$")

    ax.legend(loc='lower right', title='')

plt.suptitle(r'Centrality measures correlated to mixed-use richness at varying decompositions')
path = f'../phd-doc/doc/images/centrality/primal_decompositions.pdf'

plt.savefig(path)

# %%
'''
plots fragmentation of distributions at smaller thresholds
'''
tables = [df_full, df_50, df_20]  # , df_100
table_labels = ['no', '50m', '20m']  # , '$100m$'
targets = ['c_node_harmonic_{dist}', 'c_segment_harmonic_{dist}']
labels = ['Node harmonic closeness', 'Segment harmonic closeness']
metas = ['node_harm', 'seg_harm']
# prepare font dict
font = {'size': 7}
for target, label, meta in zip(targets, labels, metas):
    util_funcs.plt_setup()
    fig, axes = plt.subplots(3, 4, figsize=(7, 5.5))
    for t_idx, (table, table_label) in enumerate(zip(tables, table_labels)):
        for d_idx, dist in enumerate([50, 100, 200, 400]):  # , 300
            x = table[target.format(dist=dist)]
            x = x[x < np.percentile(x, 99.9)]
            bins = axes[t_idx][d_idx].hist(x,
                                           bins=100,
                                           edgecolor='white',
                                           linewidth=0.3)
            axes[t_idx][d_idx].set_xlabel(f'{table_label} decomp. ' + r'$d_{max}=' + str(dist) + r'm$')
            axes[t_idx][d_idx].set_xlim(left=0, right=np.nanmax(x))
            tr = axes[t_idx][d_idx].transAxes
            mu = round(np.nanmean(x), 2)
            var = round(np.nanvar(x), 2)
            if t_idx < 3:
                axes[t_idx][d_idx].get_xaxis().set_ticklabels([])
            if d_idx > 0:
                axes[t_idx][d_idx].get_yaxis().set_ticklabels([])
            axes[t_idx][d_idx].text(1,
                                    1,
                                    f'$\mu={mu}$',
                                    ha='right',
                                    va='top',
                                    fontdict=font,
                                    transform=tr)
            axes[t_idx][d_idx].text(1,
                                    0.9,
                                    f'$\sigma^2={var}$',
                                    ha='right',
                                    va='top',
                                    fontdict=font,
                                    transform=tr)

    plt.suptitle(f'{label} distributions at respective decompositions and distances')

    path = f'../phd-doc/doc/images/centrality/distribution_degeneration_{meta}.pdf'
    plt.savefig(path)
