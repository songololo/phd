# %%
import asyncio

import asyncpg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

columns = [
    # for table - only need specific distances (but only where not already covered in columns below)
    'id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y'
]

columns_base = [
    # using segment density weighted by distance
    'c_segment_beta_',
    'mu_hill_0',
    'mu_hill_1',
    'mu_hill_2',
    'mu_hill_branch_wt_0',
    'mu_hill_branch_wt_1',
    'mu_hill_branch_wt_2',
    # pairwise gives very similar results to branch weighted, so have removed
    'mu_hill_dispar_wt_0',
    'mu_hill_dispar_wt_1',
    'mu_hill_dispar_wt_2',
    'mu_shannon',
    'mu_gini',
    'mu_raos',
    'ac_accommodation',
    'ac_eating',
    'ac_eating_nw',
    'ac_drinking',
    'ac_commercial',
    'ac_commercial_nw',
    'ac_tourism',
    'ac_entertainment',
    'ac_government',
    'ac_manufacturing',
    'ac_retail_food',
    'ac_retail_other',
    'ac_transport',
    'ac_health',
    'ac_education',
    'ac_parks',
    'ac_cultural',
    'ac_sports',
    'ac_total'
]
col_template = '{col}[{i}] as {col}_{dist}'
distances = [50, 100, 200, 300, 400, 600, 800, 1200, 1600]
for col_base in columns_base:
    for i, d in enumerate(distances):
        columns.append(col_template.format(col=col_base, i=i + 1, dist=d))
# bandwise distances do not include 50m
distances_bandwise = distances[1:]

#  %%
# load data
df_full = util_funcs.load_data_as_pd_df(db_config,
                                        columns,
                                        'analysis.nodes_full',
                                        'WHERE city_pop_id = 1 and within = true')
df_full = df_full.set_index('id')
df_full = util_funcs.clean_pd(df_full, drop_na='all', fill_inf=np.nan)

df_100 = util_funcs.load_data_as_pd_df(db_config,
                                       columns,
                                       'analysis.nodes_100',
                                       'WHERE city_pop_id = 1 and within = true')
df_100 = df_100.set_index('id')
df_100 = util_funcs.clean_pd(df_100, drop_na='all', fill_inf=np.nan)

df_50 = util_funcs.load_data_as_pd_df(db_config,
                                      columns,
                                      'analysis.nodes_50',
                                      'WHERE city_pop_id = 1 and within = true')
df_50 = df_50.set_index('id')
df_50 = util_funcs.clean_pd(df_50, drop_na='all', fill_inf=np.nan)

df_20 = util_funcs.load_data_as_pd_df(db_config,
                                      columns,
                                      'analysis.nodes_20',
                                      'WHERE city_pop_id = 1 and within = true')
df_20 = df_20.set_index('id')
df_20 = util_funcs.clean_pd(df_20, drop_na='all', fill_inf=np.nan)

# %%
'''
plot the geographic maps showing the distribution of respective mixed use measures
'''
theme_sets = [
    ('mu_hill_0_{dist}', 'mu_hill_1_{dist}', 'mu_hill_2_{dist}'),
    ('mu_hill_branch_wt_0_{dist}', 'mu_hill_branch_wt_1_{dist}', 'mu_hill_branch_wt_2_{dist}'),
    ('mu_gini_{dist}', 'mu_shannon_{dist}', 'mu_raos_{dist}')
]
theme_label_sets = [
    (r'Hill $q=0$',
     r'Hill $q=1$',
     r'Hill $q=2$'),
    (r'Hill weighted $q=0$',
     r'Hill weighted $q=1$',
     r'Hill weighted $q=2$'),
    ('Gini-Simpson',
     'Shannon',
     'Raos Quadratic')
]
theme_metas = ('Hill', 'weighted Hill', 'traditional')
path_metas = ('hill', 'weighted_hill', 'traditional')
theme_wt = (False, True, False)

for theme_set, theme_labels, t_meta, p_meta, weighted in zip(theme_sets,
                                                             theme_label_sets,
                                                             theme_metas,
                                                             path_metas,
                                                             theme_wt):
    print(f'processing metas: {t_meta}')
    util_funcs.plt_setup()
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    for t_idx, (t, label) in enumerate(zip(theme_set, theme_labels)):
        for d_idx, (dist, beta) in enumerate(zip([200, 800],
                                                 [r'-0.02', r'-0.005'])):
            ax = axes[t_idx][d_idx]
            theme = t.format(dist=dist)
            data = df_20[theme]
            plot_funcs.plot_scatter(fig,
                                    ax,
                                    df_20.x,
                                    df_20.y,
                                    km_per_inch=3.5,
                                    vals=data,
                                    s_exp=4)
            ax.set_xlabel(f'{label} ' + r'$d_{max}=' + f'{dist}m$')
    plt.suptitle(f'Plots of {t_meta} mixed-use measures for inner London')
    path = f'../phd-doc/doc/images/diversity/diversity_comparisons_{p_meta}.pdf'
    plt.savefig(path, dpi=300)

#  %% prepare PCA
# use of bands gives slightly more defined delineations for latent dimensions
pca_columns_dist = [
    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_government_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}'
]

pca_columns_labels = []
for c in pca_columns_dist:
    lb = c.replace('ac_', '')
    lb = lb.replace('mu_hill_branch_wt_0', 'mixed_uses')
    lb = lb.replace('_{dist}', '')
    lb = lb.replace('accommodation', 'accomod.')
    lb = lb.replace('commercial', 'commerc.')
    lb = lb.replace('entertainment', 'entert.')
    lb = lb.replace('government', 'govern.')
    lb = lb.replace('infrastructure', 'infra.')
    lb = lb.replace('organisations', 'orgs.')
    lb = lb.replace('manufacturing', 'manuf.')
    lb = lb.replace('industrial', 'indust.')
    lb = lb.replace('retail_food', 'retail food')
    lb = lb.replace('retail_other', 'retail other')
    lb = lb.replace('transport', 'transp.')
    lb = lb.replace('education', 'educat.')
    lb = lb.replace('cultural', 'culture')
    lb = lb.replace('sports', 'sport')
    pca_columns_labels.append(lb)


def get_band(_df, curr_dist, target_column, seg_beta_norm=False):
    d_idx = distances.index(curr_dist)
    if d_idx == 0:
        print(f'No trailing edge for distance {curr_dist}m')
        target_band = _df.loc[:, target_column.format(dist=curr_dist)]
        seg_density_band = _df[f'c_segment_beta_{curr_dist}']
    else:
        lag_idx = d_idx - 1
        lag_dist = distances[lag_idx]
        print(f'Trailing edge: {lag_dist}m')
        target_band = \
            _df.loc[:, target_column.format(dist=curr_dist)] - _df.loc[:, target_column.format(dist=lag_dist)]
        seg_density_band = _df[f'c_segment_beta_{curr_dist}'] - _df[f'c_segment_beta_{lag_dist}']
    # whether or not to normalise values by edge lengths at equivalent betas
    if not seg_beta_norm:
        return target_band.values
    else:
        band_normalised = target_band.values / seg_density_band.values
        band_normalised[np.isnan(band_normalised)] = 0
        return band_normalised


X_raw = []
pca_models = []
pca_transformed = []
for curr_df in [df_full, df_100, df_50, df_20]:
    X_pca = {}
    for target_col in pca_columns_dist:
        # use the full distances minus 50m (no lag) so that PCA dim#2 sign doesn't flip
        for target_dist in distances:
            # bandwise and normalised gives cleanest principal components
            X_pca[target_col.format(dist=target_dist)] = get_band(curr_df,
                                                                  target_dist,
                                                                  target_col,
                                                                  seg_beta_norm=True)
    # create dataframe
    X_df = pd.DataFrame.from_dict(data=X_pca)
    X_raw.append(X_df)
    # transform
    X_df_trans = StandardScaler().fit_transform(X_df)
    # create model and fit
    model = PCA(n_components=4)
    model.fit(X_df_trans)
    pca_models.append(model)
    _transformed = model.transform(X_df_trans)
    # flip sign for PCA # 2
    _transformed[:, 1] *= -1
    pca_transformed.append(_transformed)

#  %%
'''
plot PCA components
'''
model_20 = pca_models[-1]
X_pca_20 = pca_transformed[-1]
# explained variance
exp_var = model_20.explained_variance_
exp_var_ratio = model_20.explained_variance_ratio_
# eigenvector by eigenvalue - i.e. correlation to original
loadings = model_20.components_.T * np.sqrt(exp_var)
loadings = loadings.T  # transform for slicing
# flip sign for second component
# transformed PCA are already flipped
loadings[1, :] *= -1
# clip out lower outliers to clean up plot
X_pca_20_clipped = np.clip(X_pca_20,
                           np.percentile(X_pca_20, 2),
                           np.percentile(X_pca_20, 100))
im = plot_funcs.plot_components([0, 1, 2, 3],
                                pca_columns_labels,
                                [f'{d}m' for d in distances],
                                None,  # X ignored if loadings is not None
                                X_pca_20_clipped,
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
                                figsize=(7, 6.5))
plt.suptitle(f'First 4 PCA components from POI landuse accessibilities')
path = f'../phd-doc/doc/images/diversity/PCA.pdf'
plt.savefig(path, dpi=300)

#  %%
'''
scatter and distributions for selected examples contrasting hill vs. non hill typical behaviour
'''
util_funcs.plt_setup()
fig, axes = plt.subplots(6,
                         2,
                         figsize=(7, 11),
                         gridspec_kw={'height_ratios': [4, 1, 4, 1, 4, 1]})
dist = 800
X_pca_20 = pca_transformed[-1]
_y = X_pca_20[:, 0]
y_label = r'PCA component $1$'

for ax_row, divs, div_labels in zip(
        [0, 2, 4],
        [(f'mu_hill_1_{dist}',
          f'mu_shannon_{dist}'),
         (f'mu_hill_2_{dist}',
          f'mu_gini_{dist}'),
         (f'mu_hill_dispar_wt_2_{dist}',
          f'mu_raos_{dist}')],
        [(r'Hill $q=1$',
          r'Shannon Information'),
         (r'Hill $q=2$',
          r'Gini'),
         (r'Hill class disparity wt. $q=2$',
          r'Rao class disparity wt.')]):
    for ax_col in [0, 1]:
        div_theme = divs[ax_col]
        _x = df_20.loc[:, div_theme].values
        x_label = div_labels[ax_col]
        axes[ax_row][ax_col].hexbin(_x,
                                    _y,
                                    gridsize=(70, 40),
                                    mincnt=5,
                                    bins='log',
                                    xscale='linear',
                                    yscale='linear',
                                    lw=0.2,
                                    edgecolors='w',
                                    cmap='Blues')
        axes[ax_row][ax_col].set_xlim(left=0, right=np.percentile(_x, 99.999))
        axes[ax_row][ax_col].set_ylim(bottom=0, top=np.percentile(_y, 99.999))
        axes[ax_row][ax_col].set_ylabel(y_label)
        axes[ax_row + 1][ax_col].hist(_x, 100, edgecolor='white', linewidth=0.3, log=False)
        axes[ax_row + 1][ax_col].set_xlim(left=0, right=np.percentile(_x, 99.999))
        axes[ax_row + 1][ax_col].set_ylim(bottom=0)
        axes[ax_row + 1][ax_col].set_xlabel(x_label)

plt.suptitle(f'A comparison of mixed-use measure distributions')

path = f'../phd-doc/doc/images/diversity/mixed_uses_example_distributions.pdf'
plt.savefig(path, dpi=300)

#  %%
'''
correlation matrix plots for all measures at all distances set against PCA dimensions
one page each PCA dim
1 - correlations as is, 2 - by band, 3 - as-is normalised by segments, 4 - by band normalised by segments
'''
themes = ['mu_hill_0_{dist}',
          'mu_hill_1_{dist}',
          'mu_hill_2_{dist}',
          'mu_hill_branch_wt_0_{dist}',
          'mu_hill_branch_wt_1_{dist}',
          'mu_hill_branch_wt_2_{dist}',
          'mu_hill_dispar_wt_0_{dist}',
          'mu_hill_dispar_wt_1_{dist}',
          'mu_hill_dispar_wt_2_{dist}',
          'mu_gini_{dist}',
          'mu_shannon_{dist}',
          'mu_raos_{dist}']

labels = ['$H_{0}$',
          '$H_{1}$',
          '$H_{2}$',
          '$H_{0}$ b.w.',
          '$H_{1}$ b.w.',
          '$H_{2}$ b.w.',
          '$H_{0}$ d.w.',
          '$H_{1}$ d.w.',
          '$H_{2}$ d.w.',
          'Gini',
          'Shannon',
          'Rao']

# don't include 50
for pca_dim in range(2):
    print(f'Processing dimension {pca_dim + 1}')
    # setup the plot
    util_funcs.plt_setup()
    fig, axes = plt.subplots(2, 2, figsize=(7, 10))
    # fetch the slice
    X_pca_20 = pca_transformed[-1]
    pca_slice = X_pca_20[:, pca_dim]
    # iterate the ax rows and ax cols
    for ax_row_n, (ax_row, bandwise) in enumerate(zip(axes, [False, True])):
        for ax_col_n, (ax, seg_norm) in enumerate(zip(ax_row, [False, True])):
            # create an empty numpy array for containing the correlations
            corrs = np.full((len(themes), len(distances)), np.nan)
            # calculate the correlations for each type and respective distance of mixed-use measure
            for t_idx, (theme, label) in enumerate(zip(themes, labels)):
                for d_idx, dist in enumerate(distances):
                    # if bandwise - seg_norm case handled internally
                    if bandwise:
                        v = get_band(df_20, dist, theme, seg_beta_norm=seg_norm)
                    # if not bandwise, separate the seg_norm from not seg_norm case
                    else:
                        v = df_20[theme.format(dist=dist)]
                        if seg_norm:
                            s = df_20[f'c_segment_beta_{dist}']
                            v = v.values / s.values
                            # 14 locations with 0 segment beta weight
                            v[np.isnan(v)] = 0
                    corrs[t_idx][d_idx] = spearmanr(v, pca_slice)[0]
            # plot
            im = plot_funcs.plot_heatmap(ax,
                                         corrs,
                                         row_labels=labels,
                                         col_labels=[f'{d}m' for d in distances],
                                         set_col_labels=ax_row_n == 0,
                                         set_row_labels=ax_col_n == 0,
                                         text=corrs.round(2))
            if bandwise and seg_norm:
                ax_title = f'Bandwise & length-normalised to PCA {pca_dim + 1}.'
            elif bandwise:
                ax_title = f'Bandwise mixed-uses to PCA {pca_dim + 1}.'
            elif seg_norm:
                ax_title = f'Length-normalised mixed-uses to PCA {pca_dim + 1}.'
            else:
                ax_title = f'Correlations for mixed-uses to PCA {pca_dim + 1}.'
            ax.set_xlabel(ax_title)
    cbar = fig.colorbar(im,
                        ax=axes,
                        aspect=100,
                        pad=0.01,
                        orientation='horizontal',
                        ticks=[-1.0, 0.0, 1.0])
    cbar.ax.set_xticklabels([0, r'Spearman $\rho$ correlation', 1])
    cbar.ax.xaxis.set_ticks_position('top')

    plt.suptitle(r'Spearman $\rho$ ' + f'correlations for mixed-use measures to PCA component {pca_dim + 1}')
    path = f'../phd-doc/doc/images/diversity/mixed_use_measures_correlated_pca_{pca_dim + 1}.pdf'
    plt.savefig(path)

#  %%
'''
decomposition set against full, 100m, 50m, 20m tables
'''
# plot
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 1, figsize=(7, 7))
theme = 'mu_hill_branch_wt_0_{dist}'
label = '$H_{0}$ branch weighted'
theme_wt = True
for ax, pca_dim in zip(axes, list(range(2))):
    print(f'plotting for dim: {pca_dim + 1}')
    corrs_full = []
    corrs_100 = []
    corrs_50 = []
    corrs_20 = []
    for d_idx, dist in enumerate(distances_bandwise):
        # x_theme = theme.format(dist=dist)
        # full graph
        v = get_band(df_full, dist, theme, seg_beta_norm=True)
        corrs_full.append(spearmanr(v, pca_transformed[0][:, pca_dim])[0])
        # 100 graph
        v = get_band(df_100, dist, theme, seg_beta_norm=True)
        corrs_100.append(spearmanr(v, pca_transformed[1][:, pca_dim])[0])
        # 50 graph
        v = get_band(df_50, dist, theme, seg_beta_norm=True)
        corrs_50.append(spearmanr(v, pca_transformed[2][:, pca_dim])[0])
        # 20 graph
        v = get_band(df_20, dist, theme, seg_beta_norm=True)
        corrs_20.append(spearmanr(v, pca_transformed[3][:, pca_dim])[0])

    a = 1
    lw = 1
    ls = '-'
    ms = 'x'
    #
    ax.plot(distances_bandwise, corrs_full, lw=lw, alpha=a, linestyle=ls, color='C0', marker=ms, label=f'full')
    ax.plot(distances_bandwise, corrs_100, lw=lw, alpha=a, linestyle=ls, color='C1', marker=ms, label=f'100m')
    ax.plot(distances_bandwise, corrs_50, lw=lw, alpha=a, linestyle=ls, color='C2', marker=ms, label=f'50m')
    ax.plot(distances_bandwise, corrs_20, lw=lw, alpha=a, linestyle=ls, color='C3', marker=ms, label=f'20m')
    #
    ax.set_xlim([100, 1600])
    ax.set_xticks(distances_bandwise)
    ax.set_xlabel(f'Bandwise & length-normalised correlation coefficients for {label} to PCA dim {pca_dim + 1}')
    ax.set_ylim([0, 1])
    ax.set_ylabel(r"spearman $\rho$")

    ax.legend(loc='lower right', title='')

plt.suptitle(r'Hill mixed-uses correlated to PCA at increasing network decomposition')

path = f'../phd-doc/doc/images/diversity/mixed_use_measures_corr_pca_decompositions.pdf'
plt.savefig(path)

#  %%
# RANDOM REMOVAL PLOTS
'''
6A
Choose the highest cardinality node and randomly remove species, compute each of the measures, plot
'''

db_config = {
    'host': 'localhost',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}


######################################################################################
# NOTE -> if newer data in table than date last updated then won't find correct data #
######################################################################################

# fetch all mixed uses within a crow flies 1600m radius from that point
async def fetch_data(db_config, data_lng, data_lat):
    db_con = await asyncpg.connect(**db_config)

    date_updated = await db_con.fetchval(f'''
        select date_last_updated from os.poi order by date_last_updated desc limit 1; 
    ''')

    print(f'Using {date_updated} as the DATE_LAST_UPDATED parameter.')

    class_data = await db_con.fetchrow('''
        SELECT array_agg(poi.class_code) as cc, array_agg(ST_Distance(point_geom, poi.geom)) as dists
            FROM os.poi as poi, ST_Transform(ST_SetSRID(ST_Point($1, $2), 4326), 27700) as point_geom
            WHERE date_last_updated = date($3)
                AND ST_Contains(ST_Buffer(point_geom, 1600), geom)
                ORDER BY dists ASC
                LIMIT 5000;
        ''', data_lng, data_lat, date_updated)

    await db_con.close()

    return class_data


data_lng = -0.130104
data_lat = 51.510352

class_data = asyncio.run(fetch_data(db_config, data_lng, data_lat))

class_codes = class_data['cc']
class_codes = [int(c) for c in class_codes]
class_dists = class_data['dists']

#  %%
'''
6B
'''

from numba import njit
from cityseer.algos import diversity


@njit
def deduce_unique_species(classes, distances, max_dist):
    '''
    Sifts through the classes and returns unique classes, their counts, and the nearest distance to each respective type
    Only considers classes within the max distance
    Uses the closest specimen of any particular land-use as opposed to the average distance
    e.g. if a person where trying to connect to two nearest versions of type A and B
    '''
    # check that classes and distances are the same length
    if len(classes) != len(distances):
        raise ValueError('NOTE -> the classes array and the distances array need to be the same length')
    # not using np.unique because this doesn't take into account that certain classes exceed the max distance
    # prepare arrays for aggregation
    unique_count = 0
    classes_unique_raw = np.full(len(classes), np.nan)
    classes_counts_raw = np.full(len(classes), 0)  # int array
    classes_nearest_raw = np.full(len(classes), np.inf)  # coerce to float array
    # iterate all classes
    for i in range(len(classes)):
        d = distances[i]
        # check for valid entry - in case raw array is passed where unreachable verts have be skipped - i.e. np.inf
        if not np.isfinite(d):
            continue
        # first check that this instance doesn't exceed the maximum distance
        if d > max_dist:
            continue
        # if it doesn't, get the class
        c = classes[i]
        # iterate the unique classes
        # NB -> only parallelise the inner loop
        # if parallelising the outer loop it generates some very funky outputs.... beware
        for j in range(len(classes_unique_raw)):
            u_c = classes_unique_raw[j]
            # if already in the unique list, then increment the corresponding count
            if c == u_c:
                classes_counts_raw[j] += 1
                # check if the distance to this copy is closer than the prior least distance
                if d < classes_nearest_raw[j]:
                    classes_nearest_raw[j] = d
                break
            # if no match is encountered by the end of the list (i.e. np.nan), then add it:
            if np.isnan(u_c):
                classes_unique_raw[j] = c
                classes_counts_raw[j] += 1
                classes_nearest_raw[j] = d
                unique_count += 1
                break

    classes_unique = np.full(unique_count, np.nan)
    classes_counts = np.full(unique_count, 0)
    classes_nearest = np.full(unique_count, np.inf)
    for i in range(unique_count):
        classes_unique[i] = classes_unique_raw[i]
        classes_counts[i] = classes_counts_raw[i]
        classes_nearest[i] = classes_nearest_raw[i]

    return classes_unique, classes_counts, classes_nearest


data_5 = {}
data_keys = ['A']
data_betas = [0.005]  # intentional beta vs 1600
for k, beta in zip(data_keys, data_betas):
    data_5[k] = {
        'beta': beta,
        'mu_hill_0': [],
        'mu_hill_1': [],
        'mu_hill_2': [],
        'mu_hill_branch_wt_0': [],
        'mu_hill_branch_wt_1': [],
        'mu_hill_branch_wt_2': [],
        'mu_hill_pairwise_wt_0': [],
        'mu_hill_pairwise_wt_1': [],
        'mu_hill_pairwise_wt_2': [],
        'mu_hill_dispar_wt_0': [],
        'mu_hill_dispar_wt_1': [],
        'mu_hill_dispar_wt_2': [],
        'shannon': [],
        'gini_simpson': [],
        'raos_quad': []
    }

# start calculating the mixed uses scores while randomly removing one entry with each iteration
class_code_list = class_codes.copy()
class_dist_list = class_dists.copy()

# capping frivolous computation
max_elements = 5000
print(f'Number of items to be processed: {max_elements} from total set size of {len(class_code_list)}')
while len(class_code_list):

    # take a random int from the length of the list
    rand_int = np.random.randint(0, len(class_code_list))

    # pop the random item from the class code and distances lists
    class_code_list.pop(rand_int)
    class_dist_list.pop(rand_int)

    # capping frivolous computation
    if len(class_code_list) > max_elements:
        continue

    if len(class_code_list) % 100 == 0:
        print(f'List now at {len(class_code_list)}')

    class_code_arr = np.array(class_code_list)
    dist_arr = np.array(class_dist_list)
    classes_unique, classes_counts, classes_nearest = deduce_unique_species(class_code_arr, dist_arr, max_dist=1600)

    # iterate the betas and generate the mixed use metrics
    for k, beta in zip(data_keys, data_betas):
        # run the calculations
        data_5[k]['mu_hill_0'].append(diversity.hill_diversity(classes_counts, 0))
        data_5[k]['mu_hill_1'].append(diversity.hill_diversity(classes_counts, 1))
        data_5[k]['mu_hill_2'].append(diversity.hill_diversity(classes_counts, 2))

        data_5[k]['mu_hill_branch_wt_0'].append(diversity.hill_diversity_branch_distance_wt(classes_counts,
                                                                                            classes_nearest,
                                                                                            0,
                                                                                            beta=beta))
        data_5[k]['mu_hill_branch_wt_1'].append(diversity.hill_diversity_branch_distance_wt(classes_counts,
                                                                                            classes_nearest,
                                                                                            1,
                                                                                            beta=beta))
        data_5[k]['mu_hill_branch_wt_2'].append(diversity.hill_diversity_branch_distance_wt(classes_counts,
                                                                                            classes_nearest,
                                                                                            2,
                                                                                            beta=beta))

        data_5[k]['mu_hill_pairwise_wt_0'].append(diversity.hill_diversity_pairwise_distance_wt(classes_counts,
                                                                                                classes_nearest,
                                                                                                0,
                                                                                                beta=beta))
        data_5[k]['mu_hill_pairwise_wt_1'].append(diversity.hill_diversity_pairwise_distance_wt(classes_counts,
                                                                                                classes_nearest,
                                                                                                1,
                                                                                                beta=beta))
        data_5[k]['mu_hill_pairwise_wt_2'].append(diversity.hill_diversity_pairwise_distance_wt(classes_counts,
                                                                                                classes_nearest,
                                                                                                2,
                                                                                                beta=beta))

        from src.process.metrics.landuses.landuses_poi import disparity_wt_matrix

        wt_matrix = disparity_wt_matrix(classes_unique)

        data_5[k]['mu_hill_dispar_wt_0'].append(diversity.hill_diversity_pairwise_matrix_wt(classes_unique,
                                                                                            wt_matrix,
                                                                                            0))
        data_5[k]['mu_hill_dispar_wt_1'].append(diversity.hill_diversity_pairwise_matrix_wt(classes_unique,
                                                                                            wt_matrix,
                                                                                            1))
        data_5[k]['mu_hill_dispar_wt_2'].append(diversity.hill_diversity_pairwise_matrix_wt(classes_unique,
                                                                                            wt_matrix,
                                                                                            2))

        data_5[k]['shannon'].append(diversity.shannon_diversity(classes_counts))
        data_5[k]['gini_simpson'].append(diversity.gini_simpson_diversity(classes_counts))
        data_5[k]['raos_quad'].append(diversity.raos_quadratic_diversity(classes_unique, wt_matrix))

#  %%
'''
6C
'''

util_funcs.plt_setup()

fig, axes = plt.subplots(2, 1, figsize=(7, 7))

lw_1 = 4
lw_2 = 1
lw_3 = 1
lw_4 = 1
lw_5 = 1.5

a_1 = 0.6
a_2 = 0.8
a_3 = 1

col_1 = 'C1'
col_2 = 'C2'
col_3 = 'C4'

line_1 = '-'
line_2 = '-'
line_3 = '-.'
line_4 = '-'
line_5 = ':'

marker_4 = '.'


def prep_arr(arr, max_x):
    # reverse
    arr = arr[::-1]
    # max
    arr = arr[:max_x]
    # step size
    arr = arr[::int(max_x / 100)]
    return arr


for ax, y_top, x_right in zip(axes, [None, 30], [5000, 500]):
    k = 'A'
    v = data_5['A']

    step_size = x_right / 100

    idx_arr = [step_size * i for i in range(round(x_right / step_size))]

    max_dist = 1600
    beta = data_5[k]['beta']

    ax.plot(idx_arr,
            prep_arr(v['mu_hill_0'], x_right),
            linestyle=line_1,
            lw=lw_1,
            alpha=a_1,
            color=col_1,
            label=r'$H_{0}$')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_branch_wt_0'], x_right),
            linestyle=line_2,
            lw=lw_2,
            alpha=a_2,
            color=col_1,
            label=r'$H_{0}$ b.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_pairwise_wt_0'], x_right),
            linestyle=line_3,
            lw=lw_3,
            alpha=a_2,
            color=col_1,
            label=r'$H_{0}$ p.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_dispar_wt_0'], x_right),
            linestyle=line_4,
            lw=lw_4,
            alpha=a_2,
            marker=marker_4,
            color=col_1,
            label=r'$H_{0}$ d.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_1'], x_right),
            linestyle=line_1,
            lw=lw_1,
            alpha=a_1,
            color=col_2,
            label=r'$H_{1}$')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_branch_wt_1'], x_right),
            linestyle=line_2,
            lw=lw_2,
            alpha=a_2,
            color=col_2,
            label=r'$H_{1}$ b.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_pairwise_wt_1'], x_right),
            linestyle=line_3,
            lw=lw_3,
            alpha=a_2,
            color=col_2,
            label=r'$H_{1}$ p.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_dispar_wt_1'], x_right),
            linestyle=line_4,
            lw=lw_4,
            alpha=a_2,
            marker=marker_4,
            color=col_2,
            label=r'$H_{1}$ d.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_2'], x_right),
            linestyle=line_1,
            lw=lw_1,
            alpha=a_1,
            color=col_3,
            label=r'$H_{2}$')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_branch_wt_2'], x_right),
            linestyle=line_2,
            lw=lw_2,
            alpha=a_2,
            color=col_3,
            label=r'$H_{2}$ b.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_pairwise_wt_2'], x_right),
            linestyle=line_3,
            lw=lw_3,
            alpha=a_2,
            color=col_3,
            label=r'$H_{2}$ p.w.')
    ax.plot(idx_arr,
            prep_arr(v['mu_hill_dispar_wt_2'], x_right),
            linestyle=line_4,
            lw=lw_4,
            alpha=a_2,
            marker=marker_4,
            color=col_3,
            label=r'$H_{2}$ d.w.')
    ax.plot(idx_arr,
            prep_arr(v['shannon'], x_right),
            linestyle='-',
            lw=1,
            alpha=a_2,
            color='C6',
            marker='.',
            label='Shannon')
    ax.plot(idx_arr,
            prep_arr(v['gini_simpson'], x_right),
            linestyle=':',
            lw=1.5,
            alpha=a_3,
            color='C5',
            label='Gini')
    ax.plot(idx_arr,
            prep_arr(v['raos_quad'], x_right),
            linestyle='-',
            lw=1,
            alpha=a_2,
            color='C3',
            marker='x',
            label='Rao')
    ax.set_xlim(left=0,
                right=x_right)
    fig.suptitle('Mixed use calculations for a 1600m radius from London Leicester Square')
    ax.set_xlabel('Computed diversity for sets from 1 to ' + str(x_right) + r' classification codes.')
    ax.set_ylim(bottom=0,
                top=y_top)
    ax.set_ylabel('diversity / effective number of uses')
    ax.legend(loc='upper right')

path = f'../phd-doc/doc/images/diversity/mixed_uses_random_removal.pdf'
plt.savefig(path)
