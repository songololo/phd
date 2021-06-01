# %%
'''
Correlation plots for e.g. centrality and mixed-uses by town size
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import util_funcs
from src.explore.theme_setup import data_path
from src.explore.theme_setup import generate_theme

#  %% load from disk
df_full = pd.read_feather(data_path / 'df_full_all.feather')
df_full = df_full.set_index('id')
X_raw, distances, labels = generate_theme(df_full,
                                          'all_towns',
                                          bandwise=False,
                                          add_city_pop_id=True)

#  %%
# db connection params
db_config = {
    'host': 'localhost',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

# load boundaries data
bound_data = util_funcs.load_data_as_pd_df(db_config,
                                           ['pop_id',
                                          'city_name',
                                          'city_type',
                                          'city_area',
                                          'city_area_petite',
                                          'city_population',
                                          'city_species_count',
                                          'city_species_unique',
                                          'city_streets_len',
                                          'city_intersections_count'],
                                           'analysis.city_boundaries_150',
                                           'WHERE pop_id IS NOT NULL ORDER BY pop_id')


#%%
def pop_corr_plot(city_data, theme_1, theme_2, towns_data, xlabel, sup_title):
    new_towns = []
    other_towns = []
    for i, d in bound_data.iterrows():
        if d['city_type'] in ['New Town']:
            new_towns.append(d['pop_id'])
        else:
            other_towns.append(d['pop_id'])

    util_funcs.plt_setup()
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))

    max_pop_id = city_data.city_pop_id.max()
    for n, dist in enumerate(['200', '1600']):  # , '400', '800',

        key_1 = theme_1.format(dist=dist)
        key_2 = theme_2.format(dist=dist)

        axes[n].set_ylabel(r'spearman $\rho$' + r' $d_{max}=' + f'{dist}m$')
        axes[n].set_xlabel(xlabel)

        x = []
        y = []
        s = []
        o_id = []
        o_n = []
        nt_x = []
        nt_y = []
        nt_s = []
        nt_id = []
        nt_n = []
        for pop_id in reversed(range(1, int(max_pop_id) + 1)):
            pop = towns_data[towns_data.pop_id == pop_id]['city_population'].values[0]
            t_n = towns_data[towns_data.pop_id == pop_id]['city_name'].values[0]
            d_1 = city_data[city_data.city_pop_id == pop_id][key_1]
            d_2 = city_data[city_data.city_pop_id == pop_id][key_2]
            if len(d_1):
                # d_1, d_2 = phd_util.prep_xy(d_1, d_2)
                p_r, p = stats.spearmanr(d_1, d_2)
                size = ((1 - pop_id / max_pop_id) * 20 + 5)
                if pop_id in new_towns:
                    nt_x.append(pop)
                    nt_y.append(p_r)
                    nt_s.append(size)
                    nt_id.append(pop_id)
                    nt_n.append(t_n)
                else:
                    x.append(pop)
                    y.append(p_r)
                    s.append(size)
                    o_id.append(pop_id)
                    o_n.append(t_n)

        # filter other towns to same population range
        poly_min_x = np.nanmin(nt_x)
        poly_max_x = np.nanmax(nt_x)
        other_towns_filtered = []
        for o_t in other_towns:
            # returns a dataframe (pandas doesn't know that this is a single row) so index from values
            o_t_num = towns_data[towns_data.pop_id == o_t]['city_population'].values[0]
            if o_t_num >= poly_min_x and o_t_num <= poly_max_x:
                other_towns_filtered.append(o_t)

        # get averages - don't take average of average, but compute directly to avoid ecological correlation
        nt_d_1 = city_data[city_data['city_pop_id'].isin(new_towns)][key_1]
        nt_d_2 = city_data[city_data['city_pop_id'].isin(new_towns)][key_2]
        # nt_d_1, nt_d_2 = phd_util.prep_xy(nt_d_1, nt_d_2)
        nt_corr, _p = stats.spearmanr(nt_d_1, nt_d_2)
        other_d_1 = city_data[city_data['city_pop_id'].isin(other_towns_filtered)][key_1]
        other_d_2 = city_data[city_data['city_pop_id'].isin(other_towns_filtered)][key_2]
        # other_d_1, other_d_2 = phd_util.prep_xy(other_d_1, other_d_2)
        other_corr, _p = stats.spearmanr(other_d_1, other_d_2)

        nt_col = '#d32f2f'
        o_col = '#0064b7'

        # plot
        axes[n].scatter(nt_x, nt_y, c=nt_col, s=nt_s, alpha=0.7, marker='o',
                        edgecolors='white', linewidths=0.3, zorder=3)
        axes[n].scatter(x, y, c=o_col, s=s, alpha=0.4, marker='o',
                        edgecolors='white', linewidths=0.3, zorder=2)

        # add lines
        axes[n].hlines(nt_corr, xmin=poly_min_x, xmax=poly_max_x, colors=nt_col,
                       lw=2, alpha=0.5, linestyle='-', zorder=4,
                       label=f'new towns $r={round(nt_corr, 3)}$')
        axes[n].hlines(other_corr, xmin=poly_min_x, xmax=poly_max_x, colors=o_col,
                       lw=2, alpha=0.5, linestyle='-', zorder=4,
                       label=f'equiv. towns $r={round(other_corr, 3)}$')

        y_axes_cushion = (np.nanmax(y) - np.nanmin(y)) * 0.1
        y_text_cushion = (np.nanmax(y) - np.nanmin(y)) * 0.085
        # for axes extents
        upper_y_extent = np.nanmax(y) + y_axes_cushion
        lower_y_extent = np.nanmin(y) - y_axes_cushion
        # for text and lines
        upper_y_end = np.nanmax(y) + y_text_cushion
        lower_y_end = np.nanmin(y) - y_text_cushion

        # background polygon
        axes[n].fill([poly_min_x, poly_min_x, poly_max_x, poly_max_x],
                     [lower_y_extent, upper_y_extent, upper_y_extent, lower_y_extent],
                     c='grey', lw=0, alpha=0.1, zorder=1)

        # new towns
        for t_x, t_y, t_id, t_n in zip(nt_x, nt_y, nt_id, nt_n):
            # to avoid overlap, plot certain from top and others from bottom
            # top
            if t_id in [29, 39, 63, 80, 126, 153, 194, 244]:
                align = 'top'
                y_end = upper_y_end
            # bottom
            else:
                align = 'bottom'
                y_end = lower_y_end
            axes[n].text(t_x * 1.02, y_end, t_n, rotation=90, verticalalignment=align,
                         fontdict={'size': 5}, color='#D3A1A6')
            axes[n].vlines(t_x, ymin=t_y, ymax=y_end, color='#D3A1A6', lw=0.5, alpha=0.4)

        # other towns
        for t_x, t_y, t_id, t_n in zip(x, y, o_id, o_n):
            # to avoid overlap, plot certain from top and others from bottom
            # top
            if t_id in [3, 6, 8, 10, 12]:
                align = 'top'
                y_end = upper_y_end
            # bottom
            elif t_id in [1, 2, 4, 5, 7, 9, 11, 13]:
                align = 'bottom'
                y_end = lower_y_end
            else:
                continue
            axes[n].text(t_x * 1.02, y_end, t_n, rotation=90, verticalalignment=align,
                         fontdict={'size': 5}, color='silver')
            axes[n].vlines(t_x, ymin=t_y, ymax=y_end, color='silver', lw=0.5, alpha=0.4)

        axes[n].set_xlim(left=5000, right=10 ** 7)
        axes[n].set_ylim(bottom=lower_y_extent, top=upper_y_extent)
        axes[n].set_xscale('log')
        axes[n].legend(loc=2)
    fig.suptitle(sup_title)
    path = f'../phd-doc/doc/part_3/predictive/images/corr/{theme_1.strip("_{dist}")}_{theme_2.strip("_{dist}")}.pdf'
    plt.savefig(path, dpi=300)


#  %%
bound_text = 'City population by town / city boundary'
pop_corr_plot(X_raw,
              'cens_tot_pop_{dist}',
              'c_node_harmonic_angular_{dist}',
              bound_data,
              bound_text,
              'Population correlated to closeness centrality')

pop_corr_plot(X_raw,
              'cens_tot_pop_{dist}',
              'mu_hill_branch_wt_0_{dist}',
              bound_data,
              bound_text,
              'Population correlated to mixed-uses')

pop_corr_plot(X_raw,
              'c_node_harmonic_angular_{dist}',
              'mu_hill_branch_wt_0_{dist}',
              bound_data,
              bound_text,
              'Closeness centrality correlated to mixed-uses')
