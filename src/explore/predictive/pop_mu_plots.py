# %%
'''
Average value plots for e.g. centralities / land-uses by town size
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


#  %%
def pop_plot(city_data, city_theme, city_label, bound_data, bound_label):
    new_towns = []
    other_towns = []
    for i, d in bound_data.iterrows():
        if d['city_type'] in ['New Town']:
            new_towns.append(d['pop_id'])
        else:
            other_towns.append(d['pop_id'])

    util_funcs.plt_setup()
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))

    max_pop_id = city_data.city_pop_id.max()
    for n, dist in enumerate(['200', '1600']):  # , '400', '800',

        city_key = city_theme.format(dist=dist)
        axes[n].set_ylabel(city_label + ' $d_{max}=' + f'{dist}m$')
        axes[n].set_xlabel(bound_label)

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
            # returns a dataframe (pandas doesn't know that this is a single row) so index from values
            x_d = bound_data[bound_data.pop_id == pop_id]['city_population'].values[0]
            t_n = bound_data[bound_data.pop_id == pop_id]['city_name'].values[0]
            y_d = city_data[city_data.city_pop_id == pop_id][city_key]
            if len(y_d):
                size = ((1 - pop_id / max_pop_id) * 20 + 5)
                if pop_id in new_towns:
                    nt_x.append(x_d)
                    nt_y.append(np.nanmean(y_d))
                    nt_s.append(size)
                    nt_id.append(pop_id)
                    nt_n.append(t_n)
                else:
                    x.append(x_d)
                    y.append(np.nanmean(y_d))
                    s.append(size)
                    o_id.append(pop_id)
                    o_n.append(t_n)

        # filter other towns to same population range
        poly_min_x = np.nanmin(nt_x)
        poly_max_x = np.nanmax(nt_x)
        other_towns_filtered = []
        for o_t in other_towns:
            # returns a dataframe (pandas doesn't know that this is a single row) so index from values
            o_t_num = bound_data[bound_data.pop_id == o_t]['city_population'].values[0]
            if o_t_num >= poly_min_x and o_t_num <= poly_max_x:
                other_towns_filtered.append(o_t)

        # get averages - don't take average of average, but compute directly to avoid ecological correlation
        nt_mu = np.nanmean(city_data[city_data['city_pop_id'].isin(new_towns)][city_key])
        other_mu = np.nanmean(
            city_data[city_data['city_pop_id'].isin(other_towns_filtered)][city_key])

        nt_col = '#d32f2f'
        o_col = '#0064b7'

        # plot
        axes[n].scatter(nt_x, nt_y, c=nt_col, s=nt_s, alpha=0.7, marker='o',
                        edgecolors='white', linewidths=0.3, zorder=3)
        axes[n].scatter(x, y, c=o_col, s=s, alpha=0.4, marker='o',
                        edgecolors='white', linewidths=0.3, zorder=2)

        # add lines
        axes[n].hlines(nt_mu, xmin=poly_min_x, xmax=poly_max_x, colors=nt_col,
                       lw=2, alpha=0.5, linestyle='-', zorder=4,
                       label=f'new towns $\mu={round(nt_mu, 3)}$')
        axes[n].hlines(other_mu, xmin=poly_min_x, xmax=poly_max_x, colors=o_col,
                       lw=2, alpha=0.5, linestyle='-', zorder=4,
                       label=f'equiv. towns $\mu={round(other_mu, 3)}$')

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

    path = f'../phd-admin/PhD/part_3/images/predicted/mus/{city_theme.strip("_{dist}")}.pdf'
    plt.savefig(path, dpi=300)


#  %%
# census to population plots
pop_plot(X_raw,
         'cens_tot_pop_{dist}',
         'Population $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'c_node_harmonic_angular_{dist}',
         'Closeness $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'mu_hill_branch_wt_0_{dist}',
         'Hill wt. $q=0$ $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_eating_{dist}',
         'Eat & Drink $\mu$',
         bound_data,
         'Global city population')
'''
#  %%
pop_plot(X_raw,
         'cens_dwellings_{dist}',
         'Dwellings $\mu$',
         bound_data,
         'Global city population')

# centrality

pop_plot(X_raw,
         'c_node_betweenness_beta_{dist}',
         r'Betweenness $\mu$',
         bound_data,
         'Global city population')

# landuses
pop_plot(X_raw,
         'ac_eating_{dist}',
         'Eat & Drink $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_commercial_{dist}',
         'Commercial $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_retail_food_{dist}',
         'Retail - Food $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_retail_other_{dist}',
         'Retail - Other $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_manufacturing_{dist}',
         'Manuf. $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_transport_{dist}',
         'Transport $\mu$',
         bound_data,
         'Global city population')
pop_plot(X_raw,
         'ac_education_{dist}',
         'Education $\mu$',
         bound_data,
         'Global city population')

'''
