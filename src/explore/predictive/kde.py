# %%
'''
KDE LOCAL metrics plotted at 200, 400, 800, 1600m for all towns and cities
'''
import logging


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import phd_util
from src.explore.theme_setup import data_path
from src.explore.theme_setup import generate_theme
from scipy import stats



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% load from disk
df_full = pd.read_feather(data_path / 'df_full_all.feather')
df_full = df_full.set_index('id')
X_raw, distances, labels = generate_theme(df_full, 'all_towns', bandwise=False, city_pop_id=True)


# %%
def kde_gen(_data, _col_name, _pop_id, _y_chunks):
    # select only data from city id
    # first check that column exists
    if not _col_name in _data:
        logger.info(f'column: {_col_name} does not exist?')
        return None, None
    selection = _data[_data.city_pop_id == _pop_id][_col_name].values
    # remove NaN and inf values
    selection = selection[np.isfinite(selection)]
    # some arrays will be empty or minimal, catch:
    if len(selection) < 10:
        logger.info(f'skipping small array size: {len(selection)} for city: {_pop_id} column: {_col_name}')
        return None, None
    # some arrays contain only zeros, catch:
    if selection.max() == 0:
        logger.info(f'skipping array of zeros for city: {_pop_id} column: {_col_name}')
        return None, None
    # kde
    density = stats.gaussian_kde(selection)
    # plot
    step_size = selection.max() / _y_chunks
    # prep x data
    x = np.arange(0, selection.max(), step_size)
    # deduce y data
    y = density.evaluate(x)
    return x, y


def compound_kde(df,
                 column_name,
                 city_pop_range,
                 ax,
                 y_scale,
                 y_chunks=200):
    data = df.copy(deep=True)

    # setup colormap if not set explicitly
    cmap = phd_util.cityseer_cmap()

    # plot the data for each city
    ylim_max = []
    xlim_max = []
    for city_pop_id in reversed(city_pop_range):
        # scale derived from city_pop_id
        city_scale = city_pop_id ** -0.1  # transforms 1-1000 to 1-0.5

        x, y = kde_gen(data, column_name, city_pop_id, y_chunks)
        if x is None:
            continue

        # scale
        y *= y_scale
        # offset
        y += city_pop_id
        # close the polygon
        x_l = list(x)
        x_l += [x.max(), x.min()]
        y_l = list(y)
        y_l += [y.min(), y.min()]
        # 0 - 1  # be aware that cmaps treat floats (0-1.0) differently to ints (1-255)
        c = cmap(city_scale * 2 - 1)
        ax.fill(x_l,
                y_l,
                c=c,
                lw=city_scale * 1.5 - 0.5,
                alpha=city_scale - 0.25)
        if city_pop_id == 1 or city_pop_id % 5 == 0:
            ax.plot(x,
                    y,
                    c='w',
                    lw=city_scale * 1.5 - 0.5,
                    alpha=city_scale)
        # check plot limit maxes
        ylim_max.append(y.max())
        xlim_max.append(x.max())

    return xlim_max, ylim_max


def kde_plot(data, theme, label, x_max=100, y_chunks=500):
    cmap = phd_util.cityseer_cmap()
    city_pop_range = range(1, int(data.city_pop_id.max() + 1))

    # figure out the y scale
    max_size = 200
    min_size = 20
    y_maxes = []
    for dist in ['200', '400', '800', '1600']:
        col_key = theme.format(dist=dist)
        x, y = kde_gen(data, col_key, 1, y_chunks)
        y_maxes.append(y.max())
    y_maxes = np.array(y_maxes)

    min_scale = max_size / y_maxes.max()
    # check that the smallest axes is larger than min, if not, interpolate to new range
    if min_scale * y_maxes.min() >= min_size:
        y_scales = np.full(4, min_scale)
    else:
        # the smallest y_max needs to be scaled by a larger amount
        y_scales = []
        for y_max in y_maxes:
            y_frac = (y_max - y_maxes.min()) / (y_maxes.max() - y_maxes.min())
            target = ((max_size - min_size) * y_frac) + min_size
            y_scales.append(target / y_max)

    phd_util.plt_setup()
    fig, axes = plt.subplots(1, 4, figsize=(12, 10))
    for n, (dist, y_scale) in enumerate(zip(['200', '400', '800', '1600'], y_scales)):
        col_key = theme.format(dist=dist)
        axes[n].set_xlabel(label + r' $d_{max}=' + f'{dist}m$')

        xlims, ylims = compound_kde(data,
                                    col_key,
                                    city_pop_range,
                                    axes[n],
                                    y_scale,
                                    y_chunks=y_chunks)
        axes[n].get_yaxis().set_visible(False)
        # x_max is per ax
        right_most = np.nanpercentile(xlims, x_max)
        axes[n].set_xlim(0, right_most)
        axes[n].set_ylim(0, 1100)

        # global and london mean values
        global_median = round(np.nanmedian(data[col_key]), 3)
        london_median = round(np.nanmedian(data[data.city_pop_id == 1][col_key]), 3)
        axes[n].axvline(global_median,
                        color=cmap(0.5),
                        linestyle='--',
                        alpha=1,
                        linewidth=0.7,
                        label=r'global $\widetilde{x}=' + str(global_median) + '$')
        axes[n].axvline(london_median,
                        color=cmap(1.0),
                        linestyle='--',
                        alpha=1,
                        linewidth=0.7,
                        label=r'london $\widetilde{x}=' + str(london_median) + '$')
        axes[n].legend(loc=1, prop={'size': 6})

        # add labels
        for city_id, city_label in zip([1, 51, 164], ['London', 'Cambridge', 'Letchworth']):
            c = cmap(city_id ** -0.1 * 2 - 1)
            axes[n].text(right_most,
                         city_id,
                         city_label,
                         horizontalalignment='right',
                         fontdict={'size': 6},
                         color=c)

    path = f'../../phd-admin/PhD/part_3/images/predicted/kde/{theme.strip("_{dist}")}.png'
    plt.savefig(path, dpi=300)


# %%
# census aggregation plot compound KDE
kde_plot(X_raw, 'cens_tot_pop_{dist}', 'KDE Population', x_max=99)
# kde_plot(X_raw, 'cens_dwellings_{dist}', 'KDE Dwellings', x_max=99)

# centrality plot compound KDE
kde_plot(X_raw, 'c_node_harmonic_angular_{dist}', 'Node Harmonic Angular', x_max=99)
# kde_plot(X_raw, 'c_node_betweenness_beta_{dist}', r'Node Betweenness $\beta$', x_max=99)

# landuses compound KDE
# kde_plot(X_raw, 'ac_eating_{dist}', 'Eat & Drink', x_max=99)
# kde_plot(X_raw, 'ac_commercial_{dist}', 'Commercial', x_max=99)
# kde_plot(X_raw, 'ac_retail_food_{dist}', 'Retail - Food', x_max=99)
# kde_plot(X_raw, 'ac_retail_other_{dist}', 'Retail - Other', x_max=99)
# kde_plot(X_raw, 'ac_manufacturing_{dist}', 'Manuf.', x_max=99)
# kde_plot(X_raw, 'ac_transport_{dist}', 'Transport', x_max=99)
# kde_plot(X_raw, 'ac_education_{dist}', 'Education', x_max=99)

# mixeduses compound KDE
kde_plot(X_raw, 'mu_hill_branch_wt_0_{dist}', 'Hill wt. $q=0$', x_max=99)
