'''
These were tricky plots showing changes in landuses data over time
Ultimately didn't proceed with timeseries and the plots showed various inconsistencies in the data
Presumably due to changes in third party data sources, i.e. additions and deletions to POI databases
'''

#%%
from importlib import reload
import time
import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import phd_util
reload(phd_util)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# db connection params
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

# load boundaries data
bound_data = phd_util.load_data_as_pd_df(
    db_config, [
        'pop_id',
        'city_name',
        'city_type',
        'city_area',
        'city_area_petite',
        'city_population',
        'city_species_count',
        'city_species_unique',
        'city_streets_len',
        'city_intersections_count'
    ],
    'analysis.city_boundaries_150',
    'WHERE pop_id IS NOT NULL ORDER BY pop_id')

epochs = [
    #'2006-09-01',
    ## '2006-12-01' - missing
    #'2007-03-01',
    #'2007-06-01',
    #'2007-09-01',
    #'2007-12-01',
    #'2008-03-01',
    ## '2008-06-01' - missing
    ## '2008-09-01' - missing
    ## '2008-12-01' - missing
    #'2009-03-01',
    #'2009-06-01',
    #'2009-09-01',
    ## '2009-12-01' - missing
    ## '2010-03-01' - missing
    ## '2010-06-01' - missing
    #'2010-09-01',
    #'2010-12-01',
    #'2011-03-01',
    '2011-06-01',
    '2011-09-01',
    '2011-12-01',
    '2012-03-01',
    '2012-06-01',
    '2012-09-01',
    '2012-12-01',
    '2013-03-01',
    '2013-06-01',
    '2013-09-01',
    # '2013-12-01' - missing
    # '2014-03-01' - missing
    '2014-06-01',
    '2014-09-01',
    '2014-12-01',
    '2015-03-01',
    '2015-06-01',
    '2015-09-01',
    '2015-12-01',
    '2016-03-01',
    '2016-06-01',
    '2016-09-01',
    '2016-12-01',
    '2017-03-01',
    '2017-06-01',
    '2017-09-01',
    '2017-12-01',
    '2018-03-01',
    '2018-06-01',
    '2018-09-01']

dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in epochs]

themes = ['mixed_uses',
          'accommodation',
          'eating',
          'drinking',
          'commercial',
          'tourism',
          'entertainment',
          'government',
          'manufacturing',
          'retail_food',
          'retail_other',
          'transport',
          'health',
          'education',
          'parks',
          'cultural',
          'sports',
          'total']

labels = ['Hill wt. $q=0$',
          'Accommodation',
          'Eat',
          'Drink',
          'Commercial',
          'Tourism',
          'Entertainment',
          'Government',
          'Manuf.',
          'Retail Food',
          'Retail Other',
          'Transport',
          'Health',
          'Education',
          'Parks',
          'Cultural',
          'Recreation',
          'Total'
          ]


#%%
def reduce_epochs(data_store, pop_id, dist, theme):
    '''
    fetches data for chosen theme column at specified distance
    then iterates epochs to deduce mean values per epoch
    '''
    epochs_mu = np.full(28, np.nan)

    # multi index: (uid, dist, epoch, city_pop_id)
    # storer = data_store.get_storer('df').levels
    # labelled as: ['level_0', 'level_1', 'level_2', 'level_3']
    # for performance: filter by city_pop_id first, then filter by distance
    data = data_store.select('df', where=[f'level_3 = {pop_id}', f'level_1 = {dist}'], columns=[theme])  # , f'level_2 = "{epoch}"'
    data.index.set_names(['uid', 'dist', 'epoch', 'city_pop_id'], inplace=True)
    data = data[theme]

    if len(data) / len(epochs) < 10:
        return None

    for idx, epoch in enumerate(epochs):
        try:
            sel = data.xs(epoch, level='epoch')
            sel[(sel < np.nanquantile(sel, q=0.01)) | (sel > np.nanquantile(sel, q=0.99))] = np.nan
            epochs_mu[idx] = np.nanmean(sel)
        except KeyError:
            logger.warning(f'Missing any epoch data for {epoch} on city {pop_id}')
            epochs_mu[idx] = 0

    return epochs_mu


def time_plot(data_store, theme, label, decomp, path):

    city_pop_range = range(1, 959)

    # figure out the y scale
    logger.info('Setting y scales')
    max_size = 50
    min_size = 20
    y_maxes = []
    for dist in ['200', '400', '800', '1600']:
        epochs_mu = reduce_epochs(data_store, city_pop_range[0], dist, theme)
        # get max and min for determining scale
        y_maxes.append(np.nanmax(epochs_mu) - np.nanmin(epochs_mu))

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

    logger.info('Starting all cities plot')
    phd_util.plt_setup()
    fig, axes = plt.subplots(1, 4, figsize=(10, 8))
    for n, (dist, y_scale) in enumerate(zip(['200', '400', '800', '1600'], y_scales)):

        axes[n].set_xlabel(f'{label} ' + r'$\mu$ $d_{max}=' + f'{dist}m$')

        y_maxes = []
        y_mins = []
        for city_pop_id in reversed(city_pop_range):

            if city_pop_id % 10 == 0:
                logger.info(f'Processing scale: {dist}, city: {city_pop_id}')

            city_scale = city_pop_id ** -0.1  # transforms 1-1000 to 1-0.5

            epochs_mu = reduce_epochs(data_store, city_pop_id, dist, theme)

            if epochs_mu is None:
                logger.info(f'No or minimal data for city {city_pop_id}, skipping...')
                continue

            epochs_scaled = epochs_mu * y_scale
            # enforce max size
            if np.nanmax(epochs_scaled) - np.nanmin(epochs_scaled) > max_size:
                ht = np.nanmax(epochs_scaled) - np.nanmin(epochs_scaled)
                sc = max_size / ht
                epochs_scaled = sc * epochs_scaled

            # offset so that first element starts at city_pop_id
            offset = city_pop_id - epochs_scaled[0]
            epochs_scaled += offset

            # generate colours and sizes - first item set to neutral
            cmap = LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#dddddd', '#d32f2f'])

            epochs_col = [cmap(0.5)]
            epochs_size = [0.3]
            e_range = np.nanmax(epochs_mu) - np.nanmin(epochs_mu)
            for idx in range(1, 28):
                # change
                s_ch = abs(epochs_mu[idx] - epochs_mu[idx - 1]) / e_range
                # the greater the change, the greater the size
                epochs_size.append(0.3 + s_ch * 5)
                # colour is sensitive to + or -
                col_ch = (epochs_mu[idx] - epochs_mu[idx - 1]) / e_range
                col_ch *= 2
                if col_ch > 0.5:
                    col_ch = 0.5
                elif col_ch < -0.5:
                    col_ch = -0.5
                epochs_col.append(cmap(0.5 + col_ch))

            y_maxes.append(np.nanmax(epochs_scaled))
            y_mins.append(np.nanmin(epochs_scaled))

            #for idx in range(1, len(epochs_scaled)):

            axes[n].scatter(dates,
                            epochs_scaled,
                            s=epochs_size,
                            c=epochs_col,
                            edgecolors='none',  # doesn't scale size
                            alpha=1,
                            zorder=3)

            # if overall increase, set colour to red, else blue
            if epochs_scaled[-1] == epochs_scaled[0]:
                c = 'silver'
            elif epochs_scaled[-1] > epochs_scaled[0]:
                c = cmap(1.0)
            else:
                c = cmap(0.0)

            # bold new town lines
            lw = city_scale * 0.2 + 0.2
            a = city_scale * 0.5
            if bound_data[bound_data.pop_id == city_pop_id]['city_type'].values[0] == 'New Town':
                lw = 0.75
                a = 0.75
            elif city_pop_id == 1:
                lw = 0.75
                a = 0.75
                c = '#333333'

            axes[n].plot(dates,
                     epochs_scaled,
                     color=c,
                     lw=lw,
                     alpha=a,
                     zorder=2)

            # plot min and max for first city
            if city_pop_id == city_pop_range[0]:
                mi = np.nanmin(epochs_scaled)
                ma = np.nanmax(epochs_scaled)
                mi_t = np.nanmin(epochs_mu).round(3)
                ma_t = np.nanmax(epochs_mu).round(3)
                axes[n].hlines([mi, ma],
                               xmin=dates[0],
                               xmax=dates[-1],
                               color='#333333',
                               lw=0.5,
                               alpha=0.5,
                               zorder=3,
                               linestyle='--')
                # make up date for text spacing
                axes[n].text('2011-07-01', mi - 1, 'london', fontdict={'size': 4}, color='#333333',
                             zorder=4, verticalalignment='top', horizontalalignment='left')
                axes[n].text('2018-08-01', mi - 1, mi_t, fontdict={'size': 4}, color='#333333',
                             zorder=4, verticalalignment='top', horizontalalignment='right')
                axes[n].text('2018-08-01', ma + 1, ma_t, fontdict={'size': 4}, color='#333333',
                             zorder=4, horizontalalignment='right')

        # for axes extents
        y_axes_cushion = (np.nanmax(y_maxes) - np.nanmin(y_mins)) * 0.01
        upper_y_extent = np.nanmax(y_maxes) + y_axes_cushion
        lower_y_extent = np.nanmin(y_mins) - y_axes_cushion

        axes[n].get_yaxis().set_visible(False)
        axes[n].spines['left'].set_visible(False)
        axes[n].grid(False)
        axes[n].set_xlim(left=dates[0], right=dates[-1])
        axes[n].set_ylim(bottom=lower_y_extent, top=upper_y_extent)

    if path:
        plt.savefig(path)
    else:
        plt.show()


#%%

# 200m @ full = 21m
# 200m @ 100m = 32m

decomp = 20

for theme, label in zip(themes, labels):

    print(f'Processing theme: {theme}')

    file_path = f'/Volumes/gareth 07783845654/df_epochs_{decomp}_dup.nc'

    with pd.HDFStore(file_path) as data_store:

        data_store.flush(fsync=True)

        start_time = time.time()

        path = f'/Users/gareth/dev/gitlab/songololo-phd/admin/PhD/2/images/time_series_{decomp}/{theme}_trim.png'
        time_plot(data_store, theme, label, decomp, path)

        logger.info(f'Duration: {round((time.time() - start_time) / 60, 2)}m')

#data_coords = data_store.select_as_coordinates('df', ['level_1 = 200', 'level_3=1'])
#data = data_store.select('df', where=data_coords, columns=['mixed_uses'])  #21s
#data_coords = data_store.select_as_coordinates('df', ['level_1 = 200', 'level_3=1'])
#data = data_store.select('df', where=data_coords)  #20s
#data = data_store.select('df', where=['level_1 = 200', 'level_3=1'], columns=['mixed_uses'])  # 20s
#data = data_store.select('df', where=['level_1 = 200', 'level_3=1'])  # 12s
# selecting where all greater than 0 is quite slow...
# seems to get faster as data gets smaller
# for some reason, selecting by column is slower...?
