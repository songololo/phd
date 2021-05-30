import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# data_path = Path('/Users/gareth/Documents/data/')
data_path = Path('/Users/gareth/databases/phd_data/')

# plot_path = Path('../../phd-doc/doc/part_3/images/signatures')
plot_path = Path('./src/temp_images/')

logs_path = Path('./src/temp_logs/')
weights_path = Path('./src/temp_weights/')

# use of bands gives slightly more defined delineations for latent dimensions
columns_cent = [
    # 'c_node_density_{dist}',
    # 'c_node_cycles_{dist}',
    # 'c_node_harmonic_{dist}',
    # 'c_node_beta_{dist}',
    'c_segment_density_{dist}',
    # 'c_segment_harmonic_{dist}',
    'c_segment_beta_{dist}',
    # 'c_node_harmonic_angular_{dist}',
    'c_segment_harmonic_hybrid_{dist}',
    # 'c_node_betweenness_{dist}',
    # 'c_node_betweenness_beta_{dist}',
    'c_segment_betweenness_{dist}',
    # 'c_node_betweenness_angular_{dist}',
    'c_segment_betweeness_hybrid_{dist}']
labels_cent = [
    # 'node density',
    # 'node cycles',
    # 'node harmonic',
    # 'node beta',
    'seg. density',
    # 'segment harmonic',
    r'segment $\beta$',
    # 'node harm. ang.',
    'seg. harm. hyb.',
    # 'node betw.',
    # r'node betw. $\beta$',
    'segment betw.',
    # 'seg. betw. ang.',
    'seg betw. hyb.'
]

columns_lu = [
    'mu_hill_branch_wt_0_{dist}',
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
    'ac_sports_{dist}',
    'ac_total_{dist}'
]
labels_lu = [
    'mixed_uses',
    'accomod.',
    'eating',
    'drinking',
    'commerc.',
    'tourism',
    'entert.',
    'govern.',
    'manuf.',
    'retail food',
    'retail other',
    'transp.',
    'health',
    'educat.',
    'parks',
    'culture',
    'sports',
    'total'
]

columns_cens = [
    'cens_tot_pop_{dist}',
    # 'cens_adults_{dist}',
    'cens_employed_{dist}',
    'cens_dwellings_{dist}',
    'cens_students_{dist}'
]
labels_cens = [
    'total pop.',
    # 'adults pop.',
    'employed',
    'dwellings',
    'students'
]

columns_select = [
    'c_segment_beta_{dist}',
    'c_segment_betweeness_hybrid_{dist}',
    'mu_hill_branch_wt_0_{dist}',
    'cens_tot_pop_{dist}'
]
labels_select = [
    r'segment $\beta$',
    'seg betw. hyb.',
    'mixed_uses',
    'total pop.'
]

columns_all_towns = [
    'c_node_harmonic_angular_{dist}',
    'c_node_betweenness_beta_{dist}',
    'mu_hill_branch_wt_0_{dist}',
    'ac_eating_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_total_{dist}',
    'cens_tot_pop_{dist}',
    'cens_dwellings_{dist}'
]
labels_all_towns = [
    'h. closeness',
    r'\beta betweenness',
    'mixed_uses',
    'eating',
    'drinking',
    'commerc.',
    'manuf.',
    'retail food',
    'retail other',
    'transp.',
    'total',
    'total pop.',
    'dwellings'
]

columns_pred_mixed = [
    'c_segment_density_{dist}',
    'c_segment_beta_{dist}',
    'c_segment_harmonic_hybrid_{dist}',
    'c_segment_betweenness_{dist}',
    'c_segment_betweeness_hybrid_{dist}',
    'cens_tot_pop_{dist}',
    'cens_employed_{dist}',
    'cens_dwellings_{dist}',
    'cens_students_{dist}'
    # 'ac_transport_{dist}'
]

labels_pred_mixed = [
    'seg. density',
    r'segment $\beta$',
    'seg. harm. hyb.',
    'segment betw.',
    'seg betw. hyb.',
    'total pop.',
    'employed',
    'dwellings',
    'students'
    # 'transp.'
]

columns_pred_sim = [
    'c_node_harmonic_angular_{dist}',
    'c_node_betweenness_beta_{dist}',
    'ac_commercial_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    'cens_dwellings_{dist}'
]

labels_pred_sim = [
    'node harm. ang.',
    r'node betw. $\beta$',
    'commerc.',
    'manuf.',
    'retail food',
    'dwellings'
]


template_distances = (100, 200, 300, 400, 600, 800, 1200, 1600)

def generate_theme(df,
                   theme,
                   bandwise=False,
                   add_city_pop_id=False,
                   max_dist=None):
    df_copy = df.copy(deep=True)
    if max_dist is None:
        distances = [d for d in template_distances]
    else:
        distances = [d for d in template_distances if d <= max_dist]

    if theme == 'all':
        columns = columns_cent + columns_lu + columns_cens
        labels = labels_cent + labels_lu + labels_cens
    elif theme == 'cent':
        columns = columns_cent
        labels = labels_cent
    elif theme == 'lu':
        columns = columns_lu
        labels = labels_lu
    elif theme == 'cens':
        columns = columns_cens
        labels = labels_cens
    elif theme == 'select':
        columns = columns_select
        labels = labels_select
    elif theme == 'all_towns':
        columns = columns_all_towns
        labels = labels_all_towns
    elif theme == 'pred_lu':
        columns = columns_pred_mixed
        labels = labels_pred_mixed
    elif theme == 'pred_sim':
        columns = columns_pred_sim
        labels = labels_pred_sim
    else:
        raise ValueError('Invalid theme specified for data theme.')

    if add_city_pop_id:
        # unpack the columns by distances and fetch the data
        # first add generic (non-distance) columns
        labels = ['City Population ID'] + labels
        selected_columns = [
            'city_pop_id'
        ]
    else:
        selected_columns = []
    # unpack the columns by distances and fetch the data
    for column in columns:
        for d in distances:
            selected_columns.append(column.format(dist=d))
    # if not bandwise, simply return distance based columns as they are
    if not bandwise:
        X = df_copy[selected_columns]
    # but if bandwise, first subtract foregoing distances
    else:
        print('Generating bandwise')
        for column in columns:
            for d in distances:
                print(f'Current distance leading edge: {d}m')
                d_idx = distances.index(d)
                if d_idx == 0:
                    print(f'No trailing edge for distance {d}m')
                # subsequent bands subtract the prior band
                else:
                    lag_idx = d_idx - 1
                    lag_dist = distances[lag_idx]
                    print(f'Trailing edge: {lag_dist}m')
                    df_copy.loc[:, column.format(dist=d)] = \
                        df.loc[:, column.format(dist=d)] - df.loc[:, column.format(dist=lag_dist)]
        # edited necessary columns in place, only pass those columns back
        X = df_copy[selected_columns]

    return X, distances, labels
