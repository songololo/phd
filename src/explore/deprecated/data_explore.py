import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_pd(pd_dataframe):
    # remove infinite and NaN values
    pd_dataframe = pd_dataframe.replace([np.inf, -np.inf], np.nan)
    pd_dataframe = pd_dataframe.dropna(axis=0, how='any')  # axis 0 is index / row axis
    return pd_dataframe


def remove_outliers(pd_dataframe):
    # remove outliers
    pd_dataframe = pd_dataframe[np.abs(pd_dataframe - pd_dataframe.mean()) <= (3 * pd_dataframe.std())]
    pd_dataframe = pd_dataframe[~(np.abs(pd_dataframe - pd_dataframe.mean()) > (3 * pd_dataframe.std()))]
    return pd_dataframe


def pairwise_correlations(dsn_string, schema, nodes_table, columns, path, filename=None, city_id=None, size=(10, 10)):
    with psycopg2.connect(dsn_string) as db_connection:
        node_query = f'SELECT id, {", ".join(columns)} FROM {schema}.{nodes_table}'
        if city_id:
            node_query += f' WHERE city_id = {city_id}::text'
        else:
            node_query += ' WHERE city_id IS NOT NULL AND city_id::int < 101'
        logger.info(f'query: {node_query}')
        df_nodes = pd.read_sql(
            sql=node_query,
            con=db_connection,
            index_col='id',
            coerce_float=True,
            params=None
        )
        logger.info(f'{len(df_nodes)} node rows loaded')

    df_nodes = clean_pd(df_nodes)

    sns.set(style="white")

    # Compute the correlation matrix
    corr = df_nodes.corr()
    corr = corr.round(2)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size, dpi=300)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    print(corr)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr,
                # mask=mask,
                cmap=cmap,
                annot=True,
                annot_kws={
                    'size': 4
                },
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 4}
    plt.rc('font', **font)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if city_id:
        out_file = os.path.join(path, f'city_{city_id}_{nodes_table}_pairwise_correlations')
    else:
        out_file = os.path.join(path, f'city_all_{nodes_table}_pairwise_correlations')
    if filename:
        out_file = f'{out_file}_{filename}'
    plt.savefig(out_file + '.png', dpi=300, orientation='landscape', transparent=True, bbox_inches='tight',
                pad_inches=0.1)


def pair_grid(dsn_string, schema, nodes_table, columns, path, filename=None, city_id=None):
    with psycopg2.connect(dsn_string) as db_connection:
        node_query = f'SELECT id, {", ".join([c for c, b in columns])} FROM {schema}.{nodes_table}'
        if city_id:
            node_query += f' WHERE city_id = {city_id}::text'
        else:
            node_query += ' WHERE city_id IS NOT NULL AND city_id::int < 101'
        logger.info(f'query: {node_query}')
        df_nodes = pd.read_sql(
            sql=node_query,
            con=db_connection,
            index_col='id',
            coerce_float=True,
            params=None
        )
        logger.info(f'{len(df_nodes)} node rows loaded')

    df_nodes = clean_pd(df_nodes)

    # log where log boolean is true
    for column_name, log_bool in columns:
        if log_bool:
            # doing a log plus 1 transformation
            df_nodes[column_name] = df_nodes[column_name] + 1
            df_nodes[column_name] = df_nodes[column_name].apply(np.log)  # axis 0 is apply to column
            # rename the column
            df_nodes = df_nodes.rename(columns={column_name: column_name + '_log'})

    # remove outliers and clean again
    df_nodes = remove_outliers(df_nodes)
    df_nodes = clean_pd(df_nodes)

    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15), dpi=300)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.PairGrid(df_nodes)
    g = g.map_diag(plt.hist, edgecolor="w", bins=50)
    g = g.map_offdiag(plt.scatter, edgecolor="w", s=1, alpha=0.1)
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    if city_id:
        out_file = os.path.join(path, f'city_{city_id}_{nodes_table}_pair_grid')
    else:
        out_file = os.path.join(path, f'city_all_{nodes_table}_pair_grid')
    if filename:
        out_file = f'{out_file}_{filename}'
    plt.savefig(out_file + '.png', dpi=300, orientation='landscape', transparent=True, bbox_inches='tight',
                pad_inches=0.1)


if __name__ == '__main__':

    dsn_string = f"dbname='gareth' user='gareth' host='localhost' port=5432 password=''"
    schema = 'analysis'
    out_path = ''

    general_set = [
        # interpolated census results
        'cens_cars_interp', 'cens_nocars_interp', 'cens_ttw_bike_interp', 'cens_ttw_home_interp',
        'cens_ttw_motors_interp', 'cens_ttw_peds_interp', 'cens_ttw_pubtrans_interp',
        # aggregated census results
        'cens_tot_pop_100', 'cens_tot_pop_250', 'cens_tot_pop_500', 'cens_tot_pop_750', 'cens_tot_pop_1000',
        'cens_tot_pop_2000', 'cens_tot_pop_5000',
        'cens_adults_100', 'cens_adults_250', 'cens_adults_500', 'cens_adults_750', 'cens_adults_1000',
        'cens_adults_2000', 'cens_adults_5000',
        'cens_employed_100', 'cens_employed_250', 'cens_employed_500', 'cens_employed_750', 'cens_employed_1000',
        'cens_employed_2000', 'cens_employed_5000',
        'cens_dwellings_100', 'cens_dwellings_250', 'cens_dwellings_500', 'cens_dwellings_750', 'cens_dwellings_1000',
        'cens_dwellings_2000', 'cens_dwellings_5000',
        'cens_students_100', 'cens_students_250', 'cens_students_500', 'cens_students_750', 'cens_students_1000',
        'cens_students_2000', 'cens_students_5000',
        # centrality measures
        'closeness_100', 'closeness_250', 'closeness_500', 'closeness_750', 'closeness_1000', 'closeness_2000',
        'centrality_entropy_100', 'centrality_entropy_250', 'centrality_entropy_500',
        'betweenness_100', 'betweenness_250', 'betweenness_500', 'betweenness_750', 'betweenness_1000',
        'betweenness_2000',
        # VOA properties
        'voa_count_100', 'voa_count_250', 'voa_count_500', 'voa_count_750', 'voa_count_1000', 'voa_count_2000',
        # 'voa_area_mean_100', 'voa_area_mean_250', 'voa_area_mean_500', 'voa_area_mean_750', 'voa_area_mean_1000', 'voa_area_mean_2000',
        'voa_area_mean_clean_100', 'voa_area_mean_clean_250', 'voa_area_mean_clean_500', 'voa_area_mean_clean_750',
        'voa_area_mean_clean_1000', 'voa_area_mean_clean_2000',
        # 'voa_val_mean_100', 'voa_val_mean_250', 'voa_val_mean_500', 'voa_val_mean_750', 'voa_val_mean_1000', 'voa_val_mean_2000',
        'voa_val_mean_clean_100', 'voa_val_mean_clean_250', 'voa_val_mean_clean_500', 'voa_val_mean_clean_750',
        'voa_val_mean_clean_1000', 'voa_val_mean_clean_2000',
        # 'voa_cof_val_100', 'voa_cof_val_250', 'voa_cof_val_500', 'voa_cof_val_750', 'voa_cof_val_1000', 'voa_cof_val_2000',
        'voa_cof_val_clean_100', 'voa_cof_val_clean_250', 'voa_cof_val_clean_500', 'voa_cof_val_clean_750',
        'voa_cof_val_clean_1000', 'voa_cof_val_clean_2000',
        # 'voa_rate_mean_100', 'voa_rate_mean_250', 'voa_rate_mean_500', 'voa_rate_mean_750', 'voa_rate_mean_1000', 'voa_rate_mean_2000',
        'voa_rate_mean_clean_100', 'voa_rate_mean_clean_250', 'voa_rate_mean_clean_500', 'voa_rate_mean_clean_750',
        'voa_rate_mean_clean_1000', 'voa_rate_mean_clean_2000',
        # 'voa_cof_rate_100', 'voa_cof_rate_250', 'voa_cof_rate_500', 'voa_cof_rate_750', 'voa_cof_rate_1000', 'voa_cof_rate_2000'
        'voa_cof_rate_clean_100', 'voa_cof_rate_clean_250', 'voa_cof_rate_clean_500', 'voa_cof_rate_clean_750',
        'voa_cof_rate_clean_1000', 'voa_cof_rate_clean_2000',
        # uses
        'uses_score_100', 'uses_score_250', 'uses_score_500', 'uses_score_750', 'uses_score_1000', 'uses_score_2000',
        'uses_score_primary_100', 'uses_score_primary_250', 'uses_score_primary_500', 'uses_score_primary_750',
        'uses_score_primary_1000', 'uses_score_primary_2000',
        'uses_score_secondary_100', 'uses_score_secondary_250', 'uses_score_secondary_500', 'uses_score_secondary_750',
        'uses_score_secondary_1000', 'uses_score_secondary_2000',
        'uses_score_tertiary_100', 'uses_score_tertiary_250', 'uses_score_tertiary_500', 'uses_score_tertiary_750',
        'uses_score_tertiary_1000', 'uses_score_tertiary_2000'
    ]

    compare_set = [
        # centrality measures
        'closeness_100', 'closeness_250', 'closeness_500', 'closeness_750', 'closeness_1000', 'closeness_2000',
        # 'centrality_entropy_100', 'centrality_entropy_250', 'centrality_entropy_500',
        'betweenness_100', 'betweenness_250', 'betweenness_500', 'betweenness_750', 'betweenness_1000',
        'betweenness_2000',
        # VOA properties
        'voa_count_100', 'voa_count_250', 'voa_count_500', 'voa_count_750', 'voa_count_1000', 'voa_count_2000',
        # 'voa_area_mean_100', 'voa_area_mean_250', 'voa_area_mean_500', 'voa_area_mean_750', 'voa_area_mean_1000', 'voa_area_mean_2000',
        'voa_area_mean_clean_100', 'voa_area_mean_clean_250', 'voa_area_mean_clean_500', 'voa_area_mean_clean_750',
        'voa_area_mean_clean_1000', 'voa_area_mean_clean_2000',
        # 'voa_val_mean_100', 'voa_val_mean_250', 'voa_val_mean_500', 'voa_val_mean_750', 'voa_val_mean_1000', 'voa_val_mean_2000',
        'voa_val_mean_clean_100', 'voa_val_mean_clean_250', 'voa_val_mean_clean_500', 'voa_val_mean_clean_750',
        'voa_val_mean_clean_1000', 'voa_val_mean_clean_2000',
        # 'voa_cof_val_100', 'voa_cof_val_250', 'voa_cof_val_500', 'voa_cof_val_750', 'voa_cof_val_1000', 'voa_cof_val_2000',
        'voa_cof_val_clean_100', 'voa_cof_val_clean_250', 'voa_cof_val_clean_500', 'voa_cof_val_clean_750',
        'voa_cof_val_clean_1000', 'voa_cof_val_clean_2000',
        # 'voa_rate_mean_100', 'voa_rate_mean_250', 'voa_rate_mean_500', 'voa_rate_mean_750', 'voa_rate_mean_1000', 'voa_rate_mean_2000',
        'voa_rate_mean_clean_100', 'voa_rate_mean_clean_250', 'voa_rate_mean_clean_500', 'voa_rate_mean_clean_750',
        'voa_rate_mean_clean_1000', 'voa_rate_mean_clean_2000',
        # 'voa_cof_rate_100', 'voa_cof_rate_250', 'voa_cof_rate_500', 'voa_cof_rate_750', 'voa_cof_rate_1000', 'voa_cof_rate_2000'
        'voa_cof_rate_clean_100', 'voa_cof_rate_clean_250', 'voa_cof_rate_clean_500', 'voa_cof_rate_clean_750',
        'voa_cof_rate_clean_1000', 'voa_cof_rate_clean_2000',
        # uses
        'uses_score_100', 'uses_score_250', 'uses_score_500', 'uses_score_750', 'uses_score_1000', 'uses_score_2000',
        'uses_score_primary_100', 'uses_score_primary_250', 'uses_score_primary_500', 'uses_score_primary_750',
        'uses_score_primary_1000', 'uses_score_primary_2000',
        'uses_score_secondary_100', 'uses_score_secondary_250', 'uses_score_secondary_500', 'uses_score_secondary_750',
        'uses_score_secondary_1000', 'uses_score_secondary_2000',
        'uses_score_tertiary_100', 'uses_score_tertiary_250', 'uses_score_tertiary_500', 'uses_score_tertiary_750',
        'uses_score_tertiary_1000', 'uses_score_tertiary_2000'
    ]
    # pairwise_correlations(dsn_string, schema, 'test_table', compare_set, out_path, city_id='50', size=(20, 20))

    decay_set_500 = ['betweenness_500',
                     # 'betweenness_angular_500',
                     'betweenness_weighted_500_001',
                     # 'betweenness_angular_weighted_500_001',
                     'gravity_500_001',
                     # 'gravity_angular_500_001',
                     'closeness_500',
                     # 'closeness_angular_500',
                     # 'centrality_entropy_500',
                     'cens_tot_pop_500', 'cens_adults_500', 'cens_employed_500', 'cens_dwellings_500',
                     'cens_students_500', 'cens_nocars_interp', 'cens_cars_interp', 'cens_ttw_peds_interp',
                     'cens_ttw_bike_interp', 'cens_ttw_motors_interp', 'cens_ttw_pubtrans_interp',
                     'cens_ttw_home_interp', 'listed_bldgs_count_500_001', 'voa_count_500_001', 'voa_area_mean_500_001',
                     'voa_val_mean_500_001', 'voa_cov_val_500_001', 'voa_rate_mean_500_001', 'voa_cov_rate_500_001',
                     'uses_accommodation_500_001', 'uses_eating_500_001', 'uses_commercial_500_001',
                     'uses_attractions_500_001', 'uses_entertainment_500_001', 'uses_manufacturing_500_001',
                     'uses_retail_500_001', 'uses_transport_500_001', 'uses_property_500_001', 'uses_health_500_001',
                     'uses_education_500_001', 'uses_parks_500_001', 'uses_cultural_500_001', 'uses_sports_500_001',
                     'mixed_uses_score_0_500_001', 'mixed_uses_score_5_500_001', 'mixed_uses_score_10_500_001',
                     'uses_score_primary_500_001', 'uses_score_secondary_500_001', 'uses_score_tertiary_500_001',
                     'mixed_uses_score_hill_0_500_001', 'mixed_uses_score_hill_10_500_001',
                     'mixed_uses_score_hill_20_500_001', 'mixed_uses_d_simpson_index_500_001',
                     'mixed_uses_d_species_distinctness_500_001', 'mixed_uses_d_balance_factor_500_001',
                     'mixed_uses_d_combined_500_001']

    decay_set_1000 = ['betweenness_1000',
                      # 'betweenness_angular_1000',
                      'betweenness_weighted_1000_0005',
                      # 'betweenness_angular_weighted_1000_0005',
                      'gravity_1000_0005',
                      # 'gravity_angular_1000_0005',
                      'closeness_1000',
                      # 'closeness_angular_1000',
                      # 'centrality_entropy_1000',
                      'cens_tot_pop_1000', 'cens_adults_1000', 'cens_employed_1000', 'cens_dwellings_1000',
                      'cens_students_1000', 'cens_nocars_interp', 'cens_cars_interp', 'cens_ttw_peds_interp',
                      'cens_ttw_bike_interp', 'cens_ttw_motors_interp', 'cens_ttw_pubtrans_interp',
                      'cens_ttw_home_interp', 'listed_bldgs_count_1000_0005', 'voa_count_1000_0005',
                      'voa_area_mean_1000_0005', 'voa_val_mean_1000_0005', 'voa_cov_val_1000_0005',
                      'voa_rate_mean_1000_0005', 'voa_cov_rate_1000_0005', 'uses_accommodation_1000_0005',
                      'uses_eating_1000_0005', 'uses_commercial_1000_0005', 'uses_attractions_1000_0005',
                      'uses_entertainment_1000_0005', 'uses_manufacturing_1000_0005', 'uses_retail_1000_0005',
                      'uses_transport_1000_0005', 'uses_property_1000_0005', 'uses_health_1000_0005',
                      'uses_education_1000_0005', 'uses_parks_1000_0005', 'uses_cultural_1000_0005',
                      'uses_sports_1000_0005', 'mixed_uses_score_0_1000_0005', 'mixed_uses_score_5_1000_0005',
                      'mixed_uses_score_10_1000_0005', 'uses_score_primary_1000_0005', 'uses_score_secondary_1000_0005',
                      'uses_score_tertiary_1000_0005', 'mixed_uses_score_hill_0_1000_0005',
                      'mixed_uses_score_hill_10_1000_0005', 'mixed_uses_score_hill_20_1000_0005',
                      'mixed_uses_d_simpson_index_1000_0005', 'mixed_uses_d_species_distinctness_1000_0005',
                      'mixed_uses_d_balance_factor_1000_0005', 'mixed_uses_d_combined_1000_0005']

    decay_set_2000 = ['betweenness_2000',
                      # 'betweenness_angular_2000',
                      'betweenness_weighted_2000_00025',
                      # 'betweenness_angular_weighted_2000_00025',
                      'gravity_2000_00025',
                      # 'gravity_angular_2000_00025',
                      'closeness_2000',
                      # 'closeness_angular_2000',
                      # 'centrality_entropy_2000',
                      'cens_tot_pop_2000', 'cens_adults_2000', 'cens_employed_2000', 'cens_dwellings_2000',
                      'cens_students_2000', 'cens_nocars_interp', 'cens_cars_interp', 'cens_ttw_peds_interp',
                      'cens_ttw_bike_interp', 'cens_ttw_motors_interp', 'cens_ttw_pubtrans_interp',
                      'cens_ttw_home_interp', 'listed_bldgs_count_2000_00025', 'voa_count_2000_00025',
                      'voa_area_mean_2000_00025', 'voa_val_mean_2000_00025', 'voa_cov_val_2000_00025',
                      'voa_rate_mean_2000_00025', 'voa_cov_rate_2000_00025', 'uses_accommodation_2000_00025',
                      'uses_eating_2000_00025', 'uses_commercial_2000_00025', 'uses_attractions_2000_00025',
                      'uses_entertainment_2000_00025', 'uses_manufacturing_2000_00025', 'uses_retail_2000_00025',
                      'uses_transport_2000_00025', 'uses_property_2000_00025', 'uses_health_2000_00025',
                      'uses_education_2000_00025', 'uses_parks_2000_00025', 'uses_cultural_2000_00025',
                      'uses_sports_2000_00025', 'mixed_uses_score_0_2000_00025', 'mixed_uses_score_5_2000_00025',
                      'mixed_uses_score_10_2000_00025', 'uses_score_primary_2000_00025',
                      'uses_score_secondary_2000_00025', 'uses_score_tertiary_2000_00025',
                      'mixed_uses_score_hill_0_2000_00025', 'mixed_uses_score_hill_10_2000_00025',
                      'mixed_uses_score_hill_20_2000_00025', 'mixed_uses_d_simpson_index_2000_00025',
                      'mixed_uses_d_species_distinctness_2000_00025', 'mixed_uses_d_balance_factor_2000_00025',
                      'mixed_uses_d_combined_2000_00025']

    full = ['betweenness_500', 'betweenness_1000', 'betweenness_2000', 'betweenness_angular_500',
            'betweenness_angular_1000', 'betweenness_angular_2000', 'betweenness_weighted_500_001',
            'betweenness_weighted_1000_0005', 'betweenness_weighted_2000_00025', 'betweenness_angular_weighted_500_001',
            'betweenness_angular_weighted_1000_0005', 'betweenness_angular_weighted_2000_00025', 'gravity_500_001',
            'gravity_1000_0005', 'gravity_2000_00025', 'gravity_angular_500_001', 'gravity_angular_1000_0005',
            'gravity_angular_2000_00025', 'closeness_500', 'closeness_1000', 'closeness_2000', 'closeness_angular_500',
            'closeness_angular_1000', 'closeness_angular_2000', 'centrality_entropy_500', 'centrality_entropy_1000',
            'centrality_entropy_2000', 'cens_nocars_interp', 'cens_cars_interp', 'cens_ttw_peds_interp',
            'cens_ttw_bike_interp', 'cens_ttw_motors_interp', 'cens_ttw_pubtrans_interp', 'cens_ttw_home_interp',
            'cens_tot_pop_500', 'cens_tot_pop_1000', 'cens_tot_pop_2000', 'cens_adults_500', 'cens_adults_1000',
            'cens_adults_2000', 'cens_employed_500', 'cens_employed_1000', 'cens_employed_2000', 'cens_dwellings_500',
            'cens_dwellings_1000', 'cens_dwellings_2000', 'cens_students_500', 'cens_students_1000',
            'cens_students_2000', 'listed_bldgs_count_500_001', 'listed_bldgs_count_1000_0005',
            'listed_bldgs_count_2000_00025', 'voa_count_500_001', 'voa_count_1000_0005', 'voa_count_2000_00025',
            'voa_area_mean_500_001', 'voa_area_mean_1000_0005', 'voa_area_mean_2000_00025', 'voa_val_mean_500_001',
            'voa_val_mean_1000_0005', 'voa_val_mean_2000_00025', 'voa_cov_val_500_001', 'voa_cov_val_1000_0005',
            'voa_cov_val_2000_00025', 'voa_rate_mean_500_001', 'voa_rate_mean_1000_0005', 'voa_rate_mean_2000_00025',
            'voa_cov_rate_500_001', 'voa_cov_rate_1000_0005', 'voa_cov_rate_2000_00025', 'uses_accommodation_500_001',
            'uses_accommodation_1000_0005', 'uses_accommodation_2000_00025', 'uses_eating_500_001',
            'uses_eating_1000_0005', 'uses_eating_2000_00025', 'uses_commercial_500_001', 'uses_commercial_1000_0005',
            'uses_commercial_2000_00025', 'uses_attractions_500_001', 'uses_attractions_1000_0005',
            'uses_attractions_2000_00025', 'uses_entertainment_500_001', 'uses_entertainment_1000_0005',
            'uses_entertainment_2000_00025', 'uses_manufacturing_500_001', 'uses_manufacturing_1000_0005',
            'uses_manufacturing_2000_00025', 'uses_retail_500_001', 'uses_retail_1000_0005', 'uses_retail_2000_00025',
            'uses_transport_500_001', 'uses_transport_1000_0005', 'uses_transport_2000_00025', 'uses_property_500_001',
            'uses_property_1000_0005', 'uses_property_2000_00025', 'uses_health_500_001', 'uses_health_1000_0005',
            'uses_health_2000_00025', 'uses_education_500_001', 'uses_education_1000_0005', 'uses_education_2000_00025',
            'uses_parks_500_001', 'uses_parks_1000_0005', 'uses_parks_2000_00025', 'uses_cultural_500_001',
            'uses_cultural_1000_0005', 'uses_cultural_2000_00025', 'uses_sports_500_001', 'uses_sports_1000_0005',
            'uses_sports_2000_00025', 'mixed_uses_score_0_500_001', 'mixed_uses_score_0_1000_0005',
            'mixed_uses_score_0_2000_00025', 'mixed_uses_score_5_500_001', 'mixed_uses_score_5_1000_0005',
            'mixed_uses_score_5_2000_00025', 'mixed_uses_score_10_500_001', 'mixed_uses_score_10_1000_0005',
            'mixed_uses_score_10_2000_00025', 'uses_score_primary_500_001', 'uses_score_primary_1000_0005',
            'uses_score_primary_2000_00025', 'uses_score_secondary_500_001', 'uses_score_secondary_1000_0005',
            'uses_score_secondary_2000_00025', 'uses_score_tertiary_500_001', 'uses_score_tertiary_1000_0005',
            'uses_score_tertiary_2000_00025', 'mixed_uses_score_hill_0_500_001', 'mixed_uses_score_hill_0_1000_0005',
            'mixed_uses_score_hill_0_2000_00025', 'mixed_uses_score_hill_10_500_001',
            'mixed_uses_score_hill_10_1000_0005', 'mixed_uses_score_hill_10_2000_00025',
            'mixed_uses_score_hill_20_500_001', 'mixed_uses_score_hill_20_1000_0005',
            'mixed_uses_score_hill_20_2000_00025', 'mixed_uses_d_simpson_index_500_001',
            'mixed_uses_d_simpson_index_1000_0005', 'mixed_uses_d_simpson_index_2000_00025',
            'mixed_uses_d_species_distinctness_500_001', 'mixed_uses_d_species_distinctness_1000_0005',
            'mixed_uses_d_species_distinctness_2000_00025', 'mixed_uses_d_balance_factor_500_001',
            'mixed_uses_d_balance_factor_1000_0005', 'mixed_uses_d_balance_factor_2000_00025',
            'mixed_uses_d_combined_500_001', 'mixed_uses_d_combined_1000_0005', 'mixed_uses_d_combined_2000_00025']

    pairwise_correlations(dsn_string, 'analysis', 'roadnodes_20', decay_set_500, out_path, city_id='1', size=(20, 20),
                          filename='decay_500')

    detailed_set = [
        'cens_cars_interp', 'cens_nocars_interp', 'cens_ttw_bike_interp', 'cens_ttw_home_interp',
        'cens_ttw_motors_interp', 'cens_ttw_peds_interp', 'cens_ttw_pubtrans_interp',
        # aggregated census results
        'cens_tot_pop_{dist}', 'cens_adults_{dist}', 'cens_employed_{dist}', 'cens_dwellings_{dist}',
        'cens_students_{dist}',
        # centrality measures
        'closeness_{dist}', 'betweenness_{dist}',
        # VOA properties
        'voa_count_{dist}', 'voa_area_mean_{dist}', 'voa_area_mean_clean_{dist}', 'voa_val_mean_{dist}',
        'voa_val_mean_clean_{dist}', 'voa_cof_val_{dist}', 'voa_cof_val_clean_{dist}', 'voa_rate_mean_{dist}',
        'voa_rate_mean_clean_{dist}', 'voa_cof_rate_{dist}', 'voa_cof_rate_clean_{dist}',
        # uses
        'uses_score_{dist}', 'uses_score_primary_{dist}', 'uses_score_secondary_{dist}', 'uses_score_tertiary_{dist}',
        'listed_bldgs_count_{dist}', 'uses_accommodation_{dist}', 'uses_attractions_{dist}', 'uses_commercial_{dist}',
        'uses_cultural_{dist}', 'uses_eating_{dist}', 'uses_education_{dist}', 'uses_entertainment_{dist}',
        'uses_health_{dist}', 'uses_manufacturing_{dist}', 'uses_parks_{dist}', 'uses_property_{dist}',
        'uses_retail_{dist}', 'uses_sports_{dist}', 'uses_transport_{dist}'
    ]
    for dist in [100, 250, 500, 750, 1000, 2000]:
        temp_set = [s.format(dist=dist) for s in detailed_set]
        # pairwise_correlations(dsn_string, schema, 'test_table', temp_set, out_path, filename=f'detailed_uses_{dist}', city_id='1')

    pair_grid_set = [
        ('closeness_500', False),
        ('cens_tot_pop_interp', True),
        ('uses_score_{dist}', True),
        ('voa_cof_{dist}', True)
    ]

    # for dist in [100, 250, 500, 750]:
    #    temp_set = [(k.format(dist=dist), b) for k, b in pair_grid_set]
    #    pair_grid(dsn_string, schema, 'roadnodes_50', temp_set, out_path, filename=f'_{dist}', city_id='1')
