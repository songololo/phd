#%%
# http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
import os
import logging
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan

import phd_util
reload(phd_util)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%% columns to load
columns_w = [
    'uses_transport_{dist}',
    'uses_accommodation_{dist}',
    'uses_parks_{dist}',
    'uses_health_{dist}',
    'mixed_uses_score_10_{dist}',
    'mixed_uses_score_0_{dist}',
    'uses_education_{dist}',
    'uses_cultural_{dist}',
    'mixed_uses_score_5_{dist}',
    'uses_sports_{dist}',
    'gravity_{dist}',
    'uses_commercial_{dist}',
    'uses_entertainment_{dist}',
    'uses_manufacturing_{dist}',
    'uses_retail_{dist}',
    'voa_rate_mean_{dist}',
    'voa_val_mean_{dist}',
    'voa_cov_val_{dist}',
    'voa_count_{dist}',
    'voa_cov_rate_{dist}',
    'voa_area_mean_{dist}',
    'uses_attractions_{dist}',
    'betweenness_weighted_{dist}',
    'uses_eating_{dist}',
    'mixed_uses_score_hill_0_{dist}',
    'mixed_uses_score_hill_10_{dist}',
    'mixed_uses_score_hill_20_{dist}',
    'mixed_uses_d_simpson_index_{dist}'
]
columns_nw = [
    'betweenness_{dist}',
    'cens_students_{dist}',
    'closeness_{dist}',
    'cens_dwellings_{dist}',
    'cens_tot_pop_{dist}',
    'cens_adults_{dist}',
    'cens_employed_{dist}'
]

compound_columns = [
    'id',
    'cens_nocars_interp',
    'cens_cars_interp',
    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp'
]

for d in ['200', '400', '800', '1600']:
    compound_columns += [c.format(dist=d) for c in columns_nw]

for d in ['200_002', '400_001', '800_0005', '1600_00025']:
    compound_columns += [c.format(dist=d) for c in columns_w]

#%% load data
logger.info(f'loading columns: {compound_columns}')
source_table = 'analysis.roadnodes_20_test'
ml_data_ldn = phd_util.load_data_as_pd_df(compound_columns, source_table, f'WHERE city_pop_id = 1')

#%% clean data
logger.info('cleaning data')
# be careful with cleanup...
# per correlation plot:
# remove most extreme values - 0.99 gives good balance, 0.999 is stronger for some but generally weaker?
ml_data_ldn_clean = phd_util.remove_outliers_quantile(ml_data_ldn, q=(0, 0.999))
# remove rows where all nan
ml_data_ldn_clean = phd_util.clean_pd(ml_data_ldn_clean, drop_na='all')
# for this data nans normally represent null or 0 so just fill with 0
ml_data_ldn_clean = ml_data_ldn_clean.fillna(0)

#%% columns for training / predicting
cols_ml = [
    'closeness_{dist}',
    #'gravity_{dist}',
    #'betweenness_{dist}',
    #'betweenness_weighted_{dist}',
    #'cens_dwellings_{dist}',
    #'cens_tot_pop_{dist}',
    #'cens_employed_{dist}',
    #'cens_ttw_peds_interp',
    #'cens_ttw_motors_interp',
    #'cens_ttw_pubtrans_interp',
    #'cens_ttw_home_interp',
    #'uses_transport_{dist}',
    #'uses_eating_{dist}',
    #'uses_commercial_{dist}',
    #'uses_retail_{dist}',
    #'uses_accommodation_{dist}',
    #'uses_parks_{dist}',
    #'uses_education_{dist}',
    #'uses_cultural_{dist}',
    #'uses_manufacturing_{dist}',
    #'uses_sports_{dist}',
    #'voa_count_{dist}',
    #'voa_val_mean_{dist}',
    #'voa_rate_mean_{dist}',
    #'voa_area_mean_{dist}',
    #'mixed_uses_score_hill_0_{dist}',
    'mixed_uses_score_hill_10_{dist}',
    #'mixed_uses_score_hill_20_{dist}'
]
logger.info(f'columns: {cols_ml}')

selected_cols = []
for d in [400]:
    selected_cols += [c.format(dist=d) for c in cols_ml]

data = ml_data_ldn_clean[selected_cols]

#%% clustering

# sets min size for clusters
min_cluster_size = 1000
# sets how conservative - use lower numbers to reduce noise classifications
min_samples = 5

logger.info(f'clustering with hdbscan for min cluster size: {min_cluster_size} and min samples: {min_samples} on {len(selected_cols)} dimensions')
logger.info(data.columns)
cache_dir = os.path.join(os.path.abspath(os.getcwd()), 'cache')
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, memory=cache_dir)
cluster_labels = clusterer.fit_predict(data)

#%%
# write to database
logger.info('writing results to db')

phd_util.write_col_data(source_table, cluster_labels, f'hdb_m_{min_cluster_size}_{min_samples}_{len(selected_cols)}dim', 'int', ml_data_ldn_clean.id, 'id')

logger.info('completed')
