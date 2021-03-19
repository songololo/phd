'''
Loads the data from the database and saves as feather files - dramatically faster to work with
- All metrics computed at all distances for London
- For locations smaller than London - only 20m and full are computed for all (not 50m and 100m)
'''

#%%
import phd_util
from explore.theme_setup import data_path

# db connection params 
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

#%%
#columns
columns_base = [
    'id',
    'city_pop_id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y',
    'cens_density_interp',
    'cens_nocars_interp',
    'cens_cars_interp',
    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp'
]

# centrality columns
cent_columns_d = [
    'c_node_density',
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
    'c_segment_betweeness_hybrid']

lu_columns_d = [
    'mu_hill_branch_wt_0',
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
    'ac_retail_food_nw',
    'ac_retail_other',
    'ac_retail_other_nw',
    'ac_transport',
    'ac_health',
    'ac_education',
    'ac_parks',
    'ac_cultural',
    'ac_sports',
    'ac_total',
    'ac_total_nw']

cens_columns_d = [
    'cens_tot_pop',
    'cens_adults',
    'cens_employed',
    'cens_dwellings',
    'cens_students'
]

columns = [c for c in columns_base]
col_template = '{col}[{i}] as {col}_{dist}'

'''
Loads London to feather - uses more granular distances
'''
# large distances only available on full
cent_distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
for col_interp in cent_columns_d:
    for i, d in enumerate(cent_distances):
        columns.append(col_template.format(col=col_interp, i=i + 1, dist=d))

lu_distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
for col_interp in lu_columns_d:
    for i, d in enumerate(lu_distances):
        columns.append(col_template.format(col=col_interp, i=i + 1, dist=d))

# no 50 on cens
cens_distances = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
for col_interp in cens_columns_d:
    for i, d in enumerate(cens_distances):
        columns.append(col_template.format(col=col_interp, i=i + 1, dist=d))

# load data for London only
print('loading columns')
df_full = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_full',
    'WHERE city_pop_id <= 1 and within = true')
df_full = df_full.reset_index()
df_full.to_feather(data_path / 'df_full.feather')

df_100 = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_100',
    'WHERE city_pop_id <= 1 and within = true')
df_100 = df_100.reset_index()
df_100.to_feather(data_path / 'df_100.feather')

df_50 = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_50',
    'WHERE city_pop_id <= 1 and within = true')
df_50 = df_50.reset_index()
df_50.to_feather(data_path / 'df_50.feather')

df_20 = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_20',
    'WHERE city_pop_id <= 1 and within = true')
df_20 = df_20.reset_index()
df_20.to_feather(data_path / 'df_20.feather')

#%%
'''
Load data for all towns and cities on 20m and full networks (not computed on 50m and 100m)
Uses less granular distances otherwise memory issues on 20m network
'''

#columns
columns_base = [
    'id',
    'city_pop_id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y'
]

# centrality columns
cent_columns_d = [
    #'c_node_density',
    #'c_node_cycles',
    #'c_node_harmonic',
    #'c_node_beta',
    #'c_segment_density',
    #'c_segment_harmonic',
    #'c_segment_beta',
    #'c_node_betweenness',
    'c_node_betweenness_beta',
    #'c_segment_betweenness',
    'c_node_harmonic_angular',
    #'c_segment_harmonic_hybrid',
    #'c_node_betweenness_angular',
    #'c_segment_betweeness_hybrid'
    ]

lu_columns_d = [
    'mu_hill_branch_wt_0',
    #'ac_accommodation',
    'ac_eating',
    'ac_drinking',
    'ac_commercial',
    #'ac_tourism',
    #'ac_entertainment',
    #'ac_government',
    'ac_manufacturing',
    'ac_retail_food',
    'ac_retail_other',
    'ac_transport',
    #'ac_health',
    #'ac_education',
    #'ac_parks',
    #'ac_cultural',
    #'ac_sports',
    'ac_total']

cens_columns_d = [
    'cens_tot_pop',
    'cens_dwellings'
    #'cens_students'
]

columns = [c for c in columns_base]
col_template = '{col}[{i}] as {col}_{dist}'

# large distances only available on full
cent_distances = [100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600]
for col_interp in cent_columns_d:
    for i, d in enumerate(cent_distances):
        columns.append(col_template.format(col=col_interp, i=i + 1, dist=d))

lu_distances = [100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600]
for col_interp in lu_columns_d:
    for i, d in enumerate(lu_distances):
        columns.append(col_template.format(col=col_interp, i=i + 1, dist=d))

# no 50 on cens
cens_distances = [100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600]
for col_interp in cens_columns_d:
    for i, d in enumerate(cens_distances):
        columns.append(col_template.format(col=col_interp, i=i + 1, dist=d))


#  %%
'''
Fetch all towns for full network
'''
df_full = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_full',
    'WHERE within = true')
df_full = df_full.reset_index()
df_full.to_feather(data_path / 'df_full_all.feather')

#  %%
'''
Fetch only new towns band for 20m else memory issues
'''
df_20 = phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.nodes_20',
    'WHERE city_pop_id >= 19 AND city_pop_id <= 650 AND within = true')
df_20 = df_20.reset_index()
df_20.to_feather(data_path / 'df_20_all.feather')