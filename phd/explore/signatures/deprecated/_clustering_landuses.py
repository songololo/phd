#%% for jupyter notebooks
import logging
import asyncio
import pandas as pd
import numpy as np
from importlib import reload
from sklearn.cluster import OPTICS
from sklearn.preprocessing import RobustScaler, PowerTransformer

import phd_util
reload(phd_util)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#%%
# cityseer cmap
cityseer_cmap = phd_util.cityseer_cmap()

# db connection params
db_config = {
    #'host': 'localhost',
    'host': 'host.docker.internal',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

###
# landuse columns
landuse_columns = [
    # for table - only need specific distances (but only where not already covered in columns below)
    'id'
]

landuse_columns_dist = [
    'mu_hill_0_{dist}',
    'mu_hill_1_{dist}',
    'mu_hill_2_{dist}',
    'mu_hill_branch_wt_0_{dist}',
    'mu_hill_branch_wt_1_{dist}',
    'mu_hill_branch_wt_2_{dist}',
    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    'ac_eating_nw_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    'ac_commercial_nw_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_government_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    'ac_retail_food_nw_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}',
    'ac_total_{dist}',
    'ac_total_nw_{dist}'
]

# selected landuse distances are distinct from the selected centrality columns
landuse_distances = [50, 200, 800]
for d in landuse_distances:
    landuse_columns += [c.format(dist=d) for c in landuse_columns_dist]


#%% load roadnodes data
print('loading columns')
# load landuses
df_data_landuse_full = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    landuse_columns,
    'analysis.roadnodes_full_lu',
    'WHERE city_pop_id = 1'))
df_data_landuse_full = df_data_landuse_full.set_index('id')
df_full_clean = phd_util.clean_pd(df_data_landuse_full, drop_na='all', fill_inf=np.nan)


#%%
col_theme = 'mu_hill_branch_wt_0_{dist}'
cols = []
for d in landuse_distances:
    cols.append(col_theme.format(dist=d))
# select columns
X = df_full_clean[cols]

print('shaping data')
# shape data - these return copy by default
# transforming to help splay data at smaller distances
X = PowerTransformer(standardize=False).fit_transform(X)
# using robust to better handle outliers for small distances
X = RobustScaler().fit_transform(X)

#%%
# get a sense of the distances
from scipy.spatial.distance import minkowski

for i, arr_1 in enumerate(X):
    for j, arr_2 in enumerate(X):
        print(arr_1)
        print(arr_2)
        print(arr_1 - arr_2)
        m_d = minkowski(arr_1, arr_2)
        print(m_d)
        if j == 10:
            break
    if i == 10:
        break

#%%

# #############################################################################
# Compute DBSCAN
print('clustering')
db = OPTICS(min_samples=1000, n_jobs=-1, max_eps=0.1).fit(X)

print('reviewing')
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print('labels', labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('clusters', n_clusters_)
n_noise_ = list(labels).count(-1)
print('noise', n_noise_)

#await util.write_col_data(
#    db_config,
#    'analysis.roadnodes_20',
#    y_predict,
#    f'pred_{short_target}_using_{theme_d}',
#    'real',
#    df_20_clean.index,
#    'id')
