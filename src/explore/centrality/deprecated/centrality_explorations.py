# %%


import matplotlib.pyplot as plt
import numpy as np
from src import phd_util



# %% columns to load
# weighted
columns = [
    'city_pop_id',
    'mixed_uses_score_hill_0_200_002',
    'uses_commercial_200_002',
    'uses_retail_200_002',
    'uses_eating_200_002',
    'uses_transport_200_002',
    'uses_education_200_002',
    'uses_cultural_200_002',
    'uses_manufacturing_200_002',
    'uses_parks_200_002',
    'uses_accommodation_200_002',
    'uses_health_200_002',
    'uses_sports_200_002',
    'uses_entertainment_200_002',
    'uses_attractions_200_002',
    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp',
    'cens_tot_pop_200',
    'cens_tot_pop_400',
    'cens_tot_pop_800',
    'cens_tot_pop_1600'
    # 'ST_X(geom)',
    # 'ST_Y(geom)'
]

# non weighted
columns_d = [
    'met_betw_{d}',
    'met_betw_w_{d}',
    'met_gravity_{d}',
    'met_node_count_{d}',
    'met_farness_{d}',
    'met_rt_complex_{d}'
]

distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

for d in distances:
    columns += [c.format(d=d) for c in columns_d]

# %% load roadnodes data
print(f'loading columns: {columns}')
df_data_full = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_full', 'WHERE city_pop_id = 1')
df_data_100 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_100', 'WHERE city_pop_id = 1')
# TODO: change once done calcs
df_data_50 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_100', 'WHERE city_pop_id = 1')
# TODO: change once done calcs
df_data_20 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_100', 'WHERE city_pop_id = 1')

# %%
print('cleaning data')
# be careful with cleanup...
df_data_full_clean = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)
df_data_100_clean = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)
df_data_50_clean = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)
df_data_20_clean = phd_util.clean_pd(df_data_20, drop_na='all', fill_inf=np.nan)

# angular averaged back to primary is now deprecated, but for reference:
# angular farness at 1600 with zeros imputed for NaN vs NaN manually removed from arrays:
#
# normalised mutual information 0.49558572983953136
# normalised mutual information 0.5318966971554541
#
# cuberoot pearson r 0.506658824438202
# cuberoot pearson r 0.5290957097130733
#
# boxcox shift pearson r: 0.26233280271345516 at optimal maxlog: 0.1532554630697125
# boxcox pearson r: 0.4867729536243162 at optimal maxlog: 0.4320233723743492
#
# spearman r 0.4882621571093322
# spearman r 0.4802195237028851
#
# rdc:  0.23500000968392293
# rdc:  0.5336297974343239


# %% create the closeness columns
for d in distances:
    df_data_full_clean[f'met_closeness_{d}'] = df_data_full_clean[f'met_node_count_{d}'] ** 2 / df_data_full_clean[
        f'met_farness_{d}']
    df_data_100_clean[f'met_closeness_{d}'] = df_data_100_clean[f'met_node_count_{d}'] ** 2 / df_data_100_clean[
        f'met_farness_{d}']
    df_data_50_clean[f'met_closeness_{d}'] = df_data_50_clean[f'met_node_count_{d}'] ** 2 / df_data_50_clean[
        f'met_farness_{d}']
    df_data_20_clean[f'met_closeness_{d}'] = df_data_20_clean[f'met_node_count_{d}'] ** 2 / df_data_20_clean[
        f'met_farness_{d}']

# %% correlation strategies - EXPLORATORY ONLY
x = df_data_full_clean.ang_farness_200
y = df_data_full_clean.mixed_uses_score_hill_0_200

# from minepy import MINE
# performance penalty from around 0.5
# achieves fairly high correlation by 0.3
# mine = MINE(alpha=0.5, c=15, est='mic_approx')
# mine.compute_score(x, y)
# print('max info coef', mine.mic())

# from sklearn.metrics import mutual_info_score
# bins = 1000
# c_xy = np.histogram2d(x, y, bins)[0]
# mi = mutual_info_score(None, None, contingency=c_xy)
# print('mutual information', mi)

from sklearn.metrics.cluster import normalized_mutual_info_score

nmi = normalized_mutual_info_score(x, y)
print('normalised mutual information', nmi)

# this is slow and doesn't appear to be accurate?
# from sklearn.metrics.cluster import adjusted_mutual_info_score
# ami = adjusted_mutual_info_score(x, y)
# print('adjusted mutual information', ami)

from scipy.stats import pearsonr, spearmanr, boxcox
from src.phd_util import rdc_cor

x_clean = x[np.isfinite(x)]
y_clean = y[np.isfinite(x)]

x_clean = x_clean[np.isfinite(y_clean)]
y_clean = y_clean[np.isfinite(y_clean)]

p = pearsonr(x_clean, y_clean)
print('cuberoot pearson r', p[0])

try:
    bc_x, maxlog = boxcox(x_clean)
    p = pearsonr(bc_x, y_clean)
    print(f'boxcox pearson r: {p[0]} at optimal maxlog: {maxlog}')
except:
    bc_x, maxlog = boxcox(x_clean + 0.0000000001)
    p = pearsonr(bc_x, y_clean)
    print(f'boxcox shift pearson r: {p[0]} at optimal maxlog: {maxlog}')

s = spearmanr(x_clean, y_clean)
print('spearman r', s[0])

# k=5 and s=0.5 seems to balance speed and consistency, though instabilities persist, hence n as well
print('rdc: ', rdc_cor(x, y, k=5, s=0.5))

# %%
mu = np.array([0, 0])
r = np.array([[1, 0.56], [0.56, 1]])
x, y = np.random.multivariate_normal(mu, r, size=10000).T
print('pearson', pearsonr(x, y)[0])
print('spearman', spearmanr(x, y)[0])
print('rdc: ', rdc_cor(x, y))

mu = np.array([0, 0])
r = np.array([[1, 0.21], [0.21, 1]])
x, y = np.random.multivariate_normal(mu, r, size=10000).T
print('pearson', pearsonr(x, y)[0])
print('spearman', spearmanr(x, y)[0])
print('rdc: ', rdc_cor(x, y))

# %% autocorrelation EXPLORATION
import geopandas as gpd
from shapely import geometry
from pysal.weights.Distance import Kernel

geoms = [geometry.Point(xy) for xy in zip(df_data_clean['ST_X(geom)'], df_data_clean['ST_Y(geom)'])]
crs = {'init': 'epsg:27700'}
geo_df_data_clean = gpd.GeoDataFrame(df_data_clean, crs=crs, geometry=geoms)

print('generating weights')
kernel_w = Kernel.from_dataframe(geo_df_data_clean)

# %% autocorrelation EXPLORATION
# first calculate your residuals
from scipy.stats import linregress
from pysal.esda.moran import Moran

x = geo_df_data_clean.met_node_count_800
y = geo_df_data_clean.mixed_uses_score_hill_0_200

slope, intercept, rvalue, pvalue, stderr = linregress(x=np.cbrt(x), y=np.cbrt(y))
print(rvalue, pvalue)
y_predict = (slope * np.cbrt(x) + intercept) ** 3
plt.scatter(y, y_predict)
plt.show()

# %%
residuals = y_predict - y
plt.hist(residuals, bins=100)
plt.show()

# %%
morans_I = Moran(residuals, kernel_w)
print(morans_I.I)

# on 100m graph
# met_node_count_200: 0.7473059000911748
# met_node_count_400: 0.7257693008191158
# met_node_count_800: 0.7000382165084419
# met_node_count_1600: 0.6821779777306415
# met_node_count_1600 vs mixed_uses_score_hill_0_1600: 0.9368075473934013
# met_far_norm_400: 0.7279574152608174
# met_closeness_200: 0.7460617776578066
# met_betw_200: 0.7327319238167639
# met_betw_w_200: 0.7320129875536658

# 100m vs 20m 0n 800m node count vs mixed uses hill 0
# 100m: 0.58 r value and 0.70 Moran's I
# 20m: 0.57 r value and 0.70 Moran's I

# %% write to database
print('writing residuals to db')
residuals_col_name = f'residuals_met_far_norm_800_mixed_uses_score_hill_0_200'
phd_util.write_col_data(table, residuals, residuals_col_name, 'real', geo_df_data_clean.id, 'id')
