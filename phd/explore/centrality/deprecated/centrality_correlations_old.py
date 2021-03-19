#%%
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
import phd_util
reload(phd_util)


#%% columns to load
# weighted
columns_w = [
    'mixed_uses_score_hill_0_{dist}',
    'uses_commercial_{dist}',
    'uses_retail_{dist}',
    'uses_eating_{dist}',
    'uses_transport_{dist}',
    'uses_education_{dist}',
    'uses_cultural_{dist}',
    'uses_manufacturing_{dist}',
    'uses_parks_{dist}',
    'uses_accommodation_{dist}',
    'uses_health_{dist}',
    'uses_sports_{dist}',
    'uses_entertainment_{dist}',
    'uses_attractions_{dist}',
    # 'mixed_uses_score_0_{dist}',
    # 'mixed_uses_score_5_{dist}',
    # 'mixed_uses_score_10_{dist}',
    # 'mixed_uses_score_hill_10_{dist}',
    # 'mixed_uses_score_hill_20_{dist}',
    # 'mixed_uses_d_simpson_index_{dist}',
    'met_gravity_{dist}',
    'met_betw_w_{dist}',
    'met_rt_complex_{dist}'
]
# non weighted
columns_nw = [
    'cens_tot_pop_{dist}',

    #'cens_students_{dist}',
    #'cens_dwellings_{dist}',
    #'cens_adults_{dist}',
    #'cens_employed_{dist}',

    'met_node_count_{dist}',
    'met_farness_{dist}',
    'met_betw_{dist}',
    'ang_node_count_{dist}',
    'ang_farness_{dist}',
    'ang_betw_{dist}'
]

compound_columns = [
    'id',
    'city_pop_id',
    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp',
    'ST_X(geom)',
    'ST_Y(geom)'

    #'cens_nocars_interp',
    #'cens_cars_interp',
]

for d in ['200', '400', '800', '1600']:
    compound_columns += [c.format(dist=d) for c in columns_nw]

for d in ['200_002', '400_001', '800_0005', '1600_00025']:
    compound_columns += [c.format(dist=d) for c in columns_w]

#%% load roadnodes data
print(f'loading columns: {compound_columns}')
table = 'analysis.roadnodes_20'
df_data = phd_util.load_data_as_pd_df(compound_columns, table, 'WHERE city_pop_id = 1')

#%% create the closeness columns
for d in [200, 400, 800, 1600]:
    df_data[f'met_closeness_{d}'] = df_data[f'met_node_count_{d}']**2 / df_data[f'met_farness_{d}']
    df_data[f'met_far_norm_{d}'] = df_data[f'met_farness_{d}'] / df_data[f'met_node_count_{d}'] ** 2
    # angular farness needs to be transformed
    df_data[f'ang_closeness_{d}_0'] = df_data[f'ang_node_count_{d}'] ** 2 / (df_data[f'ang_farness_{d}'])
    df_data[f'ang_closeness_{d}_1'] = df_data[f'ang_node_count_{d}'] ** 2 / (df_data[f'ang_farness_{d}'] + df_data[f'ang_node_count_{d}'])
    df_data[f'ang_closeness_{d}_10'] = df_data[f'ang_node_count_{d}']**2 / (df_data[f'ang_farness_{d}'] + df_data[f'ang_node_count_{d}'] * 10)
    df_data[f'ang_far_norm_{d}'] = df_data[f'ang_farness_{d}'] / df_data[f'ang_node_count_{d}'] ** 2

#%% clean data
print('cleaning data')
# be careful with cleanup...
# angular values on dual are averaged out to primary vertices, so dead-ends result in null values
# instead of imputing values, manually drop non finite values from arrays when calculating coefficients
# only drop only rows where all values are NaN
df_data_clean = phd_util.clean_pd(df_data, drop_na='all', fill_inf=np.nan)

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

#%% demonstrate the typical distributions
# the point is that low values preclude mixed uses but high values don't guarantee them
# the spike in the distributions is an artefact of search distances vis-a-vis straight roads with no intersections,
# in which case a certain number of adjacent nodes is reached, resulting in a stable preponderance of a certain value
# this would not be present in gravity-weighted forms of centrality

phd_util.plt_setup()

fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharey='row', sharex='col', gridspec_kw = { 'height_ratios':[4, 1] })

axes[0][0].set_title(f'{table}', size='xx-small')

y = df_data_clean.mixed_uses_score_hill_0_200
x = df_data_clean.met_node_count_800
axes[0][0].scatter(x, y, s=0.03, alpha=0.5)
axes[1][0].hist(x, np.logspace(np.log10(1), np.log10(x.max()), 150), edgecolor='white', linewidth=0.3, log=False)
axes[1][0].set_xlabel('Node Count $800m$')
axes[1][0].set_xscale('log')
axes[1][0].set_xlim(xmin=0)

x = df_data_clean.met_betw_800
axes[0][1].scatter(x, y, s=0.03, alpha=0.5)
axes[1][1].hist(x, np.logspace(np.log10(1), np.log10(x.max()), 150), edgecolor='white', linewidth=0.3, log=False)
axes[1][1].set_xlabel('Betweenness $800m$')
axes[1][1].set_xscale('log')
axes[1][1].set_xlim(xmin=0)

#shared
axes[0][0].set_ylim(ymin=0)
axes[0][0].set_ylabel(r'Mixed Use Richness $H_{0}\ _{200m}$')

plt.show()


#%% KDE plots of angular offsets
phd_util.plt_setup()

fig, axes = plt.subplots(6, 4, figsize=(8, 10))

city_pop_range = range(1, 2)

# problems:
cols = [
    'met_closeness_{dist}',
    'met_far_norm_{dist}',
    'ang_closeness_{dist}_0',
    'ang_closeness_{dist}_1',
    'ang_closeness_{dist}_10',
    'ang_far_norm_{dist}'
]
labels = [
    'Closeness $m$',
    'Farness $m$',
    'Closeness $\measuredangle$',
    'Closeness $\measuredangle+1$',
    'Closeness $\measuredangle+10$',
    'Farness $\measuredangle$'
]
for ax_row, col, label in zip(axes, cols, labels):
    for n, dist in enumerate(['200', '400', '800', '1600']):
        ax_row[n].set_xlabel(f'{label} ' + r'$d_{max}=' + f'{dist.split("_")[0]}m$')
        if n == 0:
            ax_row[n].set_ylabel(f'KDE')  # ylabel only for the first plot in each row
        phd_util.compound_kde(df_data_clean, col.format(dist=dist), city_pop_range, ax_row[n], transform=None)
plt.show()

#%% correlation strategies - EXPLORATORY ONLY
x = df_data_clean.ang_farness_200
y = df_data_clean.mixed_uses_score_hill_0_200

#from minepy import MINE
# performance penalty from around 0.5
# achieves fairly high correlation by 0.3
# mine = MINE(alpha=0.5, c=15, est='mic_approx')
# mine.compute_score(x, y)
# print('max info coef', mine.mic())

#from sklearn.metrics import mutual_info_score
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
from phd_util import rdc_cor

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


#%%
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


#%% autocorrelation EXPLORATION
import geopandas as gpd
from shapely import geometry
from pysal.weights.Distance import Kernel

geoms = [geometry.Point(xy) for xy in zip(df_data_clean['ST_X(geom)'], df_data_clean['ST_Y(geom)'])]
crs = {'init': 'epsg:27700'}
geo_df_data_clean = gpd.GeoDataFrame(df_data_clean, crs=crs, geometry=geoms)

print('generating weights')
kernel_w = Kernel.from_dataframe(geo_df_data_clean)


#%% autocorrelation EXPLORATION
# first calculate your residuals
from scipy.stats import linregress
from pysal.esda.moran import Moran

x = geo_df_data_clean.met_node_count_800
y = geo_df_data_clean.mixed_uses_score_hill_0_200

slope, intercept, rvalue, pvalue, stderr = linregress(x=np.cbrt(x), y=np.cbrt(y))
print(rvalue, pvalue)
y_predict = (slope * np.cbrt(x) + intercept)**3
plt.scatter(y, y_predict)
plt.show()

#%%
residuals = y_predict - y
plt.hist(residuals, bins=100)
plt.show()


#%%
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

#%% write to database
print('writing residuals to db')
residuals_col_name = f'residuals_met_far_norm_800_mixed_uses_score_hill_0_200'
phd_util.write_col_data(table, residuals, residuals_col_name, 'real', geo_df_data_clean.id, 'id')


#%% plot correlation table for centralities vs. uses against pearson, spearman, mutual information

from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr

cent_cols = [
    'met_node_count_200',
    'met_node_count_400',
    'met_node_count_800',
    'met_node_count_1600',
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'met_far_norm_200',
    'met_far_norm_400',
    'met_far_norm_800',
    'met_far_norm_1600',
    'ang_closeness_200_0',
    'ang_closeness_400_0',
    'ang_closeness_800_0',
    'ang_closeness_1600_0',
    'ang_closeness_200_10',
    'ang_closeness_400_10',
    'ang_closeness_800_10',
    'ang_closeness_1600_10',
    'ang_far_norm_200',
    'ang_far_norm_400',
    'ang_far_norm_800',
    'ang_far_norm_1600',
    'met_gravity_200',
    'met_gravity_400',
    'met_gravity_800',
    'met_gravity_1600',
    'met_rt_complex_200',
    'met_rt_complex_400',
    'met_rt_complex_800',
    'met_rt_complex_1600',
    'met_betw_200',
    'met_betw_400',
    'met_betw_800',
    'met_betw_1600',
    'met_betw_w_200',
    'met_betw_w_400',
    'met_betw_w_800',
    'met_betw_w_1600',
    'ang_betw_200',
    'ang_betw_400',
    'ang_betw_800',
    'ang_betw_1600'
]
cent_labels = [
    r'Node Density $_{200m}$',
    r'Node Density $_{400m}$',
    r'Node Density $_{800m}$',
    r'Node Density $_{1600m}$',
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
    r'Farness $_{200m}$',
    r'Farness $_{400m}$',
    r'Farness $_{800m}$',
    r'Farness $_{1600m}$',
    r'Closeness $\measuredangle\ _{200m}$',
    r'Closeness $\measuredangle\ _{400m}$',
    r'Closeness $\measuredangle\ _{800m}$',
    r'Closeness $\measuredangle\ _{1600m}$',
    r'Closeness $\measuredangle+10\ _{200m}$',
    r'Closeness $\measuredangle+10\ _{400m}$',
    r'Closeness $\measuredangle+10\ _{800m}$',
    r'Closeness $\measuredangle+10\ _{1600m}$',
    r'Farness $\measuredangle\ _{200m}$',
    r'Farness $\measuredangle\ _{400m}$',
    r'Farness $\measuredangle\ _{800m}$',
    r'Farness $\measuredangle\ _{1600m}$',
    r'Gravity $_{\beta=0.02}$',
    r'Gravity $_{\beta=0.01}$',
    r'Gravity $_{\beta=0.005}$',
    r'Gravity $_{\beta=0.0025}$',
    r'Route Complexity $_{200m}$',
    r'Route Complexity $_{400m}$',
    r'Route Complexity $_{800m}$',
    r'Route Complexity $_{1600m}$',
    r'Betweenness $_{200m}$',
    r'Betweenness $_{400m}$',
    r'Betweenness $_{800m}$',
    r'Betweenness $_{1600m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{400m}$',
    r'Betweenness wt. $_{800m}$',
    r'Betweenness wt. $_{1600m}$',
    r'Betweenness $\measuredangle\ _{200m}$',
    r'Betweenness $\measuredangle\ _{400m}$',
    r'Betweenness $\measuredangle\ _{800m}$',
    r'Betweenness $\measuredangle\ _{1600m}$'
]
assert len(cent_cols) == len(cent_labels)

uses_cols = [
    'mixed_uses_score_hill_0_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]
uses_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$'
]
assert len(uses_cols) == len(uses_labels)

# generate the table
table = r'''
    \begin{table}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{0.7\textheight}{!}{
    \begin{tabular}{ r | c c c c c | c c c c c | c c c c c }
    '''

# insert the column headers
for i in range(3):
    for use in uses_labels:
        table += r'''
        & \rotatebox[origin=l]{90}{''' + use + r'} '
# close the line
table += r'''
    \\
    \cline{2-16}
'''

table += r'''
    & \multicolumn{5}{ c | }{ Pearson p (cuberoot) }
    & \multicolumn{5}{ c | }{ Spearman Rank } 
    & \multicolumn{5}{ c }{ RDC } \\
    \hline
'''

count = 0
for cent, label in zip(cent_cols, cent_labels):
    count += 1
    print(label)
    x = df_data_clean[cent]
    row = label
    # pearson cuberoot
    for use in uses_cols:
        y = df_data_clean[use]
        row += f' & {round(pearsonr(np.cbrt(x[np.isfinite(x)]), np.cbrt(y[np.isfinite(x)]))[0], 2)}'
        #row += ' & x '
    # spearman rank
    for use in uses_cols:
        y = df_data_clean[use]
        row += f' & {round(spearmanr(x[np.isfinite(x)], y[np.isfinite(x)])[0], 2)}'
        #row += ' & x '
    # mutual information
    for use in uses_cols:
        y = df_data_clean[use]
        row += f' & {round(rdc_cor(x, y), 2)}'
        #row += ' & x '
    row += r' \\' # end line
    # append to data
    table += row
    if count in [4, 8, 12, 16, 20, 24, 28, 32]:
        # insert lines between batches
        table += r'''
        \hline
        '''

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{Pearson correlation, Spearman Rank correlation, and Mutual Information for Greater London network measures correlated to mixed-use measures and land-use accessibilities.}\label{table:table_corr_cent_landuses}
    \end{table}
'''

print(table) # copy and paste into included latex file

#%% plot correlation table for comparing measures to each other - VERY SLOW

from scipy.stats import spearmanr

cent_cols = [
    'met_node_count_{dist}',
    'met_far_norm_{dist}',
    'ang_far_norm_{dist}_10',
    'met_gravity_{dist}',
    'met_rt_complex_{dist}',
    'met_betw_{dist}',
    'met_betw_w_{dist}',
    'ang_betw_{dist}'
]
cent_labels = [
    'Node Density $',
    'Farness $',
    'Farness $\measuredangle+10\ ',
    'Gravity $',
    'Route Complexity $',
    'Betweenness $',
    'Betweenness wt. $',
    'Betweenness $\measuredangle\ '
]
assert len(cent_cols) == len(cent_labels)

# generate the table
table = r'''
    \begin{sidewaystable}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{\textheight}{!}{
    \begin{tabular}{ r | c c c c | c c c c | c c c c | c c c c | c c c c | c c c c | c c c c | c c c c }
    '''

# insert the column headers
for label in cent_labels:
    for d in ['200', '400', '800', '1600']:
        table += r'& \rotatebox[origin=l]{90}{' + label + r'_{' + d + 'm}$' + r'}'
# close the line
table += r'''
    \\
    \hline
'''

for c, label in zip(cent_cols, cent_labels):
    print(c)
    for d in ['200', '400', '800', '1600']:
        x = df_data_clean[c.format(dist=d)]
        row = label + r'_{' + d + 'm}$'
        # mutual information
        for cc in cent_cols:
            print(cc)
            for dd in ['200', '400', '800', '1600']:
                print(dd)
                y = df_data_clean[cc.format(dist=dd)]
                row += f' & {round(spearmanr(x[np.isfinite(x)], y[np.isfinite(x)])[0], 2)}'
                #row += ' & x'
        row += r' \\' # end line
        # append to data
        table += row
    # insert lines between batches
    table += r'''
    \hline
    '''

# remove last line
table = table[:-11]

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{Pairwise Spearman rank correlations for Greater London network measures.}\label{table:table_corr_cent_cent}
    \end{sidewaystable}
'''

print(table) # copy and paste into included latex file


#%% plot pairwise correlations for network measures - ALTERNATE TO ABOVE SLOW METHOD

phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

cols = [
    'met_node_count_200',
    'met_node_count_400',
    'met_node_count_800',
    'met_node_count_1600',
    'met_far_norm_200',
    'met_far_norm_400',
    'met_far_norm_800',
    'met_far_norm_1600',
    'ang_far_norm_200',
    'ang_far_norm_400',
    'ang_far_norm_800',
    'ang_far_norm_1600',
    'met_gravity_200',
    'met_gravity_400',
    'met_gravity_800',
    'met_gravity_1600',
    'met_rt_complex_200',
    'met_rt_complex_400',
    'met_rt_complex_800',
    'met_rt_complex_1600',
    'met_betw_200',
    'met_betw_400',
    'met_betw_800',
    'met_betw_1600',
    'met_betw_w_200',
    'met_betw_w_400',
    'met_betw_w_800',
    'met_betw_w_1600',
    'ang_betw_200',
    'ang_betw_400',
    'ang_betw_800',
    'ang_betw_1600'
]
labels = [
    r'Node Density $_{200m}$',
    r'Node Density $_{400m}$',
    r'Node Density $_{800m}$',
    r'Node Density $_{1600m}$',
    r'Farness $_{200m}$',
    r'Farness$_{400m}$',
    r'Farness$_{800m}$',
    r'Farness$_{1600m}$',
    r'Farness$\measuredangle+10\ _{200m}$',
    r'Farness$\measuredangle+10\ _{400m}$',
    r'Farness$\measuredangle+10\ _{800m}$',
    r'Farness$\measuredangle+10\ _{1600m}$',
    r'Gravity $_{\beta=0.02}$',
    r'Gravity $_{\beta=0.01}$',
    r'Gravity $_{\beta=0.005}$',
    r'Gravity $_{\beta=0.0025}$',
    r'Route Complexity $_{200m}$',
    r'Route Complexity $_{400m}$',
    r'Route Complexity $_{800m}$',
    r'Route Complexity $_{1600m}$',
    r'Betweenness $_{200m}$',
    r'Betweenness $_{400m}$',
    r'Betweenness $_{800m}$',
    r'Betweenness $_{1600m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{400m}$',
    r'Betweenness wt. $_{800m}$',
    r'Betweenness wt. $_{1600m}$',
    r'Betweenness $\measuredangle\ _{200m}$',
    r'Betweenness $\measuredangle\ _{400m}$',
    r'Betweenness $\measuredangle\ _{800m}$',
    r'Betweenness $\measuredangle\ _{1600m}$'
]
assert len(cols) == len(labels)

phd_util.pairwise_correlations(df_data_clean, ax, columns=cols, labels=labels, method='spearman')

fontsize = 7
plt.xticks(fontsize=fontsize, fontweight='bold', rotation=90)
plt.yticks(fontsize=fontsize, fontweight='bold', rotation=0)

plt.title(f'{table}: centralities', size='xx-small', fontweight='bold')
plt.show()

#%% plot correlation table for comparing selected measures to a wider range of categories - FAIRLY SLOW

from scipy.stats import spearmanr

cent_cols = [
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'met_gravity_200',
    'met_gravity_400',
    'met_gravity_800',
    'met_gravity_1600',
    'met_rt_complex_200',
    'met_rt_complex_400',
    'met_rt_complex_800',
    'met_rt_complex_1600',
    'met_betw_w_200',
    'met_betw_w_400',
    'met_betw_w_800',
    'met_betw_w_1600'
]
cent_labels = [
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
    r'Gravity $_{\beta=0.02}$',
    r'Gravity $_{\beta=0.01}$',
    r'Gravity $_{\beta=0.005}$',
    r'Gravity $_{\beta=0.0025}$',
    r'Route Complexity $_{200m}$',
    r'Route Complexity $_{400m}$',
    r'Route Complexity $_{800m}$',
    r'Route Complexity $_{1600m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{800m}$',
    r'Betweenness wt. $_{1600m}$'
]
assert len(cent_cols) == len(cent_labels)

uses_cols = [
    'mixed_uses_score_hill_0_200',
    'mixed_uses_score_hill_0_400',
    'mixed_uses_score_hill_0_800',
    'mixed_uses_score_hill_0_1600',
    'cens_tot_pop_200',
    'cens_tot_pop_400',
    'cens_tot_pop_800',
    'cens_tot_pop_1600',
    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200',
    'uses_transport_200',
    'uses_education_200',
    'uses_cultural_200',
    'uses_manufacturing_200',
    'uses_parks_200',
    'uses_accommodation_200',
    'uses_health_200',
    'uses_sports_200',
    'uses_entertainment_200',
    'uses_attractions_200',
]

uses_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Mixed Uses $H_{0}\ _{400m}$',
    r'Mixed Uses $H_{0}\ _{800m}$',
    r'Mixed Uses $H_{0}\ _{1600m}$',
    r'Population $\ _{200m}$',
    r'Population $\ _{400m}$',
    r'Population $\ _{800m}$',
    r'Population $\ _{1600m}$',
    r'Walk to work $\%$',
    r'Cycle to work $\%$',
    r'Drive to work $\%$',
    r'Transport to work $\%$',
    r'Work from home $\%$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$',
    r'Transport $\ _{\beta=0.02}$',
    r'Education $\ _{\beta=0.02}$',
    r'Cultural $\ _{\beta=0.02}$',
    r'Manufacturing $\ _{\beta=0.02}$',
    r'Parks $\ _{\beta=0.02}$',
    r'Accommodation $\ _{\beta=0.02}$',
    r'Health $\ _{\beta=0.02}$',
    r'Sports $\ _{\beta=0.02}$',
    r'Entertainment $\ _{\beta=0.02}$',
    r'Attractions $\ _{\beta=0.02}$'
]

assert len(uses_cols) == len(uses_labels)

# generate the table
table = r'''
    \begin{sidewaystable}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{\textheight}{!}{
    \begin{tabular}{ r | c c c c | c c c c | c c c c | c c c c c | c c c c c c c c c c c c c }
    '''

# insert the column headers for first batch
for use_label in uses_labels:
    table += r'''
    & \rotatebox[origin=l]{90}{''' + use_label + r'} '
# close the line
table += r'''
    \\
    \hline
'''

# first batch of columns
count = 0
for cent, label in zip(cent_cols, cent_labels):
    count += 1
    print(label)
    x = df_data_clean[cent]
    row = label
    # spearman rank
    for use in uses_cols:
        y = df_data_clean[use]
        row += f' & {round(spearmanr(x[np.isfinite(x)], y[np.isfinite(x)])[0], 2)}'
        #row += f' & x'
    row += r' \\' # end line
    # append to data
    table += row
    # insert lines between batches
    if count in [4, 8, 12]:
        table += r'''
        \hline
        '''

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{Spearman rank correlations for Greater London network measures correlated to mixed-use measures, land-use accessibilities, population intensities, and travel to work modes.}\label{table:table_corr_cent_landuses_full}
    \end{sidewaystable}
'''

print(table) # copy and paste into included latex file

#%% load 100m and 50m table data

# load data for 50m, 100m, and full networks
# full
table_full = 'analysis.roadnodes_full'
df_data_full = phd_util.load_data_as_pd_df(compound_columns, table_full, 'WHERE city_pop_id = 1')
for d in [200, 400, 800, 1600]:
    df_data_full[f'met_closeness_{d}'] = df_data_full[f'met_node_count_{d}']**2 / df_data_full[f'met_farness_{d}']
df_data_clean_full = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)
# 100m
table_100 = 'analysis.roadnodes_100'
df_data_100 = phd_util.load_data_as_pd_df(compound_columns, table_100, 'WHERE city_pop_id = 1')
for d in [200, 400, 800, 1600]:
    df_data_100[f'met_closeness_{d}'] = df_data_100[f'met_node_count_{d}']**2 / df_data_100[f'met_farness_{d}']
df_data_clean_100 = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)
# 50m
table_50 = 'analysis.roadnodes_50'
df_data_50 = phd_util.load_data_as_pd_df(compound_columns, table_50, 'WHERE city_pop_id = 1')
for d in [200, 400, 800, 1600]:
    df_data_50[f'met_closeness_{d}'] = df_data_50[f'met_node_count_{d}']**2 / df_data_50[f'met_farness_{d}']
df_data_clean_50 = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)

#%% plot correlation table for comparing selected measures to different decompositions

from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import spearmanr

cent_cols = [
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'met_gravity_200',
    'met_gravity_400',
    'met_gravity_800',
    'met_gravity_1600',
    'met_rt_complex_200',
    'met_rt_complex_400',
    'met_rt_complex_800',
    'met_rt_complex_1600',
    'met_betw_w_200',
    'met_betw_w_400',
    'met_betw_w_800',
    'met_betw_w_1600'
]
cent_labels = [
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
    r'Gravity $_{\beta=0.02}$',
    r'Gravity $_{\beta=0.01}$',
    r'Gravity $_{\beta=0.005}$',
    r'Gravity $_{\beta=0.0025}$',
    r'Route Complexity $_{200m}$',
    r'Route Complexity $_{400m}$',
    r'Route Complexity $_{800m}$',
    r'Route Complexity $_{1600m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{800m}$',
    r'Betweenness wt. $_{1600m}$'
]
assert len(cent_cols) == len(cent_labels)

uses_cols = [
    'mixed_uses_score_hill_0_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]
uses_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$',
]
assert len(uses_cols) == len(uses_labels)

# generate the table
table = r'''
    \begin{table}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{0.8\textheight}{!}{
    \begin{tabular}{ r | c c c c c | c c c c c | c c c c c | c c c c c | }
    '''

# insert the column headers
for i in range(4):
    for use in uses_labels:
        table += r'''
    & \rotatebox[origin=l]{90}{''' + use + r'''} '''
# close the line
table += r'''
    \\
    \cline{2-21}
'''

table += r'''
    & \multicolumn{20}{ c | }{ Spearman Rank } \\
    \cline{2-21}
'''

table += r'''
    & \multicolumn{5}{ c | }{ 20m decomposed } 
    & \multicolumn{5}{ c | }{ 50m decomposed }
    & \multicolumn{5}{ c | }{ 100m decomposed }
    & \multicolumn{5}{ c | }{ undecomposed }\\
    \hline
'''

# spearman rank
count = 0
for cent, label in zip(cent_cols, cent_labels):
    count += 1
    print(label)
    row = label
    for data in [df_data_clean, df_data_clean_50, df_data_clean_100, df_data_clean_full]:
        x = data[cent]
        # spearman rank
        for use in uses_cols:
            y = data[use]
            row += f' & {round(spearmanr(x[np.isfinite(x)], y[np.isfinite(x)])[0], 2)}'
            #row += ' & a'
    row += r' \\'  # end line
    # append to data
    table += row
    # insert lines between batches
    if count in [4, 8, 12]:
        table += r'''
        \hline
        '''

# add space
table += r'\noalign{\bigskip}'

table += r'''
    \cline{2-21}
    & \multicolumn{20}{ c | }{ Mutual Information } \\
    \cline{2-21}
'''

table += r'''
    & \multicolumn{5}{ c | }{ 20m decomposed }
    & \multicolumn{5}{ c | }{ 50m decomposed }
    & \multicolumn{5}{ c | }{ 100m decomposed }
    & \multicolumn{5}{ c | }{ undecomposed }\\
    \hline
'''

# mutual information
count = 0
for cent, label in zip(cent_cols, cent_labels):
    count += 1
    print(label)
    row = label
    for data in [df_data_clean, df_data_clean_50, df_data_clean_100, df_data_clean_full]:
        x = data[cent]
        # mutual information
        for use in uses_cols:
            y = data[use]
            row += f' & {round(normalized_mutual_info_score(x, y), 2)}'
            #row += ' & b'
    row += r' \\'  # end line
    # append to data
    table += row
    # insert lines between batches
    if count in [4, 8, 12]:
        table += r'''
        \hline
        '''

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{Spearman Rank correlation and Mutual Information for Greater London network measures correlated to mixed-use measures and land-use accessibilities on varying network decompositions.}\label{table:table_corr_decomp}
    \end{table}
'''

print(table) # copy and paste into included latex file