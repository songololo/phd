#%%
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from importlib import reload
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


#%% corr grid function
def pairwise_correlations(df, ax, columns=None, labels=None, city_pop_id=None):

    d = df.copy(deep=True)  # otherwise changes are made in place

    # first sort city id, won't be present after column filtering
    if city_pop_id:
        d = d[d.city_pop_id == city_pop_id]

    if columns:
        d = d[columns]

    lb = 'auto'
    if labels:
        lb = labels

    # Compute the correlation matrix - better to do own for proper preparation
    # from sklearn.metrics.cluster import normalized_mutual_info_score
    d_len = len(d.columns)
    arr = np.full((d_len, d_len), np.nan)
    for i_idx, i_col in enumerate(d.columns):
        for j_idx, j_col in enumerate(d.columns):
            x, y = phd_util.prep_xy(d[i_col], d[j_col])
            #mut_inf = normalized_mutual_info_score(x, y, average_method='arithmetic')
            arr[i_idx][j_idx] = round(stats.spearmanr(x, y)[0], 2)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(arr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = phd_util.cityseer_diverging_cmap()

    logger.info(arr)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(arr,
                ax=ax,
                cmap=cmap,
                annot=True,
                annot_kws={'size': 5.5},
                center=0,
                cbar=False,
                square=True,
                xticklabels=lb,
                yticklabels=lb,
                linewidths=.5,
                cbar_kws={"shrink": .5})


#%% columns to load
columns_d = [
    'c_gravity_{dist}',
    'c_between_wt_{dist}',
    'c_cycles_{dist}',
    'met_info_ent_{dist}',

    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    'ac_commercial_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}',

    'mu_hill_branch_wt_0_{dist}',
    'mu_hill_branch_wt_1_{dist}',

    'area_mean_wt_{dist}',
    'area_variance_wt_{dist}',
    'rate_mean_wt_{dist}',
    'rate_variance_wt_{dist}',

    'cens_tot_pop_{dist}',
    'cens_dwellings_{dist}',
    'cens_students_{dist}',
    'cens_adults_{dist}',
    'cens_employed_{dist}'
]

compound_columns = [
    'city_pop_id',
    'cens_nocars_interp',
    'cens_cars_interp',
    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp',
    'cens_density_interp'
]

for d in ['200', '400', '800', '1600']:
    compound_columns += [c.format(dist=d) for c in columns_d]


#%% load roadnodes 100 data
print(f'loading columns: {compound_columns}')
table_100 = 'ml.roadnodes_100_metrics'
corr_data_ldn_100 = phd_util.load_data_as_pd_df(db_config, compound_columns, table_100, 'WHERE city_pop_id = 1')


#%% create the coefficient of variation columns
# standard devision (sqrt(variance)) / mean
for d in [200, 400, 800, 1600]:
    corr_data_ldn_100[f'area_cov_wt_{d}'] = np.sqrt(corr_data_ldn_100[f'area_variance_wt_{d}']) / corr_data_ldn_100[f'area_mean_wt_{d}']
    corr_data_ldn_100[f'rate_cov_wt_{d}'] = np.sqrt(corr_data_ldn_100[f'rate_variance_wt_{d}']) / corr_data_ldn_100[f'rate_mean_wt_{d}']

print('cleaning data')
# be careful with cleanup...
# remove most extreme values - 0.99 gives good balance, 0.999 is stronger for some but generally weaker?
corr_data_ldn_100_clean = phd_util.remove_outliers_quantile(corr_data_ldn_100, q=(0, 0.99))
# remove rows with all nan
corr_data_ldn_100_clean = phd_util.clean_pd(corr_data_ldn_100_clean, drop_na='all')


#%% columns to plot
cols_corr = [
    'c_gravity_{dist}',
    'c_between_wt_{dist}',
    'c_cycles_{dist}',
    'met_info_ent_{dist}',

    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    'ac_commercial_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}',

    'mu_hill_branch_wt_0_{dist}',
    'mu_hill_branch_wt_1_{dist}',

    'area_mean_wt_{dist}',
    'area_cov_wt_{dist}',
    'rate_mean_wt_{dist}',
    'rate_cov_wt_{dist}',

    'cens_tot_pop_{dist}',
    'cens_density_interp',

    'cens_dwellings_{dist}',
    'cens_students_{dist}',
    'cens_employed_{dist}',

    'cens_ttw_peds_interp',
    'cens_ttw_bike_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp',
    'cens_ttw_home_interp'
]

labels_corr = [
    'Gravity',
    'Between. wt.',
    'Netw. Cycles',
    'Netw. Info.',

    'Accommod.',
    'Eating',
    'Commercial',
    'Tourism',
    'Entertain.',
    'Manuf.',
    'Retail',
    'Transport',
    'Health',
    'Education',
    'Parks',
    'Cultural',
    'Sports',

    'Mixed Uses $H_{0}$',
    'Mixed Uses $H_{1}$',

    'VOA area $\mu$',
    r'VOA area $\frac{\sigma}{\mu}$',
    'VOA rate $\mu$',
    r'VOA rate $\frac{\sigma}{\mu}$',

    'Population',
    'Density',

    'Dwellings',
    'Students',
    'Employed',

    'ttw Walk',
    'ttw Cycle',
    'ttw Motors',
    'ttw Transit',
    'ttw Home'
]


#%% plot roadnodes 100 correlations
for d in [200, 400, 800, 1600]:
    phd_util.plt_setup()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    c = [c.format(dist=d) for c in cols_corr]

    pairwise_correlations(corr_data_ldn_100_clean, ax, columns=c, labels=labels_corr)

    fontsize = 6
    plt.xticks(fontsize=fontsize, fontweight='bold', rotation=90)
    plt.yticks(fontsize=fontsize, fontweight='bold', rotation=0)
    plt.title(f'{table_100}: spearman ' + r'$\rho$' + f' correlations at distance: {d}m', size='xx-small', fontweight='bold')
    plt.show()


#%% load roadnodes 20 data
print(f'loading columns: {compound_columns}')
table_20 = 'ml.roadnodes_20_metrics'
corr_data_ldn_20 = phd_util.load_data_as_pd_df(db_config, compound_columns, table_20, 'WHERE city_pop_id = 1')


#  %% create the coefficient of variation columns
# standard devision (sqrt(variance)) / mean
print('generating cof of var columns')
for d in [200, 400, 800, 1600]:
    corr_data_ldn_20[f'area_cov_wt_{d}'] = np.sqrt(corr_data_ldn_20[f'area_variance_wt_{d}']) / corr_data_ldn_20[f'area_mean_wt_{d}']
    corr_data_ldn_20[f'rate_cov_wt_{d}'] = np.sqrt(corr_data_ldn_20[f'rate_variance_wt_{d}']) / corr_data_ldn_20[f'rate_mean_wt_{d}']

print('cleaning data')
# be careful with cleanup...
# remove most extreme values - 0.99 gives good balance, 0.999 is stronger for some but generally weaker?
corr_data_ldn_20_clean = phd_util.remove_outliers_quantile(corr_data_ldn_20, q=(0, 0.99))
# remove rows with all nan
corr_data_ldn_20_clean = phd_util.clean_pd(corr_data_ldn_20_clean)


#  %% plot roadnodes 20 correlations
for d in [200, 400, 800, 1600]:
    phd_util.plt_setup()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    c = [c.format(dist=d) for c in cols_corr]

    pairwise_correlations(corr_data_ldn_20_clean, ax, columns=c, labels=labels_corr)

    fontsize = 6
    plt.xticks(fontsize=fontsize, fontweight='bold', rotation=90)
    plt.yticks(fontsize=fontsize, fontweight='bold', rotation=0)
    plt.title(f'{table_20}: spearman ' + r'$\rho$' + f' correlations at distance: {d}m', size='xx-small', fontweight='bold')
    plt.show()


#  %% pairwise scatter columns
scatter_cols_corr = [
    'c_gravity_{dist}',
    'cens_tot_pop_{dist}',
    'rate_mean_wt_{dist}',
    'mu_hill_branch_wt_0_{dist}']

scatter_cols_corr_labels = [
    [r'Gravity $d_{max}=', '{dist}$'],
    [r'Population $d_{max}=', '{dist}$'],
    [r'Rates $d_{max}=', '{dist}$'],
    [r'Mixed Uses $H_{0}$ $d_{max}=', '{dist}$']
]


#pairwise scatter plots
# density works well at small scale, but at larger scales aggregate population counts works better
cmap = phd_util.cityseer_cmap()

for d in [200, 400, 800, 1600]:

    # using 0.99 for above correlations but 0.999 here for fuller picture of data
    corr_data_ldn_100_lite_clean = phd_util.remove_outliers_quantile(corr_data_ldn_100, q=(0, 0.999))
    # remove rows with all nan -> in this case drop rows with any otherwise histogram issues
    corr_data_ldn_100_lite_clean = phd_util.clean_pd(corr_data_ldn_100_lite_clean, drop_na='any')

    phd_util.plt_setup()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    cols = [c.format(dist=d) for c in scatter_cols_corr]
    col_labels = [c[0] + c[1].format(dist=d) for c in scatter_cols_corr_labels]
    y_logs = [False, False, True, False]
    x_logs = [False, False, True, False]

    for y, ax_row in enumerate(axes):
        for x, ax in enumerate(ax_row):
            # for last ax row, set x labels
            if y == 3:
                ax.set_xlabel(col_labels[x])
            # for first ax in each ax row, set y labels
            if x == 0:
                ax.set_ylabel(col_labels[y])
            # set x and y data
            x_data = corr_data_ldn_100_lite_clean[cols[x]]
            y_data = corr_data_ldn_100_lite_clean[cols[y]]
            # set colour
            # NOTE -> removed, confusing...
            # col = np.clip(y_data, 0, np.percentile(y_data, 99))
            # set y logged axes - this doesn't matter for hist or scatter
            # if y_logs[y]:
                # ax.set_yscale('log')
                # also transform colour accordingly
                # removed colour per above
                # col = np.log(col)
            # if i = j it is a diagonal ax

            if y == x:
                ax.hist(x_data, bins=100, linewidth=0.2, color='#0064b7', edgecolor='white')
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
                if x_logs[x]:
                    ax.set_xlim(left=1)
                    ax.set_xscale('log')
            else:
                ax.set_xlim(0, x_data.max())
                ax.set_ylim(0, y_data.max())
                ax.scatter(x_data, y_data, marker='.', s=3, c='#0064b7', alpha=0.05, edgecolors='none')
                if x_logs[x]:
                    ax.set_xlim(1, x_data.max())
                    ax.set_xscale('log')
                if y_logs[y]:
                    ax.set_ylim(1, y_data.max())
                    ax.set_yscale('log')
    plt.show()
