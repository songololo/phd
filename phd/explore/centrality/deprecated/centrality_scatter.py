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
    'mixed_uses_score_hill_10_{dist}',
    'uses_commercial_{dist}',
    'uses_retail_{dist}',
    'uses_eating_{dist}',
    'uses_transport_{dist}',
    'uses_education_{dist}',
    'uses_cultural_{dist}',
    'uses_manufacturing_{dist}',
    'uses_parks_{dist}',
    # 'uses_accommodation_{dist}',
    # 'uses_health_{dist}',
    # 'uses_sports_{dist}',
    # 'uses_entertainment_{dist}',
    # 'uses_attractions_{dist}',
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
    'city_pop_id',
    'cens_ttw_peds_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp'

    #'cens_nocars_interp',
    #'cens_cars_interp',
    #'cens_ttw_bike_interp',
    #'cens_ttw_home_interp'
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
    # angular farness needs to be transformed
    df_data[f'ang_closeness_{d}_0'] = df_data[f'ang_node_count_{d}'] ** 2 / (df_data[f'ang_farness_{d}'])
    df_data[f'ang_closeness_{d}_1'] = df_data[f'ang_node_count_{d}'] ** 2 / (df_data[f'ang_farness_{d}'] + df_data[f'ang_node_count_{d}'])
    df_data[f'ang_closeness_{d}_10'] = df_data[f'ang_node_count_{d}']**2 / (df_data[f'ang_farness_{d}'] + df_data[f'ang_node_count_{d}'] * 10)


#%% clean data
print('cleaning data')
# be careful with cleanup...
# angular values on dual are averaged out to primary vertices, so dead-ends result in null values
# preferably drop only rows where nan is all
df_data_clean = phd_util.clean_pd(df_data, drop_na='all', fill_inf=np.nan)
# corr_data_ldn_clean = util.remove_outliers_quantile(corr_data_ldn_clean, q=(0, 0.99))


#%% KDE plots of centrality distributions - EXPLORATORY ONLY
phd_util.plt_setup()

fig, axes = plt.subplots(8, 4, figsize=(8, 10))

city_pop_range = range(1, 2)

# problems:
cols = [
    'met_node_count_{dist}',
    'met_closeness_{dist}',
    'ang_closeness_{dist}_10',
    'met_gravity_{dist}',
    'met_rt_complex_{dist}',
    'met_betw_{dist}',
    'met_betw_w_{dist}',
    'ang_betw_{dist}'
]
labels = [
    'Node Count',
    'Closeness $m$',
    'Closeness$°$ + 10',
    'Gravity $m$',
    'Rt. Complex.',
    'Betw. $m$',
    'Betw. $m\ wt.$',
    'Betw. $°$'
]
for ax_row, col, label in zip(axes, cols, labels):
    for n, dist in enumerate(['200', '400', '800', '1600']):
        ax_row[n].set_xlabel(f'{label} ' + r'$d_{max}=' + f'{dist.split("_")[0]}m$')
        if n == 0:
            ax_row[n].set_ylabel(f'KDE')  # ylabel only for the first plot in each row
        phd_util.compound_kde(df_data_clean, col.format(dist=dist), city_pop_range, ax_row[n], transform=None)
plt.show()


#%% KDE plots of landuses distributions - EXPLORATORY ONLY
phd_util.plt_setup()

fig, axes = plt.subplots(10, 4, figsize=(8, 10))

city_pop_range = range(1, 2)

# problems:
cols = [
    'mixed_uses_score_hill_0_200',
    'mixed_uses_score_hill_10_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200',
    'cens_tot_pop_200',
    'cens_tot_pop_1600',
    'cens_ttw_peds_interp',
    'cens_ttw_motors_interp',
    'cens_ttw_pubtrans_interp'
]
labels = [
    'Mixed $H_{0}$',
    'Mixed $H_{1}$',
    'Commerc. $m$',
    'Retail $m$',
    'Eating $m$',
    'Population $m$',
    'Population $m$',
    'ttw Pedestrians',
    'ttw Motorvehicle',
    'ttw Transport'
]
for ax_row, col, label in zip(axes, cols, labels):
    for n, dist in enumerate(['200', '400', '800', '1600']):
        ax_row[n].set_xlabel(f'{label} ' + r'$d_{max}=' + f'{dist.split("_")[0]}m$')
        if n == 0:
            ax_row[n].set_ylabel(f'KDE')  # ylabel only for the first plot in each row
        # use global percentiles for long-tail distributed
        phd_util.compound_kde(df_data_clean, col.format(dist=dist), city_pop_range, ax_row[n])
plt.show()


#%% Scatter plots for exploring puzzling correlations for pearson vs. mutual information
cols_v = [
    #'ang_closeness_{dist}_10'
    'met_node_count_{dist}'
]

cols_v_labels = [
    #'Closeness $\measuredangle+10\ '
    'Node Count ',
]

cols_h = [
    'mixed_uses_score_hill_0_200',
    'mixed_uses_score_hill_10_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]

cols_h_labels = [
    'Mixed Uses $H_{0}\ _{200m}$',
    'Mixed Uses $H_{1}\ _{200m}$',
    'Commercial $_{200m}$',
    'Retail $_{200m}$',
    'Eat & Drink $_{200m}$'
]

# using 0.99 for above correlations but 0.999 here for fuller picture of data
#corr_data_ldn_100_lite_clean = util.remove_outliers_quantile(df_data_clean, q=(0, 0.999))
# remove rows with all nan -> in this case drop rows with any otherwise histogram issues
#corr_data_ldn_100_lite_clean = util.clean_pd(corr_data_ldn_100_lite_clean, drop_na='any')

phd_util.plt_setup()

fig, axes = plt.subplots(5, 4, figsize=(8, 8), sharex='col', sharey='row')

# plot pairwise scatter plots
v_count = 0
for d in [200, 400, 800, 1600]:

    for col_v, label_v in zip(cols_v, cols_v_labels):

        col_v = col_v.format(dist=d)
        label_v = label_v + r'_{' + str(d) + r'm}$'

        h_count = 0
        for col_h, label_h in zip(cols_h, cols_h_labels):

            print(v_count, h_count)
            print(col_v, col_h)

            # v column is on x axis
            x_data = np.cbrt(df_data_clean[col_v])
            y_data = np.cbrt(df_data_clean[col_h])

            axes[h_count][v_count].scatter(x_data, y_data, marker='.', s=8, alpha=0.1, edgecolors='none')

            ymin = 0
            if col_h in ['mixed_uses_score_hill_0_200', 'mixed_uses_score_hill_10_200']:
                ymin = 1
            axes[h_count][v_count].set_xlim(xmin=0)
            axes[h_count][v_count].set_ylim(ymin=ymin)
            axes[h_count][v_count].set_xlabel(label_v)
            axes[h_count][v_count].set_ylabel(label_h)

            h_count += 1

        v_count += 1

plt.show()



    # for y, ax_row in enumerate(axes):
    #     for x, ax in enumerate(ax_row):
    #         # for last ax row, set x labels
    #         if y == 3:
    #             ax.set_xlabel(col_labels[x])
    #         # for first ax in each ax row, set y labels
    #         if x == 0:
    #             ax.set_ylabel(col_labels[y])
    #         # set x and y data
    #         x_data = df_data_clean[cols[x]]
    #         y_data = df_data_clean[cols[y]]
    #         # set colour
    #         col = np.clip(y_data, 0, np.percentile(y_data, 99))
    #         # set y logged axes - this doesn't matter for hist or scatter
    #         if y_logs[y]:
    #             ax.set_yscale('log')
    #             # also transform colour accordingly
    #             col = np.log(col)
    #         # if i = j it is a diagonal ax
    #         if y == x:
    #             ax.hist(x_data, bins=100, linewidth=0.2, color=plt.cm.viridis(0.35), edgecolor='white')
    #         else:
    #             ax.set_ylim(round(np.percentile(y_data, 0.01)), y_data.max())
    #             ax.set_xlim(round(np.percentile(x_data, 0.01)), x_data.max())
    #             # set x logged axes only if not hist
    #             if x_logs[x]:
    #                 ax.set_xscale('log')
    #             ax.scatter(x_data, y_data, marker='.', s=8, c=col, alpha=0.2, edgecolors='none')
    # plt.show()
