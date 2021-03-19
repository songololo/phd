# %%


import numpy as np
from src import phd_util



# %% columns to load
# weighted
columns = [
    'city_pop_id',
    'mixed_uses_score_hill_0_200_002',
    'mixed_uses_score_hill_10_200_002',
    'uses_commercial_200_002',
    'uses_retail_200_002',
    'uses_eating_200_002'
]

# non weighted
columns_nw = [
    'ang_node_count_{dist}',
    'ang_farness_{dist}',
    'ang_betw_{dist}'
]

for d in ['200', '400', '800', '1600']:
    columns += [c.format(dist=d) for c in columns_nw]

# %%
print(f'loading columns: {columns}')
table = 'analysis.roadnodes_20_dual'
df_data_dual = phd_util.load_data_as_pd_df(columns, table, 'WHERE city_pop_id::int = 1')

# %% create the closeness columns
for d in [200, 400, 800, 1600]:
    df_data_dual[f'ang_closeness_{d}_0'] = df_data_dual[f'ang_node_count_{d}'] ** 2 / (df_data_dual[f'ang_farness_{d}'])
    df_data_dual[f'ang_closeness_{d}_1'] = df_data_dual[f'ang_node_count_{d}'] ** 2 / (
            df_data_dual[f'ang_farness_{d}'] + df_data_dual[f'ang_node_count_{d}'])
    df_data_dual[f'ang_closeness_{d}_10'] = df_data_dual[f'ang_node_count_{d}'] ** 2 / (
            df_data_dual[f'ang_farness_{d}'] + df_data_dual[f'ang_node_count_{d}'] * 10)

# %%
print('cleaning data')
# be careful with cleanup...
df_data_dual_clean = phd_util.clean_pd(df_data_dual, drop_na='all', fill_inf=np.nan)

# %% plot correlation table

from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr

cent_cols = [
    'ang_closeness_200_0',
    'ang_closeness_400_0',
    'ang_closeness_800_0',
    'ang_closeness_1600_0',
    'ang_closeness_200_10',
    'ang_closeness_400_10',
    'ang_closeness_800_10',
    'ang_closeness_1600_10',
    'ang_betw_200',
    'ang_betw_400',
    'ang_betw_800',
    'ang_betw_1600'
]
cent_labels = [
    r'Closeness $\measuredangle\ _{200m}$',
    r'Closeness $\measuredangle\ _{400m}$',
    r'Closeness $\measuredangle\ _{800m}$',
    r'Closeness $\measuredangle\ _{1600m}$',
    r'Closeness $\measuredangle+10\ _{200m}$',
    r'Closeness $\measuredangle+10\ _{400m}$',
    r'Closeness $\measuredangle+10\ _{800m}$',
    r'Closeness $\measuredangle+10\ _{1600m}$',
    r'Betweenness $\measuredangle\ _{200m}$',
    r'Betweenness $\measuredangle\ _{400m}$',
    r'Betweenness $\measuredangle\ _{800m}$',
    r'Betweenness $\measuredangle\ _{1600m}$'
]
assert len(cent_cols) == len(cent_labels)

uses_cols = [
    'mixed_uses_score_hill_0_200',
    'mixed_uses_score_hill_10_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]
uses_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Mixed Uses $H_{1}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$'
]
assert len(uses_cols) == len(uses_labels)

# generate the table
table = r'''
    \begin{table}[htbp!]
    \centering
    \makebox[\textwidth]{
    \resizebox{0.7\textheight}{!}{
    \begin{tabular}{ r | c c c c c | c c c c c | c c c c c }
    '''

# insert the column headers
for i in range(3):
    for use in uses_labels:
        table += r'''
    & \rotatebox[origin=l]{90}{''' + use + r'''} '''
# close the line
table += r'''
    \\
    \cline{2-16}
'''

table += r'''
    & \multicolumn{5}{ c | }{ Pearson p (cuberoot) }
    & \multicolumn{5}{ c | }{ Spearman Rank } 
    & \multicolumn{5}{ c }{ Mutual Information } \\
    \hline
'''

count = 0
for cent, label in zip(cent_cols, cent_labels):
    count += 1
    print(label)
    x = df_data_dual[cent]
    row = label
    # pearson cuberoot
    for use in uses_cols:
        y = df_data_dual[use]
        row += f' & {round(pearsonr(np.cbrt(x[np.isfinite(x)]), np.cbrt(y[np.isfinite(x)]))[0], 2)}'
    # spearman rank
    for use in uses_cols:
        y = df_data_dual[use]
        row += f' & {round(spearmanr(x[np.isfinite(x)], y[np.isfinite(x)])[0], 2)}'
    # mutual information
    for use in uses_cols:
        y = df_data_dual[use]
        row += f' & {round(normalized_mutual_info_score(x, y), 2)}'
    row += r' \\'  # end line
    # append to data
    table += row
    if count in [4, 8]:
        # insert lines between batches
        table += r'''
        \hline
        '''

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{Pearson correlation, Spearman Rank correlation, and Mutual Information for Greater London angular network measures correlated to mixed-use measures and land-use accessibilities. Calculated directly on the dual-graph.}\label{table:table_corr_cent_landuses_dual}
    \end{table}
'''

print(table)  # copy and paste into included latex file
