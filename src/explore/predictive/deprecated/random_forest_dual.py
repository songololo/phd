# %%
import logging


import numpy as np
from src import phd_util
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# %% columns to load
# weighted
columns = [
    'city_pop_id',
    'mixed_uses_score_hill_0_200_002',
    'uses_commercial_200_002',
    'uses_retail_200_002',
    'uses_eating_200_002'
]

# non weighted
columns_nw = [
    'dual_met_betweenness_{d}',
    'dual_ang_betweenness_{d}',
    'dual_node_count_{d}',
    'dual_met_farness_{d}',
    'dual_ang_farness_{d}',
    'dual_ang_farness_m_{d}',
    'dual_ratio_{d}'
]

# skip smaller distances because correlations break down
# e.g. 50m causes a lot of rows to be dropped unnecessarily
distances = [200, 300, 400, 600, 800, 1200, 1600]

for d in distances:
    columns += [c.format(d=d) for c in columns_nw]

# %%
print(f'loading columns: {columns}')
df_data_dual_full = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_full_dual', 'WHERE city_pop_id = 1')
df_data_dual_100 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_100_dual', 'WHERE city_pop_id = 1')
df_data_dual_50 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_50_dual', 'WHERE city_pop_id = 1')
df_data_dual_20 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_20_dual', 'WHERE city_pop_id = 1')

# %%
print('cleaning data')
# be careful with cleanup...
df_data_dual_full_clean = phd_util.clean_pd(df_data_dual_full, drop_na='all', fill_inf=np.nan)
df_data_dual_100_clean = phd_util.clean_pd(df_data_dual_100, drop_na='all', fill_inf=np.nan)
df_data_dual_50_clean = phd_util.clean_pd(df_data_dual_50, drop_na='all', fill_inf=np.nan)
df_data_dual_20_clean = phd_util.clean_pd(df_data_dual_20, drop_na='all', fill_inf=np.nan)

# %% sample the data so that primal and dual have same number of items
print('sampling (to match primal quantity of vertices)')
df_data_dual_full_clean = df_data_dual_full_clean.sample(n=167941, random_state=0)
df_data_dual_100_clean = df_data_dual_100_clean.sample(n=252777, random_state=0)
df_data_dual_50_clean = df_data_dual_50_clean.sample(n=433409, random_state=0)
df_data_dual_20_clean = df_data_dual_20_clean.sample(n=991629, random_state=0)

# %% generate farness data
for d in distances:
    df_data_dual_full_clean[f'dual_imp_met_far_{d}'] = df_data_dual_full_clean[f'dual_met_farness_{d}'] / \
                                                       df_data_dual_full_clean[f'dual_node_count_{d}'] ** 2
    df_data_dual_100_clean[f'dual_imp_met_far_{d}'] = df_data_dual_100_clean[f'dual_met_farness_{d}'] / \
                                                      df_data_dual_100_clean[f'dual_node_count_{d}'] ** 2
    df_data_dual_50_clean[f'dual_imp_met_far_{d}'] = df_data_dual_50_clean[f'dual_met_farness_{d}'] / \
                                                     df_data_dual_50_clean[f'dual_node_count_{d}'] ** 2
    df_data_dual_20_clean[f'dual_imp_met_far_{d}'] = df_data_dual_20_clean[f'dual_met_farness_{d}'] / \
                                                     df_data_dual_20_clean[f'dual_node_count_{d}'] ** 2

    df_data_dual_full_clean[f'dual_imp_ang_far_{d}'] = df_data_dual_full_clean[f'dual_ang_farness_{d}'] / \
                                                       df_data_dual_full_clean[f'dual_node_count_{d}'] ** 2
    df_data_dual_100_clean[f'dual_imp_ang_far_{d}'] = df_data_dual_100_clean[f'dual_ang_farness_{d}'] / \
                                                      df_data_dual_100_clean[f'dual_node_count_{d}'] ** 2
    df_data_dual_50_clean[f'dual_imp_ang_far_{d}'] = df_data_dual_50_clean[f'dual_ang_farness_{d}'] / \
                                                     df_data_dual_50_clean[f'dual_node_count_{d}'] ** 2
    df_data_dual_20_clean[f'dual_imp_ang_far_{d}'] = df_data_dual_20_clean[f'dual_ang_farness_{d}'] / \
                                                     df_data_dual_20_clean[f'dual_node_count_{d}'] ** 2

# %% compute pairwise random forest r^2 for decompositions and measures, repeat for mixed uses, commercial, retail, eat & drink

targets = [
    'mixed_uses_score_hill_0_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200']
target_labels = [
    'mixed_uses',
    'commercial',
    'retail',
    'eat_drink'
]
target_friendly_labels = [
    'Mixed Uses',
    'Commercial',
    'Retail',
    'Eating \& Drink'
]

themes = [
    'dual_node_count_{d}',
    'dual_imp_met_far_{d}',
    'dual_imp_ang_far_{d}',
    'dual_met_betweenness_{d}',
    'dual_ang_betweenness_{d}'
]
theme_labels = [
    'node_density',
    'met_farness',
    'ang_farness',
    'met_betw',
    'ang_betw'
]

tables = [
    df_data_dual_full_clean,
    df_data_dual_100_clean,
    df_data_dual_50_clean,
    df_data_dual_20_clean
]
table_labels = ['full', '100m', '50m', '20m']

results = {}

for target, target_label in zip(targets, target_labels):

    logger.info(f'TARGET COLUMN: {target_label}')
    results[target_label] = {}

    for table, table_label in zip(tables, table_labels):

        logger.info(f'CURRENT TABLE: {table_label}')
        results[target_label][table_label] = {}

        for theme, theme_label in zip(themes, theme_labels):

            cols = []
            for d in distances:
                cols.append(theme.format(d=d))

            # drop rows where any non finite values for target column
            data = table.copy(deep=True)
            start_len = len(data)
            data = data[np.isfinite(data[target])]
            if len(data) != start_len:
                logger.warning(
                    f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in target: {target}')

            # drop rows where any non finite rows for input columns
            for col in cols:
                start_len = len(data)
                data = data[np.isfinite(data[col])]
                if len(data) != start_len:
                    logger.warning(
                        f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

            data_target = data[target]
            data_input = data[cols]

            pca = PCA(n_components=6)
            data_input_reduced = pca.fit_transform(data_input)

            training_inputs, testing_inputs, training_targets, testing_targets = \
                train_test_split(data_input_reduced, data_target, train_size=0.8)

            # create
            random_forest = RandomForestRegressor(
                n_jobs=-1,
                max_features='auto',
                # max_depth=20,
                n_estimators=50
            )

            # fit
            random_forest.fit(training_inputs, training_targets)
            # test set prediction
            pred_test = random_forest.predict(testing_inputs)
            # insert into dict
            acc = metrics.r2_score(testing_targets, pred_test)
            results[target_label][table_label][theme_label] = acc
            logger.info(
                f'fitted table: {table_label} and target: {target_label} to r^2 accuracy: {round(acc * 100, 2)} for {theme_label}')

# %% build the table

table = r'''
\begin{table}[htb!]
\centering
\makebox[\textwidth]{
\resizebox{0.8\textheight}{!}{
\begin{tabular}{ r | c c c c c | c c c c c | c c c c c | c c c c c }
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Farness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Farness}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Betweenness} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Farness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Farness}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Betweenness} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Farness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Farness}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Betweenness} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Farness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Farness}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{$\measuredangle$ Betweenness} \\
\cline{2-21}
$r^{2}\ \%$
& \multicolumn{5}{ c | }{ Mixed Uses $200m$ }
& \multicolumn{5}{ c | }{ Commercial $\ _{\beta=0.02}$ }
& \multicolumn{5}{ c | }{ Retail $\ _{\beta=0.02}$ }
& \multicolumn{5}{ c }{ Eating \& Drinking $\ _{\beta=0.02}$ }\\
\hline'''

for table_label in table_labels:

    # add the table resolution
    table += r'''
            ''' + table_label

    for target_label in target_labels:

        for theme_label in theme_labels:
            table += r'''
                & ''' + str(round(results[target_label][table_label][theme_label] * 100, 1)) + r''

    # close the line
    table += r' \\'

# close the table
table += r'''
\end{tabular}
}}\caption{Example random forest ML $r^{2}$ prediction accuracies for landuses as calculated on dual graphs. (50 estimators, 6 PCA components. Downsampled to match node quantities on the primal graph.)}\label{table:pred_dual_comparisons}
\end{table}
'''

print(table)

# %% calculate the accuracies and feature importances
cols_ml = [
    'dual_node_count_{d}',
    'dual_imp_met_far_{d}',
    'dual_imp_ang_far_{d}',
    'dual_met_betweenness_{d}',
    'dual_ang_betweenness_{d}'
]
cols_ml_labels = [
    r'Node Density',
    r'Farness',
    r'$\measuredangle$ Farness',
    r'Betweenness',
    r'$\measuredangle$ Betweenness'
]

assert len(cols_ml) == len(cols_ml_labels)

logger.info(f'columns: {cols_ml}')

# TARGETS:
target_cols = [
    'mixed_uses_score_hill_0_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]

target_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$'
]

assert len(target_cols) == len(target_labels)

results_dict = {}
for target_col, target_label in zip(target_cols, target_labels):

    logger.info(f'target column: {target_col}')

    # drop rows where any nan for target column
    data = df_data_dual_20_clean.copy(deep=True)

    start_len = len(data)
    data = data[np.isfinite(data[target_col])]
    if len(data) != start_len:
        logger.warning(
            f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in target: {target_label}')

    # drop rows where any non finite rows for input columns
    for col in cols_ml:
        # process all distances:
        for d in distances:
            start_len = len(data)
            data = data[np.isfinite(data[col.format(d=d)])]
            if len(data) != start_len:
                logger.warning(
                    f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

    data_target = data[target_col]

    # process each theme individually
    pca_data = None
    for col in cols_ml:
        _cols = []
        for d in distances:
            _cols.append(col.format(d=d))
        # calculate PCA accuracy to control for curse of dimensionality
        pca = PCA(n_components=1)
        _data = data[_cols]
        _pca = pca.fit_transform(_data)
        if pca_data is None:
            pca_data = _pca
        else:
            pca_data = np.hstack((pca_data, _pca))

    training_inputs, testing_inputs, training_targets, testing_targets = \
        train_test_split(pca_data, data_target, train_size=0.8)

    # create
    random_forest = RandomForestRegressor(
        n_jobs=-1,
        max_features='auto',
        n_estimators=100
    )

    # fit
    random_forest.fit(training_inputs, training_targets)
    # test set prediction
    pred_test = random_forest.predict(testing_inputs)

    # insert into dict
    results_dict[target_col] = {
        'target': target_label,
        'accuracy': metrics.r2_score(testing_targets, pred_test),
        'features': [None] * training_inputs.shape[1],
        'scores': [None] * training_inputs.shape[1]
    }

    importances = random_forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices):
        results_dict[target_col]['features'][i] = cols_ml_labels[idx]
        results_dict[target_col]['scores'][i] = importances[idx]

print(results_dict)

# %% build the table

table = r'''
    \begin{table}[htb!]
    \centering
    \makebox[\textwidth]{
    \resizebox{0.8\textheight}{!}{
    \begin{tabular}{ r | r l | r l | r l | r l }
'''

table += r'''
    Targets'''

# add the headers
for v in results_dict.values():
    table += r'''
    & \multicolumn{2}{ c }{ ''' + v['target'] + ' }'

# close and insert a line
table += r' \\'

table += r'''
    $r^{2}$ Accuracies'''

# add the accuracies
for v in results_dict.values():
    table += r'''
    & \multicolumn{2}{ c }{ ''' + str(round(v['accuracy'] * 100, 2)) + r'\% }'

# close and insert a line
table += r' \\'
table += r'''
    \hline'''

table += r'''
    Importances'''

# add the feature importances
for i in range(5):
    for v in results_dict.values():
        table += r'''
        & ''' + str(round(v['scores'][i] * 100, 2)) + r'\% & ' + v['features'][i]
    # close the line
    table += r' \\'

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{100 estimators. Feature importances derived from random forest regression on the dual graph for different network measures. (Multiple distances from $200m$ to $1600m$ reduced using PCA to a single dimension per measure.)}\label{table:table_random_forest_pred_dual}
    \end{table}
'''

print(table)

# %%
##################
# TESTING: compute pairwise random forest r^2 for decompositions and measures, repeat for mixed uses, commercial, retail, eat & drink

targets = [
    'uses_commercial_200']
target_labels = [
    'commercial'
]
target_friendly_labels = [
    'Commercial'
]

themes = [
    'dual_node_count_{d}',
    'dual_imp_met_far_{d}',
    'dual_met_betweenness_{d}'
]
theme_labels = [
    'node_density',
    'met_farness',
    'met_betw'
]

tables = [
    df_data_dual_full_clean,
    df_data_dual_20_clean
]
table_labels = ['full', '20m']

results = {}

# skip smaller distances because correlations break down
# 50m causes a lot of rows to be dropped unnecessarily
distances = [200, 300, 400, 600, 800, 1200, 1600]

for target, target_label in zip(targets, target_labels):

    logger.info(f'TARGET COLUMN: {target_label}')
    results[target_label] = {}

    for table, table_label in zip(tables, table_labels):

        logger.info(f'CURRENT TABLE: {table_label}')
        results[target_label][table_label] = {}

        for theme, theme_label in zip(themes, theme_labels):

            cols = []
            for d in distances:
                cols.append(theme.format(d=d))

            # drop rows where any non finite values for target column
            data = table.copy(deep=True)
            start_len = len(data)
            data = data[np.isfinite(data[target])]
            if len(data) != start_len:
                logger.warning(
                    f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in target: {target}')

            # drop rows where any non finite rows for input columns
            for col in cols:
                start_len = len(data)
                data = data[np.isfinite(data[col])]
                if len(data) != start_len:
                    logger.warning(
                        f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

            data_target = data[target]
            data_input = data[cols]

            pca = PCA(n_components=6)
            data_input_reduced = pca.fit_transform(data_input)

            training_inputs, testing_inputs, training_targets, testing_targets = \
                train_test_split(data_input_reduced, data_target, train_size=0.8)

            # create
            random_forest = RandomForestRegressor(
                n_jobs=-1,
                max_features='auto',
                # max_depth=20,
                n_estimators=50
            )

            # fit
            random_forest.fit(training_inputs, training_targets)
            # test set prediction
            pred_test = random_forest.predict(testing_inputs)
            # insert into dict
            acc = metrics.r2_score(testing_targets, pred_test)
            results[target_label][table_label][theme_label] = acc
            logger.info(
                f'fitted table: {table_label} and target: {target_label} to r^2 accuracy: {round(acc * 100, 2)} for {theme_label}')

'''
# no max, n estimators 50

INFO:builtins:TARGET COLUMN: commercial

INFO:builtins:CURRENT TABLE: full
WARNING:builtins:NOTE -> dropped 0.01% rows due to non-finite values in target: uses_commercial_200
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: -6.91 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 7.05 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 28.35 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 32.9 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 29.86 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 30.53 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 27.79 for met_farness

INFO:builtins:CURRENT TABLE: 20m
WARNING:builtins:NOTE -> dropped 0.0% rows due to non-finite values in target: uses_commercial_200
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: -8.66 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 13.44 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 30.93 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 46.58 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 49.3 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 58.37 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 53.77 for met_farness

INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 23.7 for node_density
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 12.87 for node_density
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 26.87 for node_density
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 29.36 for node_density
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 35.23 for node_density
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 36.53 for node_density
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 35.62 for node_density

INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 48.8 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 14.99 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 29.71 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 44.6 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 54.53 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 55.47 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 56.13 for node_density

INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: -36.78 for met_betw
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: -6.03 for met_betw
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 5.4 for met_betw
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 11.33 for met_betw
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 9.85 for met_betw
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 10.4 for met_betw
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 10.49 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: -36.3 for met_betw
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: -0.24 for met_betw
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 14.54 for met_betw
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 22.23 for met_betw
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 29.72 for met_betw
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 29.74 for met_betw
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 31.2 for met_betw


## max depth 10, n estimators 100

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 23.67 for node_density
WARNING:builtins:NOTE -> dropped 53.9% rows due to non-finite values in var: dual_imp_met_far_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 25.35 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 12.47 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 35.2 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 31.66 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 12.61 for met_betw

## max depth None, n estimators 100 -> SLOW for 20m

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 29.73 for node_density
WARNING:builtins:NOTE -> dropped 53.9% rows due to non-finite values in var: dual_imp_met_far_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 37.11 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 7.16 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 46.0 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 42.1 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 20.63 for met_betw

## max depth 5, n estimators 100

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 22.2 for node_density
WARNING:builtins:NOTE -> dropped 53.9% rows due to non-finite values in var: dual_imp_met_far_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 27.52 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 12.72 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 26.61 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 26.3 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 10.02 for met_betw

## max depth 5, n estimators 50

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 27.75 for node_density
WARNING:builtins:NOTE -> dropped 53.9% rows due to non-finite values in var: dual_imp_met_far_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 29.69 for met_farness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 10.65 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 25.22 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 27.8 for met_farness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 8.74 for met_betw
'''

# %% test PCA impact on joint accuracies
cols_ml = [
    'dual_node_count_200',
    'dual_node_count_300',
    'dual_node_count_400',
    'dual_node_count_600',
    'dual_node_count_800',
    'dual_node_count_1200',
    'dual_node_count_1600',
    'dual_imp_met_far_200',
    'dual_imp_met_far_300',
    'dual_imp_met_far_400',
    'dual_imp_met_far_600',
    'dual_imp_met_far_800',
    'dual_imp_met_far_1200',
    'dual_imp_met_far_1600',
    'dual_imp_ang_far_200',
    'dual_imp_ang_far_300',
    'dual_imp_ang_far_400',
    'dual_imp_ang_far_600',
    'dual_imp_ang_far_800',
    'dual_imp_ang_far_1200',
    'dual_imp_ang_far_1600',
    'dual_met_betweenness_200',
    'dual_met_betweenness_300',
    'dual_met_betweenness_400',
    'dual_met_betweenness_600',
    'dual_met_betweenness_800',
    'dual_met_betweenness_1200',
    'dual_met_betweenness_1600',
    'dual_ang_betweenness_200',
    'dual_ang_betweenness_300',
    'dual_ang_betweenness_400',
    'dual_ang_betweenness_600',
    'dual_ang_betweenness_800',
    'dual_ang_betweenness_1200',
    'dual_ang_betweenness_1600'
]
cols_ml_labels = [
    r'Node Density $_{200m}$',
    r'Node Density $_{300m}$',
    r'Node Density $_{400m}$',
    r'Node Density $_{600m}$',
    r'Node Density $_{800m}$',
    r'Node Density $_{1200m}$',
    r'Node Density $_{1600m}$',
    r'Farness $_{200m}$',
    r'Farness $_{300m}$',
    r'Farness $_{400m}$',
    r'Farness $_{600m}$',
    r'Farness $_{800m}$',
    r'Farness $_{1200m}$',
    r'Farness $_{1600m}$',
    r'$\measuredangle$ Farness $_{200m}$',
    r'$\measuredangle$ Farness $_{300m}$',
    r'$\measuredangle$ Farness $_{400m}$',
    r'$\measuredangle$ Farness $_{600m}$',
    r'$\measuredangle$ Farness $_{800m}$',
    r'$\measuredangle$ Farness $_{1200m}$',
    r'$\measuredangle$ Farness $_{1600m}$',
    r'Betweenness $_{200m}$',
    r'Betweenness $_{300m}$',
    r'Betweenness $_{400m}$',
    r'Betweenness $_{600m}$',
    r'Betweenness $_{800m}$',
    r'Betweenness $_{1200m}$',
    r'Betweenness $_{1600m}$',
    r'$\measuredangle$ Betweenness $_{200m}$',
    r'$\measuredangle$ Betweenness $_{300m}$',
    r'$\measuredangle$ Betweenness $_{400m}$',
    r'$\measuredangle$ Betweenness $_{600m}$',
    r'$\measuredangle$ Betweenness $_{800m}$',
    r'$\measuredangle$ Betweenness $_{1200m}$',
    r'$\measuredangle$ Betweenness $_{1600m}$'
]

assert len(cols_ml) == len(cols_ml_labels)

logger.info(f'columns: {cols_ml}')

feature_names_dict = {}
for c, l in zip(cols_ml, cols_ml_labels):
    feature_names_dict[c] = l

# TARGETS:
target_cols = [
    'mixed_uses_score_hill_0_200',
    # 'uses_commercial_200',
    # 'uses_retail_200',
    # 'uses_eating_200'
]

target_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    # r'Commercial $\ _{\beta=0.02}$',
    # r'Retail $\ _{\beta=0.02}$',
    # r'Eat \& Drink $\ _{\beta=0.02}$'
]

assert len(target_cols) == len(target_labels)

for target_col, target_label in zip(target_cols, target_labels):

    logger.info(f'target column: {target_col}')

    # drop rows where any nan for target column
    data = df_data_dual_20_clean.copy(deep=True)

    start_len = len(data)
    data = data[np.isfinite(data[target_col])]
    if len(data) != start_len:
        logger.warning(
            f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in target: {target_label}')

    # drop rows where any non finite rows for input columns
    for col in cols_ml:
        start_len = len(data)
        data = data[np.isfinite(data[col])]
        if len(data) != start_len:
            logger.warning(
                f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

    data_target = data[target_col]
    data_input = data[cols_ml]

    training_inputs, testing_inputs, training_targets, testing_targets = \
        train_test_split(data_input, data_target, train_size=0.8)

    # create
    random_forest = RandomForestRegressor(
        n_jobs=-1,
        max_features='auto',
        # max_features=20,
        n_estimators=50
    )

    # fit
    random_forest.fit(training_inputs, training_targets)
    # test set prediction
    pred_test = random_forest.predict(testing_inputs)
    # accuracy
    acc = metrics.r2_score(testing_targets, pred_test)

    logger.info(
        f'fitted target: {target_label} to r^2 accuracy: {round(acc * 100, 2)}')

    # calculate PCA accuracy to control for curse of dimensionality
    for i in range(10, 21, 2):
        pca = PCA(n_components=i)
        data_input_reduced = pca.fit_transform(data_input)

        training_inputs, testing_inputs, training_targets, testing_targets = \
            train_test_split(data_input_reduced, data_target, train_size=0.8)

        # fit
        random_forest.fit(training_inputs, training_targets)
        # test set prediction
        pred_test = random_forest.predict(testing_inputs)
        # insert into dict
        pca_acc = metrics.r2_score(testing_targets, pred_test)

        logger.info(
            f'fitted target: {target_label} to r^2 accuracy using PCA {i}: {round(pca_acc * 100, 2)}')

"""
RESULTS:

# 100 estimators on 200, 400, 800, 1600

INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy: 68.92
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 1: -23.53
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 2: 14.34
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 3: 33.69
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 4: 44.78
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 5: 51.03
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 6: 67.99
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 7: 69.29
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 8: 69.96
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 9: 72.13
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 10: 72.28
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 11: 72.85 <-
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 12: 72.9
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 13: 70.2
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 14: 68.39

# 50 estimators on all distances

INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy: 70.15
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 10: 73.33
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 12: 75.27
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 14: 76.22
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 16: 75.87
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 18: 76.22
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 20: 76.29 <-

"""

# %% for directly comparing primal and dual graphs
cols_ml = [
    'dual_node_count_200',
    'dual_node_count_400',
    'dual_node_count_800',
    'dual_node_count_1600',
    'dual_imp_met_far_200',
    'dual_imp_met_far_400',
    'dual_imp_met_far_800',
    'dual_imp_met_far_1600',
    'dual_met_betweenness_200',
    'dual_met_betweenness_400',
    'dual_met_betweenness_800',
    'dual_met_betweenness_1600'
]
cols_ml_labels = [
    r'Node Density $_{200m}$',
    r'Node Density $_{400m}$',
    r'Node Density $_{800m}$',
    r'Node Density $_{1600m}$',
    r'Farness $_{200m}$',
    r'Farness $_{400m}$',
    r'Farness $_{800m}$',
    r'Farness $_{1600m}$',
    r'Betweenness $_{200m}$',
    r'Betweenness $_{400m}$',
    r'Betweenness $_{800m}$',
    r'Betweenness $_{1600m}$'
]

assert len(cols_ml) == len(cols_ml_labels)

logger.info(f'columns: {cols_ml}')

feature_names_dict = {}
for c, l in zip(cols_ml, cols_ml_labels):
    feature_names_dict[c] = l

# TARGETS:
target_cols = [
    # 'mixed_uses_score_hill_0_200',
    'uses_commercial_200',
    # 'uses_retail_200',
    # 'uses_eating_200'
]

target_labels = [
    # r'Mixed Uses $H_{0}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    # r'Retail $\ _{\beta=0.02}$',
    # r'Eat \& Drink $\ _{\beta=0.02}$'
]

assert len(target_cols) == len(target_labels)

for target_col, target_label in zip(target_cols, target_labels):

    logger.info(f'target column: {target_col}')

    # drop rows where any nan for target column
    data = df_data_dual_20_clean.copy(deep=True)

    start_len = len(data)
    data = data[np.isfinite(data[target_col])]
    if len(data) != start_len:
        logger.warning(
            f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in target: {target_label}')

    # drop rows where any non finite rows for input columns
    for col in cols_ml:
        start_len = len(data)
        data = data[np.isfinite(data[col])]
        if len(data) != start_len:
            logger.warning(
                f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

    data_target = data[target_col]
    data_input = data[cols_ml]

    training_inputs, testing_inputs, training_targets, testing_targets = \
        train_test_split(data_input, data_target, train_size=0.8)

    # create
    random_forest = RandomForestRegressor(
        n_jobs=-1,
        max_features='auto',
        # max_features=20,
        n_estimators=50
    )

    # fit
    random_forest.fit(training_inputs, training_targets)
    # test set prediction
    pred_test = random_forest.predict(testing_inputs)
    # accuracy
    acc = metrics.r2_score(testing_targets, pred_test)

    logger.info(
        f'fitted target: {target_label} to r^2 accuracy: {round(acc * 100, 2)}')

    # calculate PCA accuracy to control for curse of dimensionality
    for i in range(5, 13):
        pca = PCA(n_components=i)
        data_input_reduced = pca.fit_transform(data_input)

        training_inputs, testing_inputs, training_targets, testing_targets = \
            train_test_split(data_input_reduced, data_target, train_size=0.8)

        # fit
        random_forest.fit(training_inputs, training_targets)
        # test set prediction
        pred_test = random_forest.predict(testing_inputs)
        # insert into dict
        pca_acc = metrics.r2_score(testing_targets, pred_test)

        logger.info(
            f'fitted target: {target_label} to r^2 accuracy using PCA {i}: {round(pca_acc * 100, 2)}')

"""
50 estimators

Mixed Uses
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy: 61.54
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 5: 63.01
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 6: 64.05
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 7: 66.16
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 8: 66.47 <-
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 9: 62.83
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 10: 61.24
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 11: 61.0
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 12: 61.86

INFO:builtins:target column: uses_commercial_200
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy: 47.76
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 5: 50.44
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 6: 50.56
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 7: 53.66
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 8: 53.8 <-
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 9: 49.04
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 10: 46.19
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 11: 46.72
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 12: 46.85

"""
