'''
Early attempts, maybe from upgrade, to predict MU from centralities etc.

'''

# %%
import logging


import numpy as np
from src import phd_util



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# %% columns to load
# weighted
columns = [
    'city_pop_id',
    'uses_hill_0_100',
    'uses_hill_funct_wt_0_100',
    'access_eating_100',
    'access_commercial_100',
    'access_retail_100'
]

# non weighted
columns_nw = [
    'met_betw_{d}',
    'met_betw_w_{d}',
    'met_gravity_{d}',
    'met_node_count_{d}',
    'met_farness_{d}',
    'met_rt_complex_{d}',
    'met_info_ent_{d}'
]

distances = [800]

for d in distances:
    columns += [c.format(d=d) for c in columns_nw]

# %%
print(f'loading columns: {columns}')
df_data_full = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_full', 'WHERE city_pop_id = 1')
df_data_100 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_100', 'WHERE city_pop_id = 1')
df_data_50 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_50', 'WHERE city_pop_id = 1')
df_data_20 = phd_util.load_data_as_pd_df(columns, 'analysis.roadnodes_20', 'WHERE city_pop_id = 1')

# %%
print('cleaning data')
# be careful with cleanup...
df_data_full_clean = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)
df_data_100_clean = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)
df_data_50_clean = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)
df_data_20_clean = phd_util.clean_pd(df_data_20, drop_na='all', fill_inf=np.nan)

# %% create the closeness columns
# checked and don't see different behaviour for farness vs. closeness on euclidean data
for d in distances:
    df_data_full_clean[f'met_closeness_{d}'] = df_data_full_clean[f'met_node_count_{d}'] ** 2 / df_data_full_clean[
        f'met_farness_{d}']
    df_data_100_clean[f'met_closeness_{d}'] = df_data_100_clean[f'met_node_count_{d}'] ** 2 / df_data_100_clean[
        f'met_farness_{d}']
    df_data_50_clean[f'met_closeness_{d}'] = df_data_50_clean[f'met_node_count_{d}'] ** 2 / df_data_50_clean[
        f'met_farness_{d}']
    df_data_20_clean[f'met_closeness_{d}'] = df_data_20_clean[f'met_node_count_{d}'] ** 2 / df_data_20_clean[
        f'met_farness_{d}']

# %% compute linear regression for mixed uses, commercial, retail, eat & drink
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

targets = [
    'uses_hill_0_100',
    'uses_hill_funct_wt_0_100',
    'access_commercial_100',
    'access_retail_100',
    'access_eating_100'
]
target_labels = [
    'mixed_uses',
    'mixed_uses_wt',
    'commercial',
    'retail',
    'eat_drink'
]

themes = [
    'met_node_count_{d}',
    'met_closeness_{d}',
    'met_gravity_{d}',
    'met_betw_{d}',
    'met_betw_w_{d}'
]
theme_labels = [
    'node_density',
    'met_closeness',
    'met_gravity',
    'met_betw',
    'met_w_betw'
]

tables = [
    df_data_full_clean,
    df_data_100_clean,
    df_data_50_clean,
    df_data_20_clean
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
            for d in [800]:
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

            training_inputs, testing_inputs, training_targets, testing_targets = \
                train_test_split(data_input, data_target, train_size=0.8)

            # polynomial
            poly = PolynomialFeatures(degree=2)
            training_inputs_poly = poly.fit_transform(training_inputs)

            # model
            from sklearn.linear_model import ARDRegression

            model = ARDRegression().fit(training_inputs_poly, training_targets)

            # run on test set
            testing_inputs_poly = poly.fit_transform(testing_inputs)
            acc = model.score(testing_inputs_poly, testing_targets)

            results[target_label][table_label][theme_label] = acc
            logger.info(
                f'fitted table: {table_label} and target: {target_label} to r^2 accuracy: {round(acc * 100, 2)} for {theme_label}')

# %% build the table

table = r'''
\begin{table}[htb!]
\centering
\makebox[\textwidth]{
\resizebox{0.8\textheight}{!}{
\begin{tabular}{ r | c c c c c | c c c c c | c c c c c | c c c c c | c c c c c }
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Closeness}
& \rotatebox[origin=l]{90}{Gravity}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{Weighted Betw.} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Closeness}
& \rotatebox[origin=l]{90}{Gravity}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{Weighted Betw.} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Closeness}
& \rotatebox[origin=l]{90}{Gravity}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{Weighted Betw.} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Closeness}
& \rotatebox[origin=l]{90}{Gravity}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{Weighted Betw.} 
& \rotatebox[origin=l]{90}{Node Density}
& \rotatebox[origin=l]{90}{Closeness}
& \rotatebox[origin=l]{90}{Gravity}
& \rotatebox[origin=l]{90}{Betweenness}
& \rotatebox[origin=l]{90}{Weighted Betw.} \\
\cline{2-21}
$r^{2}\ \%$
& \multicolumn{5}{ c | }{ Mixed Uses $H_{0\ 100m}$ }
& \multicolumn{5}{ c | }{ Mixed Uses $H_{0\ wt.\ \beta=-0.04}$ }
& \multicolumn{5}{ c | }{ Commercial $_{\beta=-0.04}$ }
& \multicolumn{5}{ c | }{ Retail $_{\beta=-0.04}$ }
& \multicolumn{5}{ c }{ Eat \& Drink $_{\beta=-0.04}$ }\\
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
}}\caption{Example random forest ML $r^{2}$ prediction accuracies for landuses as calculated on primal graphs. (50 estimators, 6 PCA components.)}\label{table:pred_comparisons}
\end{table}
'''

print(table)
'''
& 26.2
                & 26.3
                & 26.8
                & 16.2
                & 16.9
                & 20.3
                & 21.0
                & 20.7
                & 12.7
                & 12.7
                & 11.9
                & 12.8
                & 11.8
                & 4.8
                & 5.6
                & 7.5
                & 9.3
                & 9.5
                & 5.7
                & 4.9
                & 12.4
                & 13.5
                & 13.9
                & 6.9
                & 7.7 \\
'''

# %% calculate the accuracies and feature importances
cols_ml = [
    'met_node_count_{d}',
    'met_closeness_{d}',
    'met_gravity_{d}',
    'met_info_ent_{d}',
    'met_rt_complex_{d}',
    'met_betw_{d}',
    'met_betw_w_{d}'
]
cols_ml_labels = [
    r'Node Density',
    r'Closeness',
    r'Gravity',
    r'Route Entropy',
    r'Route Complexity',
    r'Betweenness',
    r'Betweenness wt.'
]

assert len(cols_ml) == len(cols_ml_labels)

logger.info(f'columns: {cols_ml}')

# TARGETS:
target_cols = [
    'uses_hill_0_100',
    'uses_hill_funct_wt_0_100',
    'access_commercial_100',
    'access_retail_100',
    'access_eating_100'
]

target_labels = [
    r'Mixed Uses $H_{0\ 100m}$',
    r'Mixed Uses $H_{0\ wt.\ \beta=-0.04}$',
    r'Commercial $_{\beta=-0.04}$',
    r'Retail $_{\beta=-0.04}$',
    r'Eat \& Drink $_{\beta=-0.04}$'
]

assert len(target_cols) == len(target_labels)

results_dict = {}
for target_col, target_label in zip(target_cols, target_labels):

    logger.info(f'target column: {target_col}')

    # drop rows where any nan for target column
    data = df_data_20_clean.copy(deep=True)

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
    \begin{tabular}{ r | r l | r l | r l | r l | r l }
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
for i in range(7):
    for v in results_dict.values():
        table += r'''
        & ''' + str(round(v['scores'][i] * 100, 2)) + r'\% & ' + v['features'][i]
    # close the line
    table += r' \\'

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{100 estimators. Feature importances derived from random forest regression on the primal graph for different network measures. (Multiple distances from $200m$ to $1600m$ reduced using PCA to a single dimension per measure.)}\label{table:table_random_forest_pred}
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
    'Commercial']

themes = [
    'met_node_count_{d}',
    'met_closeness_{d}',
    'met_betw_{d}'
]
theme_labels = [
    'node_density',
    'met_closeness',
    'met_betw'
]

tables = [
    df_data_full_clean,
    df_data_20_clean
]
table_labels = ['full', '20m']

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

'''
## max depth 10, n estimators 100

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 33.9 for node_density
WARNING:builtins:NOTE -> dropped 40.61% rows due to non-finite values in var: met_closeness_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 32.35 for met_closeness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 9.13 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 33.12 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 31.95 for met_closeness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 11.15 for met_betw

## max depth None, n estimators 100

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 28.6 for node_density
WARNING:builtins:NOTE -> dropped 40.61% rows due to non-finite values in var: met_closeness_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 26.54 for met_closeness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 9.91 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 40.77 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 36.14 for met_closeness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 15.05 for met_betw

## max depth 5, n estimators 100

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 22.81 for node_density
WARNING:builtins:NOTE -> dropped 40.61% rows due to non-finite values in var: met_closeness_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 21.6 for met_closeness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 11.01 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 28.31 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 25.72 for met_closeness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 10.07 for met_betw

## max depth 5, n estimators 50

INFO:builtins:TARGET COLUMN: commercial
INFO:builtins:CURRENT TABLE: full
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 21.6 for node_density
WARNING:builtins:NOTE -> dropped 40.61% rows due to non-finite values in var: met_closeness_50
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 27.22 for met_closeness
INFO:builtins:fitted table: full and target: commercial to r^2 accuracy: 12.77 for met_betw
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 27.72 for node_density
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 24.75 for met_closeness
INFO:builtins:fitted table: 20m and target: commercial to r^2 accuracy: 9.6 for met_betw
'''

# %% test PCA impact on joint accuracies
cols_ml = [
    'met_node_count_200',
    'met_node_count_300',
    'met_node_count_400',
    'met_node_count_600',
    'met_node_count_800',
    'met_node_count_1200',
    'met_node_count_1600',
    'met_closeness_200',
    'met_closeness_300',
    'met_closeness_400',
    'met_closeness_600',
    'met_closeness_800',
    'met_closeness_1200',
    'met_closeness_1600',
    'met_gravity_200',
    'met_gravity_300',
    'met_gravity_400',
    'met_gravity_600',
    'met_gravity_800',
    'met_gravity_1200',
    'met_gravity_1600',
    'met_info_ent_200',
    'met_info_ent_300',
    'met_info_ent_400',
    'met_info_ent_600',
    'met_info_ent_800',
    'met_info_ent_1200',
    'met_info_ent_1600',
    'met_rt_complex_200',
    'met_rt_complex_300',
    'met_rt_complex_400',
    'met_rt_complex_600',
    'met_rt_complex_800',
    'met_rt_complex_1200',
    'met_rt_complex_1600',
    'met_betw_200',
    'met_betw_300',
    'met_betw_400',
    'met_betw_600',
    'met_betw_800',
    'met_betw_1200',
    'met_betw_1600',
    'met_betw_w_200',
    'met_betw_w_300',
    'met_betw_w_400',
    'met_betw_w_600',
    'met_betw_w_800',
    'met_betw_w_1200',
    'met_betw_w_1600'
]
cols_ml_labels = [
    r'Node Density $_{200m}$',
    r'Node Density $_{300m}$',
    r'Node Density $_{400m}$',
    r'Node Density $_{600m}$',
    r'Node Density $_{800m}$',
    r'Node Density $_{1200m}$',
    r'Node Density $_{1600m}$',
    r'Closeness $_{200m}$',
    r'Closeness $_{300m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{600m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1200m}$',
    r'Closeness $_{1600m}$',
    r'Gravity $_{\beta=0.02}$',
    r'Gravity $_{\beta=0.013}$',
    r'Gravity $_{\beta=0.01}$',
    r'Gravity $_{\beta=0.007}$',
    r'Gravity $_{\beta=0.005}$',
    r'Gravity $_{\beta=0.003}$',
    r'Gravity $_{\beta=0.0025}$',
    r'Route Entropy $_{200m}$',
    r'Route Entropy $_{300m}$',
    r'Route Entropy $_{400m}$',
    r'Route Entropy $_{600m}$',
    r'Route Entropy $_{800m}$',
    r'Route Entropy $_{1200m}$',
    r'Route Entropy $_{1600m}$',
    r'Route Complexity $_{200m}$',
    r'Route Complexity $_{300m}$',
    r'Route Complexity $_{400m}$',
    r'Route Complexity $_{600m}$',
    r'Route Complexity $_{800m}$',
    r'Route Complexity $_{1200m}$',
    r'Route Complexity $_{1600m}$',
    r'Betweenness $_{200m}$',
    r'Betweenness $_{300m}$',
    r'Betweenness $_{400m}$',
    r'Betweenness $_{600m}$',
    r'Betweenness $_{800m}$',
    r'Betweenness $_{1200m}$',
    r'Betweenness $_{1600m}$',
    r'Betweenness wt. $_{200m}$',
    r'Betweenness wt. $_{300m}$',
    r'Betweenness wt. $_{400m}$',
    r'Betweenness wt. $_{600m}$',
    r'Betweenness wt. $_{800m}$',
    r'Betweenness wt. $_{1200m}$',
    r'Betweenness wt. $_{1600m}$'
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
    data = df_data_20_clean.copy(deep=True)

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
    for i in range(14, 21, 2):
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

INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy: 64.79
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 1: -20.2
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 2: 11.35
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 3: 47.18
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 4: 53.22
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 5: 58.75
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 6: 63.15
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 7: 63.31
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 8: 65.3
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 9: 66.52
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 10: 66.23
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 11: 68.29
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 12: 68.47 <-
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 13: 68.27
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 14: 68.21

# 50 estimators on all distances

INFO:builtins:target column: mixed_uses_score_hill_0_200
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy: 65.85
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 10: 67.57
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 12: 68.82
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 14: 69.71
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 14: 69.56
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 16: 69.74 <-
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 18: 68.81
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 20: 68.66

"""

# %%
##################
# TESTING:
# for directly comparing primal and dual graphs
cols_ml = [
    'met_node_count_200',
    'met_node_count_400',
    'met_node_count_800',
    'met_node_count_1600',
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'met_betw_200',
    'met_betw_400',
    'met_betw_800',
    'met_betw_1600'
]
cols_ml_labels = [
    r'Node Density $_{200m}$',
    r'Node Density $_{400m}$',
    r'Node Density $_{800m}$',
    r'Node Density $_{1600m}$',
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
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
    data = df_data_20_clean.copy(deep=True)

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
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy: 59.33
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 5: 59.13
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 6: 60.43
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 7: 62.45
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 8: 62.6
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 9: 64.94
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 10: 65.82 <-
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 11: 65.69
INFO:builtins:fitted target: Mixed Uses $H_{0}\ _{200m}$ to r^2 accuracy using PCA 12: 64.92

INFO:builtins:target column: uses_commercial_200
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy: 42.05
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 5: 43.82
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 6: 42.97
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 7: 47.23
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 8: 46.58
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 9: 50.1
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 10: 50.68
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 11: 52.42 <-
INFO:builtins:fitted target: Commercial $\ _{\beta=0.02}$ to r^2 accuracy using PCA 12: 51.62

"""

# %%
##################
# TESTING: pipeline with multiple steps

targets = ['mixed_uses_score_hill_0_200']
target_labels = 'Mixed Uses'

themes = [
    'met_node_count_{d}',
    'met_closeness_{d}',
    'met_betw_{d}'
]
theme_labels = [
    'node_density',
    'met_closeness',
    'met_betw'
]

tables = [
    df_data_full_clean,
    df_data_20_clean
]
table_labels = ['full', '20m']

results = {}

for target, target_label in zip(targets, target_labels):

    logger.info(f'TARGET COLUMN: {target_label}')
    results[target_label] = {}

    for table, table_label in zip(tables, table_labels):

        logger.info(f'CURRENT TABLE: {table_label}')

        closeness_cols = []
        betw_cols = []
        for d in distances:
            closeness_cols.append('met_closeness_{d}'.format(d=d))
            betw_cols.append('met_betw_{d}'.format(d=d))

        # drop rows where any non finite values for target column
        data = table.copy(deep=True)
        start_len = len(data)
        data = data[np.isfinite(data[target])]
        if len(data) != start_len:
            logger.warning(
                f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in target: {target}')

        # drop rows where any non finite rows for input columns
        for col in closeness_cols:
            start_len = len(data)
            data = data[np.isfinite(data[col])]
            if len(data) != start_len:
                logger.warning(
                    f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

        for col in betw_cols:
            start_len = len(data)
            data = data[np.isfinite(data[col])]
            if len(data) != start_len:
                logger.warning(
                    f'NOTE -> dropped {round(((start_len - len(data)) / start_len) * 100, 2)}% rows due to non-finite values in var: {col}')

        data_target = data[target]

        pca = PCA(n_components=6)
        closeness_input_reduced = pca.fit_transform(data[closeness_cols])

        pca = PCA(n_components=6)
        betw_input_reduced = pca.fit_transform(data[betw_cols])

        data_reduced = np.hstack([closeness_input_reduced, betw_input_reduced])

        training_inputs, testing_inputs, training_targets, testing_targets, training_idx, testing_idx = \
            train_test_split(data_reduced, data_target, np.arange(len(data_reduced)), train_size=0.8)

        # create
        random_forest = RandomForestRegressor(
            n_jobs=-1,
            max_features='auto',
            n_estimators=50
        )

        # fit
        random_forest.fit(training_inputs[:, :6], training_targets)
        # test set prediction
        pred_test = random_forest.predict(testing_inputs[:, :6])
        # insert into dict
        acc = metrics.r2_score(testing_targets, pred_test)
        logger.info(
            f'fitted table: {table_label} and target: {target_label} to r^2 accuracy: {round(acc * 100, 2)} for closeness')

        # now betweenness
        # fit
        close_pred = random_forest.predict(data_reduced[:, :6])
        train_set = np.hstack([np.array([close_pred]).T[training_idx], training_inputs[:, 6:]])
        random_forest.fit(train_set, training_targets)
        # test set prediction
        test_set = np.hstack([np.array([close_pred]).T[testing_idx], testing_inputs[:, 6:]])
        pred_test = random_forest.predict(test_set)
        # insert into dict
        acc = metrics.r2_score(testing_targets, pred_test)
        logger.info(
            f'fitted table: {table_label} and target: {target_label} to r^2 accuracy: {round(acc * 100, 2)} for betweenness -> closeness')

'''
INFO:builtins:TARGET COLUMN: M
INFO:builtins:CURRENT TABLE: full
WARNING:builtins:NOTE -> dropped 1.28% rows due to non-finite values in var: met_closeness_200
INFO:builtins:fitted table: full and target: M to r^2 accuracy: 54.57 for closeness
INFO:builtins:fitted table: full and target: M to r^2 accuracy: 52.82 for betweenness -> closeness
INFO:builtins:CURRENT TABLE: 20m
INFO:builtins:fitted table: 20m and target: M to r^2 accuracy: 59.15 for closeness
INFO:builtins:fitted table: 20m and target: M to r^2 accuracy: 58.5 for betweenness -> closeness
'''
