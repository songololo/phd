#%%
import logging
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import feature_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve, validation_curve

import phd_util
reload(phd_util)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%% columns to load
# weighted
columns_w = [
    'mixed_uses_score_hill_0_{dist}',
    'mixed_uses_score_hill_10_{dist}',
    'uses_commercial_{dist}',
    'uses_retail_{dist}',
    'uses_eating_{dist}',

    'met_gravity_{dist}',
    'met_betw_w_{dist}',
    'met_rt_complex_{dist}'
]
# non weighted
columns_nw = [
    'met_node_count_{dist}',
    'met_farness_{dist}',
    'met_betw_{dist}',
    'ang_node_count_{dist}',
    'ang_farness_{dist}',
    'ang_betw_{dist}'
]

compound_columns = [
    'city_pop_id'
]

for d in ['200', '400', '800', '1600']:
    compound_columns += [c.format(dist=d) for c in columns_nw]

for d in ['200_002', '400_001', '800_0005', '1600_00025']:
    compound_columns += [c.format(dist=d) for c in columns_w]

#%% load data
logger.info(f'loading columns: {compound_columns}')
source_table = 'analysis.roadnodes_20'
ml_data_ldn = phd_util.load_data_as_pd_df(compound_columns, source_table, f'WHERE city_pop_id = 1')

#%% create the closeness columns
for d in [200, 400, 800, 1600]:
    ml_data_ldn[f'met_closeness_{d}'] = ml_data_ldn[f'met_node_count_{d}']**2 / ml_data_ldn[f'met_farness_{d}']
    # angular farness needs to be transformed
    ml_data_ldn[f'ang_closeness_{d}_10'] = ml_data_ldn[f'ang_node_count_{d}']**2 / (ml_data_ldn[f'ang_farness_{d}'] + ml_data_ldn[f'ang_node_count_{d}'] * 10)

#%% clean data
logger.info('cleaning data')
# be careful with cleanup...
# per correlation plot:
# remove most extreme values - 0.99 gives good balance, 0.999 is stronger for some but generally weaker?
# ml_data_ldn_clean = util.remove_outliers_quantile(ml_data_ldn, q=(0, 0.999))
# remove rows where all nan
ml_data_ldn_clean = phd_util.clean_pd(ml_data_ldn, drop_na='all')
# due to the nature of the data fill nans with 0
ml_data_ldn_clean = ml_data_ldn_clean.fillna(0)

#%% calculate the accuracies and feature importances
cols_ml = [
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'ang_closeness_200_10',
    'ang_closeness_400_10',
    'ang_closeness_800_10',
    'ang_closeness_1600_10',
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
cols_ml_labels = [
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
    r'Closeness $\measuredangle+10\ _{200m}$',
    r'Closeness $\measuredangle+10\ _{400m}$',
    r'Closeness $\measuredangle+10\ _{800m}$',
    r'Closeness $\measuredangle+10\ _{1600m}$',
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
logger.info(f'columns: {cols_ml}')
feature_names_dict = {}
for c, l in zip(cols_ml, cols_ml_labels):
    feature_names_dict[c] = l
print(feature_names_dict)

# TARGETS:
target_cols = [
    'mixed_uses_score_hill_0_200',
    'mixed_uses_score_hill_10_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]

target_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Mixed Uses $H_{1}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$'
]

results_dict = {}
for target_col, target_label in zip(target_cols, target_labels):

    logger.info(f'target column: {target_col}')

    # drop rows where any nan for target column
    d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
    logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

    data_target = d[target_col]
    data_input = d[cols_ml]

    training_inputs, testing_inputs, training_targets, testing_targets = \
        train_test_split(data_input, data_target, train_size=0.8)

    # create
    random_forest = RandomForestRegressor(
        n_jobs=6,
        #max_features=8,
        #max_depth=30,
        n_estimators=50)
    # fit
    random_forest.fit(training_inputs, training_targets)
    # test set prediction
    pred_test = random_forest.predict(testing_inputs)

    # insert into dict
    results_dict[target_col] = {
        'target': target_label,
        'accuracy': metrics.r2_score(testing_targets, pred_test),
        'features': [],
        'scores': []
    }

    # get a sense for feature importances
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    for f in range(training_inputs.shape[1]):
        results_dict[target_col]['features'].append(feature_names_dict[data_input.columns[indices[f]]])
        results_dict[target_col]['scores'].append(importances[indices[f]])
print(results_dict)

#%% build the table

table = r'''
\begin{sidewaystable}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{\textheight}{!}{
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
for i in range(25):
    for v in results_dict.values():
        table += r'''
        & ''' + str(round(v['scores'][i] * 100, 2)) + r'\% & ' + v['features'][i]
    # close the line
    table += r' \\'

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{$r^{2}$ prediction accuracies for modelled land-uses and respective feature importances derived from Random Forest regression.}\label{table:table_random_forest_pred}
    \end{sidewaystable}
'''

print(table)

r'''
\begin{sidewaystable}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{\textheight}{!}{
    \begin{tabular}{ r | r l | r l | r l | r l | r l }
    Targets
    & \multicolumn{2}{ c }{ Mixed Uses $H_{0}\ _{200m}$ }
    & \multicolumn{2}{ c }{ Mixed Uses $H_{1}\ _{200m}$ }
    & \multicolumn{2}{ c }{ Commercial $\ _{\beta=0.02}$ }
    & \multicolumn{2}{ c }{ Retail $\ _{\beta=0.02}$ }
    & \multicolumn{2}{ c }{ Eat \& Drink $\ _{\beta=0.02}$ } \\
    $r^{2}$ Accuracies
    & \multicolumn{2}{ c }{ 66.4\% }
    & \multicolumn{2}{ c }{ 51.88\% }
    & \multicolumn{2}{ c }{ 52.8\% }
    & \multicolumn{2}{ c }{ 45.25\% }
    & \multicolumn{2}{ c }{ 54.82\% } \\
    \hline
    Importances
        & 30.19\% & Gravity $_{\beta=0.0025}$
        & 19.85\% & Closeness $_{1600m}$
        & 22.4\% & Closeness $_{1600m}$
        & 12.88\% & Gravity $_{\beta=0.005}$
        & 12.34\% & Gravity $_{\beta=0.0025}$ \\
        & 10.97\% & Closeness $_{1600m}$
        & 7.75\% & Gravity $_{\beta=0.0025}$
        & 7.28\% & Gravity $_{\beta=0.0025}$
        & 7.07\% & Route Complexity $_{1600m}$
        & 10.44\% & Gravity $_{\beta=0.005}$ \\
        & 8.32\% & Gravity $_{\beta=0.005}$
        & 6.9\% & Route Complexity $_{1600m}$
        & 7.01\% & Route Complexity $_{1600m}$
        & 6.67\% & Closeness $\measuredangle+10\ _{1600m}$
        & 8.41\% & Closeness $_{1600m}$ \\
        & 6.61\% & Route Complexity $_{1600m}$
        & 5.33\% & Betweenness $\measuredangle\ _{1600m}$
        & 4.12\% & Closeness $\measuredangle+10\ _{1600m}$
        & 6.05\% & Closeness $_{1600m}$
        & 5.93\% & Route Complexity $_{1600m}$ \\
        & 3.16\% & Closeness $\measuredangle+10\ _{1600m}$
        & 4.22\% & Closeness $\measuredangle+10\ _{1600m}$
        & 3.99\% & Betweenness $\measuredangle\ _{1600m}$
        & 5.25\% & Betweenness $\measuredangle\ _{1600m}$
        & 5.17\% & Route Complexity $_{800m}$ \\
        & 3.11\% & Route Complexity $_{400m}$
        & 3.86\% & Route Complexity $_{800m}$
        & 3.87\% & Route Complexity $_{800m}$
        & 4.05\% & Route Complexity $_{800m}$
        & 5.09\% & Closeness $\measuredangle+10\ _{1600m}$ \\
        & 2.76\% & Betweenness $\measuredangle\ _{1600m}$
        & 3.72\% & Betweenness $\measuredangle\ _{800m}$
        & 3.8\% & Betweenness $\measuredangle\ _{800m}$
        & 3.85\% & Closeness $_{800m}$
        & 4.93\% & Betweenness $\measuredangle\ _{1600m}$ \\
        & 2.66\% & Route Complexity $_{800m}$
        & 3.45\% & Closeness $_{800m}$
        & 3.51\% & Closeness $_{800m}$
        & 3.76\% & Closeness $\measuredangle+10\ _{200m}$
        & 3.26\% & Closeness $\measuredangle+10\ _{200m}$ \\
        & 2.45\% & Closeness $_{800m}$
        & 3.2\% & Closeness $\measuredangle+10\ _{800m}$
        & 3.37\% & Route Complexity $_{400m}$
        & 3.7\% & Betweenness $\measuredangle\ _{800m}$
        & 3.25\% & Closeness $_{800m}$ \\
        & 2.44\% & Closeness $\measuredangle+10\ _{800m}$
        & 3.1\% & Closeness $\measuredangle+10\ _{200m}$
        & 3.24\% & Gravity $_{\beta=0.005}$
        & 3.69\% & Closeness $\measuredangle+10\ _{800m}$
        & 3.24\% & Closeness $\measuredangle+10\ _{800m}$ \\
        & 2.17\% & Closeness $\measuredangle+10\ _{200m}$
        & 3.07\% & Closeness $\measuredangle+10\ _{400m}$
        & 3.08\% & Closeness $\measuredangle+10\ _{200m}$
        & 3.56\% & Gravity $_{\beta=0.0025}$
        & 3.15\% & Betweenness $\measuredangle\ _{800m}$ \\
        & 2.16\% & Closeness $\measuredangle+10\ _{400m}$
        & 2.97\% & Betweenness $_{1600m}$
        & 3.0\% & Closeness $\measuredangle+10\ _{800m}$
        & 3.29\% & Closeness $\measuredangle+10\ _{400m}$
        & 2.94\% & Route Complexity $_{400m}$ \\
        & 2.13\% & Closeness $_{400m}$
        & 2.85\% & Betweenness $\measuredangle\ _{400m}$
        & 2.98\% & Betweenness $\measuredangle\ _{400m}$
        & 3.22\% & Betweenness $_{1600m}$
        & 2.87\% & Betweenness $_{1600m}$ \\
        & 2.11\% & Betweenness $_{1600m}$
        & 2.8\% & Closeness $_{400m}$
        & 2.96\% & Closeness $\measuredangle+10\ _{400m}$
        & 3.16\% & Betweenness $\measuredangle\ _{400m}$
        & 2.77\% & Closeness $\measuredangle+10\ _{400m}$ \\
        & 2.08\% & Betweenness $\measuredangle\ _{800m}$
        & 2.7\% & Route Complexity $_{400m}$
        & 2.67\% & Closeness $_{400m}$
        & 2.81\% & Route Complexity $_{400m}$
        & 2.74\% & Betweenness $\measuredangle\ _{400m}$ \\
        & 1.9\% & Betweenness $\measuredangle\ _{400m}$
        & 2.57\% & Gravity $_{\beta=0.005}$
        & 2.59\% & Betweenness $_{1600m}$
        & 2.73\% & Closeness $_{400m}$
        & 2.7\% & Closeness $_{400m}$ \\
        & 1.68\% & Closeness $_{200m}$
        & 2.44\% & Route Complexity $_{200m}$
        & 2.14\% & Betweenness $\measuredangle\ _{200m}$
        & 2.68\% & Route Complexity $_{200m}$
        & 2.33\% & Route Complexity $_{200m}$ \\
        & 1.64\% & Route Complexity $_{200m}$
        & 2.11\% & Betweenness $\measuredangle\ _{200m}$
        & 2.04\% & Route Complexity $_{200m}$
        & 2.43\% & Betweenness $\measuredangle\ _{200m}$
        & 2.19\% & Betweenness $\measuredangle\ _{200m}$ \\
        & 1.46\% & Betweenness $\measuredangle\ _{200m}$
        & 2.03\% & Closeness $_{200m}$
        & 1.86\% & Betweenness wt. $_{1600m}$
        & 2.35\% & Betweenness $_{800m}$
        & 2.09\% & Closeness $_{200m}$ \\
        & 1.41\% & Gravity $_{\beta=0.01}$
        & 2.01\% & Gravity $_{\beta=0.02}$
        & 1.82\% & Gravity $_{\beta=0.02}$
        & 2.21\% & Betweenness wt. $_{1600m}$
        & 1.84\% & Betweenness wt. $_{1600m}$ \\
        & 1.21\% & Betweenness $_{800m}$
        & 2.0\% & Betweenness $_{800m}$
        & 1.82\% & Betweenness wt. $_{200m}$
        & 2.18\% & Closeness $_{200m}$
        & 1.81\% & Gravity $_{\beta=0.02}$ \\
        & 1.2\% & Gravity $_{\beta=0.02}$
        & 2.0\% & Gravity $_{\beta=0.01}$
        & 1.82\% & Closeness $_{200m}$
        & 2.01\% & Gravity $_{\beta=0.01}$
        & 1.78\% & Betweenness $_{800m}$ \\
        & 1.19\% & Betweenness wt. $_{1600m}$
        & 1.88\% & Betweenness wt. $_{1600m}$
        & 1.75\% & Betweenness $_{800m}$
        & 1.91\% & Gravity $_{\beta=0.02}$
        & 1.7\% & Gravity $_{\beta=0.01}$ \\
        & 1.12\% & Betweenness $_{400m}$
        & 1.6\% & Betweenness wt. $_{200m}$
        & 1.68\% & Gravity $_{\beta=0.01}$
        & 1.82\% & Betweenness $_{400m}$
        & 1.56\% & Betweenness wt. $_{200m}$ \\
        & 1.07\% & Betweenness wt. $_{200m}$
        & 1.54\% & Betweenness wt. $_{800m}$
        & 1.42\% & Betweenness $_{400m}$
        & 1.72\% & Betweenness $_{200m}$
        & 1.53\% & Betweenness $_{400m}$ \\
    \end{tabular}
    }}
    \caption{$r^{2}$ prediction accuracies for modelled land-uses and respective feature importances derived from Random Forest regression.}\label{table:table_random_forest_pred}
    \end{sidewaystable}
'''


#%% WITH NODE COUNTS
# calculate the accuracies and feature importances
cols_ml = [
    'met_node_count_200',
    'met_node_count_400',
    'met_node_count_800',
    'met_node_count_1600',
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'ang_closeness_200_10',
    'ang_closeness_400_10',
    'ang_closeness_800_10',
    'ang_closeness_1600_10',
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
cols_ml_labels = [
    r'Node Count $_{200m}$',
    r'Node Count $_{400m}$',
    r'Node Count $_{800m}$',
    r'Node Count $_{1600m}$',
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
    r'Closeness $\measuredangle+10\ _{200m}$',
    r'Closeness $\measuredangle+10\ _{400m}$',
    r'Closeness $\measuredangle+10\ _{800m}$',
    r'Closeness $\measuredangle+10\ _{1600m}$',
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
logger.info(f'columns: {cols_ml}')
feature_names_dict = {}
for c, l in zip(cols_ml, cols_ml_labels):
    feature_names_dict[c] = l
print(feature_names_dict)

# TARGETS:
target_cols = [
    'mixed_uses_score_hill_0_200',
    'mixed_uses_score_hill_10_200',
    'uses_commercial_200',
    'uses_retail_200',
    'uses_eating_200'
]

target_labels = [
    r'Mixed Uses $H_{0}\ _{200m}$',
    r'Mixed Uses $H_{1}\ _{200m}$',
    r'Commercial $\ _{\beta=0.02}$',
    r'Retail $\ _{\beta=0.02}$',
    r'Eat \& Drink $\ _{\beta=0.02}$'
]

results_dict = {}
for target_col, target_label in zip(target_cols, target_labels):

    logger.info(f'target column: {target_col}')

    # drop rows where any nan for target column
    d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
    logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

    data_target = d[target_col]
    data_input = d[cols_ml]

    training_inputs, testing_inputs, training_targets, testing_targets = \
        train_test_split(data_input, data_target, train_size=0.8)

    # create
    random_forest = RandomForestRegressor(
        n_jobs=6,
        #max_features=8,
        #max_depth=30,
        n_estimators=50)
    # fit
    random_forest.fit(training_inputs, training_targets)
    # test set prediction
    pred_test = random_forest.predict(testing_inputs)

    # insert into dict
    results_dict[target_col] = {
        'target': target_label,
        'accuracy': metrics.r2_score(testing_targets, pred_test),
        'features': [],
        'scores': []
    }

    # get a sense for feature importances
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    for f in range(training_inputs.shape[1]):
        results_dict[target_col]['features'].append(feature_names_dict[data_input.columns[indices[f]]])
        results_dict[target_col]['scores'].append(importances[indices[f]])
print(results_dict)

#%% build the table

table = r'''
\begin{sidewaystable}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{\textheight}{!}{
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
for i in range(25):
    for v in results_dict.values():
        table += r'''
        & ''' + str(round(v['scores'][i] * 100, 2)) + r'\% & ' + v['features'][i]
    # close the line
    table += r' \\'

# close the table
table += r'''
    \end{tabular}
    }}
    \caption{$r^{2}$ prediction accuracies for modelled land-uses and respective feature importances derived from Random Forest regression.}\label{table:table_random_forest_pred}
    \end{sidewaystable}
'''

print(table)

r'''
\begin{sidewaystable}[p!]
    \centering
    \makebox[\textwidth]{
    \resizebox{\textheight}{!}{
    \begin{tabular}{ r | r l | r l | r l | r l | r l }
    Targets
    & \multicolumn{2}{ c }{ Mixed Uses $H_{0}\ _{200m}$ }
    & \multicolumn{2}{ c }{ Mixed Uses $H_{1}\ _{200m}$ }
    & \multicolumn{2}{ c }{ Commercial $\ _{\beta=0.02}$ }
    & \multicolumn{2}{ c }{ Retail $\ _{\beta=0.02}$ }
    & \multicolumn{2}{ c }{ Eat \& Drink $\ _{\beta=0.02}$ } \\
    $r^{2}$ Accuracies
    & \multicolumn{2}{ c }{ 67.21\% }
    & \multicolumn{2}{ c }{ 53.09\% }
    & \multicolumn{2}{ c }{ 52.26\% }
    & \multicolumn{2}{ c }{ 46.34\% }
    & \multicolumn{2}{ c }{ 55.39\% } \\
    \hline
    Importances
        & 30.15\% & Gravity $_{\beta=0.0025}$
        & 15.91\% & Node Count $_{1600m}$
        & 18.97\% & Closeness $_{1600m}$
        & 12.65\% & Gravity $_{\beta=0.005}$
        & 11.54\% & Gravity $_{\beta=0.0025}$ \\
        & 9.21\% & Closeness $_{1600m}$
        & 7.57\% & Route Complexity $_{1600m}$
        & 7.14\% & Route Complexity $_{1600m}$
        & 7.15\% & Closeness $\measuredangle+10\ _{1600m}$
        & 10.63\% & Gravity $_{\beta=0.005}$ \\
        & 7.87\% & Gravity $_{\beta=0.005}$
        & 7.24\% & Gravity $_{\beta=0.0025}$
        & 6.77\% & Gravity $_{\beta=0.0025}$
        & 6.57\% & Route Complexity $_{1600m}$
        & 6.18\% & Closeness $_{1600m}$ \\
        & 6.4\% & Route Complexity $_{1600m}$
        & 5.41\% & Closeness $_{1600m}$
        & 5.7\% & Node Count $_{1600m}$
        & 4.73\% & Betweenness $\measuredangle\ _{1600m}$
        & 5.7\% & Route Complexity $_{1600m}$ \\
        & 3.35\% & Node Count $_{1600m}$
        & 5.15\% & Betweenness $\measuredangle\ _{1600m}$
        & 3.8\% & Closeness $\measuredangle+10\ _{1600m}$
        & 3.97\% & Closeness $_{1600m}$
        & 4.91\% & Route Complexity $_{800m}$ \\
        & 3.01\% & Closeness $\measuredangle+10\ _{1600m}$
        & 4.11\% & Closeness $\measuredangle+10\ _{1600m}$
        & 3.57\% & Betweenness $\measuredangle\ _{1600m}$
        & 3.78\% & Route Complexity $_{800m}$
        & 4.75\% & Betweenness $\measuredangle\ _{1600m}$ \\
        & 2.97\% & Route Complexity $_{400m}$
        & 3.34\% & Route Complexity $_{800m}$
        & 3.56\% & Route Complexity $_{400m}$
        & 3.61\% & Node Count $_{1600m}$
        & 4.72\% & Node Count $_{1600m}$ \\
        & 2.54\% & Route Complexity $_{800m}$
        & 3.33\% & Betweenness $\measuredangle\ _{800m}$
        & 3.45\% & Betweenness $\measuredangle\ _{800m}$
        & 3.55\% & Gravity $_{\beta=0.0025}$
        & 4.41\% & Closeness $\measuredangle+10\ _{1600m}$ \\
        & 2.53\% & Betweenness $\measuredangle\ _{1600m}$
        & 2.89\% & Closeness $\measuredangle+10\ _{200m}$
        & 3.37\% & Route Complexity $_{800m}$
        & 3.4\% & Closeness $\measuredangle+10\ _{800m}$
        & 2.91\% & Closeness $\measuredangle+10\ _{800m}$ \\
        & 2.18\% & Closeness $\measuredangle+10\ _{800m}$
        & 2.86\% & Closeness $\measuredangle+10\ _{400m}$
        & 2.76\% & Betweenness $_{1600m}$
        & 3.23\% & Betweenness $\measuredangle\ _{800m}$
        & 2.88\% & Closeness $\measuredangle+10\ _{200m}$ \\
        & 1.96\% & Closeness $\measuredangle+10\ _{200m}$
        & 2.83\% & Closeness $\measuredangle+10\ _{800m}$
        & 2.69\% & Closeness $\measuredangle+10\ _{400m}$
        & 3.21\% & Betweenness $_{1600m}$
        & 2.88\% & Betweenness $\measuredangle\ _{800m}$ \\
        & 1.93\% & Betweenness $_{1600m}$
        & 2.68\% & Betweenness $_{1600m}$
        & 2.67\% & Closeness $\measuredangle+10\ _{200m}$
        & 3.12\% & Closeness $\measuredangle+10\ _{200m}$
        & 2.68\% & Node Count $_{800m}$ \\
        & 1.93\% & Closeness $\measuredangle+10\ _{400m}$
        & 2.61\% & Betweenness $\measuredangle\ _{400m}$
        & 2.66\% & Node Count $_{800m}$
        & 3.12\% & Closeness $\measuredangle+10\ _{400m}$
        & 2.61\% & Betweenness $_{1600m}$ \\
        & 1.91\% & Betweenness $\measuredangle\ _{800m}$
        & 2.57\% & Node Count $_{800m}$
        & 2.6\% & Gravity $_{\beta=0.005}$
        & 2.84\% & Closeness $_{800m}$
        & 2.6\% & Route Complexity $_{400m}$ \\
        & 1.86\% & Node Count $_{800m}$
        & 2.57\% & Route Complexity $_{400m}$
        & 2.59\% & Closeness $_{800m}$
        & 2.79\% & Betweenness $\measuredangle\ _{400m}$
        & 2.5\% & Betweenness $\measuredangle\ _{400m}$ \\
        & 1.71\% & Betweenness $\measuredangle\ _{400m}$
        & 2.45\% & Closeness $_{800m}$
        & 2.53\% & Betweenness $\measuredangle\ _{400m}$
        & 2.6\% & Route Complexity $_{400m}$
        & 2.5\% & Closeness $\measuredangle+10\ _{400m}$ \\
        & 1.61\% & Closeness $_{800m}$
        & 2.14\% & Node Count $_{400m}$
        & 2.45\% & Closeness $\measuredangle+10\ _{800m}$
        & 2.38\% & Node Count $_{800m}$
        & 2.13\% & Closeness $_{800m}$ \\
        & 1.55\% & Node Count $_{400m}$
        & 2.12\% & Betweenness $\measuredangle\ _{200m}$
        & 2.13\% & Node Count $_{400m}$
        & 2.29\% & Node Count $_{400m}$
        & 2.08\% & Route Complexity $_{200m}$ \\
        & 1.48\% & Route Complexity $_{200m}$
        & 2.1\% & Route Complexity $_{200m}$
        & 1.89\% & Route Complexity $_{200m}$
        & 2.24\% & Route Complexity $_{200m}$
        & 2.0\% & Node Count $_{400m}$ \\
        & 1.42\% & Closeness $_{400m}$
        & 2.02\% & Gravity $_{\beta=0.005}$
        & 1.88\% & Betweenness $\measuredangle\ _{200m}$
        & 2.19\% & Betweenness $\measuredangle\ _{200m}$
        & 1.92\% & Betweenness $\measuredangle\ _{200m}$ \\
        & 1.33\% & Betweenness $\measuredangle\ _{200m}$
        & 1.95\% & Closeness $_{400m}$
        & 1.82\% & Closeness $_{400m}$
        & 2.16\% & Betweenness wt. $_{1600m}$
        & 1.88\% & Closeness $_{400m}$ \\
        & 1.23\% & Closeness $_{200m}$
        & 1.8\% & Gravity $_{\beta=0.02}$
        & 1.79\% & Gravity $_{\beta=0.02}$
        & 2.1\% & Closeness $_{400m}$
        & 1.63\% & Betweenness wt. $_{1600m}$ \\
        & 1.17\% & Gravity $_{\beta=0.01}$
        & 1.76\% & Betweenness $_{800m}$
        & 1.57\% & Betweenness wt. $_{1600m}$
        & 2.04\% & Betweenness $_{800m}$
        & 1.59\% & Gravity $_{\beta=0.02}$ \\
        & 1.11\% & Node Count $_{200m}$
        & 1.75\% & Betweenness wt. $_{1600m}$
        & 1.52\% & Closeness $_{200m}$
        & 1.81\% & Closeness $_{200m}$
        & 1.58\% & Closeness $_{200m}$ \\
        & 1.1\% & Betweenness $_{800m}$
        & 1.58\% & Closeness $_{200m}$
        & 1.49\% & Betweenness wt. $_{200m}$
        & 1.72\% & Gravity $_{\beta=0.01}$
        & 1.56\% & Betweenness $_{800m}$ \\
    \end{tabular}
    }}
    \caption{$r^{2}$ prediction accuracies for modelled land-uses and respective feature importances derived from Random Forest regression.}\label{table:table_random_forest_pred}
    \end{sidewaystable}
'''