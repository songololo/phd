# %%
import logging


import numpy as np
from src import phd_util
from sklearn import metrics
from sklearn.model_selection import train_test_split



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'comp_met_betweenness_{dist}',
    'comp_ang_betweenness_{dist}',
    'comp_node_count_{dist}',
    'comp_met_farness_{dist}',
    'comp_ang_farness_{dist}',
    'comp_ang_farness_m_{dist}',
    'comp_ratio_{dist}'
]

distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

for d in distances:
    columns += [c.format(dist=d) for c in columns_nw]

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

# %% generate farness data
for d in distances:
    df_data_dual_full_clean[f'comp_imp_met_far_{d}'] = df_data_dual_full_clean[f'comp_met_farness_{d}'] / \
                                                       df_data_dual_full_clean[f'comp_node_count_{d}'] ** 2
    df_data_dual_100_clean[f'comp_imp_met_far_{d}'] = df_data_dual_100_clean[f'comp_met_farness_{d}'] / \
                                                      df_data_dual_100_clean[f'comp_node_count_{d}'] ** 2
    df_data_dual_50_clean[f'comp_imp_met_far_{d}'] = df_data_dual_50_clean[f'comp_met_farness_{d}'] / \
                                                     df_data_dual_50_clean[f'comp_node_count_{d}'] ** 2
    df_data_dual_20_clean[f'comp_imp_met_far_{d}'] = df_data_dual_20_clean[f'comp_met_farness_{d}'] / \
                                                     df_data_dual_20_clean[f'comp_node_count_{d}'] ** 2

    df_data_dual_full_clean[f'comp_imp_ang_far_{d}'] = df_data_dual_full_clean[f'comp_ang_farness_{d}'] / \
                                                       df_data_dual_full_clean[f'comp_node_count_{d}'] ** 2
    df_data_dual_100_clean[f'comp_imp_ang_far_{d}'] = df_data_dual_100_clean[f'comp_ang_farness_{d}'] / \
                                                      df_data_dual_100_clean[f'comp_node_count_{d}'] ** 2
    df_data_dual_50_clean[f'comp_imp_ang_far_{d}'] = df_data_dual_50_clean[f'comp_ang_farness_{d}'] / \
                                                     df_data_dual_50_clean[f'comp_node_count_{d}'] ** 2
    df_data_dual_20_clean[f'comp_imp_ang_far_{d}'] = df_data_dual_20_clean[f'comp_ang_farness_{d}'] / \
                                                     df_data_dual_20_clean[f'comp_node_count_{d}'] ** 2

# %% testing different classifiers
target = 'mixed_uses_score_hill_0_200'
predictor = 'comp_node_count_{d}'
table = df_data_dual_50_clean

cols = []
for d in distances:
    cols.append(predictor.format(d=d))
logger.info(f'column names: {cols}')

# drop rows where any non finite values for target column
data = table.copy()
data = data[np.isfinite(data[target])]
if len(data) != len(table):
    logger.warning(f'NOTE -> dropped {len(data) - len(table)} rows due to non-finite values in {target}')

# drop rows where any non finite rows for input columns
for col in cols:
    start_len = len(data)
    data = data[np.isfinite(data[col])]
    if len(data) != start_len:
        logger.warning(f'NOTE -> dropped {len(data) - start_len} rows due to non-finite values in {col}')

data_target = data[target]
data_input = data[cols]

# pca = PCA(n_components=2)
# data_input = pca.fit_transform(data_input)

training_inputs, testing_inputs, training_targets, testing_targets = \
    train_test_split(data_input, data_target, train_size=0.8)

# create
# mod = linear_model.LinearRegression()  # 39.28%
# mod = linear_model.Ridge()  # 39.5%
# mod = linear_model.Lasso()  # 39.77%
# mod = linear_model.BayesianRidge()  # 39.62%
# mod = svm.SVR()  # -14.74%
# mod = GaussianProcessRegressor()
# mod = RandomForestRegressor(n_jobs=-1, max_features='auto', max_depth=20, n_estimators=30)  #53.56%

# mod = MLPRegressor()  # 37.84%

# fit
mod.fit(training_inputs, training_targets)
# test set prediction
pred_test = mod.predict(testing_inputs)
# r2
logger.info(f'r^2 accuracy: {round(metrics.r2_score(testing_targets, pred_test) * 100, 2)}%')
