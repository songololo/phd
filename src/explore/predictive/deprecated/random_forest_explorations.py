# %%
import logging


import numpy as np
from src import phd_util
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% columns to load
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

# %% load data
logger.info(f'loading columns: {compound_columns}')
source_table = 'analysis.roadnodes_20'
ml_data_ldn = phd_util.load_data_as_pd_df(compound_columns, source_table, f'WHERE city_pop_id = 1')

# %% create the closeness columns
for d in [200, 400, 800, 1600]:
    ml_data_ldn[f'met_closeness_{d}'] = ml_data_ldn[f'met_node_count_{d}'] ** 2 / ml_data_ldn[f'met_farness_{d}']
    # angular farness needs to be transformed
    ml_data_ldn[f'ang_closeness_{d}_10'] = ml_data_ldn[f'ang_node_count_{d}'] ** 2 / (
            ml_data_ldn[f'ang_farness_{d}'] + ml_data_ldn[f'ang_node_count_{d}'] * 10)

# %% clean data
logger.info('cleaning data')
# be careful with cleanup...
# per correlation plot:
# remove most extreme values - 0.99 gives good balance, 0.999 is stronger for some but generally weaker?
# ml_data_ldn_clean = util.remove_outliers_quantile(ml_data_ldn, q=(0, 0.999))
# remove rows where all nan
ml_data_ldn_clean = phd_util.clean_pd(ml_data_ldn, drop_na='all')
# due to the nature of the data fill nans with 0
ml_data_ldn_clean = ml_data_ldn_clean.fillna(0)

# %% EXPERIMENTAL
# calculate the accuracies and feature importances
cols_ml = [
    'met_closeness_200',
    'met_closeness_400',
    'met_closeness_800',
    'met_closeness_1600',
    'met_rt_complex_200',
    'met_rt_complex_400',
    'met_rt_complex_800',
    'met_rt_complex_1600',
    'met_betw_200',
    'met_betw_400',
    'met_betw_800',
    'met_betw_1600'
]
cols_ml_labels = [
    r'Closeness $_{200m}$',
    r'Closeness $_{400m}$',
    r'Closeness $_{800m}$',
    r'Closeness $_{1600m}$',
    r'Route Complexity $_{200m}$',
    r'Route Complexity $_{400m}$',
    r'Route Complexity $_{800m}$',
    r'Route Complexity $_{1600m}$',
    r'Betweenness $_{200m}$',
    r'Betweenness $_{400m}$',
    r'Betweenness $_{800m}$',
    r'Betweenness $_{1600m}$'
]
logger.info(f'columns: {cols_ml}')
feature_names_dict = {}
for c, l in zip(cols_ml, cols_ml_labels):
    feature_names_dict[c] = l
print(feature_names_dict)

# TARGETS:
target_col = 'mixed_uses_score_hill_0_200'

target_labels = r'Mixed Uses $H_{0}\ _{200m}$'

# %% RANDOM FOREST EXPERIMENT

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
    n_jobs=-1,
    # max_features=8,
    # max_depth=10,
    n_estimators=100,
    oob_score=True)
# fit
random_forest.fit(training_inputs, training_targets)
# print OOB score
logger.info(f'OOB score: {random_forest.oob_score_}')
# test set prediction
pred_test = random_forest.predict(testing_inputs)
logger.info(f'r2 Score: {metrics.r2_score(testing_targets.values, pred_test)}')

'''
50 estimators:
INFO:__main__:OOB score: 0.5106835290878611
INFO:__main__:r2 Score: 0.5149982035059039

100 estimators:
INFO:__main__:OOB score: 0.6187930901560419
INFO:__main__:r2 Score: 0.6247741562486424

'''

# %% CROSS VAL RANDOM FOREST

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
    n_jobs=-1,
    n_estimators=50,
    oob_score=True)
# fit
cv_score = cross_val_score(
    random_forest,
    training_inputs,
    training_targets,
    n_jobs=-1
)
logger.info(f'cross val predict: {cv_score}')
# don't yet understand cross_val_predict vs cross_val_score

'''
this is against validation data so higher than vault test data...?
INFO:__main__:cross val predict: [0.59700871 0.59740385 0.59367786]
'''

# %% GRADIENT BOOST EXPERIMENT

from sklearn.ensemble import GradientBoostingRegressor

logger.info(f'target column: {target_col}')

# drop rows where any nan for target column
d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

data_target = d[target_col]
data_input = d[cols_ml]

training_inputs, testing_inputs, training_targets, testing_targets = \
    train_test_split(data_input, data_target, train_size=0.8)

# create
gradient_boost = GradientBoostingRegressor()
# fit
gradient_boost.fit(training_inputs, training_targets)
# print OOB score
logger.info(f'OOB score: {random_forest.oob_score_}')
# test set prediction
pred_test = gradient_boost.predict(testing_inputs)
logger.info(f'r2 Score: {metrics.r2_score(testing_targets.values, pred_test)}')

# insert into dict
results_dict[target_col] = {
    'target': target_label,
    'accuracy': metrics.r2_score(testing_targets, pred_test),
    'features': [],
    'scores': []
}
'''
INFO:__main__:OOB score: 0.6187930901560419
INFO:__main__:r2 Score: 0.4863801347504012
'''

# %% PCA EXPERIMENT

logger.info(f'target column: {target_col}')

# drop rows where any nan for target column
d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

data_target = d[target_col]
data_input = d[cols_ml]

from sklearn.decomposition import PCA

pca = PCA(n_components=7)
data_input_reduced = pca.fit_transform(data_input)
print(data_input_reduced.shape)

'''
n_components = 2
INFO:__main__:OOB score: 0.08163829599113548
INFO:__main__:r2 Score: 0.10695331475077685

n_components = 3
INFO:__main__:OOB score: 0.18986606584415378
INFO:__main__:r2 Score: 0.219924858470944

n_components = 4
INFO:__main__:OOB score: 0.23865409984044583
INFO:__main__:r2 Score: 0.2646916576032272

n_components = 5
INFO:__main__:OOB score: 0.5256753173561167
INFO:__main__:r2 Score: 0.5453126811297413

n_components = 6
INFO:__main__:OOB score: 0.5616174311025728
INFO:__main__:r2 Score: 0.580771358003247

n_components = 7
INFO:__main__:OOB score: 0.5721957748032339
INFO:__main__:r2 Score: 0.5895602115776086
'''

training_inputs, testing_inputs, training_targets, testing_targets = \
    train_test_split(data_input_reduced, data_target, train_size=0.8)

# create
random_forest = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=50,
    oob_score=True)
# fit
random_forest.fit(training_inputs, training_targets)
# print OOB score
logger.info(f'OOB score: {random_forest.oob_score_}')
# test set prediction
pred_test = random_forest.predict(testing_inputs)
logger.info(f'r2 Score: {metrics.r2_score(testing_targets.values, pred_test)}')

# %% PCA EXPERIMENT

logger.info(f'target column: {target_col}')

# drop rows where any nan for target column
d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

data_target = d[target_col]
data_input = d[cols_ml]

from sklearn.decomposition import PCA

pca = PCA(n_components=6)
data_input_reduced = pca.fit_transform(data_input)
print(data_input_reduced.shape)

training_inputs, testing_inputs, training_targets, testing_targets = \
    train_test_split(data_input_reduced, data_target, train_size=0.9)

# create
random_forest = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=200,
    oob_score=True)
# fit
random_forest.fit(training_inputs, training_targets)
# print OOB score
logger.info(f'OOB score: {random_forest.oob_score_}')
# test set prediction
pred_test = random_forest.predict(testing_inputs)
logger.info(f'r2 Score: {metrics.r2_score(testing_targets.values, pred_test)}')

'''
INFO:__main__:OOB score: 0.5931388392423118
INFO:__main__:r2 Score: 0.6027495787782489
'''

# %% LOCALLY LINEAR EMBEDDING EXPERIMENT

# %% PCA EXPERIMENT

logger.info(f'target column: {target_col}')

# drop rows where any nan for target column
d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

data_target = d[target_col]
data_input = d[cols_ml]

from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=6, n_neighbors=10)
data_input_reduced = lle.fit_transform(data_input)
print(data_input_reduced.shape)

'''
n_components = 2
INFO:__main__:OOB score: 0.10967029946550311
INFO:__main__:r2 Score: 0.11603280828152451

n_components = 6
INFO:__main__:OOB score: 0.09774585420780646
INFO:__main__:r2 Score: 0.11700414485757149

'''

training_inputs, testing_inputs, training_targets, testing_targets = \
    train_test_split(data_input_reduced, data_target, train_size=0.8)

# create
random_forest = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=50,
    oob_score=True)
# fit
random_forest.fit(training_inputs, training_targets)
# print OOB score
logger.info(f'OOB score: {random_forest.oob_score_}')
# test set prediction
pred_test = random_forest.predict(testing_inputs)
logger.info(f'r2 Score: {metrics.r2_score(testing_targets.values, pred_test)}')

# %% MDS EXPERIMENT

logger.info(f'target column: {target_col}')

# drop rows where any nan for target column
d = ml_data_ldn_clean[ml_data_ldn_clean[target_col] != np.nan]
logger.info(f'Dropping rows where target data is nan. Start rows: {len(ml_data_ldn_clean)}. End rows: {len(d)}.')

data_target = d[target_col]
data_input = d[cols_ml]

from sklearn.manifold import MDS

mds = MDS(n_components=6)
data_input_reduced = mds.fit_transform(data_input)
print(data_input_reduced.shape)

'''
n_components = 2

'''

training_inputs, testing_inputs, training_targets, testing_targets = \
    train_test_split(data_input_reduced, data_target, train_size=0.8)

# create
random_forest = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=50,
    oob_score=True)
# fit
random_forest.fit(training_inputs, training_targets)
# print OOB score
logger.info(f'OOB score: {random_forest.oob_score_}')
# test set prediction
pred_test = random_forest.predict(testing_inputs)
logger.info(f'r2 Score: {metrics.r2_score(testing_targets.values, pred_test)}')
