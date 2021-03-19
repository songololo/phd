# %%
'''

'''
import asyncio
import logging


import numpy as np
import pandas as pd
from src import phd_util
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import power_transform


from src.explore import plot_funcs

reload(plot_funcs)

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cityseer cmap
cityseer_cmap = phd_util.cityseer_cmap()
cityseer_div_cmap = phd_util.cityseer_diverging_cmap()

# db connection params
db_config = {
    'host': 'localhost',
    # 'host': 'host.docker.internal',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}

# %%

centrality_columns = [
    'id',
    'city_pop_id'
]

centrality_columns_dist = [
    'c_node_dens_{dist}',
    'c_harm_close_{dist}',
    'c_imp_close_{dist}',
    'c_gravity_{dist}',
    'c_cycles_{dist}',
    'c_between_{dist}',
    'c_between_wt_{dist}',
    'met_info_ent_{dist}',
    'area_mean_wt_{dist}',
    'area_variance_wt_{dist}',
    'val_mean_wt_{dist}',
    'val_variance_wt_{dist}',
    'rate_mean_wt_{dist}',
    'rate_variance_wt_{dist}',
    'cens_tot_pop_{dist}',
    'cens_employed_{dist}',
    'cens_dwellings_{dist}'
]

# landuse columns
landuse_columns = [
    # for table - only need specific distances (but only where not already covered in columns below)
    'id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y'
]

landuse_columns_dist = [
    # 'mu_hill_0_{dist}',
    # 'mu_hill_1_{dist}',
    # 'mu_hill_2_{dist}',
    'mu_hill_branch_wt_0_{dist}',
    'mu_hill_branch_wt_1_{dist}',
    'mu_hill_branch_wt_2_{dist}',
    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    # 'ac_eating_nw_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    # 'ac_commercial_nw_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_government_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    # 'ac_retail_food_nw_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}',
    'ac_total_{dist}',
    # 'ac_total_nw_{dist}'
]

# selected landuse distances are distinct from the selected centrality columns
distances = [200, 400, 800, 1600]

for d in distances:
    centrality_columns += [c.format(dist=d) for c in centrality_columns_dist]
    landuse_columns += [c.format(dist=d) for c in landuse_columns_dist]

columns = centrality_columns + landuse_columns

# %% load data

print(f'loading columns: {columns}')
df_data_cent_full = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_full', 'WHERE city_pop_id = 1'))
df_data_cent_full = df_data_cent_full.set_index('id')
df_data_landuse_full = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_full_lu', 'WHERE city_pop_id = 1'))
df_data_landuse_full = df_data_landuse_full.set_index('id')
df_data_full = pd.concat([df_data_cent_full, df_data_landuse_full], axis=1, verify_integrity=True)

df_data_cent_100 = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_100', 'WHERE city_pop_id = 1'))
df_data_cent_100 = df_data_cent_100.set_index('id')
df_data_landuse_100 = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_100_lu', 'WHERE city_pop_id = 1'))
df_data_landuse_100 = df_data_landuse_100.set_index('id')
df_data_100 = pd.concat([df_data_cent_100, df_data_landuse_100], axis=1, verify_integrity=True)

df_data_cent_50 = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_50', 'WHERE city_pop_id = 1'))
df_data_cent_50 = df_data_cent_50.set_index('id')
df_data_landuse_50 = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_50_lu', 'WHERE city_pop_id = 1'))
df_data_landuse_50 = df_data_landuse_50.set_index('id')
df_data_50 = pd.concat([df_data_cent_50, df_data_landuse_50], axis=1, verify_integrity=True)

df_data_cent_20 = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, centrality_columns, 'analysis.roadnodes_20', 'WHERE city_pop_id = 1'))
df_data_cent_20 = df_data_cent_20.set_index('id')
df_data_landuse_20 = asyncio.run(
    phd_util.load_data_as_pd_df(db_config, landuse_columns, 'analysis.roadnodes_20_lu', 'WHERE city_pop_id = 1'))
df_data_landuse_20 = df_data_landuse_20.set_index('id')
df_data_20 = pd.concat([df_data_cent_20, df_data_landuse_20], axis=1, verify_integrity=True)

#  %% clean data
df_full_clean = phd_util.clean_pd(df_data_full, drop_na='all', fill_inf=np.nan)
df_100_clean = phd_util.clean_pd(df_data_100, drop_na='all', fill_inf=np.nan)
df_50_clean = phd_util.clean_pd(df_data_50, drop_na='all', fill_inf=np.nan)
df_20_clean = phd_util.clean_pd(df_data_20, drop_na='all', fill_inf=np.nan)

# %% prepare columns
selected_columns_raw = [
    # 'c_node_dens_{dist}',
    # 'c_harm_close_{dist}',
    # 'c_imp_close_{dist}',
    # 'c_gravity_{dist}',
    # 'c_cycles_{dist}',
    # 'c_between_{dist}',
    # 'c_between_wt_{dist}',
    # 'met_info_ent_{dist}',
    # 'area_mean_wt_{dist}',
    # 'area_variance_wt_{dist}',
    # 'val_mean_wt_{dist}',
    # 'val_variance_wt_{dist}',
    # 'rate_mean_wt_{dist}',
    # 'rate_variance_wt_{dist}',
    # 'cens_tot_pop_{dist}',
    # 'cens_employed_{dist}',
    # 'cens_dwellings_{dist}',
    'mu_hill_branch_wt_0_{dist}',
    'ac_accommodation_{dist}',
    'ac_eating_{dist}',
    'ac_drinking_{dist}',
    'ac_commercial_{dist}',
    'ac_tourism_{dist}',
    'ac_entertainment_{dist}',
    'ac_government_{dist}',
    'ac_manufacturing_{dist}',
    'ac_retail_food_{dist}',
    'ac_retail_other_{dist}',
    'ac_transport_{dist}',
    'ac_health_{dist}',
    'ac_education_{dist}',
    'ac_parks_{dist}',
    'ac_cultural_{dist}',
    'ac_sports_{dist}',
    'ac_total_{dist}'
]

selected_columns = []
for c in selected_columns_raw:
    for d in distances:
        selected_columns.append(c.format(dist=d))

selected_columns_labels = []
for c in selected_columns_raw:
    lb = c.replace('ac_', '')
    lb = lb.replace('c_', '')
    lb = lb.replace('wt_', '')
    lb = lb.replace('mu_hill_branch_wt_0', 'mixed_uses')
    lb = lb.replace('_{dist}', '')
    lb = lb.replace('accommodation', 'accommod.')
    lb = lb.replace('total', 'all_landuses')
    selected_columns_labels.append(lb)

# %%
'''
2A - Choose level of network decomposition and plot heatmaps:
Show amount of correlation and respective ordering vs. strongest PCA components
'''

table = df_data_20
X = table[selected_columns]

# impute missing values for smaller agged census stats
imp = SimpleImputer()
X = imp.fit_transform(X)

# PCA centers the data but doesn't scale - use whiten param to scale to unit variance
X = power_transform(X, method='yeo-johnson', standardize=True)
n_components = 12

model = PCA(n_components=n_components, whiten=True)
theme = 'PCA_lu_only'

# X = minmax_scale(X)
# from sklearn.decomposition import NMF
# model = NMF(n_components=n_components)
# theme = 'NMF_no_cent'

# from sklearn.decomposition import SparsePCA
# model = SparsePCA(n_components=n_components)
# theme = 'sparse_PCA'

# from sklearn.decomposition import MiniBatchSparsePCA
# model = MiniBatchSparsePCA(n_components=n_components)
# theme = 'minibatch_sparse_PCA'

# from sklearn.decomposition import FastICA
# model = FastICA(n_components=n_components)
# theme = 'fast_ICA'

# CRASHES
# from sklearn.decomposition import KernelPCA
# model = KernelPCA(n_components=n_components, kernel='poly')
# theme = 'kernel_PCA'

# reduce
X_reduced = model.fit_transform(X, )
assert X_reduced.shape[1] == n_components

# project back - not available with all models
# X_restored = model.inverse_transform(X_reduced)


# %%
'''
2B - Plot
'''
path = f'./explore/F_feature_extraction/exploratory_plots/all_dim_reduct_{theme}.png'
plot_funcs.plot_components(n_components,
                           selected_columns_labels,
                           distances,
                           X,
                           X_reduced,
                           table.x,
                           table.y,
                           path,
                           double_row=True,
                           title=theme)

# %%
'''
3 - autoencoder
- transform the data, otherwise betweenness exaggerates a small error / loss
- unconstrained - should be similar to PCA
- add regularisation - not helping on shallow
- add dropout - seems to hurt rather than help
- batch normalisation seems to help a bit
- deep - helps, a bit
- try variational
'''
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Dense, Input, Reshape, Conv2D, MaxPooling2D, Flatten, UpSampling2D
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime
import tensorflow as tf

log_dir = './tensorboard_logs/feature_extraction_lu/' + datetime.now().strftime('%Y%m%d-%H%M%S')
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

tensor_board = TensorBoard(log_dir=log_dir,
                           histogram_freq=0,
                           write_graph=True,
                           write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=0.001)

table = df_data_20

X_raw = table[selected_columns]
X_trans = power_transform(X_raw, method='yeo-johnson', standardize=True)

raw_dims = X_trans.shape[1]
# inner_dims = 12
double_rows = False

# %%
for inner_dims in list(range(1, 13)):

    #  %% SIMPLE
    double_rows = False
    if inner_dims > 6:
        double_rows = True

    autoencoder_simple = Sequential()
    autoencoder_simple.add(Dense(inner_dims,
                                 activation='elu',
                                 input_shape=(raw_dims,)))
    autoencoder_simple.add(Dense(raw_dims,
                                 activation='linear'))

    autoencoder_simple.compile(optimizer='adam',
                               loss='mae',
                               metrics=['mae', 'mse'])

    autoencoder_simple.fit(X_trans, X_trans,
                           epochs=5,
                           batch_size=512,
                           shuffle=True,
                           validation_split=0.1,
                           callbacks=[tensor_board, early_stopping])

    encoder_simple = Sequential()
    encoder_simple.add(autoencoder_simple.layers[0])

    decoder_simple = Sequential()
    decoder_simple.add(autoencoder_simple.layers[1])

    #  %%
    theme = 'autoencoder_simple'

    print('encoding')
    encoded_data_simple = encoder_simple.predict(X_trans)

    print('decoding')
    decoded_data_simple = decoder_simple.predict(encoded_data_simple)

    print('plotting')
    path = f'./explore/F_feature_extraction/exploratory_plots/lu_dim_reduct_{theme}_{inner_dims}.png'
    plot_funcs.plot_components(inner_dims,
                               selected_columns_labels,
                               distances,
                               X_trans,
                               encoded_data_simple,
                               table.x,
                               table.y,
                               path,
                               double_row=double_rows)

    #  %% DEEP
    df = 'channels_last'
    data_total_dim = X_trans.shape[1]
    data_rows = len(selected_columns_labels)
    data_cols = len(distances)

    data_input = Input(shape=(data_total_dim,))
    l = Reshape((data_rows, data_cols, 1))(data_input)
    l = Conv2D(16, (1, 5), activation='elu', padding='same', data_format=df)(l)
    l = MaxPooling2D((1, 2), padding='same', data_format=df)(l)
    l = Conv2D(8, (1, 3), activation='elu', padding='same', data_format=df)(l)
    l = Flatten()(l)
    l = Dense(64, activation='elu')(l)
    l = Dense(32, activation='elu')(l)
    encoded = Dense(inner_dims, activation='elu')(l)

    l = Dense(32, activation='elu')(encoded)
    l = Dense(64, activation='elu')(l)
    l = Dense(720, activation='elu')(l)
    l = Reshape((18, 5, 8))(l)
    l = Conv2D(16, (1, 3), activation='elu', padding='same', data_format=df)(l)
    l = UpSampling2D((1, 2), data_format=df)(l)
    l = Conv2D(1, (1, 5), activation='linear', padding='same', data_format=df)(l)
    decoded = Reshape((data_total_dim,))(l)

    autoencoder_conv = Model(data_input, decoded)
    autoencoder_conv.compile(optimizer='adam',
                             loss='mae',
                             metrics=['mae', 'mse'])

    autoencoder_conv.fit(X_trans, X_trans,
                         epochs=5,
                         batch_size=512,
                         shuffle=True,
                         validation_split=0.1,
                         callbacks=[tensor_board, early_stopping],
                         verbose=True)

    #  %%
    encoder_conv = Sequential()
    for layer in autoencoder_conv.layers[:9]:
        encoder_conv.add(layer)

    theme = 'autoencoder_conv'

    print('encoding')
    encoded_data_conv = encoder_conv.predict(X_trans)

    print('plotting')
    path = f'./explore/F_feature_extraction/exploratory_plots/lu_dim_reduct_{theme}_{inner_dims}.png'
    plot_funcs.plot_components(inner_dims,
                               selected_columns_labels,
                               distances,
                               X_trans,
                               encoded_data_conv,
                               table.x,
                               table.y,
                               path,
                               double_row=double_rows)
