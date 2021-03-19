'''
This was an attempt to map centrality data directly to the latents of a variational autoencoder

ONLY WORKS FOR RELATIVELY SIMPLE 2d CASES - OTHER MORE COMPLEX CASES WOULD REQUIRE MORE DATA
Basically, certain landuse nuances are relatively complex and path-dependent and do not depend on x, y, z
So only simple cases of general district diversity and high-street diversity (to a lesser extent) can be estimated
'''

# %%
import os
import logging
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer

from keras.layers import Lambda, Input, Dense, Reshape, Flatten, Permute, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector
from keras.models import Model
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K

from importlib import reload
import phd_util

reload(phd_util)
from explore import plot_funcs

reload(plot_funcs)

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cityseer_cmap = phd_util.cityseer_cmap()

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
columns = [
    'id',
    'ST_X(geom) as x',
    'ST_Y(geom) as y'
]

cens_columns_dist = [
    'cens_tot_pop_{dist}',
    'cens_employed_{dist}',
    'cens_dwellings_{dist}'
]
for dist in [200, 400, 800, 1600]:
    for cens_column in cens_columns_dist:
        columns.append(cens_column.format(dist=dist))

voa_columns_dist = [
    'area_mean_wt_{dist}',
    'area_variance_wt_{dist}',
    #'val_mean_wt_{dist}',
    #'val_variance_wt_{dist}',
    #'rate_mean_wt_{dist}',
    #'rate_variance_wt_{dist}'
]
for dist in [50, 100, 200, 400, 800, 1600]:
    for column in voa_columns_dist:
        columns.append(column.format(dist=dist))

cent_columns_dist = [
    #'met_info_ent_{dist}',
    'c_imp_close_{dist}',
    'c_gravity_{dist}',
    'c_between_{dist}',
    'c_between_wt_{dist}',
    #'c_cycles_{dist}'
]
for dist in [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
    for column in cent_columns_dist:
        columns.append(column.format(dist=dist))


#%%
# load data
df_data_cent_20 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    columns,
    'analysis.roadnodes_20',
    'WHERE city_pop_id = 1'))
df_data_cent_20 = df_data_cent_20.set_index('id')


lu_columns = [
    'id'
]
lu_columns_dist = [
    'ac_transport_{dist}'
]
for dist in [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
    for column in lu_columns_dist:
        lu_columns.append(column.format(dist=dist))

df_data_lu_20 = asyncio.run(phd_util.load_data_as_pd_df(
    db_config,
    lu_columns,
    'analysis.roadnodes_20_lu',
    'WHERE city_pop_id = 1'))
df_data_lu_20 = df_data_lu_20.set_index('id')

merged_data = pd.concat([df_data_cent_20, df_data_lu_20], axis=1, verify_integrity=True)
merged_data = phd_util.clean_pd(merged_data, drop_na='all', fill_inf=np.nan)
merged_data.sort_index(inplace=True)


# %%
# prepare landuses VAE model
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from datetime import datetime
from keras.callbacks import TensorBoard, EarlyStopping
from explore.F_feature_extraction import vae_model

reload(vae_model)

latent_dim = 2
theme = 'simple'
vae_features = 15
vae_distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]
encoder, decoder, vae, model_key = vae_model.prepare_vae_model(vae_features,
                                                               len(vae_distances),
                                                               theme,
                                                               latent_dim)
encoder.load_weights(f'./explore/F_feature_extraction/vae/model_{model_key}_encoder.h5')
decoder.load_weights(f'./explore/F_feature_extraction/vae/model_{model_key}_decoder.h5')
vae.load_weights(f'./explore/F_feature_extraction/vae/model_{model_key}.h5')

# load landuses transformed data
vae_X_trans = np.load(f'./explore/F_feature_extraction/data/lu_power_transformed_raw_data.npy')
# encode data - this will be used for training
encoded_vae = encoder.predict(vae_X_trans)
zs = encoded_vae[2]
assert len(zs) == len(merged_data)


#%% prepare examplars data (from vae_workflow.py)
vae_columns = [
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
    'ac_cultural_{dist}',
    'ac_sports_{dist}'
]

vae_column_labels = []
for c in vae_columns:
    lb = c.replace('ac_', '')
    lb = lb.replace('mu_hill_branch_wt_0', 'mixed_uses')
    lb = lb.replace('_{dist}', '')
    lb = lb.replace('accommodation', 'accomod.')
    lb = lb.replace('entertainment', 'entert.')
    lb = lb.replace('manufacturing', 'manuf.')
    vae_column_labels.append(lb)

locations = [
    'seven_dials',
    'oxford_circus',
    'primrose_hill',
    'dartmouth',
    'crouch_end',
    'chatsworth_road',
    'hackney',
    'angel',
    'farringdon',
    'mahmed'
]

indices = [
    2286249,
    3012772,
    2994647,
    3067461,
    3363559,
    2531836,
    2620342,
    2363541,
    2309160,
    2285682
]

location_keys = {}
for loc, idx in zip(locations, indices):
    location_keys[loc] = {
        'idx': idx,
        'iloc': merged_data.index.get_loc(idx),
        'x': merged_data.x.iloc[merged_data.index == idx],
        'y': merged_data.y.iloc[merged_data.index == idx]
    }


#%% prints example decodings
maxes = np.max(np.abs(vae_X_trans), axis=0)
for z_key in [(-0.6, 3.59), (1.82, 2.45)]:
    z_arr = np.array([[*z_key]])
    phd_util.plt_setup(dark=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    arr = decoder.predict(z_arr)
    arr /= maxes
    arr = np.reshape(arr, (len(vae_column_labels), len(vae_distances)))
    plot_funcs.plot_heatmap(ax, arr, vae_column_labels, vae_distances, set_labels=False, dark=True)
    ax.set_yticks(list(range(len(vae_column_labels))))
    ax.set_yticklabels(vae_column_labels, rotation='horizontal', fontsize='x-large')
    ax.set_xticks(list(range(len(vae_distances))))
    ax.set_xticklabels(vae_distances, rotation='vertical', fontsize='x-large')
    path = f'./explore/G_supervised_learning/plots/lu_decode_example_{z_key[0]}_{z_key[1]}_t{theme}_d{latent_dim}.png'
    plt.savefig(path, dpi=300)
    plt.show()


# %%
# prepare centrality data
selected_columns = []
selected_columns_dist = [
    'cens_tot_pop_{dist}',
    #'cens_employed_{dist}',
    #'cens_dwellings_{dist}',
    #'area_mean_wt_{dist}',
    #'area_variance_wt_{dist}',
    'ac_transport_{dist}',
    'c_imp_close_{dist}',
    #'c_gravity_{dist}',
    'c_between_{dist}',
    #'c_between_wt_{dist}'
]
for column in selected_columns_dist:
    for dist in [200, 400, 800, 1600]:
        selected_columns.append(column.format(dist=dist))

X_raw = merged_data[selected_columns]
imp = SimpleImputer()
X_raw = imp.fit_transform(X_raw)
X_trans = PowerTransformer().fit_transform(X_raw)
raw_dims = len(selected_columns)


# %%
model_key = f'cent_vae_link'
early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=3)
log_key = f'{datetime.now().strftime("%Hh%Mm")}_{model_key}'
log_dir = f'./tensorboard_logs/supervised/{log_key}'
tensor_board = TensorBoard(log_dir=log_dir, write_graph=False)

data_format = 'channels_last'
model_key = f'vae_{theme}_{latent_dim}'

input_data = Input(shape=(raw_dims,), name='input')
reg = Dense(32, activation='relu')(input_data)
reg = BatchNormalization()(reg)
reg = Dense(64, activation='relu')(reg)
reg = Dropout(0.1)(reg)
reg = BatchNormalization()(reg)
reg = Dense(32, activation='relu')(reg)
reg = BatchNormalization()(reg)
reg = Dense(latent_dim, activation='linear')(reg)

model = Model(input_data, reg, name='encoder')
model.compile(optimizer='adam', metrics=['mae', 'mse'], loss='mae')
model.summary()

model.fit(X_trans,
          zs,
          batch_size=1024,
          epochs=20,
          shuffle=True,
          validation_split=0.1,
          callbacks=[tensor_board, early_stopping],
          verbose=True)

pred_latent_dims = model.predict(X_trans)


#%% prints the actual vs. centrality predicted vae encodings
for loc, loc_data in location_keys.items():
    iloc = loc_data['iloc']
    print(loc)
    print(zs[iloc].round(2))
    print(pred_latent_dims[iloc].round(2))


#%% plots the centrality to VAE latent dims mappings
for dim in range(latent_dim):
    phd_util.plt_setup(dark=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    plot_funcs.plot_scatter(axes[0], merged_data.x, merged_data.y, -pred_latent_dims[:, dim], s=1.5)
    plot_funcs.plot_scatter(axes[1], merged_data.x, merged_data.y, -zs[:, dim], s=1.5)

    plt.suptitle('')
    path = f'./explore/G_supervised_learning/plots/pred_vae_dim_{dim}.png'
    plt.savefig(path, dpi=300)
    plt.show()


#%% prints some example centrality - vae mappings in decoded form
# [200, 400, 800, 1600]
grad_a = [1, 1, 0.75, 0.5]
grad_b = [1, 1, 1, 1]
grad_c = [0, 0, 0.5, 1]

for label, grad in zip(['near', 'all', 'far'], [grad_a, grad_b, grad_c]):
    gr = np.array(grad)
    inputs = []
    counter = 0
    for c in selected_columns_dist:
        for g in gr:
            d = X_trans[counter]
            d_min = np.nanmin(d)
            d_max = np.nanmax(d)
            inputs.append((d_max - d_min) * g + d_min)
            counter += 1
    assert counter == len(selected_columns)
    z_arr = model.predict(np.array([[*inputs]]))
    phd_util.plt_setup(dark=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    arr = decoder.predict(z_arr)
    arr /= maxes
    arr = np.reshape(arr, (len(vae_column_labels), len(vae_distances)))
    plot_funcs.plot_heatmap(ax, arr, vae_column_labels, vae_distances, set_labels=False, dark=True)
    ax.set_yticks(list(range(len(vae_column_labels))))
    ax.set_yticklabels(vae_column_labels, rotation='horizontal', fontsize='x-large')
    ax.set_xticks(list(range(len(vae_distances))))
    ax.set_xticklabels(vae_distances, rotation='vertical', fontsize='x-large')
    z_0 = str(round(z_arr[0][0], 2))
    z_1 = str(round(z_arr[0][1], 2))
    path = f'./explore/G_supervised_learning/plots/cent_vae_lu_example_{label}_{z_0}_{z_1}.png'
    plt.savefig(path, dpi=300)
    plt.show()