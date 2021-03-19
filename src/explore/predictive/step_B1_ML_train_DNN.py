# %%
'''
Trains three ML models for three respective landuses:
- aims to predict landuse probability based on approximated population flows

100 epochs back to non-split
r2 predicted accuracy for ac_commercial_400: 0.93
r2 predicted accuracy for ac_manufacturing_400: 0.15
r2 predicted accuracy for ac_retail_food_400: 0.86
r2 predicted accuracy for ac_accommodation_400: 0.49

r2 predicted accuracy for ac_commercial_800: 0.98
r2 predicted accuracy for ac_manufacturing_800: 0.83
r2 predicted accuracy for ac_retail_food_800: 0.96
r2 predicted accuracy for ac_accommodation_800: 0.74

r2 predicted accuracy for ac_commercial_1600: 0.99
r2 predicted accuracy for ac_manufacturing_1600: 0.96
r2 predicted accuracy for ac_retail_food_1600: 0.98
r2 predicted accuracy for ac_accommodation_1600: 0.94

100 epochs with different distances
r2 predicted accuracy for ac_commercial_400: 0.93
r2 predicted accuracy for ac_manufacturing_800: 0.83
r2 predicted accuracy for ac_retail_food_200: 0.6
r2 predicted accuracy for ac_accommodation_1600: 0.94

r2 predicted accuracy for ac_commercial_800: 0.98
r2 predicted accuracy for ac_manufacturing_1600: 0.93
r2 predicted accuracy for ac_retail_food_400: 0.82

r2 predicted accuracy for ac_commercial_400: 0.92
r2 predicted accuracy for ac_manufacturing_800: 0.76
r2 predicted accuracy for ac_eating_200: 0.71
'''
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
from src import phd_util
from src.explore.predictions import pred_models
from src.explore.theme_setup import data_path, logs_path, weights_path, generate_theme
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, TerminateOnNaN, \
    ModelCheckpoint

#  %%
# load data and prep
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
X_raw, distances, labels = generate_theme(df_20, 'pred_sim', bandwise=False)

# run once for each target
epochs = 100
targets = ['ac_commercial_{dist}', 'ac_manufacturing_{dist}', 'ac_eating_{dist}']
target_distances = [400, 800, 200]
theme = 'london'
column_order = ['c_node_harmonic_angular_{dist}',
                'c_node_betweenness_beta_{dist}',
                'ac_commercial_{dist}',
                'ac_manufacturing_{dist}',
                'ac_eating_{dist}',
                'cens_dwellings_{dist}']
for target, target_dist in zip(targets, target_distances):
    # get y
    target_w_dist = target.format(dist=target_dist)
    print(f'Training for target column: {target_w_dist}')
    y_all = df_20[target_w_dist].values
    # drop y from X
    drop_cols = []
    for dist in distances:
        drop_cols.append(target.format(dist=dist))
    print(f'Dropping ancillary columns: {drop_cols}')
    # transform X
    X_dropped = X_raw.drop(columns=drop_cols)
    transformer = StandardScaler()
    X_trans_all = transformer.fit_transform(X_dropped).astype(np.float32)
    # test split - use spatial splitting - 300 modulo gives about 11%
    xy_test_idx = phd_util.train_test_idxs(df_20, 300)
    X_trans_train = X_trans_all[~xy_test_idx]
    X_trans_test = X_trans_all[xy_test_idx]
    y_train = y_all[~xy_test_idx]
    y_test = y_all[xy_test_idx]
    # validation split - 200 modulo gives about 25%
    xy_val_idx = phd_util.train_test_idxs(df_20[~xy_test_idx], 200)
    X_trans_val = X_trans_train[xy_val_idx]  # do first before repurposing variable name
    X_trans_train = X_trans_train[~xy_val_idx]
    y_val = y_train[xy_val_idx]  # do first before repurposing variable name
    y_train = y_train[~xy_val_idx]
    #
    reg = pred_models.LandUsePredictor(theme_base=f'{target_w_dist}_e{epochs}_{theme}_{target_dist}')


    # r2 metric
    def r2(y_true, y_pred):
        num = K.sum(K.square(y_true - y_pred))
        den = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - num / (den + K.epsilon()))


    # compile
    reg.compile(optimizer='adam',
                loss=losses.MeanSquaredError(),
                metrics=['mean_squared_error', r2])
    # prepare path and check for previously trained model
    dir_path = pathlib.Path(weights_path / f'{reg.theme}')
    history_path = dir_path / 'history.json'
    if not dir_path.exists():
        # prepare callbacks
        callbacks = [
            TensorBoard(
                log_dir=str(
                    logs_path / f'{datetime.now().strftime("%Hh%Mm%Ss")}_{reg.theme}'),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=2,
                embeddings_freq=0,
                embeddings_metadata=None),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0),
            TerminateOnNaN(),
            ModelCheckpoint(
                str(dir_path / 'weights'),
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                save_freq='epoch')
        ]
        # train
        history = reg.fit(x=X_trans_train,
                          y=y_train,
                          batch_size=256,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(X_trans_val, y_val),
                          shuffle=True,
                          callbacks=callbacks)
        with open(history_path, 'w') as json_out:
            hist_float = history.history
            for k in hist_float:
                hist_float[k] = [float(v) for v in hist_float[k]]
            json.dump(hist_float, json_out)
    else:
        with open(history_path) as h_file:
            hist_float = json.load(h_file)
        reg.load_weights(str(dir_path / 'weights'))

    #  %%
    y_hat_test = reg.predict(X_trans_test)
    y_score_r2 = round(r2_score(y_test, y_hat_test), 2)
    print(f'r2 predicted accuracy for {target_w_dist}: {y_score_r2}')
