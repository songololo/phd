# %%
'''
Predicts a selected land use vibrancy from centralities and census data
'''
import json
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from src.explore.predictions import pred_models
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, TerminateOnNaN, \
    ModelCheckpoint

from src import phd_util
from src.explore import plot_funcs
from src.explore.theme_setup import data_path, logs_path, weights_path
from src.explore.theme_setup import generate_theme

#  %%
# load data and prep
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
# generate theme
X_raw, distances, labels = generate_theme(df_20, 'pred_lu', bandwise=True)
# transform X
X_trans_all = StandardScaler().fit_transform(X_raw).astype(np.float32)
# get y
y_all = df_20['ac_eating_400'].values
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

#  %%
epochs = 100
reg = pred_models.LandUsePredictor(theme_base=f'eating_e{epochs}')


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
# split score between 400 vs 1600 threshold
y_score_r2 = round(r2_score(y_test, y_hat_test), 2)
print('r2 accuracy on test set at 400m', y_score_r2)

#  %%
x_extents = (-4000, 6500)
y_extents = (-8000, 10000)
extents_idx, _, _, _, _ = plot_funcs.view_idx(df_20.x,
                                              df_20.y,
                                              x_extents=x_extents,
                                              y_extents=y_extents,
                                              relative_extents=True)
x_extents = (-2000, 4500)
y_extents = (-6000, 8000)
X_trans_idx = X_trans_all[extents_idx]
y_all_idx = y_all[extents_idx]
y_pred = reg.predict(X_trans_idx, verbose=1).flatten()
y_diff = y_pred - y_all_idx

#  %%
phd_util.plt_setup()
fig, axes = plt.subplots(1, 3, figsize=(12, 8))
for ax_idx, (ax, vals, title) in enumerate(zip(axes,
                                               (y_all_idx, y_pred, y_diff),
                                               ('Observed eateries',
                                                'Predicted eateries',
                                                'Differenced'))):
    if ax_idx == 2:
        mm = 20
        vmin, vmax = -mm, mm
        cmap = phd_util.cityseer_diverging_cmap()
        c = vals
        s = 1
    else:
        # have to use same min max for observed vs predicted
        mm = max([abs(np.nanmin(y_pred)), np.nanmax(y_pred),
                  abs(np.nanmin(y_all_idx)), np.nanmax(y_all_idx)])
        vmin, vmax = 0, mm
        cmap = LinearSegmentedColormap.from_list('reds', ['#ffffff', '#ff6659', '#d32f2f'])
        pow = 1.25
        c = vals ** pow
        s = 0.01 + (vals ** pow / vmax * 1.5)

    im = plot_funcs.plot_scatter(ax,
                                 df_20.x[extents_idx],
                                 df_20.y[extents_idx],
                                 c=c,
                                 s=s,
                                 x_extents=x_extents,
                                 y_extents=y_extents,
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=cmap)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    plt.colorbar(im, cax=cax)

plt.suptitle(f'Observed, predicted, differenced eateries at 400m threshold. Test set accuracy: {y_score_r2:.1%} r2')
path = f'../phd-admin/PhD/part_3/images/predicted/eatery_predictions_400_{reg.theme}.png'
plt.savefig(path, dpi=300)

#  %%
phd_util.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

x_arr = np.array(range(1, len(hist_float['mean_squared_error']) + 1))
# adjust training metric by half - computed during epoch
ax.plot(x_arr - 0.5,
        hist_float['r2'],
        color='blue',
        marker='o',
        label='training')
# validation loss is computed after epoch
ax.plot(x_arr,
        hist_float['val_r2'],
        color='green',
        linestyle='--',
        marker='x',
        label='validation')

ax.set_xlabel('Epochs')
ax.set_xlim(0, np.max(x_arr))
ax.set_ylabel('$r^2$')
ax.set_ylim(0.5, 1)
ax.legend(loc='upper right', prop={'size': 5})
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

plt.suptitle(f'Learning Curve and Validation Curve')
path = f'../phd-admin/PhD/part_3/images/predicted/validation_curve_{reg.theme}.png'
plt.savefig(path, dpi=300)
