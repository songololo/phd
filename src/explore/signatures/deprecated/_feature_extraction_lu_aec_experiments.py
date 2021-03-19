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

from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, LSTM, RepeatVector, Permute
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime

early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)
# X_trans = X_trans_filtered
raw_dims = X_trans.shape[1]

# %%
# inner_dims = 3
for inner_dims in list(range(1, 13)):

    model_key = datetime.now().strftime('%Hh%Mm') + f'_simple_{inner_dims}_dims'
    log_dir = f'./tensorboard_logs/feature_extraction_lu/{model_key}'
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=False)

    #  %% SIMPLE
    double_rows = False
    if inner_dims > 6:
        double_rows = True

    # encoder
    input_data = Input(shape=(X_trans.shape[1],))
    # non-linear
    encoded = Dense(inner_dims, activation='elu')(input_data)
    # map to linear output for visualisation
    encoded = Dense(inner_dims, activation='linear')(encoded)

    # decoder
    decoded = Dense(raw_dims, activation='elu')(encoded)
    # map to linear for output
    decoded = Dense(raw_dims, activation='linear')(decoded)

    autoencoder_simple = Model(input_data, decoded)
    autoencoder_simple.compile(optimizer='adam',
                               loss='mae',
                               metrics=['mae', 'mse'])

    autoencoder_simple.fit(X_trans, X_trans,
                           epochs=10,
                           batch_size=1024,
                           shuffle=True,
                           validation_split=0.1,
                           callbacks=[tensor_board, early_stopping])

    #  %%
    print('encoding')
    encoder_simple = Model(input_data, encoded)
    encoded_data_simple = encoder_simple.predict(X_trans)

    print('plotting')
    theme = f'autoencoder_simple_{inner_dims}_dims'
    path = f'./explore/5_signatures/exploratory_plots/lu_dim_reduct_{theme}.png'
    mae = autoencoder_simple.history.history['mean_absolute_error'][-1]
    mae = round(mae, 4)
    title = f'{theme}_MAE:{mae}'
    plot_funcs.plot_components(inner_dims,
                               selected_columns_labels,
                               distances,
                               X_trans,
                               encoded_data_simple,
                               table.x,
                               table.y,
                               path,
                               double_row=double_rows,
                               title=title)

    #  %% DEEP
    model_key = datetime.now().strftime('%Hh%Mm') + f'_conv_{inner_dims}_dims'
    log_dir = f'./tensorboard_logs/feature_extraction_lu/{model_key}'
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=False)

    n_feat = 16
    n_dist = 10

    data_format = 'channels_last'
    # encoder
    input_data = Input(shape=(X_trans.shape[1],))
    encoded = Reshape((n_feat, n_dist, 1))(input_data)
    encoded = Conv2D(16, (1, 5), activation='elu', padding='same', data_format=data_format)(encoded)
    encoded = MaxPooling2D((1, 2), padding='same', data_format=data_format)(encoded)
    encoded = Conv2D(8, (1, 3), activation='elu', padding='same', data_format=data_format)(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(64, activation='elu')(encoded)
    encoded = Dense(32, activation='elu')(encoded)
    # map to linear output for visualisation
    encoded = Dense(inner_dims, activation='linear')(encoded)

    # decoder
    decoded = Dense(32, activation='elu')(encoded)
    decoded = Dense(64, activation='elu')(decoded)
    decoded = Dense(n_feat * 5 * 8, activation='elu')(decoded)
    decoded = Reshape((n_feat, 5, 8))(decoded)
    # map to linear for output
    decoded = Conv2D(16, (1, 3), activation='elu', padding='same', data_format=data_format)(decoded)
    decoded = UpSampling2D((1, 2), data_format=data_format)(decoded)
    decoded = Conv2D(1, (1, 5), activation='linear', padding='same', data_format=data_format)(decoded)
    decoded = Reshape((n_feat * n_dist,))(decoded)

    autoencoder_conv = Model(input_data, decoded)

    autoencoder_conv.compile(optimizer='adam',
                             loss='mae',
                             metrics=['mae', 'mse'])

    autoencoder_conv.fit(X_trans, X_trans,
                         epochs=10,
                         batch_size=1024,
                         shuffle=True,
                         validation_split=0.1,
                         callbacks=[tensor_board, early_stopping],
                         verbose=True)

    #  %%
    print('encoding')
    encoder_conv = Model(input_data, encoded)
    encoded_data_conv = encoder_conv.predict(X_trans)

    print('plotting')
    theme = f'autoencoder_conv_{inner_dims}_dims'
    path = f'./explore/5_signatures/exploratory_plots/lu_dim_reduct_{theme}.png'
    mae = autoencoder_conv.history.history['mean_absolute_error'][-1]
    mae = round(mae, 4)
    title = f'{theme}_MAE:{mae}'
    plot_funcs.plot_components(inner_dims,
                               selected_columns_labels,
                               distances,
                               X_trans,
                               encoded_data_conv,
                               table.x,
                               table.y,
                               path,
                               double_row=double_rows,
                               title=title)

    #  %% Sequence to Sequence RNN over distance dimensions
    model_key = datetime.now().strftime('%Hh%Mm') + f'_lstm_{inner_dims}_dims'
    log_dir = f'./tensorboard_logs/feature_extraction_lu/{model_key}'
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=False)

    n_feat = 16
    n_dist = 10

    # encoder
    data_input = Input(shape=(X_trans.shape[1],))
    # reshape to number of features by number of distances
    encoded = Reshape((n_feat, n_dist))(data_input)
    # LSTM requires steps (distances) by features order
    encoded = Permute((2, 1), input_shape=(n_feat, n_dist))(encoded)
    encoded = LSTM(64)(encoded)
    # map to linear for visualisation
    encoded = Dense(inner_dims, activation='linear')(encoded)
    # decode steps
    decoded = Dense(n_feat)(encoded)
    decoded = RepeatVector(n_dist)(decoded)
    decoded = LSTM(64, return_sequences=True)(decoded)
    # permute back to features by distances order
    decoded = Permute((2, 1), input_shape=(n_dist, n_feat))(decoded)
    # flatten to original
    decoded = Flatten()(decoded)
    # map to linear for output
    decoded = Dense(n_feat * n_dist, activation='linear')(decoded)

    autoencoder_lstm = Model(data_input, decoded)
    autoencoder_lstm.compile(optimizer='adam',
                             loss='mae',
                             metrics=['mae', 'mse'])

    autoencoder_lstm.fit(X_trans, X_trans,
                         epochs=10,
                         batch_size=1024,
                         shuffle=True,
                         validation_split=0.1,
                         callbacks=[tensor_board, early_stopping],
                         verbose=True)

    #  %%
    print('encoding')
    encoder_lstm = Model(data_input, encoded)
    encoded_data_lstm = encoder_lstm.predict(X_trans)

    print('plotting')
    theme = f'autoencoder_lstm_{inner_dims}_dims'
    path = f'./explore/5_signatures/exploratory_plots/lu_dim_reduct_{theme}.png'
    mae = autoencoder_lstm.history.history['mean_absolute_error'][-1]
    mae = round(mae, 4)
    title = f'{theme}_MAE:{mae}'
    plot_funcs.plot_components(inner_dims,
                               selected_columns_labels,
                               distances,
                               X_trans,
                               encoded_data_lstm,
                               table.x,
                               table.y,
                               path,
                               double_row=double_rows,
                               title=title)

# %%
from sklearn.manifold import TSNE

perplexity = 350
manifold = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=20, angle=0.5)
man = manifold.fit_transform(zs)
zs = man

# %%
theme = f'man_{perplexity}'

# for simple 2 latent dimensional cases
util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(zs[:, 0], zs[:, 1], s=1, alpha=0.05)
for loc, vals in location_keys.items():
    idx = vals['filtered_iloc']
    ax.scatter(zs[idx][0], zs[idx][1], s=3, alpha=1, label=loc)
ax.legend()
path = f'./explore/5_signatures/exploratory_plots/vae_{theme}_b{beta}_l{latent_dim}_lstm.png'
plt.suptitle(theme)
plt.savefig(path, dpi=300)
plt.show()

# %%
for d in range(zs.shape[1]):
    util.plt_setup(dark=True)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    plot_funcs.plot_scatter(ax, table.x, table.y, zs[:, d], constrain=True)
    theme = f'vae_LSTM_lat_dim_{d}'
    path = f'./explore/5_signatures/exploratory_plots/{theme}_b{beta}_l{latent_dim}.png'
    plt.suptitle(theme)
    plt.savefig(path, dpi=300)
    plt.show()

# %%
util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(man[:, 0], man[:, 1], s=1, alpha=0.05)
plt.show()

util.plt_setup(dark=True)
fig, axes = plt.subplots(1, 2, figsize=(20, 20))
axes[0].hist(man[:, 0], bins=1000, alpha=0.6)
axes[1].hist(man[:, 1], bins=1000, alpha=0.6)
plt.show()

# %%
util.plt_setup(dark=True)
fig, ax_rows = plt.subplots(3, 3, figsize=(20, 20))
counter = 0
zd = encoded_vae[2]
for ax_row in ax_rows:
    for ax in ax_row:
        name = location_keys[counter][0]
        z_key = location_keys[counter][1]
        arr = decoder.predict(z_key)
        arr = np.reshape(arr, (len(selected_columns_labels), len(distances)))
        plot_funcs.plot_heatmap(ax, arr, selected_columns_labels, distances, set_ylabels=True)
        counter += 1
plt.show()

# %%
restored = decoder.predict(zs)

util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
plot_funcs.plot_scatter(ax, table.x, table.y, zs[:, 0], constrain=False)
plt.show()

# %%
from sklearn.mixture import GaussianMixture

components = 10
cluster = GaussianMixture(n_components=components,
                          n_init=1,
                          max_iter=100)
z_clusters = cluster.fit_predict(zs[:, 4:6])

# %%
from palettable.cartocolors.qualitative import Bold_10

cmap = Bold_10.mpl_colormap

util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(zs[:, 4], zs[:, 5], s=1, c=z_clusters, alpha=0.05, cmap=cmap)
plt.show()

# %%
util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
plot_funcs.plot_scatter(ax, table.x, table.y, z_clusters, cmap=cmap, constrain=False)

theme = f'vae_LSTM_cluster_gaussian_{components}_4and5'
path = f'./explore/5_signatures/exploratory_plots/{theme}_b{beta}_l{latent_dim}.png'
plt.suptitle(theme)
plt.savefig(path, dpi=450)
plt.show()

# %%
from sklearn.mixture import BayesianGaussianMixture

components = 6
cluster = BayesianGaussianMixture(n_components=components,
                                  n_init=1,
                                  max_iter=100,
                                  # weight_concentration_prior=0.05,
                                  weight_concentration_prior_type='dirichlet_process')
z_clusters = cluster.fit_predict(zs)

# %%
from palettable.cartocolors.qualitative import Bold_6

cmap = Bold_6.mpl_colormap
theme = f'vae_LSTM_cluster_bayesian_{components}'

util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(zs[:, 0], zs[:, 1], s=1, c=z_clusters, alpha=0.05, cmap=cmap)
plt.suptitle(theme)
plt.show()

# %%
util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
plot_funcs.plot_scatter(ax, table_filtered.x, table_filtered.y, z_clusters, cmap=cmap, constrain=False)

path = f'./explore/5_signatures/exploratory_plots/{theme}_b{beta}_l{latent_dim}.png'
plt.suptitle(theme)
plt.savefig(path, dpi=450)
plt.show()

# %%
from joblib import Memory
import hdbscan

# ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'haversine']
cluster_size = 2000
min_samples = 1
leaf_size = 40
method = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                      min_samples=min_samples,
                      metric='euclidean',
                      leaf_size=leaf_size,
                      cluster_selection_method=method,
                      alpha=1.0,
                      core_dist_n_jobs=7,
                      memory=Memory(cachedir='./explore/5_signatures/vae/'))
hdb.fit(zs)
z_clusters = hdb.labels_
labels = set(z_clusters)
print(labels)
print(len(labels))

# %%
from palettable.cartocolors.qualitative import Bold_7

cmap = Bold_7.mpl_colormap
theme = f'vae_LSTM_cluster_hdbscan_{cluster_size}_{min_samples}_{leaf_size}_{method}'

util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(zs[:, 0], zs[:, 1], c=z_clusters, s=1, alpha=0.05, cmap=cmap)
for loc, vals in location_keys.items():
    idx = vals['filtered_iloc']
    ax.scatter(zs[idx][0], zs[idx][1], s=3, alpha=1, label=loc)
ax.legend()
path = f'./explore/5_signatures/exploratory_plots/vae_hdbscan_asdf_b{beta}_l{latent_dim}.png'
plt.suptitle(theme)
plt.savefig(path, dpi=300)
plt.show()

# %%
util.plt_setup(dark=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
plot_funcs.plot_scatter(ax, table_filtered.x, table_filtered.y, z_clusters, cmap=cmap, constrain=False)

path = f'./explore/5_signatures/exploratory_plots/{theme}_full.png'
plt.suptitle(theme)
plt.savefig(path, dpi=450)
plt.show()
