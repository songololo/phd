# %%
'''
Trains three ML models for three respective landuses

linearSVM C=0.5
r2 predicted accuracy for ac_eating_400 on test set: 0.49
r2 predicted accuracy for ac_commercial_400 on test set: 0.64
r2 predicted accuracy for ac_manufacturing_400 on test set: 0.1
'''
import pathlib
from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

import phd_util
from explore import plot_funcs
from explore.theme_setup import data_path, weights_path, generate_theme
from explore.predictions import pred_tools

seed = 123
# load data and prep
df_20 = pd.read_feather(data_path / 'df_full_all.feather')
df_20 = df_20.set_index('id')
X_raw, distances, labels = generate_theme(df_20, 'pred_sim', bandwise=False)

targets = ['ac_eating_{d}', 'ac_commercial_{d}', 'ac_manufacturing_{d}']
target_labels = ['eating', 'commerc.', 'manuf.']
target_dist = 400


#%%
# plot PCA for info
phd_util.plt_setup()
max_plotted_components = 7
fig, axes = plt.subplots(3, max_plotted_components, figsize=(10, 4))
for ax_row, target, label in zip(axes, targets, target_labels):

    # drop target columns
    drop_cols = []
    for dist in distances:
        drop_cols.append(target.format(d=dist))
    print(f'Dropping ancillary columns: {drop_cols}')
    X_dropped = X_raw.drop(columns=drop_cols)

    # find index of label and remove
    drop_labels = [l for l in labels]
    drop_labels.remove(label)
    print(f'Dropping ancillary labels: {drop_labels}')

    # PCA
    X_trans = StandardScaler().fit_transform(X_dropped)
    model = PCA(n_components=0.9, random_state=seed)
    X_latent = model.fit_transform(X_trans)
    print(f'Number of components: {model.components_.shape[0]}')
    assert model.components_.shape[0] <= max_plotted_components

    # explained variance
    exp_var = model.explained_variance_
    exp_var_ratio = model.explained_variance_ratio_
    # eigenvector by eigenvalue - i.e. correlation to original
    loadings = model.components_.T * np.sqrt(exp_var)
    loadings = loadings.T  # transform for slicing
    exp_var_str = [f'{e_v:.1%}' for e_v in exp_var_ratio]
    # plot
    for i in range(max_plotted_components):
        if i + 1 > model.components_.shape[0]:
            ax_row[i].set_axis_off()
        else:
            l = loadings[i]
            l = l.reshape(len(drop_labels), len(distances))
            plot_funcs.plot_heatmap(ax_row[i],
                                    l,
                                    row_labels=drop_labels,
                                    col_labels=distances,
                                    set_row_labels=(i == 0),
                                    set_col_labels=True,
                                    cbar=False)
            # label
            ax_row[i].set_xlabel(f'{label} comp: {i + 1}')

plt.suptitle('PCA of training variables for respective target variables')
path = f'../../phd-admin/PhD/part_3/images/predicted/pca.png'
plt.savefig(path, dpi=300)


#%% pipeline
pipeline = Pipeline([('standardise', StandardScaler()),
                     ('pca', PCA(n_components=0.9, random_state=seed)),
                     ('model', svm.SVR(C=0.5, verbose=1))])  # random state for linear

#  %%
# run once for each target
for target in targets:

    # get y
    target_w_dist = target.format(d=target_dist)
    print(f'Training for target column: {target_w_dist}')
    y_all = df_20[target_w_dist].values

    # drop y from X
    drop_cols = []
    for dist in distances:
        drop_cols.append(target.format(d=dist))
    print(f'Dropping ancillary columns: {drop_cols}')
    X_dropped = X_raw.drop(columns=drop_cols)

    # test split - use spatial splitting - 300 modulo gives about 11%
    xy_test_idx = phd_util.train_test_idxs(df_20, 300)
    X_train = X_dropped[~xy_test_idx]
    X_test = X_dropped[xy_test_idx]
    y_train = y_all[~xy_test_idx]
    y_test = y_all[xy_test_idx]

    # prepare path and check for previously trained model
    path = pathlib.Path(weights_path / f'{target_w_dist}_SVR.skmodel')
    if not path.exists():
        print('training')
        pipeline.fit(X_train, y_train)
        dump(pipeline, path)
    else:
        print('loading')
        pipeline = load(path)

    # score on test data
    y_score_r2 = round(pipeline.score(X_test, y_test), 2)
    print(f'r2 predicted accuracy for {target_w_dist} on test set: {y_score_r2}')
