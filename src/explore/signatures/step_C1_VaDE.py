# %%
'''
Runs VaDE model
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import util_funcs
from src.explore import plot_funcs
from src.explore.signatures import sig_models, sig_model_runners
from src.explore.theme_setup import data_path, logs_path, weights_path, columns_lu, columns_cent, columns_cens
from src.explore.theme_setup import generate_theme

#  %% load data and prep
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True, max_dist=800)

# can't do bandwise if masking...!
# if masking - experimental
# for col_name in columns_cent:
#     for dist in (100, 200, 300):
#         X_raw.loc[:, col_name.format(dist=dist)] = 0.0
# for col_name in columns_lu:
#     for dist in (1200, 1600):
#         X_raw.loc[:, col_name.format(dist=dist)] = 0.0
# for col_name in columns_cens:
#     for dist in (100,):
#         X_raw.loc[:, col_name.format(dist=dist)] = 0.0

X_trans = StandardScaler().fit_transform(X_raw)
test_idx = util_funcs.train_test_idxs(df_20, 200)  # gives about 25%

# %% setup paramaters
seed = 0
latent_dim = 32
dropout = 0
epochs = 25
n_components = 21
theme_base = f'VaDE'
vade = sig_models.VaDE(raw_dim=X_trans.shape[1],
                       latent_dim=latent_dim,
                       n_components=n_components,
                       theme_base=theme_base,
                       seed=seed,
                       dropout=dropout)
dir_path = pathlib.Path(weights_path / f'signatures_weights/seed_{seed}/{vade.theme}_d{latent_dim}_e{epochs}')
if not dir_path.exists():
    trainer = sig_model_runners.VaDE_trainer(model=vade,
                                             X_samples=X_trans,
                                             labels=labels,
                                             distances=distances,
                                             logs_path=logs_path,
                                             epochs=epochs,
                                             best_loss=True,
                                             save_path=dir_path,
                                             test_indices=test_idx)

    trainer.train()
else:
    vade.load_weights(str(dir_path / 'weights'))

# %% classify
cluster_probs_VaDE = vade.classify(X_trans).numpy()
cluster_assign_VaDE = np.argmax(cluster_probs_VaDE, axis=1)
#  %% plot cluster matrix
plot_funcs.plot_prob_clusters(X_raw,
                              cluster_probs_VaDE,
                              n_components,
                              f'{vade.theme}_d{latent_dim}_e{epochs}_comp{n_components}_clust_ind',
                              table.x,
                              table.y)
# plot combined map
util_funcs.plt_setup()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colours, sizes = plot_funcs.map_diversity_colour(X_raw, cluster_assign_VaDE, n_components=n_components)
plot_funcs.plot_scatter(ax,
                        table.x,
                        table.y,
                        c=colours,
                        s=sizes,
                        x_extents=(-7000, 11500),
                        y_extents=(-4500, 7500),
                        rasterized=True)
path = f'../phd-doc/doc/part_3/signatures/images/{vade.theme}_d{latent_dim}_e{epochs}_comp{n_components}_clust_comb.pdf'
plt.savefig(path, dpi=300)
