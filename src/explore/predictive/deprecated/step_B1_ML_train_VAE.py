# %%
'''

'''
import pathlib

import numpy as np
import pandas as pd
from src import phd_util
from src.explore import plot_funcs
from src.explore.signatures import sig_models, sig_model_runners
from src.explore.theme_setup import data_path, logs_path, weights_path, generate_theme
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

#  %%
# load data and prep
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
X_raw, distances, labels = generate_theme(df_20, 'pred_sim', bandwise=False)
#
epochs = 3
latent_dim = 16
theme = 'vae'
beta = 4
cap = 0
seed = 0
batch = 256
lr = 1e-3
transformer = StandardScaler()
X_trans = transformer.fit_transform(X_raw).astype(np.float32)
test_idx = phd_util.train_test_idxs(df_20, 200)  # 200 gives about 25%
#
vae = sig_models.VAE(X_raw.shape[1],
                     latent_dim=latent_dim,
                     beta=beta,
                     capacity=cap,
                     epochs=epochs,
                     theme_base=theme,
                     seed=seed)
dir_path = pathlib.Path(weights_path / f'{vae.theme}')
dir_path.mkdir(exist_ok=True, parents=True)
l_path = logs_path
if seed > 1:
    l_path = None

trainer = sig_model_runners.VAE_trainer(model=vae,
                                        X_samples=X_trans,
                                        labels=labels,
                                        distances=distances,
                                        logs_path=l_path,
                                        epochs=epochs,
                                        batch=batch,
                                        val_split=0.2,
                                        best_loss=True,
                                        lr=lr,
                                        save_path=dir_path,
                                        test_indices=test_idx)
trainer.train()
#  %%
Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)
# single latent kl divergence, i.e. don't sum over latents
kl_loss = -0.5 * (1 + Z_log_var - np.square(Z_mu) - np.exp(Z_log_var.numpy().astype('float64')))
# average kl divergence per latent
kl_vec = np.mean(kl_loss, axis=0)
# latent kl labels
kl_labels = []
for idx in range(latent_dim):
    kl_labels.append(f'Dim. {idx + 1} - ' + r'$D_{KL}=' + f'{kl_vec[idx]:.1f}$')
#  %% plot
plot_funcs.plot_components(list(range(latent_dim)),
                           labels,
                           distances,
                           X_trans,
                           Z_mu.numpy(),
                           df_20.x,
                           df_20.y,
                           tag_values=kl_labels)

plt.suptitle(r"Geographic mappings and correlations for the latents of the VAE model")
path = f'../../phd-admin/PhD/part_3/images/predicted/{vae.theme}.png'
plt.savefig(path, dpi=300)
