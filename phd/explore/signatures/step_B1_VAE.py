# %%
'''
Trains the VAE models

- transform the data, otherwise betweenness exaggerates a small error / loss
- VAE is a strong form of regularisation - therefore (except for b=0) regularisation (dropout etc.) is not necessary.
- batch normalisation added on deep networks - pay attention to training flag
- Zero initialisation on log_var (1 via exp + noise) works fairly well
- Tried Zero, RandomNormal(stddev=0.001), GlorotNormal for mean initialisation
  smaller start values gives 'smoother' but 'smaller' outcomes - but probably converge more slowly.
'''
import pathlib
from importlib import reload
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from explore.theme_setup import data_path, logs_path, weights_path
from explore.theme_setup import generate_theme
from explore.signatures import sig_models, sig_model_runners
import phd_util
from explore import plot_funcs

reload(sig_models)
reload(sig_model_runners)
reload(phd_util)
reload(plot_funcs)

# load data and prep
df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')
table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)
test_idx = phd_util.train_test_idxs(df_20, 200)  # 200 gives about 25%

# setup paramaters
epochs = 25
batch = 256
theme_base = f'VAE_e{epochs}'
n_d = len(distances)
split_input_dims = (int(5 * n_d), int(18 * n_d), int(4 * n_d))
split_latent_dims = (6, 8, 2)
split_hidden_layer_dims = ([128, 128, 128],
                           [256, 256, 256],
                           [64, 64, 64])
latent_dim = 8
seed = 0
beta = 4
cap = 12
lr = 1e-3

# iterate
vae = sig_models.SplitVAE(raw_dim=X_trans.shape[1],
                          latent_dim=latent_dim,
                          beta=beta,
                          capacity=cap,
                          epochs=epochs,
                          split_input_dims=split_input_dims,
                          split_latent_dims=split_latent_dims,
                          split_hidden_layer_dims=split_hidden_layer_dims,
                          theme_base=theme_base,
                          seed=seed,
                          name='VAE')
dir_path = pathlib.Path(weights_path / f'seed_{seed}/{vae.theme}_epochs_{epochs}_batch_{batch}_train')
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
                                        best_loss=True,
                                        lr=lr,
                                        save_path=dir_path,
                                        test_indices=test_idx)
trainer.train()
#
if seed <= 1:
    Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)
    # single latent kl divergence, i.e. don't sum over latents
    kl_loss = -0.5 * (1 + Z_log_var - np.square(Z_mu) - np.exp(Z_log_var.numpy().astype('float64')))
    # average kl divergence per latent
    kl_vec = np.mean(kl_loss, axis=0)
    # plot
    phd_util.plt_setup()
    fig, axes = plt.subplots(2, 8, figsize=(12, 8))
    for ax_row, inverse in zip(axes, [False, True]):
        L = np.copy(Z_mu)
        if inverse:
            L *= -1
        for latent_idx in range(latent_dim):
            ax = ax_row[latent_idx]
            plot_funcs.plot_scatter(ax,
                                    table.x,
                                    table.y,
                                    L[:, latent_idx],
                                    s_min=0,
                                    s_max=0.8,
                                    c_exp=3,
                                    s_exp=5)
            ax.set_title(f'Dim. {latent_idx + 1} - ' + r'$D_{KL}=' + f'{kl_vec[latent_idx]:.1f}$')
            if inverse:
                ax.set_title(f'Dim. {latent_idx + 1} - Inverse')
    plt.suptitle(r"Geographic mappings (and inverses) for the latents of the VAE model")
    path = f'../../phd-admin/PhD/part_3/images/signatures/latents_map_{beta}_{cap}_{seed}_{theme_base}.png'
    plt.savefig(path, dpi=300)
