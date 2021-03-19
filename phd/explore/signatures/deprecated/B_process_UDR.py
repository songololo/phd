# %%
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from explore.theme_setup import data_path, weights_path
from explore.theme_setup import generate_theme
from explore.signatures.udr import udr_custom

df_20 = pd.read_feather(data_path / 'df_20.feather')
df_20 = df_20.set_index('id')

table = df_20
X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
X_trans = StandardScaler().fit_transform(X_raw)

random_state = np.random.RandomState(0)
latent_dim = 6
inner_theme = 'simple'
epochs = 10
gammas = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # , 512
caps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
seeds = list(range(1, 9))
udr_dict = {}
losses = ['val_loss',
          'val_vae_loss',
          'val_rec_mse_metric',
          'val_kl_metric',
          'val_kl_metric_beta',
          'val_kl_metric_cap']

for gamma in gammas:
    for cap in caps:

        theme = f'vae_autoencoder_all_{inner_theme}_{latent_dim}d_gamma_{gamma}_cap_{cap}_new'
        print('')
        print(theme)

        inferred_model_reps = []
        kl_vecs = []

        key = f'all_simple_{latent_dim}d_g{gamma}_c{cap}'
        for seed in seeds:
            print(f'...loading data from model seed {seed}')

            X_latent = np.load(weights_path / f'data/all_simple_{latent_dim}d_g{gamma}_c{cap}_s{seed}_latent.npy')
            z_mu_output = np.load(weights_path / f'data/all_simple_{latent_dim}d_g{gamma}_c{cap}_s{seed}_z_mu.npy')
            z_log_var_output = np.load(weights_path / f'data/all_simple_{latent_dim}d_g{gamma}_c{cap}_s{seed}_z_log_var.npy')

            with open(f'temp_weights/set seed {seed}/model_{theme}_history.json') as f:
                hist_data = json.load(f)

            '''
            getting average kl divergence per individual latent:

            See equation 3 on page 5 of paper, which matches typical VAE closed-form equation for KL divergence...
            e.g.: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
            −DKL(qϕ(z|x)||pθ(z))=1/2∑(1+log(σ2)−μ2−σ2)
            Seems that this is ordinarily summed over the latent dimensions... but notice that the paper is PER latent

            Further, see the code comments in udr.py where the function signature states that the kl divergence vector
            should be: "a vector of the average kl divergence per latent"
            '''

            # single latent kl divergence, i.e. don't sum over latents
            kl_loss = -0.5 * (1 + z_log_var_output - np.square(z_mu_output) - np.exp(z_log_var_output))
            # average kl divergence per latent
            kl_vector = np.mean(kl_loss, axis=0)

            inferred_model_reps.append(X_latent)
            kl_vecs.append(kl_vector)
            
            if key not in udr_dict:
                udr_dict[key] = {}
            for loss in losses:
                if loss not in udr_dict[key]:
                    udr_dict[key][loss] = []
                loss_keys = [k for k in hist_data[loss].keys()]
                max_iter = loss_keys[-1]
                udr_dict[key][loss].append(hist_data[loss][max_iter])

        # filter out non finite values
        before_len = inferred_model_reps[0].shape[0]
        not_finite_idx = np.zeros(before_len, dtype=bool)
        for inferred_model in inferred_model_reps:
            for l_idx in range(latent_dim):
                not_finite_idx = np.logical_or(not_finite_idx, ~np.isfinite(inferred_model[:, l_idx]))
        after_len = np.sum(~not_finite_idx)
        if after_len == 0:
            print(f'NO FINITE VALUES: UNABLE TO PROCESS {theme}')
            continue
        elif after_len != before_len:
            print(f'DROPPED {before_len - after_len} NON FINITE ROWS...')
        # filter out
        for i, inferred_model in enumerate(inferred_model_reps):
            inferred_model_reps[i] = inferred_model[~not_finite_idx, :]
            assert inferred_model_reps[i].shape[0] == after_len

        print('...calculating UDR')
        udr = udr_custom.compute_udr_sklearn(inferred_model_reps,
                                             kl_vecs,
                                             random_state,
                                             correlation_matrix="lasso",
                                             filter_low_kl=True,
                                             include_raw_correlations=True,
                                             kl_filter_threshold=25)

        udr_dict[key]['udr'] = udr['model_scores']
        print('model scores:', udr['model_scores'])

idx = [k for k in udr_dict.keys()]
cols = None
data = []
for key in idx:
    vals = udr_dict[key]
    # columns are nested one deep
    if cols is None:
        cols = [k for k in vals.keys()]
    # unpack the values
    theme_data = [vals[c] for c in cols]
    data.append(theme_data)

df = pd.DataFrame(data=data, columns=cols, index=idx)
print(df)
df.to_json(f'./model_scores.json')
