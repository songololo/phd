# %%
'''
- Encodes latents for various VAE hyperparameters and saves to data files
- Calculates UDR entangling metric and saves to data files
'''
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.explore.signatures import sig_models
from src.explore.theme_setup import data_path, weights_path
from src.explore.theme_setup import generate_theme
from src.explore.signatures.udr import udr_custom


# %% load from disk
def process_latents_vae():
    df_20 = pd.read_feather(data_path / 'df_20.feather')
    df_20 = df_20.set_index('id')
    '''
    Processes and saves latents
    '''
    table = df_20
    seeds = list(range(1, 11))
    latent_dims = [8]
    X_raw, distances, labels = generate_theme(table, 'all', bandwise=True)
    X_trans = StandardScaler().fit_transform(X_raw)
    # prepare the splits
    n_d = len(distances)
    epochs = 10
    batch = 256
    split_input_dims = (int(5 * n_d), int(18 * n_d), int(4 * n_d))
    split_latent_dims = (6, 8, 2)
    split_hidden_layer_dims = ([128, 128, 128],
                               [256, 256, 256],
                               [64, 64, 64])
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    theme_base = f'VAE_e{epochs}'
    # iterate
    for latent_dim in latent_dims:
        for beta in betas:
            caps = [0, 4, 8, 12, 16, 20]
            if beta == 0:
                caps = [0]
            for cap in caps:
                for seed in seeds:
                    print(f'...collecting data from model seed {seed}')
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
                    print(vae.theme)
                    dir_path = pathlib.Path(
                        weights_path / f'seed_{seed}/{vae.theme}_epochs_{epochs}_batch_{batch}_train')
                    # visualise
                    vae.load_weights(str(dir_path / 'weights'))
                    print('preparing latents...')
                    # save the latents
                    Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)
                    # paths
                    p = str(weights_path / f'data/model_{vae.theme}')
                    np.save(p + '_latent', Z)
                    np.save(p + '_z_mu', Z_mu)
                    np.save(p + '_z_log_var', Z_log_var)


# %%
if __name__ == '__main__':
    process_latents_vae()

# %%
'''
Calculate UDR and set in ndarray
'''


def generate_udr_grid(latent_dim,
                      epochs,
                      seeds,
                      kl_threshold: float = 0.01,
                      random_state=np.random.RandomState(0)):
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    arr = np.full((len(betas), 6, len(seeds)), np.nan)
    mask_count_arr = np.full((len(betas), 6, len(seeds)), np.nan)
    for beta_idx, (beta) in enumerate(betas):
        caps = [0, 4, 8, 12, 16, 20]
        if beta == 0:
            caps = [0]
        for cap_idx, cap in enumerate(caps):
            # gather the latent representations and the latent kl divergences
            inferred_model_reps = []
            kl_vecs = []
            for seed in seeds:
                key = f'model_VAE_e{epochs}_d{latent_dim}_b{beta}_c{cap}_s{seed}'
                print(f'...loading data for {key}')
                Z = np.load(weights_path / f'data/{key}_latent.npy')
                inferred_model_reps.append(Z)
                '''
                getting average kl divergence per individual latent:
    
                See equation 3 on page 5 of paper, which matches typical VAE closed-form equation for KL divergence...
                e.g.: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
                −DKL(qϕ(z|x)||pθ(z))=1/2∑(1+log(σ2)−μ2−σ2)
                Seems that this is ordinarily summed over the latent dimensions... but notice that the paper is PER latent
    
                Further, see the code comments in udr.py where the function signature states that the kl divergence vector
                should be: "a vector of the average kl divergence per latent"
                '''
                Z_mu = np.load(weights_path / f'data/{key}_z_mu.npy')
                Z_log_var = np.load(weights_path / f'data/{key}_z_log_var.npy')
                # single latent kl divergence, i.e. don't sum over latents
                kl_loss = -0.5 * (1 + Z_log_var - np.square(Z_mu) - np.exp(Z_log_var))
                # average kl divergence per latent
                kl_vector = np.mean(kl_loss, axis=0)
                kl_vecs.append(kl_vector)
            # filter out non finite values
            before_len = inferred_model_reps[0].shape[0]
            not_finite_idx = np.zeros(before_len, dtype=bool)
            for inferred_model in inferred_model_reps:
                for l_idx in range(latent_dim):
                    not_finite_idx = np.logical_or(not_finite_idx, ~np.isfinite(inferred_model[:, l_idx]))
            after_len = np.sum(~not_finite_idx)
            if after_len == 0:
                print(f'NO FINITE VALUES: UNABLE TO PROCESS for beta: {beta} and cap: {cap}')
                continue
            elif after_len != before_len:
                print(f'DROPPED {before_len - after_len} NON FINITE SAMPLES...')
            # filter out
            for i, inferred_model in enumerate(inferred_model_reps):
                inferred_model_reps[i] = inferred_model[~not_finite_idx, :]
                assert inferred_model_reps[i].shape[0] == after_len
            print('...calculating UDR')
            udr = udr_custom.compute_udr_sklearn(inferred_model_reps,
                                                 kl_vecs,
                                                 random_state,
                                                 correlation_matrix="spearman",  # lasso throws convergence errors
                                                 filter_low_kl=True,
                                                 include_raw_correlations=True,
                                                 kl_filter_threshold=kl_threshold)
            arr[beta_idx][cap_idx] = udr['model_scores']
            mask_count_arr[beta_idx][cap_idx] = np.sum(np.array(kl_vecs) > kl_threshold, axis=1)
    return arr, mask_count_arr


# %%
'''
Calculate clustering scores and set in ndarray
Calinski and Harabasz score (much faster than silhoette score)
Also known as Variance Ratio Criterion.
'''


# https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4
# from https://stackoverflow.com/a/26079963/1190200
def gmm_js(gmm_p, gmm_q, n_samples):
    X, _labels = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y, _labels = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2


def generate_comp_grid(latent_dim,
                       epochs,
                       seeds,
                       betas,
                       caps,
                       components=tuple(range(2, 13)),
                       vade=False,
                       reg_covar=1e-6):
    cluster_scores = {
        'bic': np.full((len(betas), len(caps), len(components), len(seeds)), np.nan),
        'aic': np.full((len(betas), len(caps), len(components), len(seeds)), np.nan),
        'ch': np.full((len(betas), len(caps), len(components), len(seeds)), np.nan),
        'js': np.full((len(betas), len(caps), len(components), len(seeds)), np.nan)
    }
    for beta_idx, beta in enumerate(betas):
        for cap_idx, cap in enumerate(caps):
            # no point iterating capacity if gamma is zero...
            if beta == 0 and cap != 0:
                continue
            for seed in seeds:
                if not vade:
                    key = f'model_VAE_split_e{epochs}_d{latent_dim}_b{beta}_c{cap}_s{seed}'
                else:
                    key = f'model_VaDE_e{epochs}_d{latent_dim}_b{beta}_c{cap}_s{seed}'
                print(f'...loading data for {key}')
                Z = np.load(weights_path / f'data/{key}_latent.npy')
                # prepare sample indices - randomise index
                attempt = 0
                while attempt < 3:
                    try:
                        sample_idx = np.random.choice(Z.shape[0], int(Z.shape[0] / 2), replace=False)
                        # slice random index in half - for a quarter of all samples
                        mid_p = int(len(sample_idx) / 2 - 1)
                        sample_set_A = sample_idx[:mid_p]
                        sample_set_B = sample_idx[mid_p:]
                        # prepare samples
                        Za = Z[sample_set_A]
                        Zb = Z[sample_set_B]
                        for n_comp, n_components in enumerate(components):
                            print(f'Components: {n_components}')
                            # compute respective GMMs
                            gmmA = GaussianMixture(n_components=n_components,
                                                   covariance_type='diag',
                                                   reg_covar=reg_covar,  # up from default - avoids errors
                                                   random_state=0)
                            gmmA_labels = gmmA.fit_predict(Za)
                            gmmB = GaussianMixture(n_components=n_components,
                                                   covariance_type='diag',
                                                   reg_covar=reg_covar,
                                                   random_state=0)
                            gmmB.fit(Zb)
                            cluster_scores['bic'][beta_idx][cap_idx][n_comp][seed - 1] = gmmA.bic(Za)
                            cluster_scores['aic'][beta_idx][cap_idx][n_comp][seed - 1] = gmmA.aic(Za)
                            cluster_scores['ch'][beta_idx][cap_idx][n_comp][seed - 1] = calinski_harabasz_score(Za,
                                                                                                                gmmA_labels)
                            cluster_scores['js'][beta_idx][cap_idx][n_comp][seed - 1] = gmm_js(gmmA, gmmB, mid_p)
                    except ValueError as e:
                        print(e)
                        print('attempt', attempt)
                        attempt += 1
                        continue
                    break
    return cluster_scores
