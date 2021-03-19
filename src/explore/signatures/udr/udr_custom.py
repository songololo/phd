# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SIMPLIFIED VERSION OF udr.py

BASED ON:
Methods for computing the UDR and UDR-A2A scores specified in "Unsupervised
Model Selection for Variational Disentangled Representation Learning"
(https://arxiv.org/abs/1905.12614)

Github Repo:
https://github.com/google-research/disentanglement_lib/evaluation/udr/metrics/udr.py

This version does not do batch sampling but uses entirety of dataset
The batch sampling is based on a complicated workflow that is hard to emulate
Original paper says entire ordered dataset can be used...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import logging
from src.explore.signatures.udr import udr
from sklearn import preprocessing


def compute_udr_sklearn(inferred_model_reps,
                        kl_vecs,
                        random_state,
                        correlation_matrix="lasso",
                        filter_low_kl=True,
                        include_raw_correlations=True,
                        kl_filter_threshold=0.01):
    """Computes the UDR score using scikit-learn.

    Args:
      inferred_model_reps -> MODIFIED -> manually prepare list of latent reps
      kl_vecs -> MODIFIED -> manually prepare list of kl diverges per latent rep

      random_state: numpy random state used for randomness.
      batch_size: Number of datapoints to compute in a single batch. Useful for
        reducing memory overhead for larger models.
      num_data_points: total number of representation datapoints to generate for
        computing the correlation matrix.
      correlation_matrix: Type of correlation matrix to generate. Can be either
        "lasso" or "spearman".
      filter_low_kl: If True, filter out elements of the representation vector
        which have low computed KL divergence.
      include_raw_correlations: Whether or not to include the raw correlation
        matrices in the results.
      kl_filter_threshold: Threshold which latents with average KL divergence
        lower than the threshold will be ignored when computing disentanglement.

    Returns:
      scores_dict: a dictionary of the scores computed for UDR with the following
      keys:
        raw_correlations: (num_models, num_models, latent_dim, latent_dim) -  The
          raw computed correlation matrices for all models. The pair of models is
          indexed by axis 0 and 1 and the matrix represents the computed
          correlation matrix between latents in axis 2 and 3.
        pairwise_disentanglement_scores: (num_models, num_models, 1) - The
          computed disentanglement scores representing the similarity of
          representation between pairs of models.
        model_scores: (num_models) - List of aggregated model scores corresponding
          to the median of the pairwise disentanglement scores for each model.
    """

    num_models = len(inferred_model_reps)
    logging.info("Number of Models: %s", num_models)

    logging.info("Training sklearn models.")
    latent_dim = inferred_model_reps[0].shape[1]
    corr_matrix_all = np.zeros((num_models, num_models, latent_dim, latent_dim))

    # Normalize and calculate mask based off of kl divergence to remove
    # uninformative latents.
    kl_mask = []
    for i in range(len(inferred_model_reps)):
        scaler = preprocessing.StandardScaler()
        scaler.fit(inferred_model_reps[i])
        inferred_model_reps[i] = scaler.transform(inferred_model_reps[i])
        inferred_model_reps[i] = inferred_model_reps[i] * np.greater(kl_vecs[i], 0.01)
        kl_mask.append(kl_vecs[i] > kl_filter_threshold)

    disentanglement = np.zeros((num_models, num_models, 1))
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue

            if correlation_matrix == "lasso":
                corr_matrix = udr.lasso_correlation_matrix(inferred_model_reps[i], inferred_model_reps[j], random_state)
            else:
                corr_matrix = udr.spearman_correlation_conv(inferred_model_reps[i], inferred_model_reps[j])

            corr_matrix_all[i, j, :, :] = corr_matrix
            if filter_low_kl:
                corr_matrix = corr_matrix[kl_mask[i], ...][..., kl_mask[j]]
            """
            MODIFIED - added check for zero latent dimensions - throws error for zero dimensions otherwise
            """
            if not corr_matrix.shape[0] or not corr_matrix.shape[1]:
                disentanglement[i, j] = np.nan
            else:
                disentanglement[i, j] = udr.relative_strength_disentanglement(corr_matrix)

    scores_dict = {}
    if include_raw_correlations:
        scores_dict["raw_correlations"] = corr_matrix_all.tolist()
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()

    model_scores = []
    for i in range(num_models):
        model_scores.append(np.median(np.delete(disentanglement[:, i], i)))

    scores_dict["model_scores"] = model_scores

    return scores_dict
