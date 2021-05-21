# Gareth Simons' PhD codebase

_Detection and prediction of urban archetypes at the pedestrian scale: computational toolsets, morphological metrics, and machine learning methods._

Completed in 2021, though this work was developed over the span of the preceding five years during the PhD.

This work is also available through the accompanying papers released in the `arXiv` preprint repository:
- links pending.

### Dataset preparation:

Data is primarily derived from:
- _Ordnance Survey_ _Open Roads_
- _Ordnance Survey_ _Points of Interest_
- _Office for National Statistics_ census data

Workflows (not included in this repository) load the data into a `PostGIS` enabled `Postgres` database from which the following steps proceed.

### Processing of morphological measures 
A series of scripts processes the files and saves the derived data to a PostGIS enabled Postgres database.

- Network Centralities:
  - [src/process/metrics/centrality/centrality_primal.py](src/process/metrics/centrality/centrality_primal.py)
  - [src/process/metrics/centrality/centrality_dual.py](src/process/metrics/centrality/centrality_dual.py)
- Landuse accessibilities and mixed-uses:
  - [src/process/metrics/landuses/landuses_poi.py](src/process/metrics/landuses/landuses_poi.py)
  - [src/process/metrics/landuses/poi_randomiser.py](src/process/metrics/landuses/poi_randomiser.py)
- Census aggregations:
  - [src/process/metrics/radial/census_agg.py](src/process/metrics/radial/census_agg.py)
  - [src/process/metrics/radial/census_interpolation.py](src/process/metrics/radial/census_interpolation.py)

### Exploration of data derived from the measures

- `cityseer-api` plots:
  - See [https://cityseer.benchmarkurbanism.com](https://cityseer.benchmarkurbanism.com) for the `cityseer-api` docs.
  - See [https://github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api) for the `cityseer-api` repo.
  - [src/explore/cityseer/beta_comparisons.py](src/explore/cityseer/beta_comparisons.py)
  - [src/explore/cityseer/cityseer_plots.py](src/explore/cityseer/cityseer_plots.py)
- Landuse accessibilities and mixed-uses:
  - [src/explore/diversity/global_props.py](src/explore/diversity/global_props.py)
  - [src/explore/diversity/mixed_uses_os_poi.py](src/explore/diversity/mixed_uses_os_poi.py)
- Street network centralities:
  - [src/explore/centrality/centrality_plots.py](src/explore/centrality/centrality_plots.py)
  - [src/explore/centrality/centrality_plots_dual.py](src/explore/centrality/centrality_plots_dual.py)
- Unsupervised machine learning methods:
  - [src/explore/signatures/sig_model_runners.py](src/explore/signatures/sig_model_runners.py)
  - [src/explore/signatures/sig_models.py](src/explore/signatures/sig_models.py)
  - [src/explore/signatures/step_A_PCA_explained_variance.py](src/explore/signatures/step_A_PCA_explained_variance.py)
  - [src/explore/signatures/step_B1_VAE.py](src/explore/signatures/step_B1_VAE.py)
  - [src/explore/signatures/step_B2_vae_latents_UDR.py](src/explore/signatures/step_B2_vae_latents_UDR.py)
  - [src/explore/signatures/step_B3_AE_plots.py](src/explore/signatures/step_B3_AE_plots.py)
  - [src/explore/signatures/step_B4_examples.py](src/explore/signatures/step_B4_examples.py)
  - [src/explore/signatures/step_C1_VaDE.py](src/explore/signatures/step_C1_VaDE.py)
  - [src/explore/signatures/step_C2_cluster_plots.py](src/explore/signatures/step_C2_cluster_plots.py)
- Supervised machine learning methods:
  - [src/explore/predictive/global.py](src/explore/predictive/global.py)
  - [src/explore/predictive/global.py](src/explore/predictive/global.py)
  - [src/explore/predictive/kde.py](src/explore/predictive/kde.py)
  - [src/explore/predictive/pop_corr_plots.py](src/explore/predictive/pop_corr_plots.py)
  - [src/explore/predictive/pop_mu_plots.py](src/explore/predictive/pop_mu_plots.py)
  - [src/explore/predictive/pred_models.py](src/explore/predictive/pred_models.py)
  - [src/explore/predictive/pred_tools.py](src/explore/predictive/pred_tools.py)
  - [src/explore/predictive/step_A1_new_town_classifications.py](src/explore/predictive/step_A1_new_town_classifications.py)
  - [src/explore/predictive/step_A2_classifier.py](src/explore/predictive/step_A2_classifier.py)
  - [src/explore/predictive/step_A3_M2.py](src/explore/predictive/step_A3_M2.py)
  - [src/explore/predictive/step_A4_ot_nt_plots.py](src/explore/predictive/step_A4_ot_nt_plots.py)
  - [src/explore/predictive/step_B1_ML_train_DNN.py](src/explore/predictive/step_B1_ML_train_DNN.py)
  - [src/explore/predictive/step_B2_MMML_DNN.py](src/explore/predictive/step_B2_MMML_DNN.py)
  - [src/explore/predictive/step_B3_plot.py](src/explore/predictive/step_B3_plot.py)
  - [src/explore/predictive/step_D_pred_LU_vibrancy.py](src/explore/predictive/step_D_pred_LU_vibrancy.py)
- Toy Models:
  - [src/explore/toy_models/step_C1_graphs.py](src/explore/toy_models/step_C1_graphs.py)
  - [src/explore/toy_models/step_C2_MMM.py](src/explore/toy_models/step_C2_MMM.py)
  - [src/explore/toy_models/step_C3_plot.py](src/explore/toy_models/step_C3_plot.py)
