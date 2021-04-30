# %%
'''
STEP 3:
M1 + M2 doesn't work well - latent representations don't lend themselves to OT vs NT
Hybrid optimisation of M1 + M2 works better - but numerical instability led to issues
M2 takes longer to optimise, but works, and is stable...

Experiment with clipnorm, batch sizes, learning rate to reduce instability.

Outputs from Step 2 classifier are pruned to only the strongest probabilities of OT and NT
These are then used as labelled inputs to this model, with the remainder of samples treated as unlabelled

Pruning is tricky so as not to bias classifier - see notes for prune_y() method...

Also, be cognisant that aggressive pruning may require increase of batch sizes...

5.16% samples above upper threshold
25.72% samples below lower threshold
removed 69.12% labels
preparing balanced set
equalising lower band by removing 1073122 samples
kept 269381 in upper and lower bands for given threshold
sample size for balanced: 10.32%
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from src.explore.predictive import pred_tools

from src import util_funcs
from src.explore import plot_funcs

#  %%
# load boundaries
selected_data = pred_tools.load_bounds()
# load data
max_dist = 1600
df_20, X_raw, X_clean, distances, labels = pred_tools.load_data(selected_data,
                                                                max_dist=max_dist)
X_trans = StandardScaler().fit_transform(X_clean)
# test train split
test_idx = util_funcs.train_test_idxs(df_20, 200)  # 200 gives about 25%
X_trans_train = X_trans[~test_idx]
X_trans_test = X_trans[test_idx]
y_train = X_raw.targets[~test_idx]
y_test = X_raw.targets[test_idx]
#  %%
'''
Prepare classifier
'''
epochs = 50
batch = 256
seed = 0
conf_tail = 25
dnn_dropout = 0.5
theme_base = f'CLF_new_towns_clf_e{epochs}_batch_{batch}_seed_{seed}_max_dist_{max_dist}'
#
clf = pred_tools.generate_clf(X_trans_train,
                              y_train,
                              X_trans_test,
                              y_test,
                              theme_base,
                              epochs=epochs,
                              batch=batch,
                              seed=seed,
                              dropout=dnn_dropout)
clf_y_pred = clf.predict(X_trans, verbose=1)
y_pruned, y_pruned_balanced = pred_tools.prune_y(clf_y_pred, confidence_tail=conf_tail)

#  %%
"""
M2
"""
max_dist = 800
df_20, X_raw, X_clean, distances, labels = pred_tools.load_data(selected_data,
                                                                max_dist=max_dist)
X_trans = StandardScaler().fit_transform(X_clean)

seed = 0
lr = 1e-4
latent_dim = 8
batch = 512  # use larger batches so that more balanced number of labelled instances available
epochs = 10
m2_dropout = 0.5
theme_base = f'M2_ONLY_prob{conf_tail}_batch{batch}_lr{lr}_l{latent_dim}_final_balanced_md{max_dist}'
# combine X and y
X_combined = np.hstack([X_trans, y_pruned_balanced])
# spatial validation set
test_idx = util_funcs.train_test_idxs(df_20, 200)  # 200 gives about 25%

m2 = pred_tools.generate_M2(X_combined,
                            raw_dim=X_trans.shape[1],
                            latent_dim=latent_dim,
                            theme_base=theme_base,
                            epochs=epochs,
                            batch=batch,
                            lr=lr,
                            seed=seed,
                            labels=labels,
                            distances=distances,
                            test_indices=test_idx,
                            dropout=m2_dropout)
#  %% predict classifications - two batches for memory
m2_y_pred = np.full((X_trans.shape[0], 2), np.nan)
mid_p = int(X_trans.shape[0] / 2)
m2_y_pred[:mid_p] = m2.classify(X_trans[:mid_p]).numpy()
m2_y_pred[mid_p:] = m2.classify(X_trans[mid_p:]).numpy()
# ignore the first axis - assuming interested in new town probabilities
m2_pred_nt = m2_y_pred[:, 1]

'''
#  %%
# select * from analysis.nodes_20 where city_pop_id =Any(ARRAY[40, 30, 27, 161, 73, 49, 52, 22, 473]);
db_config = {
    'host': 'localhost',
    'port': 5433,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}
phd_util.write_col_data(db_config,
                        'analysis.nodes_20',
                        m2_pred_nt.flatten(),
                        'y_M2_pred_nt',
                        'real',
                        X_raw.index,
                        'id')
'''

#  %%
'''
Overall cluster max plot - exploratory
'''
for x, y, town_name in [(485970, 236920, 'Milton Keynes'),
                        (451261, 206284, 'Oxford'),
                        (545700, 258980, 'Cambridge'),
                        (375040, 164860, 'Bath'),
                        (414500, 130130, 'Salisbury'),
                        (460570, 451740, 'York'),
                        (383920, 432900, 'Burnley'),
                        (348920, 405980, 'Skelmersdale'),
                        (342390, 398160, 'Kirkby'),
                        (442722, 393035, 'Rotherham'),
                        (442485, 314070, 'Coalville'),
                        (475167, 260066, 'Northampton'),
                        (523485, 210865, 'Hatfield & Welwyn G.C.'),
                        (524564, 224429, 'Stevenage'),
                        (560504, 178688, 'Grays'),
                        (590800, 164025, 'Sittingbourne'),
                        (526922, 136503, 'Crawley'),
                        (517575, 131430, 'Horsham'),
                        (443992, 120620, 'Chandler\'s Ford'),
                        (371460, 182616, 'Yate'),
                        (522373, 233054, 'Letchworth'),
                        (518716, 229308, 'Hitchin'),
                        (368917, 90211, 'Poundbury'),
                        (438397, 371143, 'Chesterfield'),
                        (409352, 425145, 'Halifax'),
                        (497561, 371383, 'Lincoln'),
                        (635633, 170829, 'Margate'),
                        (439743, 557041, 'Sunderland'),
                        (623184, 308525, 'Norwich'),
                        (353876, 429315, 'Preston'),
                        (330891, 436268, 'Blackpool')]:
    x_extents = (x - 2500, x + 2500)
    y_extents = (y - 5000, y + 5000)

    util_funcs.plt_setup()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    mappable = plot_funcs.plot_scatter(ax,
                                       df_20.x,
                                       df_20.y,
                                       # color range is from 0.0 to 1.0
                                       # don't manipulate in this context
                                       c=m2_pred_nt,
                                       cmap=util_funcs.cityseer_cmap(),
                                       s=1,
                                       x_extents=x_extents,
                                       y_extents=y_extents,
                                       relative_extents=False)
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    plt.colorbar(mappable, cax=cax, aspect=80)
    plt.suptitle(f'{town_name} - 20m resolution "New Town" predicted probabilities')
    path = f'../phd-admin/PhD/part_3/images/predicted/nt_hyperlocal_pred_{town_name}_{m2.theme}_{clf.theme}.pdf'
    plt.savefig(path, dpi=300)

#  %%
'''
Scatter plot of distribution and new town probabilities for clf vs M2
'''
import matplotlib.ticker as mtick

# create columns for percentages
for pop_id, town_row in selected_data.iterrows():
    # clf
    clf_pred = clf_y_pred[X_raw.city_pop_id == pop_id]
    selected_data.loc[pop_id, 'clf_nt_points_perc'] = np.sum(clf_pred) / len(
        clf_pred) * 100
    # M2
    m2_pred = m2_pred_nt[X_raw.city_pop_id == pop_id]
    selected_data.loc[pop_id, 'm2_nt_points_perc'] = np.sum(m2_pred) / len(
        m2_pred) * 100
# plot
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# sizes
pop = selected_data.city_population.to_numpy(dtype='int')
pop_norm = plt.Normalize()(pop)
#
clf_nt_points_perc = selected_data.clf_nt_points_perc
clf_nt_points_perc_norm = plt.Normalize()(clf_nt_points_perc)
#
m2_nt_points_perc = selected_data.m2_nt_points_perc
m2_nt_points_perc_norm = plt.Normalize()(m2_nt_points_perc)
#
for ax_row, ax_theme, y_pred, points_perc, points_perc_norm in zip(
        axes,
        ['DNN', 'M2'],
        [clf_y_pred, m2_pred_nt],
        [clf_nt_points_perc, m2_nt_points_perc],
        [clf_nt_points_perc_norm, m2_nt_points_perc_norm]):
    # plot distribution of nt probs
    N, bin_lower_limits, patches = ax_row[0].hist(y_pred, bins=100, edgecolor='w', linewidth=0.1)
    # set bin colours
    cityseer_cmap = util_funcs.cityseer_cmap()
    for bll, patch in zip(bin_lower_limits, patches):
        patch.set_facecolor(cityseer_cmap(bll))
    ax_row[0].set_xlabel(f'{ax_theme} - Distribution of new (vs. historic) probabilities')
    ax_row[0].set_xlim(0, 1)
    ax_row[0].set_ylabel('$n$')
    ax_row[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # plot percentage classified as new towns
    ax_row[1].scatter(x=pop,
                      y=points_perc,
                      c=points_perc_norm,
                      cmap=cityseer_cmap,
                      s=pop_norm * 80 + 10,
                      marker='.',
                      edgecolors='white',
                      linewidths=0.2,
                      zorder=2,
                      rasterized=True)
    ax_row[1].set_xlabel('City Population')
    ax_row[1].set_xlim(8000, 310000)
    ax_row[1].set_ylabel(f'{ax_theme} - Average of new (vs. historic) probabilities by town')
    ax_row[1].set_ylim(0, 100)

plt.suptitle(
    'New (vs. historic) probabilities compared for towns between 8,200 and 290,000 people.')
path = f'../phd-admin/PhD/part_3/images/predicted/nt_dists_probs_{m2.theme}_{clf.theme}.pdf'
plt.savefig(path, dpi=300)
