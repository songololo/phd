# %%
'''
STEP 4:
A collection of plots visualising outputs from previous steps
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.preprocessing import StandardScaler

from src import util_funcs
from src.explore import plot_funcs
from src.explore.predictive import pred_tools

#  %%
# load boundaries
selected_data = pred_tools.load_bounds()
# load data
max_dist = 1600
df_20_DNN, X_raw_DNN, X_clean_DNN, distances_DNN, labels_DNN = pred_tools.load_data(selected_data,
                                                                                    max_dist=max_dist)
X_trans_DNN = StandardScaler().fit_transform(X_clean_DNN)
#  %% test train split
test_idx = util_funcs.train_test_idxs(df_20_DNN, 200)  # 200 gives about 25%
X_trans_train_DNN = X_trans_DNN[~test_idx]
X_trans_test_DNN = X_trans_DNN[test_idx]
y_train_DNN = X_raw_DNN.targets[~test_idx]
y_test_DNN = X_raw_DNN.targets[test_idx]

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
clf = pred_tools.generate_clf(X_trans_train_DNN,
                              y_train_DNN,
                              X_trans_test_DNN,
                              y_test_DNN,
                              theme_base,
                              epochs=epochs,
                              batch=batch,
                              seed=seed,
                              dropout=dnn_dropout)
clf_y_pred = clf.predict(X_trans_DNN, verbose=1)
y_pruned, y_pruned_balanced = pred_tools.prune_y(clf_y_pred, confidence_tail=conf_tail)

#  %%
'''
M2
'''
max_dist = 800
df_20, X_raw, X_clean, distances, labels = pred_tools.load_data(selected_data,
                                                                max_dist=max_dist)
X_trans_M2 = StandardScaler().fit_transform(X_clean)

seed = 0
lr = 1e-4
latent_dim = 8
batch = 512  # use larger batches so that more balanced number of labelled instances available
epochs = 10
m2_dropout = 0.5
theme_base = f'M2_ONLY_prob{conf_tail}_batch{batch}_lr{lr}_l{latent_dim}_final_balanced_md{max_dist}'
# combine X and y
X_combined = np.hstack([X_trans_M2, y_pruned_balanced])
# spatial validation set
test_idx = util_funcs.train_test_idxs(df_20, 200)  # 200 gives about 25%

m2 = pred_tools.generate_M2(X_combined,
                            raw_dim=X_trans_M2.shape[1],
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
m2_y_pred = np.full((X_trans_M2.shape[0], 2), np.nan)
mid_p = int(X_trans_M2.shape[0] / 2)
m2_y_pred[:mid_p] = m2.classify(X_trans_M2[:mid_p]).numpy()
m2_y_pred[mid_p:] = m2.classify(X_trans_M2[mid_p:]).numpy()
# ignore the first axis - assuming interested in new town probabilities
m2_pred_ot = m2_y_pred[:, 0]
m2_pred_nt = m2_y_pred[:, 1]
# reshape so that indexing works (1d vs 0d)
m2_pred_ot = m2_pred_ot.reshape((m2_pred_ot.shape[0], 1))
m2_pred_nt = m2_pred_nt.reshape((m2_pred_nt.shape[0], 1))

#  %%
'''
Scatter plot of distribution and new town probabilities for clf vs M2
'''
# create columns for percentages
for pop_id, town_row in selected_data.iterrows():
    # clf
    clf_pred = clf_y_pred[X_raw_DNN.city_pop_id == pop_id]
    selected_data.loc[pop_id, 'clf_nt_points_perc'] = np.sum(clf_pred) / len(
        clf_pred) * 100
    # M2
    m2_pred = m2_pred_nt[X_raw.city_pop_id == pop_id]
    selected_data.loc[pop_id, 'm2_nt_points_perc'] = np.sum(m2_pred) / len(m2_pred) * 100
# plot
util_funcs.plt_setup()
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
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
    N, bin_lower_limits, patches = ax_row[0].hist(y_pred, bins=100, edgecolor='w',
                                                  linewidth=0.1)
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

    # add town names
    for pop_id, town_row in selected_data.iterrows():
        # find all new town like
        # if town_row.targets:
        if ax_theme == 'DNN':
            y = town_row.clf_nt_points_perc
        else:
            y = town_row.m2_nt_points_perc
        x = town_row.city_population
        name = town_row.city_name
        # ax_row[1].text(x, y, pop_id, fontdict={'size': 5}, rotation=45)
        # top
        if pop_id in [289, 162, 91, 73, 61, 45, 42, 40, 34, 31, 30, 27, 26, 25, 22, 19]:
            align = 'top'
            y_end = 99
        # bottom
        elif pop_id in [163, 138, 122, 112, 90, 70, 57, 50, 46, 41, 39, 35, 33, 32, 29, 21, 20]:
            align = 'bottom'
            y_end = 1
        else:
            continue
        ax_row[1].text(x - 5,
                       y_end,
                       name,
                       rotation=90,
                       verticalalignment=align,
                       horizontalalignment='right',
                       fontdict={'size': 5},
                       color='#D3A1A6')
        ax_row[1].vlines(x, ymin=y, ymax=y_end, color='#D3A1A6', lw=0.5, alpha=0.4)

plt.suptitle(
    'New (vs. historic) probabilities compared for towns between 8,200 and 290,000 people.')
path = f'../phd-doc/doc/images/predictive/nt_dists_probs.pdf'
plt.savefig(path, dpi=300)

# %%
'''
Comparative classification plots
Two sets - Old vs. New
'''
for set, set_theme, path_theme in zip([[((460329.88, 451894.37), 'York', 41),
                                        ((545222.64, 258326.31), 'Cambridge', 49),
                                        ((518569.11, 229294.88), 'Hitchin', 194),
                                        ((368361.7, 90442.5), 'Poundbury / Dorchester', 332)],
                                       [((485415.7, 237912.4), 'Milton Keynes', 27),
                                        ((523733.1, 224586.3), 'Stevenage', 73),
                                        ((522071.7, 232874.5), 'Letchworth', 161),
                                        ((527016.8, 136693.7), 'Crawley', 61)]],
                                      ('Historic towns', 'New Development'),
                                      ('historic', 'new')):
    cmap = util_funcs.cityseer_cmap()
    s = 0.625
    util_funcs.plt_setup()
    fig, axes = plt.subplots(4, 4, figsize=(7, 10))
    # iterate each town for a given set
    for ax_col_idx, (centre, town_name, town_pop_id) in enumerate(set):
        # map 1) Boundary designation
        # targets will be 1 vs. 0 for new town like or not
        c = cmap(float(selected_data.loc[town_pop_id, 'targets']))
        ax = axes[0, ax_col_idx]
        ax.set_xlabel(f'{town_name}')
        if ax_col_idx == 0:
            ax.set_ylabel(f'Predictions by boundary')
        plot_funcs.plot_scatter(fig,
                                ax,
                                df_20.x,
                                df_20.y,
                                c=c,
                                cmap=cmap,
                                s=s,
                                centre=centre,
                                km_per_inch=3)
        # map 2) classifier
        ax = axes[1, ax_col_idx]
        ax.set_xlabel(f'{town_name}')
        if ax_col_idx == 0:
            ax.set_ylabel(f'DNN local predictions')
        plot_funcs.plot_scatter(fig,
                                ax,
                                df_20.x,
                                df_20.y,
                                # color range is from 0.0 to 1.0
                                # don't manipulate in this context
                                c=clf_y_pred,
                                cmap=cmap,
                                s=s,
                                centre=centre,
                                km_per_inch=3)
        # map 3) classifier cropped
        ax = axes[2, ax_col_idx]
        ax.set_xlabel(f'{town_name}')
        if ax_col_idx == 0:
            ax.set_ylabel(f'DNN extracted labels (input to M2)')
        # intentionally using full y_pruned instead of balanced
        y_col = np.copy(y_pruned)
        y_col[y_col == -1] = np.nan
        plot_funcs.plot_scatter(fig,
                                ax,
                                df_20.x,
                                df_20.y,
                                # color range is from 0.0 to 1.0
                                # don't manipulate in this context
                                c=y_col,
                                vmin=0.0,  # in some cases plt doesn't correct infer extents
                                vmax=1.0,
                                cmap=cmap,
                                s=s,
                                centre=centre,
                                km_per_inch=3)
        # map 4) M2 predict
        ax = axes[3, ax_col_idx]
        ax.set_xlabel(f'{town_name}')
        if ax_col_idx == 0:
            ax.set_ylabel(f'M2 local predictions')
        # pass explicit vmin vmax keywords for colorbar
        im = plot_funcs.plot_scatter(fig,
                                     ax,
                                     df_20.x,
                                     df_20.y,
                                     # color range is from 0.0 to 1.0
                                     # don't manipulate in this context
                                     c=m2_pred_nt,
                                     cmap=cmap,
                                     s=s,
                                     centre=centre,
                                     km_per_inch=3,
                                     vmin=0,
                                     vmax=1)
    cbar = fig.colorbar(im,
                        ax=axes,
                        aspect=50,
                        pad=0.01,
                        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        orientation='horizontal',
                        shrink=0.5)
    cbar.ax.set_xticklabels([f'{n / 10:.0%}' for n in range(0, 11, 2)])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label('Probability of "artificial" development')
    plt.suptitle(f"{set_theme}: localised predictions of newer vs. historic development")
    path = f'../phd-doc/doc/images/predictive/hyperlocal_pred_{path_theme}.pdf'
    plt.savefig(path, dpi=300)

# %%
'''
Plot pre-hats and latents
'''
# DNN - pre-hats
DNN_pred = clf.pre_hat(X_trans_DNN).numpy()
DNN_titles = [f'DNN #{i + 1}' for i in range(8)]

#  M2 - latents
X_y = np.hstack([X_trans_M2, m2_y_pred.round()])
Z_mu, _, _ = m2.encode(X_y, training=False)
Z_mu = Z_mu.numpy()

# %%
M2_titles = [f'M2 #{i + 1}' for i in range(8)]
# M2 OLD
ot_idx = m2_pred_ot.flatten() > np.percentile(m2_pred_ot, 95)
Z_ot = Z_mu[ot_idx]
X_tr_M2_ot = X_trans_M2[ot_idx]
M2_ot_titles = [f'M2 OT #{i + 1}' for i in range(8)]
# M2 NEW
nt_idx = m2_pred_nt.flatten() > np.percentile(m2_pred_nt, 95)
Z_nt = Z_mu[nt_idx]
X_tr_m2_nt = X_trans_M2[nt_idx]
M2_nt_titles = [f'M2 NT #{i + 1}' for i in range(8)]

sample_size = min(len(X_trans_DNN), len(X_trans_M2), len(X_tr_M2_ot), len(X_tr_m2_nt), 300000)

# %%
util_funcs.plt_setup()
fig, axes = plt.subplots(3, 8, figsize=(10, 7))
for ax_idx, (ax_row, preds, X_tr, titles, dists) in enumerate(
        zip(axes,
            [DNN_pred, Z_ot, Z_nt],
            [X_trans_DNN, X_tr_M2_ot, X_tr_m2_nt],
            [DNN_titles, M2_ot_titles, M2_nt_titles],
            [distances_DNN, distances, distances])):
    # use sample of dataset otherwise trying to run spearmanr on massive dataset will bork
    sample_idx = np.random.choice(len(X_tr), size=sample_size)
    for latent_idx in range(preds.shape[1]):
        print(ax_idx, latent_idx)

        heatmap_corrs = plot_funcs.correlate_heatmap(len(labels),
                                                     len(dists),
                                                     X_tr[sample_idx],
                                                     preds[sample_idx, latent_idx])
        im = plot_funcs.plot_heatmap(ax_row[latent_idx],
                                     heatmap=heatmap_corrs,
                                     row_labels=labels,
                                     col_labels=dists,
                                     set_row_labels=latent_idx == 0,
                                     set_col_labels=ax_idx < 2)
        ax_row[latent_idx].set_xlabel(titles[latent_idx])
cbar = fig.colorbar(im,
                    ax=axes,
                    aspect=50,
                    pad=0.01,
                    orientation='horizontal',
                    shrink=0.5)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('bottom')
cbar.set_label('Spearman $\\rho$ correlations against source variables')
plt.suptitle('DNN "pre-hat", M2 latents split by "historic" (OT), M2 latents split by "artificial" (NT)')
plt.savefig(f'../phd-doc/doc/images/predictive/dnn_m2_latents.pdf', dpi=300)

# %%
'''
Plot correlations for predictions against source variables
VERY FICKLE DEPENDING ON CALIBRATION OF PROBABILITIES... REQUIRES FURTHER INVESTIGATION

# and correlations for NT / OT predictions against source variables (lower row)
phd_util.plt_setup()
fig, axes = plt.subplots(1, 2, figsize=(8, 6))
for ax_idx, (t_pred, t_title) in enumerate(zip(
        (m2_pred_ot.flatten(), m2_pred_nt.flatten()),
        ('Historic development', 'New development'))):
    preds_corr = plot_funcs.correlate_heatmap(len(labels),
                                              len(distances),
                                              X_trans,
                                              t_pred)
    plot_funcs.plot_heatmap(axes[ax_idx],
                            heatmap=preds_corr,
                            row_labels=labels,
                            col_labels=distances,
                            set_row_labels=ax_idx == 0,
                            cbar=True)
    axes[ax_idx].set_title(t_title)
#
plt.suptitle(
    'M2 predictions for new and historic towns correlated against source variables.')
plt.savefig(
    f'../phd-doc/doc/images/predictive/nt_hist_corrs.pdf',
    dpi=300)
'''
