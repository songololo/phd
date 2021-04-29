# %%
'''
STEP 2:
Basic DNN classifier for initial crude 20m sample point estimates of new towns vs old towns
Uses Step 1 OT vs NT designation as training targets from overall town boundary classifications.
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from src.explore.predictive import pred_tools

from src import util_funcs
from src.explore import plot_funcs

#  %%
# load boundaries
selected_data = pred_tools.load_bounds()
# load data
max_dist = 1600
df_20, X_raw, X_clean, distances, labels = pred_tools.load_data(selected_data, max_dist=max_dist)
X_trans = StandardScaler().fit_transform(X_clean)
# test train split
test_idx = util_funcs.train_test_idxs(df_20, 200)  # 200 gives about 25%
X_trans_train = X_trans[~test_idx]
X_trans_test = X_trans[test_idx]
y_train = X_raw.targets[~test_idx]
y_test = X_raw.targets[test_idx]

'''
Prepare classifier
'''
epochs = 50
batch = 256
seed = 0
dropout = 0.5
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
                              dropout=dropout)

y_pred = clf.predict(X_trans, verbose=1)
print(classification_report(X_raw.targets, y_pred.round().astype('int')))

'''
              precision    recall  f1-score   support
           0       0.79      0.89      0.84   3265757
           1       0.76      0.61      0.68   1953766
    accuracy                           0.78   5219523
   macro avg       0.78      0.75      0.76   5219523
weighted avg       0.78      0.78      0.78   5219523
'''
'''
#  %%
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'gareth',
    'database': 'gareth',
    'password': ''
}
phd_util.write_col_data(db_config,
                        'analysis.nodes_20',
                        y_pred.flatten(),
                        'y_clf_pred',
                        'real',
                        X_raw.index,
                        'id')
'''
# %%
'''
pred plot
'''
y_plot = np.copy(y_pred)
y_plot = np.reshape(y_plot, (-1, 1))
y_conf = 15
y_pruned, y_pruned_balanced = pred_tools.prune_y(y_plot, confidence_tail=y_conf)
y_col = y_pruned
y_col[y_col == -1] = np.nan

for x, y, town_name in [(485970, 236920, 'Milton Keynes'),
                        (545700, 258980, 'Cambridge'),
                        (635633, 170829, 'Margate')]:
    x_extents = (x - 2500, x + 2500)
    y_extents = (y - 5000, y + 5000)

    util_funcs.plt_setup()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_funcs.plot_scatter(ax,
                            df_20.x,
                            df_20.y,
                            # color range is from 0.0 to 1.0
                            # don't manipulate in this context
                            c=y_col,
                            cmap=util_funcs.cityseer_cmap(),
                            s=1,
                            vmin=0.0,  # in some cases plt doesn't correct infer extents
                            vmax=1.0,
                            x_extents=x_extents,
                            y_extents=y_extents,
                            relative_extents=False)
    plt.suptitle(f"{town_name} hyperlocal predictions")
    path = f'../phd-admin/PhD/part_3/images/predicted/stage_2_DNN_{town_name}_{clf.theme}_yc{y_conf}.pdf'
    plt.savefig(path, dpi=300)

#  %%
# create columns for percentages
for pop_id, town_row in selected_data.iterrows():
    pred = y_plot[X_raw.city_pop_id == pop_id]
    selected_data.loc[pop_id, 'clf_nt_points_perc'] = np.sum(pred) / len(pred) * 100
# plot
util_funcs.plt_setup()
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
# sizes
pop = selected_data.city_population.to_numpy(dtype='int')
pop_norm = plt.Normalize()(pop)
#
clf_nt_points_perc = selected_data.clf_nt_points_perc
clf_nt_points_perc_norm = plt.Normalize()(clf_nt_points_perc)
#
for ax_theme, y_pred, points_perc, points_perc_norm in zip(
        ['DNN'],
        [y_plot],
        [clf_nt_points_perc],
        [clf_nt_points_perc_norm]):
    # plot distribution of nt probs
    N, bin_lower_limits, patches = ax[0].hist(y_pred, bins=100, edgecolor='w', linewidth=0.1)
    # set bin colours
    cityseer_cmap = util_funcs.cityseer_cmap()
    for bll, patch in zip(bin_lower_limits, patches):
        patch.set_facecolor(cityseer_cmap(bll))
    ax[0].set_xlabel(f'{ax_theme} - Distribution of New Town probabilities')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylabel('$n$')
    ax[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # plot percentage classified as new towns
    ax[1].scatter(x=pop,
                  y=points_perc,
                  c=points_perc_norm,
                  cmap=cityseer_cmap,
                  s=pop_norm * 80 + 10,
                  marker='.',
                  edgecolors='white',
                  linewidths=0.2,
                  zorder=2,
                  rasterized=True)
    ax[1].set_xlabel('City Population')
    ax[1].set_xlim(8000, 310000)
    ax[1].set_ylabel(f'{ax_theme} - Average of New Town probabilities by town')
    ax[1].set_ylim(0, 100)

plt.suptitle(
    'New Town probabilities compared for towns between 8,200 and 290,000 people.')
path = f'../phd-admin/PhD/part_3/images/predicted/nt_dists_probs_{clf.theme}.pdf'
plt.savefig(path, dpi=300)
