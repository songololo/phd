'''
EXPLORE scoring plots to figure out behaviour of one vs. rest binarised classifiers

Adopting a binarised OVR approach:
- better performance - e.g. see "One-Vs-All Binarization Technique in the Context of Random Forest"
- easier analysis of respective recall / precision / brier-score-loss / roc_auc etc.
- easier derivation of feature importances for respective classes

ROC AUC - is scale invariant - provides clues into performance that don't depend on probability calibration
Calibration - use with caution per: https://developers.google.com/machine-learning/crash-course/classification/prediction-bias
Though may be unavoidable for unbalanced classes?
Reliability curve - insight to prediction bias - under the line is over prediction

from sklearn:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
Precision-recall curves are typically used in binary classification to study the output of a classifier.
In order to extend the precision-recall curve and average precision to multi-class or multi-label classification,
it is necessary to binarize the output. One curve can be drawn per label,
but one can also draw a precision-recall curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging).

Balanced (vs. None) class weights give better ROC AUC and Brier Score Loss results, and avoids artefacts in precision / recall

RobustScaler vs PowerTransformer - quite similar, class 2 does ever so slightly better with PT vs. 1 better with RS

ExtraTrees is faster than Random Forest, but otherwise similar.

The classifiers have to be treated separately because OneVsRestClassifier will normalise the probabilities.
Further, only interested in predicting 1, 2, 3 and not 0, which is the assumed defacto.

Condensed multilabel approach technically performs better than singular binary approach,
but this is only because the "proper" accuracy scores operate within an easier situation.
For the same reason, the multilabel approach becomes weaker relying on betweenness for class "2",
i.e. information about what makes these classes different, has gone missing
consequently, blue shifts from horizontal to vertical band.

CALIBRATION - need to calibrate probabilities to match original data - otherwise floods

'''



import matplotlib.pyplot as plt
# %%
import numpy as np
from src import phd_util
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from palettable.colorbrewer.qualitative import Set1_4
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import label_binarize, minmax_scale



import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def adjust_undersampled_probs(y_probs, beta):
    # return y_probs
    return (beta * y_probs) / (beta * y_probs - y_probs + 1)


def plot_clf(clf, key, X, y, classes):
    X = PowerTransformer(standardize=True).fit_transform(X)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.5)

    num_classes = len(classes)

    y_train_binarised = label_binarize(y_train, classes=classes)
    y_val_binarised = label_binarize(y_val, classes=classes)

    # if only two classes you have to split manually
    if num_classes == 2:
        y_train_binarised = np.hstack([y_train_binarised, y_train_binarised])
        y_train_binarised[:, 1] = 1 - y_train_binarised[:, 1]
        y_val_binarised = np.hstack([y_val_binarised, y_val_binarised])
        y_val_binarised[:, 1] = 1 - y_val_binarised[:, 1]

    colours = ['#aaaaaaCC',
               Set1_4.hex_colors[2] + 'CC',
               Set1_4.hex_colors[1] + 'CC',
               Set1_4.hex_colors[0] + 'CC']

    colours = colours[:num_classes]
    clfs = [clone(clf)] * num_classes

    val_probs = np.full((len(y_val), num_classes), np.nan)

    phd_util.plt_setup(dark=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 6), dpi=300)
    for i, colour in enumerate(colours):
        clfs[i].fit(X_train, y_train_binarised[:, i])
        val_probs[:, i] = clfs[i].predict_proba(X_val)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_val_binarised[:, i], val_probs[:, i])
        axes[0].plot(recalls, precisions, c=colour, label=f'class {i}')
        axes[2].plot(thresholds, recalls[:-1], c=colour, ls='--', label=f'class {i} - recall')
        axes[2].plot(thresholds, precisions[:-1], c=colour, label=f'class {i} - precision')

        fprs, tprs, thresholds = roc_curve(y_val_binarised[:, i], val_probs[:, i])
        brier = round(brier_score_loss(y_val_binarised[:, i], val_probs[:, i]), 3)
        roc_auc = round(roc_auc_score(y_val_binarised[:, i], val_probs[:, i]), 3)
        log = round(log_loss(y_val_binarised[:, i], val_probs[:, i]), 3)
        axes[1].plot(fprs, tprs, c=colour, label=f'roc auc: {roc_auc}')

        if i == 0:
            axes[3].plot([0, 1], [0, 1], color='black', lw=3, alpha=0.2)

        n_bins = 10
        try:
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_val_binarised[:, i], val_probs[:, i], n_bins=n_bins, strategy='quantile')
            axes[3].plot(mean_predicted_value, fraction_of_positives, c=colour,
                         label=f'brier loss: {brier}, log loss: {log})')
            # axes[1][1].hist(y_probas[:, i], range=(0, 1), bins=n_bins, label=key, color=colour, histtype="step", lw=1)
        except ValueError as e:
            logger.error(e)

    axes[0].set_ylabel('precision')
    axes[0].set_xlabel('recall')
    axes[0].legend(loc='center right')
    axes[0].set_title('precision-recall')
    axes[0].set_ylim(bottom=0, top=1)
    axes[0].set_xlim(left=0, right=1)
    axes[0].set_aspect('equal')

    axes[1].set_ylabel('true positive rate')
    axes[1].set_xlabel('false positive rate')
    axes[1].legend(loc='center right')
    axes[1].set_title('roc curve')
    axes[1].set_ylim(bottom=0, top=1)
    axes[1].set_xlim(left=0, right=1)
    axes[1].set_aspect('equal')

    axes[2].set_ylabel('score')
    axes[2].set_xlabel('threshold')
    axes[2].legend(loc='center left')
    axes[2].set_title('precision-recall vs threshold')
    axes[2].set_ylim(bottom=0, top=1)
    axes[2].set_xlim(left=0, right=1)
    axes[2].set_aspect('equal')

    axes[3].set_xlabel('predicted probability')
    axes[3].set_ylabel('fraction of positives')
    axes[3].legend(loc='upper left')
    axes[3].set_title('reliability curve')
    axes[3].set_ylim(bottom=0, top=1)
    axes[3].set_xlim(left=0, right=1)
    axes[3].set_aspect('equal')

    plt.suptitle(key)
    # plt.savefig(
    #    f'./explore/1-centrality/exploratory_plots/cent_pred_lu_clf_{key}_{table_name_short}_{base_cols_str}_150.png')
    plt.show()


def analyse_clf(clf, key, X, y):
    X = PowerTransformer(standardize=True).fit_transform(X)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.5)

    y_train_binarised = label_binarize(y_train, classes=[0, 1, 2, 3])
    y_val_binarised = label_binarize(y_val, classes=[0, 1, 2, 3])

    colours = ['#aaaaaaCC',
               Set1_4.hex_colors[2] + 'CC',
               Set1_4.hex_colors[1] + 'CC',
               Set1_4.hex_colors[0] + 'CC']

    clfs = [clone(clf),
            clone(clf),
            clone(clf),
            clone(clf)]

    val_probs = np.full((len(y_val), 4), np.nan)

    phd_util.plt_setup()
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=300, gridspec_kw={'height_ratios': [5, 5, 1]})
    for i, colour in enumerate(colours):

        clfs[i].fit(X_train, y_train_binarised[:, i])
        val_probs[:, i] = clfs[i].predict_proba(X_val)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_val_binarised[:, i], val_probs[:, i])
        axes[0][0].plot(recalls, precisions, c=colour, label=f'class {i}')
        axes[0][2].plot(thresholds, recalls[:-1], c=colour, ls='--', label=f'class {i} - recall')
        axes[0][2].plot(thresholds, precisions[:-1], c=colour, label=f'class {i} - precision')

        fprs, tprs, thresholds = roc_curve(y_val_binarised[:, i], val_probs[:, i])
        brier = round(brier_score_loss(y_val_binarised[:, i], val_probs[:, i]), 3)
        roc_auc = round(roc_auc_score(y_val_binarised[:, i], val_probs[:, i]), 3)
        log = round(log_loss(y_val_binarised[:, i], val_probs[:, i]), 3)
        axes[0][1].plot(fprs, tprs, c=colour, label=f'roc auc: {roc_auc}')

        if i == 0:
            axes[0][3].plot([0, 1], [0, 1], color='black', lw=3, alpha=0.2)

        n_bins = 10
        try:
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_val_binarised[:, i], val_probs[:, i], n_bins=n_bins, strategy='quantile')
            axes[0][3].plot(mean_predicted_value, fraction_of_positives, c=colour,
                            label=f'brier loss: {brier}, log loss: {log})')
            # axes[1][1].hist(y_probas[:, i], range=(0, 1), bins=n_bins, label=key, color=colour, histtype="step", lw=1)
        except ValueError as e:
            logger.error(e)

    axes[0][0].set_ylabel('precision')
    axes[0][0].set_xlabel('recall')
    axes[0][0].legend(loc='center right')
    axes[0][0].set_title('precision-recall')
    axes[0][0].set_ylim(bottom=0, top=1)
    axes[0][0].set_xlim(left=0, right=1)
    axes[0][0].set_aspect('equal')

    axes[0][1].set_ylabel('true positive rate')
    axes[0][1].set_xlabel('false positive rate')
    axes[0][1].legend(loc='center right')
    axes[0][1].set_title('roc curve')
    axes[0][1].set_ylim(bottom=0, top=1)
    axes[0][1].set_xlim(left=0, right=1)
    axes[0][1].set_aspect('equal')

    axes[0][2].set_ylabel('score')
    axes[0][2].set_xlabel('threshold')
    axes[0][2].legend(loc='center left')
    axes[0][2].set_title('precision-recall vs threshold')
    axes[0][2].set_ylim(bottom=0, top=1)
    axes[0][2].set_xlim(left=0, right=1)
    axes[0][2].set_aspect('equal')

    axes[0][3].set_xlabel('predicted probability')
    axes[0][3].set_ylabel('fraction of positives')
    axes[0][3].legend(loc='upper left')
    axes[0][3].set_title('reliability curve')
    axes[0][3].set_ylim(bottom=0, top=1)
    axes[0][3].set_xlim(left=0, right=1)
    axes[0][3].set_aspect('equal')

    #### PREPARE DECISION SURFACE GRID:
    # keep order consistent with columns above
    idx_a = selected_cols.index('c_gravity_1600')
    idx_b = selected_cols.index('c_between_wt_1600')

    sz = 1000
    prob_grid = np.full((sz * sz, 2), 0.0)

    min_x = X[:, idx_a].min()
    max_x = X[:, idx_a].max()
    inc_x = (max_x - min_x) / sz

    min_y = X[:, idx_b].min()
    max_y = X[:, idx_b].max()
    inc_y = (max_y - min_y) / sz
    count = 0
    # default origin is upper, override to lower
    # proceeds in row, column order
    # so place x on j index and in increasing order
    # place y on i index and increasing order
    for i in range(sz):
        for j in range(sz):
            vals = [np.nan, np.nan]
            vals[idx_a] = min_x + inc_x * j
            vals[idx_b] = min_y + inc_y * i
            prob_grid[count] = vals
            count += 1

    #### PREPARE DECISION SURFACE PROBS
    im_probs = np.full((sz * sz, 4), 0.0)
    im_probs[:, 0] = clfs[0].predict_proba(prob_grid)[:, 1]
    im_probs[:, 1] = clfs[1].predict_proba(prob_grid)[:, 1]
    im_probs[:, 2] = clfs[2].predict_proba(prob_grid)[:, 1]
    im_probs[:, 3] = clfs[3].predict_proba(prob_grid)[:, 1]

    #### PREPARE TEST PROBS:
    test_probs = np.full((len(y_test), 4), np.nan)
    test_probs[:, 0] = clfs[0].predict_proba(X_test)[:, 1]
    test_probs[:, 1] = clfs[1].predict_proba(X_test)[:, 1]
    test_probs[:, 2] = clfs[2].predict_proba(X_test)[:, 1]
    test_probs[:, 3] = clfs[3].predict_proba(X_test)[:, 1]

    # plot original classes for comparison
    x_scaled = minmax_scale(X_test) * sz
    for n, (cl, colour) in enumerate(zip([0, 1, 2, 3], colours)):
        plot_x = x_scaled[y_test == cl, 0]
        plot_y = x_scaled[y_test == cl, 1]
        opacity = 1
        size = 1
        lw = 0.15
        if n == 0:
            opacity = 0.5
            size = 0.5
            lw = 0.1
        for m in [0, 1, 2, 3]:
            axes[1][m].scatter(plot_x,
                               plot_y,
                               s=size,
                               c=colour,
                               edgecolors='white',
                               lw=lw,
                               alpha=opacity,
                               zorder=n + 1)

    #### SIMPLE MAJORITY PROBABILITY
    print('simple')
    y_hat = np.argmax(test_probs, axis=1)
    axes[2][0].text(0.05,
                    0.95,
                    classification_report_imbalanced(y_test, y_hat),
                    transform=axes[2][0].transAxes,
                    fontsize=4,
                    verticalalignment='top')
    axes[2][0].axis('off')
    # generate image
    im_hat = np.argmax(im_probs, axis=1)
    im = np.full((sz * sz, 3), 0.0, dtype=float)
    im[im_hat == 0] = (0.25, 0.25, 0.25)
    im[im_hat == 1] = Set1_4.mpl_colors[2]
    im[im_hat == 2] = Set1_4.mpl_colors[1]
    im[im_hat == 3] = Set1_4.mpl_colors[0]
    im = im.reshape((sz, sz, 3))
    axes[1][0].imshow(im, origin='lower', zorder=0, alpha=1.0)
    axes[1][0].imshow(im, origin='lower', zorder=6, alpha=0.25)

    #### BLENDER METHOD
    print('blender')
    blender = ExtraTreesClassifier(n_estimators=100,
                                   criterion='entropy',
                                   max_depth=20,
                                   min_samples_split=0.1,
                                   max_features='auto',
                                   class_weight='balanced',
                                   n_jobs=-1)
    # fit on validation data
    blender.fit(val_probs, y_val)
    # test and report
    y_hat = blender.predict(test_probs)
    axes[2][1].text(0.05,
                    0.95,
                    classification_report_imbalanced(y_test, y_hat),
                    transform=axes[2][1].transAxes,
                    fontsize=4,
                    verticalalignment='top')
    axes[2][1].axis('off')
    # generate decision surface
    im_hat = blender.predict(im_probs)
    im = np.full((sz * sz, 3), 0.0, dtype=float)
    im[im_hat == 0] = (0.25, 0.25, 0.25)
    im[im_hat == 1] = Set1_4.mpl_colors[2]
    im[im_hat == 2] = Set1_4.mpl_colors[1]
    im[im_hat == 3] = Set1_4.mpl_colors[0]
    im = im.reshape((sz, sz, 3))
    axes[1][1].imshow(im, origin='lower', zorder=0, alpha=1.0)
    axes[1][1].imshow(im, origin='lower', zorder=6, alpha=0.25)

    ### PROBABILITY SHIFT METHOD
    print('shifted probs')
    _iter_probs = np.zeros_like(val_probs)
    best_settings = [1, 1, 1, 1]
    best_val = 0
    betas = [i / 100 for i in range(10, 101, 20)]
    values = []
    for l, b0 in enumerate(betas):
        for b1 in betas:
            for b2 in betas:
                for b3 in betas:
                    _iter_probs[:, 0] = adjust_undersampled_probs(val_probs[:, 0], b0)
                    _iter_probs[:, 1] = adjust_undersampled_probs(val_probs[:, 1], b1)
                    _iter_probs[:, 2] = adjust_undersampled_probs(val_probs[:, 2], b2)
                    _iter_probs[:, 3] = adjust_undersampled_probs(val_probs[:, 3], b3)
                    y_hat = np.argmax(_iter_probs, axis=1)
                    b_s = geometric_mean_score(y_val, y_hat, average='weighted')
                    if b_s >= best_val:
                        best_val = b_s
                        values.append(best_val)
                        best_settings = [b0, b1, b2, b3]
    print('best', best_settings)
    # full: [0.85, 0.85, 0.85, 0.95]

    # measure best
    shifted_probs = np.full((len(y_test), 4), np.nan)
    for n in range(4):
        shifted_probs[:, n] = adjust_undersampled_probs(test_probs[:, n], best_settings[n])
    y_hat = np.argmax(shifted_probs, axis=1)
    axes[2][2].text(0.05,
                    0.95,
                    classification_report_imbalanced(y_test, y_hat),
                    transform=axes[2][2].transAxes,
                    fontsize=4,
                    verticalalignment='top')
    axes[2][2].axis('off')

    shifted_im_probs = np.full((sz * sz, 4), 0.0)
    for n in range(4):
        shifted_im_probs[:, n] = adjust_undersampled_probs(im_probs[:, n], best_settings[n])
    im = np.full((sz * sz, 3), 0.0, dtype=float)
    idx_grey = np.argmax(shifted_im_probs, axis=1) == 0
    idx_green = np.argmax(shifted_im_probs, axis=1) == 1
    idx_blue = np.argmax(shifted_im_probs, axis=1) == 2
    idx_red = np.argmax(shifted_im_probs, axis=1) == 3
    im[idx_grey] = (0.25, 0.25, 0.25)
    im[idx_green] = Set1_4.mpl_colors[2]
    im[idx_blue] = Set1_4.mpl_colors[1]
    im[idx_red] = Set1_4.mpl_colors[0]
    im = im.reshape((sz, sz, 3))
    axes[1][2].imshow(im, origin='lower', zorder=0, alpha=1.0)
    axes[1][2].imshow(im, origin='lower', zorder=6, alpha=0.25)

    #### REFERENCE
    print('reference')
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_test)
    axes[2][3].text(0.05,
                    0.95,
                    classification_report_imbalanced(y_test, y_hat),
                    transform=axes[2][3].transAxes,
                    fontsize=4,
                    verticalalignment='top')
    axes[2][3].axis('off')
    # generate image
    im_hat = clf.predict(prob_grid)
    im = np.full((sz * sz, 3), 0.0, dtype=float)
    im[im_hat == 0] = (0.25, 0.25, 0.25)
    im[im_hat == 1] = Set1_4.mpl_colors[2]
    im[im_hat == 2] = Set1_4.mpl_colors[1]
    im[im_hat == 3] = Set1_4.mpl_colors[0]
    im = im.reshape((sz, sz, 3))
    axes[1][3].imshow(im, origin='lower', zorder=0, alpha=1.0)
    axes[1][3].imshow(im, origin='lower', zorder=6, alpha=0.25)

    # specify axes
    for n, k in zip([0, 1, 2, 3], ['simple', 'blender', 'tuned', 'reference']):
        axes[1][n].set_title(f'decision surface - {k}')
        axes[1][n].set_ylabel('betweenness wt 1600')
        axes[1][n].set_xlabel('gravity 1600')
        axes[1][n].set_ylim(bottom=0, top=1000)
        axes[1][n].set_xlim(left=0, right=1000)
        axes[1][n].set_aspect('equal')

    table_name_short = table_name.split('.')[-1]
    fig.suptitle = f'classifier {key} analysis on table: {table_name_short} col: {base_cols_str}'
    plt.savefig(
        f'./explore/1-centrality/exploratory_plots/cent_pred_lu_clf_{key}_{table_name_short}_{base_cols_str}_150.png')
    plt.show()
