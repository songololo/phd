'''
Early explorations where I had used various methods in attempts to predict classifications
The classifications were based on mixed uses and were not as robust as current VaDE / GMM methods
There may be several useful functions in this file for visualising decision surfaces
'''


# %% setup functions

def set_colours(_y):
    colours = np.full(len(_y), '#0000000D')
    colours[_y == 1] = Set1_4.hex_colors[2] + 'CC'
    colours[_y == 2] = Set1_4.hex_colors[1] + 'CC'
    colours[_y == 3] = Set1_4.hex_colors[0] + 'CC'
    return colours


def set_gradient(_y, _step):
    base_colour = (0.25, 0.25, 0.25)
    if _step == 1:
        base_colour = list(Set1_4.mpl_colors[2])
    elif _step == 2:
        base_colour = list(Set1_4.mpl_colors[1])
    elif _step == 3:
        base_colour = list(Set1_4.mpl_colors[0])
    colours = []
    for y in _y:
        c = list(base_colour)
        c.append(y)
        colours.append(tuple(c))
    return colours


def plot_reduce(_X, _y_test, _y_hat, _label=None):
    X_reduce = PCA(n_components=2).fit_transform(_X)

    phd_util.plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    axes[0].scatter(X_reduce[:, 0],
                    X_reduce[:, 1],
                    c=set_colours(_y_test),
                    s=8,
                    edgecolors='none')
    axes[0].title.set_text('test set actual')

    axes[1].scatter(X_reduce[:, 0],
                    X_reduce[:, 1],
                    c=set_colours(_y_hat),
                    s=8,
                    edgecolors='none')
    axes[1].title.set_text('test set predicted')

    if _label is not None:
        plt.suptitle(_label)


def plot_simple(_X_1, _X_2, _y_test, _y_hat, _label_x1=None, _label_x2=None, _label=None):
    phd_util.plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    axes[0].scatter(_X_1[:, 0],
                    _X_2[:, 1],
                    c=set_colours(_y_test),
                    s=8,
                    edgecolors='none')
    if _label_x1 is not None:
        axes[0].set_xlabel(_label_x1)
    if _label_x2 is not None:
        axes[0].set_ylabel(_label_x2)
    axes[0].title.set_text('test set actual')

    axes[1].scatter(_X_1[:, 0],
                    _X_2[:, 1],
                    c=set_colours(_y_hat),
                    s=8,
                    edgecolors='none')
    if _label_x1 is not None:
        axes[1].set_xlabel(_label_x1)
    if _label_x2 is not None:
        axes[1].set_ylabel(_label_x2)
    axes[1].title.set_text('test set predicted')

    if _label is not None:
        plt.suptitle(_label)


def plot_probas(_X, _y_test, _y_hat_prob, _step, _label=None):
    X_reduce = PCA(n_components=2).fit_transform(_X)

    phd_util.plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    axes[0].scatter(X_reduce[:, 0],
                    X_reduce[:, 1],
                    c=set_colours(_y_test),
                    s=8,
                    edgecolors='none')
    axes[0].title.set_text('test set actual')

    axes[1].scatter(X_reduce[:, 0],
                    X_reduce[:, 1],
                    c=set_gradient(_y_hat_prob, _step),
                    s=8,
                    edgecolors='none')
    axes[1].title.set_text('train set predicted')

    if _label is not None:
        plt.suptitle(_label)


# %% custom scorer
# use "proper" scorers - the classification metrics just create confusion

def brier_loss_macro_func(y, y_probs):
    brier_scores = []
    y_binarised = label_binarize(y, classes=[0, 1, 2, 3])
    for i in range(4):
        brier_scores.append(brier_score_loss(y_binarised[:, i], y_probs[:, i]))
    return np.nanmean(np.array(brier_scores))


brier_loss_macro = make_scorer(brier_loss_macro_func, greater_is_better=False, needs_proba=True)


def brier_loss_binary_func(y, y_probs):
    return brier_score_loss(y, y_probs)


brier_loss_binary = make_scorer(brier_loss_binary_func, greater_is_better=False, needs_proba=True)


def roc_auc_macro_func(y, y_probs):
    roc_auc_scores = []
    y_binarised = label_binarize(y, classes=[0, 1, 2, 3])
    for i in range(4):
        roc_auc_scores.append(roc_auc_score(y_binarised[:, i], y_probs[:, i]))
    return np.nanmean(np.array(roc_auc_scores))


roc_auc_macro = make_scorer(roc_auc_macro_func, greater_is_better=True, needs_proba=True)


# %%
def table_factory(columns, distances=(50, 100, 200, 400, 800, 1600), single=True, single_index=0):
    selected_columns = []
    for d in distances:
        for c in columns:
            selected_columns.append(c.format(d=d))

    simple_columns = [col.split('_')[1] for col in columns]
    simple_columns = [col[:5] for col in simple_columns]
    simple_column_names = '_'.join(simple_columns)

    tables = [df_data_full, df_data_100, df_data_50, df_data_20]
    table_names = ['analysis.roadnodes_full',
                   'analysis.roadnodes_100',
                   'analysis.roadnodes_50',
                   'analysis.roadnodes_20']

    if single:
        assert isinstance(single_index, int)
        tables = [tables[single_index]]
        table_names = [table_names[single_index]]

    for table, table_name in zip(tables, table_names):
        yield table, table_name, selected_columns, simple_column_names


# %%
def split_y_labels(y):
    # convert y to multilabel
    y_multilabel = label_binarize(y, classes=[0, 1, 2, 3])
    # 3 is also 2 and 1
    y3_idx = np.where(y_multilabel[:, 3] == 1)
    y_multilabel[:, 2][y3_idx] = 1
    y_multilabel[:, 1][y3_idx] = 1
    # 2 is also 1
    y2_idx = np.where(y_multilabel[:, 2] == 1)
    y_multilabel[:, 1][y2_idx] = 1

    return y_multilabel


# %%
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}'
]
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     distances=(1600,),
                                                                     single=False,
                                                                     single_index=0):
    X = table[selected_cols]
    y = table.lu_cluster_manual

    logistic_clf = LogisticRegression(penalty='elasticnet',
                                      l1_ratio=0.5,
                                      class_weight='balanced',
                                      solver='saga',
                                      C=1,
                                      tol=0.0001,
                                      max_iter=100,
                                      n_jobs=-1)
    analyse_clf(logistic_clf, 'logistic', X, y)

    sgd_svm_clf = SGDClassifier(loss='modified_huber',
                                penalty='elasticnet',
                                l1_ratio=0.5,
                                alpha=0.00001,
                                max_iter=1000,
                                tol=0.001,
                                early_stopping=False,
                                class_weight='balanced',
                                n_jobs=-1)
    analyse_clf(sgd_svm_clf, 'sgd_svm', X, y)

    et_clf = ExtraTreesClassifier(n_estimators=100,
                                  criterion='entropy',
                                  max_depth=10,
                                  min_samples_split=0.2,
                                  max_features='auto',
                                  class_weight='balanced',
                                  n_jobs=-1)
    analyse_clf(et_clf, 'extra_trees', X, y)

    rf_clf = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    max_depth=20,
                                    min_samples_split=0.1,
                                    max_features='auto',
                                    class_weight='balanced',
                                    n_jobs=-1)
    analyse_clf(rf_clf, 'random_forest', X, y)

    naive_bayes = GaussianNB()
    analyse_clf(naive_bayes, 'naive_bayes', X, y)

    nearest_nb = KNeighborsClassifier(n_jobs=-1)
    analyse_clf(nearest_nb, 'nearest_nb', X, y)

    ada_clf = AdaBoostClassifier()
    analyse_clf(ada_clf, 'ada_boost', X, y)

    gb_clf = GradientBoostingClassifier(loss='deviance',
                                        learning_rate=0.1,
                                        n_estimators=100,
                                        subsample=1.0,
                                        criterion='friedman_mse',
                                        min_samples_split=0.2)
    analyse_clf(gb_clf, 'gradient_boost', X, y)

    bag_clf = BaggingClassifier(n_estimators=100, n_jobs=-1, max_samples=0.1)
    analyse_clf(bag_clf, 'bagging_clf', X, y)

# %%
'''
ONE VS REST COMPARISONS FOR 1, 2, 3

Classifiers:
- non linear and non-batch-descent SVC methods perform too poorly
- k nearest neighbours is also a bit slow
- random forests, extra trees, SGD (log), SGD (modified_huber) offer the best blend of characteristics
- SGD log loss is logistic
- SGD modified huber is SVM but gives probabilities

Methods:
- avoiding random sampling, random undersampling, tuned class weights
  these issue really come down to issues pertaining to how scoring is quantified
  i.e. use "proper" scorers and don't mess around too much
- Separated binary classifiers performed better (for random forestS) than joint multiclass
- Multilabelised approach performs better than singular multiclasses approach
'''


def classifier_probs(clf,
                     X,
                     y,
                     key,
                     table_name,
                     column_names,
                     multi_label=False,
                     db_write=False):
    table_name_short = table_name.split('.')[-1]
    logger.info(f'Processing key: {key} and cols: {column_names} from: {table_name_short}')

    X = PowerTransformer(standardize=True).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if multi_label:
        y_test_binarised = split_y_labels(y_test)
        y_train_binarised = split_y_labels(y_train)
    else:
        y_train_binarised = label_binarize(y_train, classes=[0, 1, 2, 3])
        y_test_binarised = label_binarize(y_test, classes=[0, 1, 2, 3])

    shifts = [0.95, 0.35, 0.25, 0.15]
    y_compound = np.full((len(y_test), 4), np.nan)
    y_write = np.full((len(y), 4), np.nan)
    brier = []
    roc = []
    for i in range(4):
        clf.fit(X_train, y_train_binarised[:, i])
        y_probas = clf.predict_proba(X_test)
        y_compound[:, i] = adjust_undersampled_probs(y_probas[:, 1], shifts[i])
        brier.append(brier_score_loss(y_test_binarised[:, i], y_probas[:, 1]))
        roc.append(roc_auc_score(y_test_binarised[:, i], y_probas[:, 1]))
        y_probas_write = clf.predict_proba(X)
        y_write[:, i] = adjust_undersampled_probs(y_probas_write[:, 1], shifts[i])
    # cross validate returns brier scores in negative form (minimised instead of maximised)
    macro_brier = np.mean(brier).round(3)
    macro_roc_auc = np.mean(roc).round(3)
    # no need to normalise probabilities prior to assigning
    y_hat = np.argmax(y_compound, axis=1)

    lb = f'{key} predicting lu on table: {table_name_short} col: {column_names} ' \
         f'macro brier loss: {macro_brier} / {np.round(brier, 2)} macro roc auc: {macro_roc_auc} / {np.round(roc, 2)}'
    plot_reduce(X_test, y_test, y_hat, lb)
    plt.savefig(f'./explore/1-centrality/exploratory_plots/cent_pred_lu_{key}_{column_names}.png')
    plt.show()

    if db_write:
        y_hat = np.argmax(y_write, axis=1)
        asyncio.run(phd_util.write_col_data(
            db_config,
            table_name,
            y_hat,
            f'pred_lu_w_{key}_{column_names}',
            'int',
            y.index,
            'id'))


# %%
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
db_write = True

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=2):
    X = table[selected_cols]
    y = table.lu_cluster_manual

    ### SGD logistic
    logistic_clf = LogisticRegression(penalty='none',
                                      class_weight='balanced',
                                      solver='saga',
                                      C=0.1,
                                      tol=0.0001,
                                      max_iter=100,
                                      n_jobs=-1)
    # classifier_probs(logistic_clf, X, y, 'logistic', table_name, base_cols_str, db_write=db_write)
    # classifier_probs(logistic_clf, X, y, 'logistic_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

    ### SGD SVM (with probabilities)
    sgd_svm_clf = SGDClassifier(loss='modified_huber',
                                penalty='l2',
                                alpha=0.001,
                                max_iter=10000,
                                tol=1,
                                early_stopping=True,
                                class_weight='balanced',
                                n_jobs=-1)
    # classifier_probs(sgd_svm_clf, X, y, 'sgd_svm', table_name, base_cols_str, db_write=db_write)
    # classifier_probs(logistic_clf, X, y, 'sgd_svm_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

    ### Extra Trees
    et_clf = ExtraTreesClassifier(n_estimators=100,
                                  criterion='entropy',
                                  max_depth=10,
                                  min_samples_split=0.1,
                                  max_features='auto',
                                  class_weight='balanced',
                                  n_jobs=-1)
    classifier_probs(et_clf, X, y, 'et', table_name, base_cols_str, db_write=db_write)
    # classifier_probs(logistic_clf, X, y, 'et_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

    ### Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    max_depth=None,
                                    class_weight='balanced',
                                    max_features='auto',
                                    n_jobs=-1)
    # classifier_probs(rf_clf, X, y, 'rf', table_name, base_cols_str, db_write=db_write)
    # classifier_probs(logistic_clf, X, y, 'rf_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

    clf = AdaBoostClassifier()
    # classifier_probs(clf, X, y, 'adaboost_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

    clf = GradientBoostingClassifier()
    # classifier_probs(clf, X, y, 'gradient_boosting_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

    clf = BaggingClassifier()
    # classifier_probs(clf, X, y, 'bagging_split', table_name, base_cols_str, multi_label=True, db_write=db_write)

# %%
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
db_write = False
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=0):
    X = table[selected_cols]
    y = table.lu_cluster_manual

    clf = RandomForestClassifier(n_estimators=100,
                                 criterion='entropy',
                                 max_depth=20,
                                 class_weight='balanced',
                                 max_features='auto',
                                 n_jobs=-1)
    classifier_chain(clf, X, y, 'rf', table_name, base_cols_str, db_write=db_write)

# %% inherently multiclass
'''
CalibratedClassifierCV seems to shift the smaller classes to lower end of the threshold spectrum. 
This means that the thresholds then have to be set manually to compensate...

OutputCodeClassifier is very instable (for this data) from run to run.
To resolve, requires a high number of classifiers - e.g. 10 - 20.
Outcome seems no better than simply using the raw classifier as an ensemble with a sufficient number of estimators

'''


def multiclass_classifier(clf,
                          X,
                          y,
                          key,
                          table_name,
                          column_names,
                          cv=3,
                          db_write=False):
    table_name_short = table_name.split('.')[-1]
    logger.info(f'Processing key: {key} and cols: {column_names} from: {table_name_short}')

    X = PowerTransformer(standardize=True).fit_transform(X)

    # # scores
    # y_scores = cross_validate(clf, X, y, cv=cv, n_jobs=-1, scoring={
    #     'brier_loss': brier_loss_macro,
    #     'roc_auc': roc_auc_macro
    # })
    # # cross validate returns brier scores in negative form (minimised instead of maximised)
    # macro_brier = np.mean(-y_scores['test_brier_loss']).round(3)
    # macro_roc_auc = np.mean(y_scores['test_roc_auc']).round(3)

    # probabilities and prediction
    y_hat = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    lb = f'{key} predicting lu on table: {table_name_short} col: {column_names} '
    # f'macro brier loss: {macro_brier} macro roc auc: {macro_roc_auc}'
    plot_reduce(X, y, y_hat, lb)
    plt.savefig(f'./explore/1-centrality/exploratory_plots/cent_pred_lu_{key}_{column_names}.png')
    plt.show()

    if db_write:
        asyncio.run(phd_util.write_col_data(
            db_config,
            table_name,
            y_hat,
            f'pred_lu_w_{key}_{column_names}',
            'int',
            y.index,
            'id'))


columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
db_write = False
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=0):
    X = table[selected_cols]
    y = table.lu_cluster_manual

    clf = ExtraTreesClassifier(n_estimators=20,
                               criterion='entropy',
                               max_depth=10,
                               min_samples_split=100,
                               max_features=None,
                               class_weight='balanced')
    # multiclass_classifier(clf, X, y, 'extra_trees', table_name, base_cols_str, db_write=db_write)

    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    # multiclass_classifier(clf, X, y, 'gaussian_nb', table_name, base_cols_str, db_write=db_write)

    # from sklearn.neighbors import KNeighborsClassifier
    # clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    # multiclass_classifier(clf, X, y, 'k_nearest', table_name, base_cols_str, db_write=db_write)

    # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    # clf = QuadraticDiscriminantAnalysis()
    # multiclass_classifier(clf, X, y, 'quad_discrim', table_name, base_cols_str, db_write=db_write)

    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(alpha=0, )
    # multiclass_classifier(clf, X, y, 'MLP', table_name, base_cols_str, db_write=db_write)

# %% compare some variations of landuses
sets = (
    ['c_gravity_{d}', 'c_between_wt_{d}'],
    ['c_gravity_{d}', 'c_between_wt_{d}', 'c_cycles_{d}'],
    ['met_info_ent_{d}'],
    ['c_gravity_{d}'],
    ['c_between_wt_{d}'],
    ['c_cycles_{d}'],
    ['c_node_dens_{d}'],
    ['c_farness_{d}'],
    ['c_harm_close_{d}'],
    ['c_imp_close_{d}']
)
db_write = True
for columns in sets:
    for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                         single=True,
                                                                         single_index=2):
        X = table[selected_cols]
        y = table.lu_cluster_manual

        clf = RandomForestClassifier(n_estimators=100,
                                     criterion='entropy',
                                     max_depth=20,
                                     max_features='auto',
                                     class_weight='balanced',
                                     n_jobs=-1)
        classifier_probs(clf, X, y, 'col_comp', table_name, base_cols_str, db_write=db_write)

# %% logistic_regression GRID SEARCh
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=2):
    X = table[selected_cols]
    X = PowerTransformer().fit_transform(X)

    y = table.lu_cluster_manual
    y = label_binarize(y, classes=[2])

    # SGD exploration
    parameters = {
        # 'penalty': ('none', 'elasticnet', 'l2', 'l1'),
        # 'C': (10, 1, 0.1),
        # 'tol': (0.001, 0.0001, 0.00001),
        'max_iter': (100, 1000),  # don't go too high!!! SLOW
    }

    # NOTE: overriding best class_weight to None
    # per ROC plots - 'balanced' is better for classes 1, 2 whereas None is better for class 3
    clf = LogisticRegression(penalty='none',
                             class_weight='balanced',
                             solver='saga',
                             C=0.1,
                             tol=0.0001,
                             max_iter=100,
                             n_jobs=-1)

    grid_search = GridSearchCV(clf,
                               parameters,
                               scoring=brier_loss_binary,
                               n_jobs=-1,
                               iid=False,
                               cv=3)
    grid_search.fit(X, y)

    print('best estimator', grid_search.best_estimator_)
    print('best params', grid_search.best_params_)

# %% sgd_svm_clf GRID SEARCH
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=2):
    X = table[selected_cols]
    X = PowerTransformer().fit_transform(X)

    y = table.lu_cluster_manual
    y = label_binarize(y, classes=[2])

    # SGD exploration
    parameters = {
        'penalty': ('l1', 'l2'),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 10000),
        'tol': (1, 0.1, 0.01),
        'early_stopping': (False, True)
    }

    # NOTE: overriding best class_weight to None
    # per ROC plots - 'balanced' is better for classes 1, 2 whereas None is better for class 3
    sgd_svm_clf = SGDClassifier(loss='modified_huber',
                                penalty='l2',
                                alpha=0.001,
                                max_iter=10000,
                                tol=1,
                                early_stopping=True,
                                class_weight='balanced',
                                n_jobs=-1)

    grid_search = GridSearchCV(sgd_svm_clf,
                               parameters,
                               scoring=brier_loss_binary,
                               n_jobs=-1,
                               iid=False,
                               cv=3)
    grid_search.fit(X, y)

    print('best estimator', grid_search.best_estimator_)
    print('best params', grid_search.best_params_)

# %% sgd_svm_clf GRID SEARCH
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=2):
    X = table[selected_cols]
    X = PowerTransformer().fit_transform(X)

    y = table.lu_cluster_manual
    y = label_binarize(y, classes=[2])

    # SGD exploration
    parameters = {
        'penalty': ('none', 'l2', 'l1'),
        'alpha': (0.01, 0.001, 0.0001),
        'max_iter': (1000, 10000),
        'tol': (1, 0.1, 0.01),
        'early_stopping': (False, True)
    }

    # NOTE: overriding best class_weight to None
    # per ROC plots - 'balanced' is better for classes 1, 2 whereas None is better for class 3
    clf = SGDClassifier(loss='modified_huber',
                        penalty='l2',
                        alpha=0.0001,
                        max_iter=10000,
                        tol=0.1,
                        early_stopping=False,
                        class_weight='balanced')

    grid_search = GridSearchCV(clf,
                               parameters,
                               scoring=brier_loss_macro,
                               n_jobs=-1,
                               iid=False,
                               cv=3)
    grid_search.fit(X, y)

    print('best estimator', grid_search.best_estimator_)
    print('best params', grid_search.best_params_)

# %% RANDOM FOREST GRID SEARCH
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=0):
    X = table[selected_cols]
    X = PowerTransformer().fit_transform(X)

    y = table.lu_cluster_manual
    y = label_binarize(y, classes=[2])

    # SGD exploration
    parameters = {
        # 'n_estimators': (50),
        'max_features': ('auto', 'log2'),
        'criterion': ('gini', 'entropy'),
        'max_depth': (20, None),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1000, 10000),
        # 'min_weight_fraction_leaf': (),
        'max_leaf_nodes': (100, None)
    }

    rf_clf = RandomForestClassifier(n_estimators=50,
                                    criterion='gini',
                                    max_depth=None,
                                    max_features='log2',
                                    max_leaf_nodes=100,
                                    min_samples_leaf=1000,
                                    min_samples_split=2,
                                    class_weight='balanced',
                                    n_jobs=-1)

    grid_search = GridSearchCV(rf_clf,
                               parameters,
                               scoring=brier_loss_binary,
                               n_jobs=-1,
                               iid=False,
                               cv=3)
    grid_search.fit(X, y)

    print('best estimator', grid_search.best_estimator_)
    print('best params', grid_search.best_params_)

# %% EXTRA TREES GRID SEARCH
columns = [
    'c_gravity_{d}',
    'c_between_wt_{d}',
    'c_cycles_{d}'
]
for table, table_name, selected_cols, base_cols_str in table_factory(columns=columns,
                                                                     single=True,
                                                                     single_index=2):
    X = table[selected_cols]
    X = PowerTransformer().fit_transform(X)

    y = table.lu_cluster_manual
    y = label_binarize(y, classes=[2])

    # SGD exploration
    parameters = {
        'n_estimators': (50, 100),
        'criterion': ('gini', 'entropy'),
        'max_depth': (5, 10, 20, None),
        'min_samples_split': (2, 10, 100)
    }

    et_clf = ExtraTreesClassifier(n_estimators=100,
                                  criterion='entropy',
                                  max_depth=None,
                                  class_weight='balanced',
                                  max_features='auto',
                                  n_jobs=-1)

    grid_search = GridSearchCV(et_clf,
                               parameters,
                               scoring=brier_loss_binary,
                               n_jobs=-1,
                               iid=False,
                               cv=3)
    grid_search.fit(X, y)

    print('best estimator', grid_search.best_estimator_)
    print('best params', grid_search.best_params_)
