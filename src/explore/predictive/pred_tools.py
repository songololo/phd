'''
General tools used by other scripts in this module:
Mostly related to the automation of the creation of models and plots
'''

import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.explore.predictive import pred_models
from tensorflow.keras import losses
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, TerminateOnNaN, \
    ModelCheckpoint

from src import util_funcs
from src.explore import plot_funcs
from src.explore.signatures import sig_model_runners
from src.explore.theme_setup import data_path, logs_path, weights_path
from src.explore.theme_setup import generate_theme


def load_bounds():
    '''
    Load town boundary data
    '''
    # db connection params
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }
    # load boundaries data
    bound_data = util_funcs.load_data_as_pd_df(db_config,
                                               ['pop_id',
                                              'city_name',
                                              'city_type',
                                              'city_population'],
                                             'analysis.city_boundaries_150',
                                             'WHERE pop_id IS NOT NULL ORDER BY pop_id')
    # add indices for city-wide data
    bound_data.set_index('pop_id', inplace=True)
    bound_data.sort_index(inplace=True)
    # label dataset according to classifications
    town_classifications = ['New Town',
                            'New Town Like',
                            'Expanded Town',
                            'Expanded Town Like']
    selected_data = bound_data.copy()
    new_towns = selected_data[selected_data.city_type.isin(town_classifications)]
    largest = new_towns.city_population.max()
    smallest = new_towns.city_population.min()
    """
    # 298701.0 - largest pop
    # 8186.0 - smallest pop
    # 650 - largest city id
    # 19 - smallest city id (largest by size)
    """
    selected_data = selected_data[np.logical_and(selected_data.city_population >= smallest,
                                                 selected_data.city_population <= largest)]
    # prepare targets
    selected_data['targets'] = 0
    for pop_id, city_row in selected_data.iterrows():
        if (city_row.city_type in town_classifications):
            selected_data.loc[pop_id, 'targets'] = 1

    return selected_data


def load_data(selected_data, max_dist=None):
    '''
    Map boundary level targets to individual points
    '''
    # load data and prep new town band data
    # the feather dataset only contains towns 19 through to 650 to save on space
    # i.e. smallest through largest new town like locations
    df_20 = pd.read_feather(data_path / 'df_20_all.feather')
    df_20 = df_20.set_index('id')
    X_raw, distances, labels = generate_theme(df_20,
                                              'all_towns',
                                              bandwise=True,
                                              add_city_pop_id=True,
                                              max_dist=max_dist)
    # create targets column on df_20
    X_raw['targets'] = 0
    # map from boundaries
    for pop_idx, row in selected_data.iterrows():
        X_raw.loc[X_raw.city_pop_id == pop_idx, 'targets'] = row.targets
    # generate the standardised data, but exclude city_pop_id
    X_clean = X_raw.drop(['city_pop_id', 'targets'], axis=1)
    # also drop city_pop_id from labels
    labels = labels[1:]
    return df_20, X_raw, X_clean, distances, labels


def generate_clf(X_train,
                 y_train,
                 X_test,
                 y_test,
                 theme_base,
                 epochs,
                 batch,
                 seed,
                 dropout):
    clf = pred_models.NewTownClassifier(theme_base=theme_base, seed=seed, dropout=dropout)
    clf.compile(optimizer='adam',
                loss=losses.BinaryCrossentropy(),
                metrics=['binary_accuracy'])
    # prepare path and check for previously trained model
    dir_path = pathlib.Path(weights_path / f'{clf.theme}')
    if not dir_path.exists():
        # prepare callbacks
        callbacks = [
            TensorBoard(
                log_dir=str(
                    logs_path / f'{datetime.now().strftime("%Hh%Mm%Ss")}_{clf.theme}'),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=2,
                embeddings_freq=0,
                embeddings_metadata=None),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0),
            TerminateOnNaN(),
            ModelCheckpoint(
                str(dir_path / 'weights'),
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                save_freq='epoch')
        ]
        # train
        clf.fit(x=X_train,
                y=y_train,
                batch_size=batch,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test),
                shuffle=True,
                callbacks=callbacks)
    else:
        clf.load_weights(str(dir_path / 'weights'))
    #
    return clf


# %%
def prune_y(clf_y_pred, confidence_tail=5):
    '''
    Can't use percentile:
    The side of the distribution with the longer tail will then overwhelm the distribution
    with the shorter tail. (Rounding the targets will pull the classification in that direction).
    BUT:
    Using probability threshold also doesn't work well:
    Can lead to unbalanced classes, with the larger class then overwhelming the smaller class
    SO:
    Using probability threshold, then randomly reducing the number of larger class to match smaller.
    '''
    if clf_y_pred.ndim != 2:
        print('y predictions are not a two-d array?')
    else:
        y_pruned = np.copy(clf_y_pred)
        lower = confidence_tail / 100
        upper = (100 - confidence_tail) / 100
        upper_n = (y_pruned > upper).sum()
        lower_n = (y_pruned < lower).sum()
        print(
            f'{(y_pruned > upper).sum() / y_pruned.shape[0]:.2%} samples above upper threshold')
        print(
            f'{(y_pruned < lower).sum() / y_pruned.shape[0]:.2%} samples below lower threshold')
        # snip out samples between the lower and upper thresholds
        y_pruned[np.logical_and(y_pruned > lower, y_pruned < upper)] = -1
        removed_n = (y_pruned == -1).sum()
        print(f'removed {removed_n / y_pruned.shape[0]:.2%} labels')
        # balance classes
        print('preparing balanced set')
        y_pruned_balanced = np.copy(y_pruned)
        # figure out if upper or lower tails need thinning
        if upper_n > lower_n:
            diff = upper_n - lower_n
            print(f'equalising upper band by removing {diff} samples')
            idx = np.where(y_pruned > upper)[0]
        else:
            diff = lower_n - upper_n
            print(f'equalising lower band by removing {diff} samples')
            idx = np.where(np.logical_and(y_pruned < lower, y_pruned != -1))[0]
        rdm_idx = np.random.choice(idx, diff, replace=False)
        y_pruned_balanced[rdm_idx] = -1
        assert (y_pruned_balanced > upper).sum() == (np.logical_and(y_pruned_balanced < lower,
                                                                    y_pruned_balanced != -1)).sum()
        print(
            f'kept {(y_pruned_balanced > upper).sum()} in upper and lower bands for given threshold')
        print(
            f'sample size for balanced: {(y_pruned_balanced != -1).sum() / y_pruned_balanced.shape[0]:.2%}')
        return y_pruned.round(), y_pruned_balanced.round()


# %%
def generate_M2(X_y,
                raw_dim,
                latent_dim,
                theme_base,
                epochs,
                batch,
                lr,
                seed,
                labels,
                distances,
                test_indices,
                dropout):
    theme_base = f'{theme_base}_e{epochs}_s{seed}'
    m2 = pred_models.M2(raw_dim=raw_dim,
                        latent_dim=latent_dim,
                        n_samples=X_y.shape[0] - test_indices.sum(),
                        theme_base=theme_base,
                        seed=seed,
                        dropout=dropout,
                        name='VAE')
    dir_path = pathlib.Path(weights_path / f'{m2.theme}')
    if not dir_path.exists():
        dir_path.mkdir(exist_ok=True, parents=True)
        l_path = logs_path
        trainer = sig_model_runners.Trainer(model=m2,
                                            X_samples=X_y,
                                            labels=labels,
                                            distances=distances,
                                            logs_path=l_path,
                                            epochs=epochs,
                                            batch=batch,
                                            lr=lr,
                                            clip_norm=1.0,
                                            best_loss=True,
                                            save_path=dir_path,
                                            test_indices=test_indices)
        trainer.train()
    else:
        m2.load_weights(str(dir_path / 'weights'))
    return m2


def plot_latents(df,
                 X_trans,
                 Z_mu,
                 labels,
                 distances,
                 latent_titles=None,
                 suptitle=None,
                 path=None):
    '''
    plot latents
    '''
    util_funcs.plt_setup()
    fig, axes = plt.subplots(3, 8, figsize=(12, 8))
    # plot correlations
    for latent_idx, ax in enumerate(axes[0]):
        heatmap_corrs = plot_funcs.correlate_heatmap(len(labels),
                                                     len(distances),
                                                     X_trans,
                                                     Z_mu[:, latent_idx])
        plot_funcs.plot_heatmap(ax,
                                heatmap=heatmap_corrs,
                                row_labels=labels,
                                col_labels=distances,
                                set_row_labels=latent_idx == 0,
                                cbar=True)
        if latent_titles is not None:
            t = latent_titles[latent_idx]
            ax.set_title(t)
    # plot maps
    for city_ax_row, city_name, x, y in zip([axes[1], axes[2]],
                                            ['Cambridge', 'Milton Keynes'],
                                            (545700, 485970),
                                            (258980, 236920)):
        x_extents = (x - 2500, x + 2500)
        y_extents = (y - 4500, y + 4500)
        for latent_idx, ax in enumerate(city_ax_row):
            plot_funcs.plot_scatter(ax,
                                    df.x,
                                    df.y,
                                    Z_mu[:, latent_idx],
                                    x_extents=x_extents,  # 10000
                                    y_extents=y_extents,  # 10000
                                    relative_extents=False,
                                    s_min=0,
                                    s_max=0.8,
                                    c_exp=5,
                                    s_exp=3.5)
            ax.set_xlabel(f'{city_name}  #{latent_idx}')

    if suptitle is not None:
        plt.suptitle(suptitle)
    if path is not None:
        plt.savefig(path, dpi=300)
    else:
        plt.show()
