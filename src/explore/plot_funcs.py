import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from sklearn.preprocessing import minmax_scale

from src import util_funcs


def view_idx(xs, ys, x_extents, y_extents, relative_extents):
    if relative_extents:
        x_mu = np.mean(xs)
        x_left = x_mu + x_extents[0]
        x_right = x_mu + x_extents[1]
        y_mu = np.mean(ys)
        y_bottom = y_mu + y_extents[0]
        y_top = y_mu + y_extents[1]
    else:
        x_left = x_extents[0]
        x_right = x_extents[1]
        y_bottom = y_extents[0]
        y_top = y_extents[1]

    select = xs > x_left
    select = np.logical_and(select, xs < x_right)
    select = np.logical_and(select, ys > y_bottom)
    select = np.logical_and(select, ys < y_top)
    select_idx = np.where(select)[0]

    return select_idx, x_left, x_right, y_bottom, y_top


def prepare_v(vals):
    # don't reshape distribution: emphasise larger values if necessary using exponential
    # i.e. amplify existing distribution rather than using a reshaped normal or uniform distribution
    # clip out outliers
    v = np.clip(vals, np.nanpercentile(vals, .1), np.nanpercentile(vals, 99.9))
    # scale colours to [0, 1]
    v = minmax_scale(v, feature_range=(0, 1))
    return v


# %%
'''
provide vals, otherwise provide explicit c and s params to override
'''
def plot_scatter(ax,
                 xs,
                 ys,
                 vals=None,
                 dark=False,
                 x_extents=(-1250, 3750),
                 y_extents=(-5000, 7000),
                 relative_extents=True,
                 s_min=0,
                 s_max=0.6,
                 c_exp=1,
                 s_exp=1,
                 cmap=None,
                 rasterized=True,
                 **kwargs):
    '''
    :param relative_extents: Uses the centrepoint of all points as a starting point.
                             Set to False when using explicit extents.
    '''
    if vals is not None and vals.ndim == 2:
        raise ValueError('Please pass a single dimensional array')

    if cmap is None:
        cmap = util_funcs.cityseer_cmap_red(dark=dark)

    select_idx, x_left, x_right, y_bottom, y_top = view_idx(xs,
                                                            ys,
                                                            x_extents,
                                                            y_extents,
                                                            relative_extents)

    if 'c' in kwargs and isinstance(kwargs['c'], (list, np.ndarray)):
        c = np.array(kwargs['c'])
        kwargs['c'] = c[select_idx]
    elif 'c' in kwargs and isinstance(kwargs['c'], str):
        pass
    elif vals is not None:
        v = prepare_v(vals)
        # apply exponential - still [0, 1]
        c = v ** c_exp
        kwargs['c'] = c[select_idx]

    if 's' in kwargs and isinstance(kwargs['s'], (list, np.ndarray)):
        s = np.array(kwargs['s'])
        kwargs['s'] = s[select_idx]
    elif vals is not None:
        v = prepare_v(vals)
        s = v ** s_exp
        # rescale s to [s_min, s_max]
        s = minmax_scale(s, feature_range=(s_min, s_max))
        kwargs['s'] = s[select_idx]

    im = ax.scatter(xs[select_idx],
                    ys[select_idx],
                    linewidths=0,
                    edgecolors='none',
                    cmap=cmap,
                    rasterized=rasterized,
                    **kwargs)

    ax.set_xlim(left=x_left, right=x_right)
    ax.set_ylim(bottom=y_bottom, top=y_top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    return im


def correlate_heatmap(n_columns, n_distances, X_raw, X_latent_arr):
    heatmap_corrs = np.full((n_columns, n_distances), 0.0)
    counter = 0
    for i in range(n_columns):
        for j in range(n_distances):
            heatmap_corrs[i][j] = pearsonr(X_raw[:, counter], X_latent_arr)[0]
            counter += 1
    return heatmap_corrs


def plot_heatmap(heatmap_ax,
                 heatmap: np.ndarray = None,
                 row_labels: list = None,
                 col_labels: list = None,
                 set_row_labels: bool = True,
                 set_col_labels: bool = True,
                 dark: bool = False,
                 constrain: tuple = (-1, 1),
                 cbar: bool = False,
                 text: np.ndarray = None,
                 cmap=None,
                 tick_fontsize='x-small',
                 grid_fontsize='xx-small'):
    '''
    Modified to permit text only plots
    '''
    if heatmap is None and text is None:
        raise ValueError('Pass either a heatmap or a text grid as a parameter')
    if set_row_labels and row_labels is None:
        raise ValueError('Pass row labels if setting to True')
    if set_col_labels and col_labels is None:
        raise ValueError('Pass column labels if setting to True')
    # plot
    if heatmap is not None:
        if cmap is None:
            cmap = util_funcs.cityseer_diverging_cmap(dark=dark)
        im = heatmap_ax.imshow(heatmap,
                               cmap=cmap,
                               vmin=constrain[0],
                               vmax=constrain[1],
                               origin='upper')
    # when doing text only plots, use a sham heatmap plot
    else:
        arr = np.full((len(row_labels), len(col_labels), 3), 1.0)
        im = heatmap_ax.imshow(arr, origin='upper')

    # set axes
    if row_labels is not None:
        heatmap_ax.set_yticks(list(range(len(row_labels))))
    if col_labels is not None:
        heatmap_ax.set_xticks(list(range(len(col_labels))))

    if row_labels is not None and set_row_labels:
        y_labels = [str(l) for l in row_labels]
        heatmap_ax.set_yticklabels(y_labels,
                                   rotation='horizontal',
                                   fontsize=tick_fontsize)
    else:
        heatmap_ax.set_yticklabels([])

    if col_labels is not None and set_col_labels:
        x_labels = [str(l) for l in col_labels]
        heatmap_ax.set_xticklabels(x_labels,
                                   rotation='vertical',
                                   fontsize=tick_fontsize)
    else:
        heatmap_ax.set_xticklabels([])

    if cbar:
        divider = make_axes_locatable(heatmap_ax)
        cax = divider.append_axes('top', size='2%', pad=0.05)
        cbar = plt.colorbar(im,
                            cax=cax,
                            aspect=100,
                            pad=0.02,
                            orientation='horizontal',
                            ticks=[constrain[0], 0.0, constrain[1]])
        title = r"Pearson's $\rho$ correlation"
        cbar.ax.set_xticklabels([constrain[0], title, constrain[1]])
        cbar.ax.xaxis.set_ticks_position('top')
    if text is not None:
        for row_idx in range(text.shape[0]):
            for col_idx in range(text.shape[1]):
                t = text[row_idx][col_idx]
                c = 'black'
                if heatmap is not None:
                    v = heatmap[row_idx][col_idx]
                    if isinstance(t, float):
                        t = round(t, 2)
                    # use white colour on darker backgrounds
                    if abs(v) > 0.5:
                        c = 'w'
                heatmap_ax.text(col_idx,
                                row_idx,
                                t,
                                ha='center',
                                va='center',
                                color=c,
                                fontsize=grid_fontsize)


# %%
def plot_components(component_idxs,
                    feature_labels,
                    distances,
                    X,
                    X_latent,
                    xs,
                    ys,
                    tag_string='',
                    tag_values=None,
                    loadings=None,
                    dark=False,
                    label_all=True,
                    s_min=0,
                    s_max=0.6,
                    c_exp=1,
                    s_exp=1,
                    cbar=False,
                    figsize=None,
                    rasterized=True):
    n_rows = 2
    n_cols = len(component_idxs)

    if figsize is None:
        figsize = (n_cols * 1.5, 8)

    util_funcs.plt_setup(dark=dark)
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=figsize,
                             gridspec_kw={
                                 'height_ratios': [1] * n_rows,
                                 'width_ratios': [1] * n_cols
                             })

    # create heatmaps for original vectors plotted against the top PCA components
    for n, comp_idx in enumerate(component_idxs):
        print(f'processing component {comp_idx + 1}')

        # split rows
        heatmap_ax = axes[0][n]
        map_ax = axes[1][n]

        if loadings is not None:
            l = loadings[comp_idx]
            heatmap_corr = l.reshape(len(feature_labels), len(distances))
        else:
            heatmap_corr = correlate_heatmap(len(feature_labels),
                                             len(distances),
                                             X,
                                             X_latent[:, comp_idx])
        # corr matrix
        plot_heatmap(heatmap_ax,
                     heatmap_corr,
                     row_labels=feature_labels,
                     col_labels=distances,
                     set_row_labels=(label_all or n == 0),
                     set_col_labels=True,
                     dark=dark,
                     cbar=cbar)
        col_data = X_latent[:, comp_idx]
        # map
        plot_scatter(map_ax,
                     xs,
                     ys,
                     col_data,
                     dark=dark,
                     s_min=s_min,
                     s_max=s_max,
                     c_exp=c_exp,
                     s_exp=s_exp,
                     rasterized=rasterized)
        map_ax.set_title(f'Latent {comp_idx + 1}')
        if tag_values is not None:
            map_ax.set_xlabel(f'{tag_string} {tag_values[comp_idx]}')


def plot_prob_clusters(X_raw,
                       cluster_probs,
                       n_components,
                       path_theme,
                       xs,
                       ys,
                       x_extents=(0, 3500),
                       y_extents=(0, 6000),
                       max_only=False,
                       plt_cmap='gist_ncar',
                       shape_exp=0.5,
                       suptitle='GMM VaDE',
                       rasterized=True):
    # get the assignments based on maximum probability
    cluster_assignments = np.argmax(cluster_probs, axis=1)
    # get the colours for each cluster based on mean mixed uses
    m_m = mu_mus(X_raw, cluster_assignments, n_components, shape_exp)
    # print the axes in order of the strength of m_m
    sorted_cluster_idx = np.argsort(m_m)[::-1]
    # plot the axes
    util_funcs.plt_setup()
    fig, axes = plt.subplots(3, 7, figsize=(12, 8))
    counter = 0
    cmap = plt.cm.get_cmap(plt_cmap)
    for ax_row in axes:
        for ax in ax_row:
            if counter < cluster_probs.shape[1]:
                cluster_idx = sorted_cluster_idx[counter]
                c = cmap(m_m[cluster_idx])
                vals = cluster_probs[:, cluster_idx]
                if max_only:
                    max_idx = (cluster_assignments == cluster_idx)
                    # shape c and s manually
                    v = np.full(len(cluster_probs), 0.0)
                    v[max_idx] = vals[max_idx]
                    s = np.copy(v)
                    s[max_idx] *= 0.75
                    s[max_idx] += 0.25
                else:
                    s = vals
                s **= 1
                # override vals with explicit "c" and "s"
                plot_scatter(ax,
                             xs,
                             ys,
                             c=c,
                             s=s,
                             x_extents=x_extents,
                             y_extents=y_extents,
                             rasterized=rasterized)
                ax.set_title(f'Cluster {cluster_idx + 1}')
            counter += 1
    plt.suptitle(suptitle)
    path = f'../phd-doc/doc/part_3/signatures/images/{path_theme}_cluster_composite'
    if max_only:
        path += '_max'
    path += '.pdf'
    plt.savefig(path, dpi=300)


def mu_mus(X_raw, cluster_assignments, n_components, shape_exp=0.5):
    '''
    prepare matrix for generating colours from mean mixed uses for each cluster
    '''
    mu_mus = []
    for n in range(n_components):
        mu_mus.append(X_raw['mu_hill_branch_wt_0_100'][cluster_assignments == n].mean())
    # normalise
    mu_mus = np.array(mu_mus)
    mu_mus = (mu_mus - np.nanmin(mu_mus)) / (np.nanmax(mu_mus) - np.nanmin(mu_mus))
    # shift to 0.1 - 0.9
    mu_mus = mu_mus * 0.8 + 0.05
    # reshape so that outliers don't bunch low-lying colours
    mu_mus **= shape_exp
    return mu_mus


def map_diversity_colour(X_raw,
                         cluster_assignments,
                         n_components,
                         plt_cmap='gist_ncar',
                         shape_exp=0.5):
    # get the colours based on mean mixed uses
    m_m = mu_mus(X_raw, cluster_assignments, n_components, shape_exp)
    # set colours in the full array - length of samples x rgba
    colours = np.full(shape=(cluster_assignments.shape[0], 4), fill_value=np.nan)
    sizes = np.full(shape=(cluster_assignments.shape[0], 1), fill_value=np.nan)
    cmap = plt.cm.get_cmap(plt_cmap)
    for n in range(n_components):
        colours[cluster_assignments == n] = cmap(m_m[n])
        sizes[cluster_assignments == n] = m_m[n]
    return colours, sizes
