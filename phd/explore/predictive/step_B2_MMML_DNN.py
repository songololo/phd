import os
import numpy as np
from cityseer.metrics import networks, layers
from tqdm import tqdm
import matplotlib.pyplot as plt

import phd_util
from explore.MMM._blocks import generate_data_layer, plotter

os.environ['CITYSEER_QUIET_MODE'] = '1'


def mmml_phd(_graph,
             _iters: int,
             distances: list,
             avg_dwell_dens: float,
             models: dict,
             column_order: list,
             cap_step: float,
             cap_jitter: float,
             max_cap: float,
             new_threshold: float,
             random_seed: int):
    '''
    landuse states is based on the number of nodes: kept simple for visualisation

    don't confuse with the data layer, which is POI based
    for the sake of the simulation, all POI's assume the same x, y locations as assigned nodes

    column order must be filtered to exclude the currently targeted layer

    the tension between existing stores and latent competition (for given capacity) is important to dynamics
    the non-linearity between activation and deactivation is also important
    '''
    _spans = len(_graph)

    # set random seed for reproducibility and comparison across scenarios
    np.random.seed(random_seed)

    # build network layer
    Netw_Layer = networks.Network_Layer_From_nX(_graph, distances=distances)
    # calculate centralities - static
    Netw_Layer.compute_centrality(['node_harmonic_angular'], angular=True)
    Netw_Layer.compute_centrality(['node_betweenness_beta'], angular=False)

    # generate flat data layer - network assignment happens internally - note randomised=False
    Flat_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    # calculate dwelling density - static
    dwellings = np.full(_spans, avg_dwell_dens)
    Flat_Layer.compute_stats_single('population_density', dwellings)
    # the dwellings map is static in this case, so may as well set up front
    dwellings_map = np.full((_iters, _spans), float(avg_dwell_dens))

    # prepare the arrays for timeline maps that keep track of computations
    # one per layer / model
    landuse_maps = []
    capacitance_maps = []
    for _ in models:
        landuse_maps.append(np.full((_iters, _spans), 0.0))
        capacitance_maps.append(np.full((_iters, _spans), 0.0))
    # per layer
    n_layers = len(models)
    landuse_states = np.full((n_layers, _spans), 0.0)
    capacitances = np.full((n_layers, _spans), 0.0)
    measured_accessibility = np.full((n_layers, _spans), 0.0)
    predicted_accessibility = np.full((n_layers, _spans), 0.0)

    # iterate iters
    for n in tqdm(range(_iters)):
        # build a landuse map reflecting the current state
        # shape is sum of all landuses by x, y, nearest, next nearest
        data_uids = []
        lu_data_map = np.full((int(landuse_states.sum()), 4), np.nan)
        landuse_labels = []
        # iterate each model's layer
        current_lu_counter = 0
        for model_idx, model_key in enumerate(models.keys()):
            # get the node indices of each active landuse location (for given model's layer)
            current_active = np.where(landuse_states[model_idx] != 0)[0]
            # iterate the active node locations and add to the current landuses data structure
            for active_idx in current_active:
                # create one instance per landuse
                for active_n in range(int(landuse_states[model_idx][active_idx])):
                    # get the x and y
                    x, y = (Netw_Layer.x_arr[active_idx], Netw_Layer.y_arr[active_idx])
                    # add to uids
                    data_uids.append(current_lu_counter)
                    # add to the data structure: nearest node will be the current node
                    lu_data_map[current_lu_counter] = [x, y, active_idx, np.nan]
                    current_lu_counter += 1
                    # add the landuse code associated with the model
                    landuse_labels.append(model_key)
        # the current landuses data structure can now be used for landuse accessibilities
        # use empty placeholders if no landuse datapoints
        # first iter needs to create keys
        if not lu_data_map.shape[0]:
            for key in models.keys():
                if not key in Netw_Layer.metrics['accessibility']['weighted']:
                    Netw_Layer.metrics['accessibility']['weighted'][key] = {}
                for dist in distances:
                    if not dist in Netw_Layer.metrics['accessibility']['weighted'][key]:
                        Netw_Layer.metrics['accessibility']['weighted'][key][dist] = {}
                    Netw_Layer.metrics['accessibility']['weighted'][key][dist] = np.full(
                        _spans, 0.0)
        else:
            # generate a datalayer
            DL = layers.Data_Layer(data_uids, lu_data_map)
            # manually assign the network layer
            DL._Network = Netw_Layer
            # compute accessibilities for label of each datapoint against list of target keys
            DL.compute_accessibilities(landuse_labels, list(models.keys()))

        # for each model, build the input data array fed to the model, then predict
        for model_idx, (model_key, model_vals) in enumerate(models.items()):
            # number of columns is columns * distances - target columns * distances
            col_num = len(column_order) * len(distances) - len(distances)
            # placeholder array
            X_arr = np.full((_spans, col_num), np.nan)
            # iterate the columns, then distances, and add to input data array
            col_num_idx = 0
            for col in column_order:
                # target columns
                if col == model_key:
                    continue
                elif col == 'c_node_harmonic_angular_{dist}':
                    c = Netw_Layer.metrics['centrality']['node_harmonic_angular']
                elif col == 'c_node_betweenness_beta_{dist}':
                    c = Netw_Layer.metrics['centrality']['node_betweenness_beta']
                elif col == 'cens_dwellings_{dist}':
                    c = Netw_Layer.metrics['stats']['population_density']['sum_weighted']
                else:
                    c = Netw_Layer.metrics['accessibility']['weighted'][col]
                for dist in distances:
                    X_arr[:, col_num_idx] = c[dist]
                    col_num_idx += 1
            assert col_num_idx == col_num
            # predict y_hat
            X_trans = model_vals['transformer'].transform(X_arr)
            predicted_accessibility[model_idx] = model_vals['model'].predict(X_trans).flatten()

        # extract the target column at the target distance for comparison to predicted
        for model_idx, (model_key, model_vals) in enumerate(models.items()):
            d = model_vals['target_dist']
            measured_accessibility[model_idx] = \
                Netw_Layer.metrics['accessibility']['weighted'][model_key][d]

        # adjust capacitances and landuse states
        for i in range(len(models)):
            # calculate potentials
            potentials = predicted_accessibility[i] - measured_accessibility[i]
            # currently occupied landuse locations for given layer
            occupied_loc = landuse_states[i] == 1
            # squash landuses - do this inside loop to capture previous on / off state changes
            squashed_landuses = landuse_states.sum(axis=0)
            """
            EXISTING LOCATIONS
            """
            # existing locations - OK as long as potentials are greater than measured
            exist_pot = np.copy(potentials)
            # normalise to smoothen and avoid batch-wise blocks
            exist_pot /= np.abs(potentials).max()  # intentionally using potentials
            # scale to capacitance step
            exist_pot *= cap_step
            # only applies to current landuse locations
            capacitances[i][occupied_loc] += exist_pot[occupied_loc]
            """
            POTENTIAL LOCATIONS
            """
            future_pot = np.copy(potentials)
            # future potentials - must exceed new_threshold to trigger new location
            future_pot -= new_threshold
            # normalise to smoothen and avoid batch-wise blocks
            future_pot /= np.abs(potentials).max()  # intentionally using potentials
            # scale to capacitance step
            future_pot *= cap_step
            # only applies to available locations
            capacitances[i][~occupied_loc] += future_pot[~occupied_loc]
            """
            LANDUSE STATES
            """
            # add some stochasticity - scale to capacitance step
            capacitances[i] += np.random.randn(_spans) * cap_step * cap_jitter
            # clip capacitances
            capacitances[i] = np.clip(capacitances[i], 0, max_cap)
            # turn off per current capacitance for current landuse
            off_idx = np.logical_and(capacitances[i] <= 0, landuse_states[i] == 1)
            landuse_states[i][off_idx] = 0
            # turn on if no landuses currently at this index
            on_idx = np.logical_and(capacitances[i] >= 1, squashed_landuses == 0)
            landuse_states[i][on_idx] = 1
            # record landuse and capacitance states
            capacitance_maps[i][n] = capacitances[i]
            landuse_maps[i][n] = landuse_states[i]

    return dwellings_map, landuse_maps, capacitance_maps


def style_ax(_ax, _title, _iters, dark=False):
    _ax.set_title(_title)
    if dark:
        _ax.set_facecolor('#2e2e2e')
    for side in ['top', 'bottom', 'left', 'right']:
        _ax.spines[side].set_visible(False)
    y_ticks = _ax.get_yticks()
    # the y_ticks are amplified by the number of nested rows, so need to be adjusted for labels
    y_tick_factor = max(y_ticks) / _iters
    # convert to string and reverse
    y_tick_labels = reversed([str(int(l / y_tick_factor)) for l in y_ticks])
    _ax.set_yticklabels(y_tick_labels)


def mmm_single(_graph,
               _iters,
               _layer_specs,
               _title='',
               random_seed=0,
               figsize=(12, 20),
               theme=None,
               path=None,
               dark=False):
    xs = []
    for n, d in _graph.nodes(data=True):
        xs.append(d['x'])

    phd_util.plt_setup(dark=dark)

    if isinstance(_layer_specs, dict):
        pop_map, landuse_maps, capacitance_maps = mmml_phd(_graph,
                                                           _iters,
                                                           distances,
                                                           avg_dwell_dens,
                                                           models,
                                                           column_order,
                                                           cap_step,
                                                           cap_jitter,
                                                           max_cap,
                                                           new_threshold,
                                                           random_seed)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        caps = capacitance_maps[0]
        lus = landuse_maps[0]
        flows = flow_maps[0]
        plotter(ax, _iters, xs, _res_factor=1,
                _plot_maps=[pop_map, caps, lus, flows],
                _plot_scales=(1, 1.5, 0.75, 1))
        title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
        style_ax(ax, title, _iters, dark=dark)

    else:
        assert isinstance(_layer_specs, (list, tuple))
        n_ax = len(_layer_specs)
        fig, axes = plt.subplots(1, n_ax, figsize=figsize)
        if isinstance(_title, str):
            _title = [_title] * n_ax
        else:
            assert isinstance(_title, (list, tuple))
            assert len(_title) == len(_layer_specs)
        for ax, title, layer_spec in zip(axes, _title, _layer_specs):
            pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_phd(_graph,
                                                                                   _iters,
                                                                                   _layer_specs=layer_spec,
                                                                                   random_seed=random_seed)
            caps = capacitance_maps[0]
            lus = landuse_maps[0]
            flows = flow_maps[0]
            plotter(ax, _iters, xs, _res_factor=1, _plot_maps=[pop_map, caps, lus, flows])
            title = title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
            style_ax(ax, title, _iters, dark=dark)

    if theme is not None:
        fig.suptitle(theme)

    if path is not None:
        plt.savefig(path, dpi=300)


def mmm_nested_split(_graph,
                     _iters,
                     _layer_specs,
                     _title='',
                     random_seed=0,
                     figsize=(30, 20),
                     theme=None,
                     path=None,
                     dark=False):
    xs = []
    for n, d in _graph.nodes(data=True):
        xs.append(d['x'])

    phd_util.plt_setup(dark=dark)

    assert isinstance(_layer_specs, (list, tuple))
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_phd(_graph,
                                                                           _iters,
                                                                           _layer_specs=
                                                                           _layer_specs[0],
                                                                           random_seed=random_seed)
    plotter(axes[0], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            capacitance_maps[0],
                                                            landuse_maps[0]],
            _plot_colours=['#555555', '#d32f2f', '#2e7d32'])
    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[0], title, _iters, dark=dark)

    # both
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_phd(_graph,
                                                                           _iters,
                                                                           _layer_specs=_layer_specs,
                                                                           random_seed=random_seed)
    plotter(axes[1], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            np.sum(capacitance_maps, axis=0),
                                                            landuse_maps[0],
                                                            landuse_maps[1]],
            _plot_colours=['#555555', '#d32f2f', '#2e7d32', '#0064b7'])
    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[1], title, _iters, dark=dark)

    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                         _iters,
                                                                         _layer_specs=
                                                                         _layer_specs[1],
                                                                         random_seed=random_seed)
    plotter(axes[2], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            capacitance_maps[0],
                                                            landuse_maps[0]],
            _plot_colours=['#555555', '#d32f2f', '#0064b7'])
    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[2], title, _iters, dark=dark)

    if theme is not None:
        fig.suptitle(theme)

    if path is not None:
        plt.savefig(path, dpi=300)


def mmm_nested_doubles(_graph,
                       _iters,
                       _layer_specs,
                       _title='',
                       random_seed=0,
                       figsize=(30, 20),
                       theme=None,
                       path=None,
                       dark=False):
    xs = []
    for n, d in _graph.nodes(data=True):
        xs.append(d['x'])

    phd_util.plt_setup(dark=dark)

    assert isinstance(_layer_specs, (list, tuple))
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # both
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                         _iters,
                                                                         _layer_specs=_layer_specs,
                                                                         random_seed=random_seed)

    plotter(axes[1], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            np.sum(capacitance_maps, axis=0),
                                                            landuse_maps[0],
                                                            landuse_maps[1]],
            _plot_colours=['#555555', '#d32f2f', '#2e7d32', '#0064b7'])

    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[1], title, _iters, dark=dark)

    fast_layers = (_layer_specs[0].copy(), _layer_specs[0].copy())
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                         _iters,
                                                                         _layer_specs=fast_layers,
                                                                         random_seed=random_seed)

    plotter(axes[0], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            np.sum(capacitance_maps, axis=0),
                                                            landuse_maps[0],
                                                            landuse_maps[1]],
            _plot_colours=['#555555', '#d32f2f', '#2e7d32', '#0064b7'])
    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[0], title, _iters, dark=dark)

    slow_layers = (_layer_specs[1].copy(), _layer_specs[1].copy())
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                         _iters,
                                                                         _layer_specs=slow_layers,
                                                                         random_seed=random_seed)
    plotter(axes[2], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            np.sum(capacitance_maps, axis=0),
                                                            landuse_maps[0],
                                                            landuse_maps[1]],
            _plot_colours=['#555555', '#d32f2f', '#2e7d32', '#0064b7'])
    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[2], title, _iters, dark=dark)

    if theme is not None:
        fig.suptitle(theme)

    if path is not None:
        plt.savefig(path, dpi=300)
