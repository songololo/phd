import numpy as np
from cityseer.metrics import networks
from tqdm import tqdm
import matplotlib.pyplot as plt

import phd_util
from explore.MMM._blocks import generate_data_layer, plotter

def mmm_layercake_b(_graph,
                    _iters=200,
                    _layer_specs=(),
                    seed=False,
                    random_seed=0):

    if isinstance(_layer_specs, dict):
        _layer_specs = (_layer_specs,)

    if not len(_layer_specs):
        raise AttributeError('''
            No layer specs provided: e.g. ({
                cap_step=0.5,
                dist_threshold=800,
                pop_threshold=800,
                explore_rate=0.5
            })
        ''')

    for l_s in _layer_specs:
        for k in ['cap_step', 'dist_threshold', 'pop_threshold', 'explore_rate']:
            if k not in l_s:
                raise AttributeError(f'Missing key {k}')

    _spans = len(_graph)

    # generate the backbone Network Layer
    # include 1 (local only) for directly counting items assigned to current node
    # also include np.inf so that all node combinations are considered in spite of wrapping around back
    distances = [l_s['dist_threshold'] for l_s in _layer_specs]
    distances = list(set(distances))
    Netw_Layer = networks.Network_Layer_From_nX(_graph, distances=distances)
    # generate population data layer - assignment happens internally - note randomised=False
    Pop_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    # population state and map
    # for this experiment assignments are not changing
    pop_state = np.full(len(Pop_Layer.uids), 20.0)  # use floats!
    # for plotting density onto to network nodes
    pop_map = np.full((_iters, _spans), 0.0)
    # generate the landuse substrate
    # keep to 1 location per node for visualisation sake
    Landuse_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    landuse_maps = []
    capacitance_maps = []
    for _ in _layer_specs:
        landuse_maps.append(np.full((_iters, _spans), 0.0))
        capacitance_maps.append(np.full((_iters, _spans), 0.0))

    n_layers = len(_layer_specs)

    landuse_states = np.full((n_layers, len(Landuse_Layer.uids)), 0.0)  # per layer
    capacitances = np.full((n_layers, _spans), 0.0)
    assigned_trips_actual = np.full((n_layers, _spans), 0.0)
    assigned_trips_potential = np.full((n_layers, _spans), 0.0)
    netw_flow_actual = np.full((n_layers, _spans), 0.0)
    netw_flow_potential = np.full((n_layers, _spans), 0.0)

    # left and right gradients
    gradient = np.arange(_spans)
    gradient = gradient / np.max(gradient)
    right_gradient = np.full(_spans, 0.0)
    mid = int(_spans / 2)
    right_gradient[mid:] = gradient[:mid]
    right_gradient[:mid] = gradient[mid:]
    left_gradient = np.flip(right_gradient)

    # for keeping track of fifths for different gradients etc.
    inc_5 = int(_iters / 5)

    # set random seed for reproducibility and comparison across scenarios
    np.random.seed(random_seed)
    # seed
    if seed:
        for n in range(n_layers):
            rand_idx = np.random.randint(0, _spans)
            landuse_states[n][rand_idx] = 1

    # iterate
    for n in tqdm(range(_iters)):

        # calculate neighbourhood density
        pop_intensity = np.copy(pop_state)
        if n < inc_5:
            pass
        elif n < inc_5 * 2:
            pop_intensity *= left_gradient
        elif n < inc_5 * 3:
            pop_intensity *= right_gradient
        elif n < inc_5 * 4:
            pop_intensity *= 2
        else:
            pop_intensity /= 2
        # record current state - actually doesn't change for this experiment...
        pop_map[n] = pop_intensity

        # apportion flow - once per layer
        assigned_trips_actual.fill(0)
        netw_flow_actual.fill(0)
        assigned_trips_potential.fill(0)
        netw_flow_potential.fill(0)
        flat_lu = np.full(_spans, 1.0)
        for i, l_s in enumerate(_layer_specs):

            # record current landuse state
            landuse_maps[i][n] = landuse_states[i]

            # compute singly constrained realised flows
            Landuse_Layer.model_singly_constrained('assigned_flows', Pop_Layer._data, Landuse_Layer._data, pop_intensity, landuse_states[i])
            assigned_trips_actual[i] += Netw_Layer.metrics['models']['assigned_flows'][l_s['dist_threshold']]['assigned_trips']
            netw_flow_actual[i] += Netw_Layer.metrics['models']['assigned_flows'][l_s['dist_threshold']]['network_flows']

            # compute potential flows (for exploration)
            # this is an even surface and doesn't need smoothing
            Landuse_Layer.model_singly_constrained('potential_flows', Pop_Layer._data, Landuse_Layer._data, pop_intensity, flat_lu)
            assigned_trips_potential[i] += Netw_Layer.metrics['models']['potential_flows'][l_s['dist_threshold']]['assigned_trips']
            netw_flow_potential[i] += Netw_Layer.metrics['models']['potential_flows'][l_s['dist_threshold']]['network_flows']

        # compute caps
        squashed_netw_flow_actual = np.sum(netw_flow_actual, axis=0)
        squashed_netw_flow_potential = np.sum(netw_flow_potential, axis=0)
        for i, l_s in enumerate(_layer_specs):
            # deduce flows and update capacitances
            flows = np.copy(squashed_netw_flow_actual)
            flows -= l_s['pop_threshold']
            flows /= l_s['pop_threshold']
            flows *= l_s['cap_step']
            flows = np.clip(flows, -l_s['cap_step'], l_s['cap_step'])
            capacitances[i] += flows
            # constrain capacitance
            capacitances[i] = np.clip(capacitances[i], 0, 1)

            # deactivate dead landuses and activate new
            # TODO: the non-linearity of the activation / deactivation seems significant...?
            off_idx = np.intersect1d(np.where(capacitances[i] <= 0), np.where(landuse_states[i] == 1))
            landuse_states[i][off_idx] = 0
            on_idx = np.intersect1d(np.where(capacitances[i] >= 1), np.where(landuse_states[i] == 0))
            landuse_states[i][on_idx] = 1
            # record capacitance state
            capacitance_maps[i][n] = capacitances[i]

            # only seed per seed_rate
            potential = np.nansum(assigned_trips_potential[i]) / l_s['pop_threshold']
            existing = np.nansum(assigned_trips_actual[i]) > l_s['pop_threshold']
            # VS: existing = np.nansum(landuse_states[i] == 1)
            new = int(potential - existing)
            if new > 0:
                blended = squashed_netw_flow_actual * (1 - l_s['explore_rate']) + squashed_netw_flow_potential * l_s['explore_rate']
                jitter = np.random.randn(_spans) * l_s['cap_step'] - l_s['cap_step'] / 2
                blended += jitter
                # sort by highest jittered flow
                seed_idx = np.argsort(blended, kind='mergesort')[::-1]
                # can't use np.intersect1d because it will sort indices
                seed_idx = seed_idx[np.in1d(seed_idx, np.where(landuse_states[i] == 0)[0])]
                seed_idx = seed_idx[:new]
                # normalise seeds and then add to capacitances for continuous changes
                # for single length arrays, normalised min will have max = 0 - use nanmax or treat separately
                if len(seed_idx) == 1:
                    capacitances[i][seed_idx] += 1
                elif len(seed_idx) != 0:
                    seed_vals = blended[seed_idx]
                    seed_vals -= np.nanmin(seed_vals)
                    seed_vals /= np.nanmax(seed_vals)
                    #seed_vals *= l_s['cap_step']
                    capacitances[i][seed_idx] += seed_vals



    return pop_map, landuse_maps, capacitance_maps


def style_ax(_ax, _title, _iters):
    _ax.set_title(_title)
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
               seed=False,
               random_seed=0,
               figsize=(12, 20),
               theme=None,
               path=None,
               dark=False):

    xs = []
    for n, d in _graph.nodes(data=True):
        xs.append(d['x'])

    background_col = '#ffffff'
    if dark:
        background_col = '#2e2e2e'

    phd_util.plt_setup()

    if isinstance(_layer_specs, dict):
        pop_map, landuse_maps, capacitance_maps = mmm_layercake_b(_graph,
                                                                  _iters,
                                                                  _layer_specs=_layer_specs,
                                                                  seed=seed,
                                                                  random_seed=random_seed)
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=background_col)
        caps = capacitance_maps[0]
        lus = landuse_maps[0] * caps
        plotter(ax, _iters, xs, _res_factor=1, _plot_maps=[pop_map, caps, lus])
        title = _title + f'l.u: {landuse_maps[0].sum()}'
        style_ax(ax, title, _iters)

    else:
        assert isinstance(_layer_specs, (list, tuple))
        n_ax = len(_layer_specs)
        fig, axes = plt.subplots(1, n_ax, figsize=figsize, facecolor=background_col)
        if isinstance(_title, str):
            _title = [_title] * n_ax
        else:
            assert isinstance(_title, (list, tuple))
            assert len(_title) == len(_layer_specs)
        for ax, title, layer_spec in zip(axes, _title, _layer_specs):
            pop_map, landuse_maps, capacitance_maps = mmm_layercake_b(_graph,
                                                                      _iters,
                                                                      _layer_specs=layer_spec,
                                                                      seed=seed,
                                                                      random_seed=random_seed)
            caps = capacitance_maps[0]
            lus = landuse_maps[0] * caps
            plotter(ax, _iters, xs, _res_factor=1, _plot_maps=[pop_map, caps, lus])
            title = title + f'l.u: {landuse_maps[0].sum()}'
            style_ax(ax, title, _iters)

    if theme is not None:
        fig.suptitle(theme)

    if path is not None:
        plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300, transparent=True)

    plt.show()
