import matplotlib.pyplot as plt
import numpy as np
from cityseer.metrics import networks
from tqdm import tqdm

from src import util_funcs
from src.explore.toy_models import generate_data_layer, plotter


def mmm_layercake_d(_graph,
                    _iters=200,
                    _layer_specs=(),
                    random_seed=0):
    if isinstance(_layer_specs, dict):
        _layer_specs = (_layer_specs,)

    if not len(_layer_specs):
        raise AttributeError('''
            No layer specs provided: e.g. ({
                cap_step=0.5,
                dist_threshold=800,
                pop_threshold=800,
                explore_rate=0.5,
                comp_rate=0.5,
                echo_rate=0.1
            })
        ''')

    for l_s in _layer_specs:
        for k in ['cap_step', 'dist_threshold', 'pop_threshold', 'explore_rate', 'comp_rate', 'echo_rate']:
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
    flow_maps = []
    for _ in _layer_specs:
        landuse_maps.append(np.full((_iters, _spans), 0.0))
        capacitance_maps.append(np.full((_iters, _spans), 0.0))
        flow_maps.append(np.full((_iters, _spans), 0.0))

    n_layers = len(_layer_specs)
    # per layer
    landuse_states = np.full((n_layers, len(Landuse_Layer.uids)), 0.0)
    capacitances = np.full((n_layers, _spans), 0.0)

    assigned_trips_actual = np.full((n_layers, _spans), 0.0)
    assigned_trips_potential = np.full((n_layers, _spans), 0.0)
    assigned_trips_echos = np.full((n_layers, _spans), 0.0)

    netw_flow_actual = np.full((n_layers, _spans), 0.0)
    netw_flow_potential = np.full((n_layers, _spans), 0.0)
    netw_flow_echos = np.full((n_layers, _spans), 0.0)

    agged_trips_actual = np.full((n_layers, _spans), 0.0)
    agged_flows_actual = np.full((n_layers, _spans), 0.0)
    agged_trips_potential = np.full((n_layers, _spans), 0.0)
    agged_flows_potential = np.full((n_layers, _spans), 0.0)

    # left and right spatial gradients
    right_gradient = np.zeros_like(pop_state)
    left_gradient = np.zeros_like(pop_state)
    # get max and min
    ma = 0
    mi = np.inf
    for n, d in _graph.nodes(data=True):
        x = d['x']
        if x > ma:
            ma = x
        if x < mi:
            mi = x
    # set gradient strength
    for i, (n, d) in enumerate(_graph.nodes(data=True)):
        x = d['x']
        right_gradient[i] = (x - mi) / (ma - mi)
        left_gradient[i] = 1 - right_gradient[i]

    # for keeping track of fifths for different gradients etc.
    inc_5 = int(_iters / 5)

    # set random seed for reproducibility and comparison across scenarios
    np.random.seed(random_seed)

    # iterate
    for n in tqdm(range(_iters)):

        # calculate neighbourhood density
        if n < inc_5:
            pop_intensity = pop_state
        elif n < inc_5 * 2:
            pop_intensity = right_gradient * pop_state
        elif n < inc_5 * 3:
            pop_intensity = left_gradient * pop_state
        elif n < inc_5 * 4:
            pop_intensity = pop_state * 2
        else:
            pop_intensity = pop_state / 2
        # record current state - actually doesn't change for this experiment...
        pop_map[n] = pop_intensity

        # apportion flow - once per layer
        flat_lu = np.full(_spans, 1.0)
        for i, l_s in enumerate(_layer_specs):
            # record current landuse state
            landuse_maps[i][n] = landuse_states[i] * capacitances[i]

            ### ACTUAL
            # compute singly constrained realised flows
            # NOTE: do not use a capacitance weighted version:
            # singly-constrained already effectively takes competition vis-a-vis flow potentials into account!!!
            Landuse_Layer.model_singly_constrained('trips',
                                                   Pop_Layer._data,
                                                   Landuse_Layer._data,
                                                   pop_intensity,
                                                   landuse_states[i])
            assigned_trips_actual[i] = Netw_Layer.metrics['models']['trips'][l_s['dist_threshold']]['assigned_trips']
            netw_flow_actual[i] = Netw_Layer.metrics['models']['trips'][l_s['dist_threshold']]['network_flows']
            # aggregate PREVIOUS iterations echos to current trips BEFORE computing new echos
            agged_trips_actual[i] = assigned_trips_actual[i] + assigned_trips_echos[i]
            agged_flows_actual[i] = netw_flow_actual[i] + netw_flow_echos[i]

            # record flow state
            flow_maps[i][n] = netw_flow_actual[i]

            ### POTENTIALS
            # compute potential flows (for exploration)
            Landuse_Layer.model_singly_constrained('potentials',
                                                   Pop_Layer._data,
                                                   Landuse_Layer._data,
                                                   pop_intensity,
                                                   flat_lu)
            assigned_trips_potential[i] = Netw_Layer.metrics['models']['potentials'][l_s['dist_threshold']][
                'assigned_trips']
            netw_flow_potential[i] = Netw_Layer.metrics['models']['potentials'][l_s['dist_threshold']]['network_flows']
            # aggregate PREVIOUS iterations echos to current potentials BEFORE computing new echos
            agged_trips_potential[i] = assigned_trips_potential[i] + assigned_trips_echos[i]
            agged_flows_potential[i] = netw_flow_potential[i] + netw_flow_echos[i]

            ### echos
            # compute singly constrained echos (spillovers)
            # uses echo rate - compounds current trip origins with previous echos loop
            echos = assigned_trips_actual[i] * l_s['echo_rate'] + assigned_trips_echos[i] * l_s['echo_rate']
            Landuse_Layer.model_singly_constrained('echos',
                                                   Landuse_Layer._data,  # landuse to landuse
                                                   Landuse_Layer._data,  # landuse to landuse
                                                   echos,
                                                   landuse_states[i])
            assigned_trips_echos[i] = Netw_Layer.metrics['models']['echos'][l_s['dist_threshold']]['assigned_trips']
            netw_flow_echos[i] = Netw_Layer.metrics['models']['echos'][l_s['dist_threshold']]['network_flows']

        # compute
        # the tension between existing stores and latent competition (for given capacity) is important to dynamics
        # the non-linearity between activation and deactivation is also important
        squashed_flows_actual = np.sum(agged_flows_actual, axis=0)
        squashed_flows_potential = np.sum(agged_flows_potential, axis=0)
        for i, l_s in enumerate(_layer_specs):

            blended = squashed_flows_actual * (1 - l_s['explore_rate']) + squashed_flows_potential * l_s['explore_rate']
            jitter = np.random.randn(_spans) * l_s['cap_step'] - l_s['cap_step'] / 2
            blended += jitter

            success_idx = np.where(agged_trips_actual[i] >= l_s['pop_threshold'])[0]
            if len(success_idx) > 0:
                success_threshold = np.nanmin(agged_flows_actual[i][success_idx])
                caps = np.copy(blended)
                caps -= success_threshold
                caps /= success_threshold
                caps *= l_s['cap_step']
                caps = np.clip(caps, -l_s['cap_step'], l_s['cap_step'])
                capacitances[i] += caps
            else:
                capacitances[i] -= l_s['cap_step']

            # determine latent competition - start by measuring actual and potential landuse intensity
            potential = np.nansum(agged_trips_potential[i])
            actual = np.nansum(agged_trips_actual[i])
            # high potential = full competition for flows: new entrants attempting to capture from non-competitive existing
            success_n = len(success_idx)
            high_potential = potential / l_s['pop_threshold'] - success_n
            # low potential represents weak competition for flows, new entrants only trying to capture untapped trips
            low_potential = (potential - actual) / l_s['pop_threshold']
            # apply competition rate
            new = int(low_potential * (1 - l_s['comp_rate']) + high_potential * l_s['comp_rate'])
            if new > 0:
                # sort by highest jittered flow
                seed_idx = np.argsort(blended, kind='mergesort')[::-1]
                # can't use np.intersect1d because it will sort indices
                # trying to steal flows from existing buildings, so can't populate currently existing landuse activations
                seed_idx = seed_idx[np.in1d(seed_idx, np.where(landuse_states[i] == 0)[0])]
                # snip off at new
                seed_idx = seed_idx[:new]
                # normalise seeds and then add to capacitances for continuous changes
                # for single length arrays, normalised min will have max = 0 - use nanmax or treat separately
                if len(seed_idx) == 1:
                    capacitances[i][seed_idx] += 1
                elif len(seed_idx) != 0:
                    seed_vals = blended[seed_idx]
                    seed_vals -= np.nanmin(seed_vals)
                    seed_vals /= np.nanmax(seed_vals)
                    # seed_vals *= l_s['cap_step']
                    capacitances[i][seed_idx] += seed_vals

            capacitances[i] = np.clip(capacitances[i], 0, 1)
            off_idx = np.intersect1d(np.where(capacitances[i] == 0), np.where(landuse_states[i] == 1))
            landuse_states[i][off_idx] = 0
            on_idx = np.intersect1d(np.where(capacitances[i] == 1), np.where(landuse_states[i] == 0))
            landuse_states[i][on_idx] = 1
            # record capacitance state
            capacitance_maps[i][n] = capacitances[i]

    return pop_map, landuse_maps, capacitance_maps, flow_maps


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

    util_funcs.plt_setup(dark=dark)

    if isinstance(_layer_specs, dict):
        pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                             _iters,
                                                                             _layer_specs=_layer_specs,
                                                                             random_seed=random_seed)
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
            pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
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

    util_funcs.plt_setup(dark=dark)

    assert isinstance(_layer_specs, (list, tuple))
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                         _iters,
                                                                         _layer_specs=_layer_specs[0],
                                                                         random_seed=random_seed)
    plotter(axes[0], _iters, xs, _res_factor=1, _plot_maps=[pop_map,
                                                            capacitance_maps[0],
                                                            landuse_maps[0]],
            _plot_colours=['#555555', '#d32f2f', '#2e7d32'])
    title = _title + f'l.u: {np.round(np.sum(landuse_maps), 2)}'
    style_ax(axes[0], title, _iters, dark=dark)

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

    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_d(_graph,
                                                                         _iters,
                                                                         _layer_specs=_layer_specs[1],
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

    util_funcs.plt_setup(dark=dark)

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
