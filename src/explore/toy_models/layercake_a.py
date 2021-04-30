import matplotlib.pyplot as plt
import numpy as np
from cityseer.metrics import networks
from tqdm import tqdm

from src import util_funcs
from src.explore.toy_models import generate_data_layer, plotter


def mmm_layercake_a(_graph,
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
                spill_rate=1.1
            })
        ''')

    for l_s in _layer_specs:
        for k in ['cap_step', 'dist_threshold', 'pop_threshold', 'spill_rate']:
            if k not in l_s:
                raise AttributeError(f'Missing key {k}')

    _spans = len(_graph)

    # generate the backbone Network Layer
    # include 1 (local only) for directly counting items assigned to current node
    # also include np.inf so that all node combinations are considered in spite of wrapping around back
    distances = [l_s['dist_threshold'] for l_s in _layer_specs]
    distances = list(set(distances))
    Netw_Layer = networks.Network_Layer_From_nX(_graph, distances=distances)
    # generate population data layer
    Pop_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    # population state and map
    # for this experiment assignments are not changing
    pop_state = np.full(len(Pop_Layer.uids), 20.0)  # use floats!
    # for plotting density onto to network nodes
    pop_map = np.full((_iters, _spans), 0.0)
    # generate the landuse substrate
    # keep to 1 location per node for visualisation sake
    # randomisation is immaterial because all will be instanced as 0.0
    Landuse_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    landuse_maps = []
    capacitance_maps = []
    spillover_maps = []
    for _ in _layer_specs:
        landuse_maps.append(np.full((_iters, _spans), 0.0))
        capacitance_maps.append(np.full((_iters, _spans), 0.0))
        spillover_maps.append(np.full((_iters, _spans), 0.0))

    n_layers = len(_layer_specs)
    # per layer
    landuse_states = np.full((n_layers, len(Landuse_Layer.uids)), 0.0)
    capacitances = np.full((n_layers, _spans), 0.0)
    assigned_trips_actual = np.full((n_layers, _spans), 0.0)
    constrained_spillovers = np.full((n_layers, _spans), 0.0)
    netw_flows_actual = np.full((n_layers, _spans), 0.0)

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

        # identify locations for landuse development
        # enforce single landuse per location - identify free parcels
        squashed_landuse_states = np.sum(landuse_states, axis=0)

        # resets
        assigned_trips_actual.fill(0)
        constrained_spillovers.fill(0)

        for i, l_s in enumerate(_layer_specs):
            # record current landuse state
            # strength = landuse_states[i] * capacitances[i]
            landuse_maps[i][n] = landuse_states[i]

            # compute singly constrained realised flows
            Landuse_Layer.model_singly_constrained('assigned_flows',
                                                   Pop_Layer._data,
                                                   Landuse_Layer._data,
                                                   pop_intensity,
                                                   landuse_states[i])
            assigned_trips_actual[i] = Netw_Layer.metrics['models']['assigned_flows'][l_s['dist_threshold']][
                'assigned_trips']
            flows = Netw_Layer.metrics['models']['assigned_flows'][l_s['dist_threshold']]['network_flows']

            # reciprocate spillovers
            spills = assigned_trips_actual[i] * l_s['spill_rate']
            Landuse_Layer.model_singly_constrained('assigned_spillovers',
                                                   Landuse_Layer._data,
                                                   Landuse_Layer._data,
                                                   spills,
                                                   squashed_landuse_states)
            constrained_spillovers[i] = Netw_Layer.metrics['models']['assigned_spillovers'][l_s['dist_threshold']][
                'assigned_trips']
            spillovers = Netw_Layer.metrics['models']['assigned_spillovers'][l_s['dist_threshold']]['assigned_trips']

            # TODO:
            # important: distributing flows this way causes different behaviour to below manner
            # below manner immediately seeds to the lagging edge of landuses - causing folds or pulses
            # on the other hand this manner mainly seeds on the leading edge nearer the actual flows
            # i.e. back-fill situations more likely to head in other direction
            # netw_flows_actual[i] = flows + spillovers
            # spillover_maps[i][n] = spillovers

        squashed_flows = np.sum(assigned_trips_actual, axis=0)
        squashed_spillovers = np.sum(constrained_spillovers, axis=0)
        for i, l_s in enumerate(_layer_specs):
            # distribute flows & spillovers - does not strictly take competition into account... but easier than furness process
            # TODO - spatial version: assymetrical betweenness or some or other gravity wouldn't take k competition into account...?

            Landuse_Layer.compute_stats_single('flow_intensity', squashed_flows)
            flows = Netw_Layer.metrics['stats']['flow_intensity']['mean_weighted'][l_s['dist_threshold']]

            Landuse_Layer.compute_stats_single('spillover_intensity', squashed_spillovers)
            spillovers = Netw_Layer.metrics['stats']['spillover_intensity']['mean_weighted'][l_s['dist_threshold']]

            # TODO: see above
            netw_flows_actual[i] = flows + spillovers
            spillover_maps[i][n] = spillovers

        # compute caps
        for i, l_s in enumerate(_layer_specs):
            # squash inside the loop to capture changes from previous iter
            squashed_landuse_states = np.sum(landuse_states, axis=0)
            # deduce flows and update capacitances
            flows = np.copy(netw_flows_actual[i])
            flows -= l_s['pop_threshold']
            flows /= l_s['pop_threshold']
            flows *= l_s['cap_step']
            flows = np.clip(flows, -l_s['cap_step'], l_s['cap_step'])
            capacitances[i] += flows
            # only seed per seed_rate
            t = netw_flows_actual[i]
            potential = int(np.nansum(t) / l_s['pop_threshold'])
            existing = np.nansum(t > l_s['pop_threshold'])
            new = potential - existing
            if new <= 0:
                rd_idx = np.random.randint(0, _spans)
                capacitances[i][rd_idx] = 1
            else:
                # prepare jitter - only scale jitter if not all == 0, e.g. first iter, otherwise use plain jitter
                # jitter = np.random.random(_spans)
                # jitter_scale = np.nanmax(flows) - np.nanmin(flows)
                # if jitter_scale:
                # jitter *= jitter_scale  # * l_s['explore_rate']
                # add jitter to flow
                # jittered = flows + jitter
                # sort by highest jittered flow
                seed_idx = np.argsort(flows, kind='mergesort')[::-1]
                # can't use np.intersect1d because it will sort indices
                seed_idx = seed_idx[np.in1d(seed_idx, np.where(squashed_landuse_states == 0))]
                seed_idx = seed_idx[:new]
                # normalise seeds and then add to capacitances
                # normalise seeds and then add to capacitances for continuous changes
                # for single length arrays, normalised min will have max = 0 - use nanmax or treat separately
                if len(seed_idx) == 1:
                    capacitances[i][seed_idx] += 1
                elif len(seed_idx) != 0:
                    seed_vals = flows[seed_idx]
                    seed_vals -= np.nanmin(seed_vals)
                    seed_vals /= np.nanmax(seed_vals)
                    # seed_vals *= l_s['cap_step']
                    capacitances[i][seed_idx] += seed_vals

            # constrain capacitance
            capacitances[i] = np.clip(capacitances[i], 0, 1)
            # deactivate dead landuses and activate new
            off_idx = np.intersect1d(np.where(capacitances[i] <= 0), np.where(squashed_landuse_states == 1))
            on_idx = np.intersect1d(np.where(capacitances[i] >= 1), np.where(squashed_landuse_states == 0))
            landuse_states[i][off_idx] = 0
            landuse_states[i][on_idx] = 1
            # record capacitance state
            capacitance_maps[i][n] = capacitances[i]

    return pop_map, landuse_maps, capacitance_maps, spillover_maps


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
        pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
                                                                             _iters,
                                                                             _layer_specs=_layer_specs,
                                                                             random_seed=random_seed)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        caps = capacitance_maps[0]
        lus = landuse_maps[0]
        flows = flow_maps[0]
        plotter(ax,
                _iters,
                xs,
                _res_factor=1,
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
            pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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

    plt.show()


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

    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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

    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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

    plt.show()


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
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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
    pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_a(_graph,
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

    plt.show()
