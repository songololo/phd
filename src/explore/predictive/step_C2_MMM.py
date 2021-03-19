import matplotlib.pyplot as plt
import numpy as np
from cityseer.metrics import networks
from tqdm import tqdm

from src import phd_util
from src.explore.MMM._blocks import generate_data_layer, plotter


def mmm_layercake_phd(_graph,
                      _iters=200,
                      echo_rate=0.2,
                      echo_distance=400,
                      competition_factor=1.1,
                      death_rate=0.01,
                      flow_jitter=0.2,
                      cap_jitter=0.1,
                      _layer_specs=(),
                      random_seed=0):
    if isinstance(_layer_specs, dict):
        _layer_specs = (_layer_specs,)

    if not len(_layer_specs):
        raise AttributeError('''
            No layer specs provided: e.g. ({
                cap_step=0.5,
                dist_threshold=800,
                pop_threshold=800
            })
        ''')

    for l_s in _layer_specs:
        for k in ['cap_step', 'dist_threshold', 'pop_threshold']:
            if k not in l_s:
                raise AttributeError(f'Missing key {k}')

    _spans = len(_graph)

    # generate the backbone Network Layer
    # include 1 (local only) for directly counting items assigned to current node
    # also include np.inf so that all node combinations are considered
    # in spite of wrapping around back
    distances = [l_s['dist_threshold'] for l_s in _layer_specs]
    distances = list(set(distances))
    # add echo distance
    distances.append(echo_distance)
    distances = list(set(distances))
    # build network layer
    Netw_Layer = networks.Network_Layer_From_nX(_graph, distances=distances)
    # generate population data layer - assignment happens internally - note randomised=False
    Pop_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    # population state and timeline map
    # for this experiment assignments are not changing
    pop_state = np.full(len(Pop_Layer.uids), 1.0)  # use floats!
    pop_map = np.full((_iters, _spans), 0.0)
    # generate the landuse substrate
    # keep to 1 location per node for visualisation sake
    Landuse_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)
    # prepare the arrays for timeline maps that keep track of computations - one per layer
    landuse_maps = []
    capacitance_maps = []
    for _ in _layer_specs:
        landuse_maps.append(np.full((_iters, _spans), 0.0))
        capacitance_maps.append(np.full((_iters, _spans), 0.0))
    # flow map is based on squashed flows - only one copy necessary
    flow_map = np.full((_iters, _spans), 0.0)

    n_layers = len(_layer_specs)
    # per layer
    landuse_states = np.full((n_layers, len(Landuse_Layer.uids)), 0.0)
    capacitances = np.full((n_layers, _spans), 0.0)

    assigned_trips_actual = np.full((n_layers, _spans), 0.0)
    # assigned_trips_potential = np.full((n_layers, _spans), 0.0)
    assigned_trips_echos = np.full(_spans, 0.0)  # echos is squashed

    netw_flow_actual = np.full((n_layers, _spans), 0.0)
    # netw_flow_potential = np.full((n_layers, _spans), 0.0)
    netw_flow_echos = np.full(_spans, 0.0)  # echos is squashed

    agged_trips_actual = np.full((n_layers, _spans), 0.0)
    agged_flows_actual = np.full((n_layers, _spans), 0.0)
    # agged_trips_potential = np.full((n_layers, _spans), 0.0)
    # agged_flows_potential = np.full((n_layers, _spans), 0.0)

    # set random seed for reproducibility and comparison across scenarios
    np.random.seed(random_seed)
    # set seed landuse location (otherwise no initial flows)
    for i in range(n_layers):
        rdm_idx = np.random.choice(_spans, 1)
        landuse_states[i][rdm_idx] = 1

    # iterate
    for n in tqdm(range(_iters)):
        pop_intensity = pop_state
        # record current state - actually doesn't change for this experiment...
        pop_map[n] = pop_intensity
        # apportion flow - once per layer
        for i, l_s in enumerate(_layer_specs):
            ### ACTUAL
            # compute singly constrained realised flows
            # NOTE: do not use a capacitance weighted version:
            # singly-constrained already effectively takes competition vis-a-vis flow
            # potentials into account!!!
            Landuse_Layer.model_singly_constrained('trips',
                                                   Pop_Layer._data,
                                                   Landuse_Layer._data,
                                                   pop_intensity,
                                                   landuse_states[i])
            assigned_trips_actual[i] = \
                Netw_Layer.metrics['models']['trips'][l_s['dist_threshold']]['assigned_trips']
            netw_flow_actual[i] = Netw_Layer.metrics['models']['trips'][l_s['dist_threshold']][
                'network_flows']
            # aggregate PREVIOUS iterations echos to current trips BEFORE computing new echos
            agged_trips_actual[i] = assigned_trips_actual[i] + assigned_trips_echos
            agged_flows_actual[i] = netw_flow_actual[i] + netw_flow_echos

        # squashing
        squashed_trips = agged_trips_actual.sum(axis=0)
        squashed_flows = agged_flows_actual.sum(axis=0)
        squashed_landuses = landuse_states.sum(axis=0)
        flow_map[n] = squashed_flows  # record flow state

        ### ECHOS
        # compute singly constrained echos (spillovers) - combined for all layers
        echos = squashed_trips * echo_rate
        Landuse_Layer.model_singly_constrained('echos',
                                               Landuse_Layer._data,  # landuse to landuse
                                               Landuse_Layer._data,  # landuse to landuse
                                               echos,
                                               squashed_landuses)
        assigned_trips_echos = Netw_Layer.metrics['models']['echos'][echo_distance][
            'assigned_trips']
        netw_flow_echos = Netw_Layer.metrics['models']['echos'][echo_distance]['network_flows']

        # compute
        # the tension between existing stores and latent competition (for given capacity) is
        # important to dynamics
        # the non-linearity between activation and deactivation is also important
        for i, l_s in enumerate(_layer_specs):
            # update squashed landuses inside loop to capture changes
            squashed_landuses = landuse_states.sum(axis=0)
            '''
            FITNESS - this is based on actual trips and assessed for this layer only
            '''
            # the fitness function:
            # trips to specific location + conversion of adjacent flows
            up_locations = np.logical_and(agged_trips_actual[i] >= l_s['pop_threshold'],
                                          landuse_states[i] == 1)
            capacitances[i][up_locations] += l_s['cap_step']
            # decreases otherwise
            down_locations = np.logical_and(agged_trips_actual[i] < l_s['pop_threshold'],
                                            landuse_states[i] == 1)
            capacitances[i][down_locations] -= l_s['cap_step']
            '''
            POTENTIALS - this is based on potentials
            '''
            # identify n potential: total population / threshold for landuse
            potential = pop_state.sum() / l_s['pop_threshold']
            # multiply by competition factor and cast to int
            potential = int(potential * competition_factor)
            # identify latent demand by subtracting actual number of current landuses
            actual = landuse_states[i].sum()
            latent = potential - actual
            latent = int(np.clip(latent, 0, np.abs(latent)))
            # check for available locations
            available_locations = squashed_landuses == 0
            # identify n highest (jittered) flows at unoccupied locations
            jitter = np.random.randn(_spans) * squashed_flows.max() * flow_jitter
            sorted_flows_idx = np.argsort(squashed_flows + jitter, kind='mergesort')[::-1]
            # sort available locations accordingly and filter
            sorted_locations = available_locations[sorted_flows_idx]
            filtered_flows_idx = sorted_flows_idx[sorted_locations]
            # snip per latent demand
            filtered_flows_idx = filtered_flows_idx[:latent]
            capacitances[i][filtered_flows_idx] += l_s['cap_step']
            '''
            CAPACITANCE - non-linearity
            '''
            # add some stochasticity - scale to capacitance step
            capacitances[i] += np.random.randn(_spans) * l_s['cap_step'] * cap_jitter
            # clip capacitances to range
            capacitances[i] = np.clip(capacitances[i], 0, 1)
            # turn off and on per current capacitance
            off_idx = np.intersect1d(np.where(capacitances[i] == 0),
                                     np.where(landuse_states[i] == 1))  # use layer
            landuse_states[i][off_idx] = 0
            # cull based on death rate probability
            active_idx = np.where(landuse_states[i] == 1)[0]
            cull = np.random.choice([0, 1], active_idx.shape[0],
                                    p=[death_rate, 1 - death_rate])
            cull_idx = active_idx[cull == 0]
            landuse_states[i][cull_idx] = 0
            capacitances[i][cull_idx] = 0
            # turn on per current capacitance
            on_idx = np.intersect1d(np.where(capacitances[i] == 1),
                                    np.where(squashed_landuses == 0))  # use squashed
            landuse_states[i][on_idx] = 1
            # record landuse and capacitance states
            capacitance_maps[i][n] = capacitances[i]
            landuse_maps[i][n] = landuse_states[i]

    return pop_map, landuse_maps, capacitance_maps, flow_map


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
        pop_map, landuse_maps, capacitance_maps, flow_maps = mmm_layercake_phd(_graph,
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
