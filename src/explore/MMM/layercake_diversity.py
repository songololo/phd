import numpy as np
from cityseer.algos import data, checks
from cityseer.metrics import networks
from numba import njit

from src.explore.MMM._blocks import generate_data_layer, plotter


# numba doesn't yet support summing by axis, though might be added soon per #1269
@njit
def dim_0_maxer(multi_dim_arr):
    arr = np.full(multi_dim_arr.shape[1], 0.0)
    for d in range(multi_dim_arr.shape[0]):
        for idx, val in enumerate(multi_dim_arr[d]):
            if arr[idx] < val:
                arr[idx] = val
    return arr


@njit
def dim_0_squasher(multi_dim_arr):
    arr = np.full(multi_dim_arr.shape[1], 0.0)
    for d in range(multi_dim_arr.shape[0]):
        for idx, val in enumerate(multi_dim_arr[d]):
            arr[idx] += val
    return arr


# @njit
def compute_iter(_n_iters,
                 _node_map,
                 _edge_map,
                 _landuse_data_map,
                 _distances,
                 _betas,
                 _tiers,
                 _bands,
                 _spans,
                 _dist_idx,
                 _spill_rate,
                 _pop_threshold,
                 _cap_step,
                 _explore_rate,
                 _innovation_rate,
                 _pop_intensity):
    landuse_state = np.full((_n_iters, _tiers, _spans), 0)  # the actual lu code - bands are derived from these
    landuse_activation = np.full((_tiers, _bands, _spans), 0.0)  # precompute available factors per band
    landuse_capacitance = np.full((_tiers, _spans), 0.0)
    constrained_flows = np.full((_n_iters, _tiers, _bands, _spans), 0.0)  # store all iters because spillovers need last
    constrained_up_spillovers = np.full((_tiers, _bands, _spans), 0.0)
    constrained_generic_spillovers = np.full((_spans), 0.0)  # squashed across tiers and bands
    potential_flows = np.full((_tiers, _bands, _spans), 0.0)
    total_flows = np.full((_tiers, _bands, _spans), 0.0)

    code_base = 1
    for i in range(_n_iters):

        checks.progress_bar(i, _n_iters, 100)

        # reset spillovers
        constrained_up_spillovers.fill(0)
        constrained_generic_spillovers.fill(0)
        # apportion flow - per tier and per base code
        for tier_idx in range(_tiers):
            # spillovers and flows are computed separately for each base code
            # iterate the base codes, e.g. 1, 2, 3
            for band_idx in range(_bands):
                # UP SPILLOVERS
                # apportion previous iteration's spillovers
                # up spills are applied from the previous iteration to the current iteration's next higher tier
                # therefore, can't apply on the first iter or on the uppermost tier
                if i != 0 and tier_idx != 0:
                    j_flows = data.singly_constrained(_node_map,
                                                      _edge_map,
                                                      _landuse_data_map,
                                                      _distances,
                                                      _betas,
                                                      i_weights=constrained_flows[i - 1][tier_idx - 1][band_idx] * (
                                                              1 - _spill_rate),
                                                      j_weights=landuse_activation[tier_idx][band_idx] *
                                                                landuse_capacitance[tier_idx],
                                                      suppress_progress=True)
                    constrained_up_spillovers[tier_idx][band_idx] = j_flows[_dist_idx]
                # GENERIC SPILLOVERS
                # generic spillovers are gathered from all tiers and dumped to a common channel
                # harvested from previous iteration, so can't apply to first iter
                # TODO: spillover should be added separately each iter, i.e. not compounded every iter
                # this should prevent explosion - i.e. each iter's contribution should decay from iter to iter
                #
                '''
                if i != 0:
                    j_flows = data.singly_constrained(_node_map,
                                                      _edge_map,
                                                      _landuse_data_map,
                                                      _distances,
                                                      _betas,
                                                      i_weights=constrained_flows[i - 1][tier_idx][band_idx] * _spill_rate,
                                                      j_weights=landuse_activation[tier_idx][band_idx] * landuse_capacitance[tier_idx],
                                                      suppress_progress=True)
                    # NB - no tier or band - so remember += instead of =
                    constrained_generic_spillovers += j_flows[_dist_idx]
                '''
        # BASE FLOWS
        # flows can now be computed - these are constrained - i.e. competition is taken into account
        for tier_idx in range(_tiers):
            for band_idx in range(_bands):
                if tier_idx == 0:
                    source_flows = _pop_intensity + constrained_generic_spillovers
                else:
                    source_flows = constrained_up_spillovers[tier_idx][band_idx]
                j_flows = data.singly_constrained(_node_map,
                                                  _edge_map,
                                                  _landuse_data_map,
                                                  _distances,
                                                  _betas,
                                                  i_weights=source_flows,
                                                  j_weights=landuse_activation[tier_idx][band_idx] *
                                                            landuse_capacitance[tier_idx],
                                                  suppress_progress=True)
                constrained_flows[i][tier_idx][band_idx] = j_flows[_dist_idx]

        # DISTRIBUTED FLOWS
        # simply uses mean_weighted which achieves the same thing as even distribution -> singly constrained...
        for tier_idx in range(_tiers):
            # compute potential flows (for exploration)
            # pre-create array for numba
            # don't use same variable name to above, avoids numba typing conflicts!!!
            dist_flows = np.full((_bands, _spans), np.nan)
            for band_idx in range(_bands):
                if tier_idx == 0:
                    dist_flows[band_idx] = _pop_intensity
                else:
                    dist_flows[band_idx] = _pop_intensity * (1 - _spill_rate) ** tier_idx
            # use distinct temporary names to avoid numba typing issues
            a, b, c, d, stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, \
            stats_max, stats_min = data.local_aggregator(_node_map,
                                                         _edge_map,
                                                         _landuse_data_map,
                                                         _distances,
                                                         _betas,
                                                         numerical_arrays=dist_flows,
                                                         suppress_progress=True)
            for band_idx in range(_bands):
                potential_flows[tier_idx][band_idx] = stats_mean_wt[band_idx][_dist_idx]
            # distribute flows & spillovers - does not strictly take competition into account... but easier than furness process
            # TODO - spatial version: assymetrical betweenness or some or other gravity wouldn't take k competition into account...?
            # use distinct temporary names to avoid numba typing issues
            a, b, c, d, stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, \
            stats_max, stats_min = data.local_aggregator(_node_map,
                                                         _edge_map,
                                                         _landuse_data_map,
                                                         _distances,
                                                         _betas,
                                                         numerical_arrays=constrained_flows[i][tier_idx],
                                                         suppress_progress=True)
            for band_idx in range(_bands):
                total_flows[tier_idx][band_idx] = stats_mean_wt[band_idx][_dist_idx]

        # ASSIGN CELLS
        for tier_idx in range(_tiers):
            # review each landuse location - step up if overflowing else down if underflowing
            # numba doesn't yet support summing by axis, though might be added soon per #1269
            squashed_flows = dim_0_squasher(constrained_flows[i][tier_idx])

            # adjust capacitances
            squashed_flows -= _pop_threshold
            squashed_flows /= _pop_threshold
            squashed_flows *= _cap_step
            # numpy clip not supported by numba
            squashed_flows[squashed_flows < -_cap_step] = -_cap_step
            squashed_flows[squashed_flows > _cap_step] = _cap_step
            landuse_capacitance[tier_idx] += squashed_flows
            # constrain capacitance
            landuse_capacitance[tier_idx][landuse_capacitance[tier_idx] < 0] = 0.0
            landuse_capacitance[tier_idx][landuse_capacitance[tier_idx] > 1] = 1.0

            # seed new locations
            # maximum potential flow - normalise by bands to maintain consistency across different band sizes
            # higher tiers have a lower capacity based on the spill rate cascading upwards
            # however, this is already effectively taken into account when calculating potential flows
            potential = np.sum(potential_flows[tier_idx]) / _bands / _pop_threshold
            existing = np.sum(landuse_state[i][tier_idx] != 0)  # count all existing landuses
            new = int(potential - existing)
            # blend based on exploration rate
            totals = dim_0_squasher(total_flows[tier_idx])
            potentials = dim_0_squasher(potential_flows[tier_idx])
            blended_flows = totals * (1 - _explore_rate) + potentials * _explore_rate
            # when starting, flows may be the same everywhere if no current activations
            # jitter (instead of rounding errors)
            if np.round(np.max(blended_flows) - np.min(blended_flows)) == 0:
                jitter = np.random.randn(_spans)
                blended_flows += jitter
            # sort by highest flow
            seed_idx = np.argsort(blended_flows, kind='mergesort')[::-1]
            # TODO: numpy in1d not supported by numba
            seed_idx_new = seed_idx[np.in1d(seed_idx, np.where(landuse_state[i][tier_idx] == 0))]
            if new > 0:
                seed_idx_new = seed_idx_new[:new]
                # normalise seeds and then add to capacitances
                seed_vals = blended_flows[seed_idx_new]
                seed_vals -= np.min(seed_vals)
                seed_vals /= np.max(seed_vals)
                landuse_capacitance[tier_idx][seed_idx_new] += seed_vals
            # get successful landuses
            seed_idx_exist = seed_idx[np.in1d(seed_idx, np.where(landuse_state[i][tier_idx] != 0))]
            # squashed flows are based on constrained flows which are already centred to population threshold
            successful_idx = seed_idx_exist[squashed_flows[seed_idx_exist] > 0]
            # leaving this as non unique to reflect preponderances
            successful_landuses = landuse_state[i][tier_idx][successful_idx]

            # deactivate dead landuses
            off_idx = np.intersect1d(np.where(landuse_capacitance[tier_idx] <= 0),
                                     np.where(landuse_state[i][tier_idx] == 1))
            landuse_state[i][tier_idx][off_idx] = 0
            # activate new landuses
            on_idx = np.intersect1d(np.where(landuse_capacitance[tier_idx] >= 1),
                                    np.where(landuse_state[i][tier_idx] == 0))
            for on_i in on_idx:
                # seed - check for innovation trigger
                coin_toss = np.random.rand()
                # if within innovation rate - then randomly combine two landuses
                if coin_toss <= _innovation_rate:
                    state = landuse_state[i][tier_idx]
                    candidates = np.unique(state[state != 0])
                    # if not enough candidates, just seed new
                    if len(candidates) < 2:
                        landuse_state[i][tier_idx][on_i] = np.random.randint(code_base, _bands + code_base)
                    # otherwise, pick two randomly
                    else:
                        while True:
                            idx_a, idx_b = np.random.randint(0, len(candidates), 2)
                            # don't add two of the same
                            if idx_a == idx_b:
                                continue
                            landuse_state[i][tier_idx][on_i] = candidates[idx_a] + candidates[idx_b]
                            # print('tier', tier_idx, 'new code', landuse_state[i][tier_idx][idx])
                            break
                # if no existing code, then pick randomly from bands
                elif len(successful_landuses) == 0 or coin_toss < 0.5:
                    landuse_state[i][tier_idx][on_i] = np.random.randint(code_base, _bands + code_base)
                # otherwise, randomly assign either from bands or from existing
                else:
                    random_new_idx = np.random.randint(0, len(successful_landuses))
                    landuse_state[i][tier_idx][on_i] = successful_landuses[random_new_idx]
                # update the associated activations
                for band_idx, band_factor in enumerate(range(code_base, _bands + code_base)):
                    # check for factors
                    if landuse_state[i][tier_idx][on_i] % band_factor == 0:
                        landuse_activation[tier_idx][band_idx][on_i] = 1

    return landuse_state


def mmm_layercake_diversity(_graph,
                            _layer_spec,
                            _n_iters=200,
                            _spans=200,
                            _bands=4,
                            _tiers=2,
                            random_seed=0):
    np.random.seed(random_seed)

    for k in ['cap_step', 'dist_threshold', 'pop_threshold', 'spill_rate', 'explore_rate']:
        if k not in _layer_spec:
            raise AttributeError(f'Missing key {k}')

    # generate the backbone Network Layer
    # include 1 (local only) for directly counting items assigned to current node
    # also include np.inf so that all node combinations are considered in spite of wrapping around back
    distances = list({1, np.inf, _layer_spec['dist_threshold']})
    Netw_Layer = networks.Network_Layer_From_nX(_graph, distances=distances)

    # generate population data layer
    Pop_Layer = generate_data_layer(_spans, 20, Netw_Layer, _randomised=False)
    # population state and map
    # for this experiment assignments are not changing
    pop_state = np.full(len(Pop_Layer.uids), 1.0)  # use floats!

    # generate the landuse substrate
    # keep to 1 location per node for visualisation sake
    # randomisation is immaterial because all will be instanced as 0.0
    Landuse_Layer = generate_data_layer(_spans, 1, Netw_Layer, _randomised=False)

    # calculate neighbourhood density
    # move into loop if using dynamic populations
    Pop_Layer.compute_stats_single('pop_intensity', pop_state)
    pop_intensity = np.copy(Netw_Layer.metrics['stats']['pop_intensity']['sum'][1])

    # iterate
    cap_step = _layer_spec['cap_step']
    dist_threshold = _layer_spec['dist_threshold']
    pop_threshold = _layer_spec['pop_threshold']
    spill_rate = _layer_spec['spill_rate']
    explore_rate = _layer_spec['explore_rate']
    innovation_rate = _layer_spec['innovation_rate']
    landuse_state = compute_iter(_n_iters,
                                 Netw_Layer._nodes,
                                 Netw_Layer._edges,
                                 Landuse_Layer._data,
                                 np.array(Netw_Layer.distances),
                                 np.array(Netw_Layer.betas),
                                 _tiers,
                                 _bands,
                                 _spans,
                                 Netw_Layer.distances.index(dist_threshold),
                                 spill_rate,
                                 pop_threshold,
                                 cap_step,
                                 explore_rate,
                                 innovation_rate,
                                 pop_intensity)

    return landuse_state


def mmm_wrap(_ax,
             _graph,
             _layer_spec,
             _n_iters,
             _spans,
             _bands=9,
             _tiers=3,
             _title='',
             _random_seed=0):
    landuse_state = mmm_layercake_diversity(_graph,
                                            _layer_spec,
                                            _n_iters,
                                            _spans,
                                            _bands=_bands,
                                            _tiers=_tiers,
                                            random_seed=_random_seed)

    print('done')

    xs = []
    for n, d in _graph.nodes(data=True):
        xs.append(d['x'])

    plotter(_ax, _n_iters, xs, _res_factor=1,
            _plot_maps=[pop_map, capacitance_maps[0], landuse_maps[0], spillover_maps[0]])

    _ax.set_title(_title + f' sum l.u: {landuse_maps[0].sum()}')
