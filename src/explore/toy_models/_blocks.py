# %%
import os

import networkx as nx
import numpy as np
from numba import njit

os.environ['CITYSEER_QUIET_MODE'] = '1'
from cityseer.util import graphs
from cityseer.metrics import layers

# %%
# TODO: add plots for energy / entropy / diversity?
'''

'''


def generate_graph(_spans=200, _stepped=False):
    # the basic graph
    spine_nx = nx.Graph()

    step_size = 20
    step_sizes = [step_size] * _spans

    prior = None
    agg = 0
    for n, step_size in enumerate(step_sizes):
        spine_nx.add_node(n, x=agg, y=0)
        agg += step_size
        if prior is not None:
            spine_nx.add_edge(prior, n)
        prior = n

    '''
    To have a circular graph:
    - manually connect first to last node
    - override associated edge impedance
    - set the largest distance to infinity or greater than the x, y distance from end to end
    '''

    # manually connect the first to the last
    spine_nx.add_edge(_spans - 1, 0)
    spine_nx = graphs.nX_simple_geoms(spine_nx)

    # override the impedance between the first and last edge
    imp_factor = step_size / step_size * _spans  # e.g. 20 / 4000 = 0.005 impedance factor
    spine_nx[_spans - 1][0]['imp_factor'] = imp_factor

    # if stepped, override the distances for second half
    if _stepped:
        for s, e, d in spine_nx.edges(data=True):
            if s < len(spine_nx) / 2:
                continue
            spine_nx[s][e]['imp_factor'] = 2

    print(nx.info(spine_nx))
    return spine_nx


def generate_graph_lattice(sections=10, sub_segments=4, segment_sizes=20):
    if not sub_segments % 2 == 0:
        raise ValueError('sub segments must be divisible by two')

    g = nx.Graph()

    base_x = sub_segments / 2 * segment_sizes
    base_y = sub_segments / 2 * segment_sizes
    for i in range(sections):
        for j in range(sections):
            step_x = int(base_x + i * sub_segments * segment_sizes)
            step_y = int(base_y + j * sub_segments * segment_sizes)
            for k in range(sub_segments):
                x = int(step_x + k * segment_sizes)
                g.add_node(f'{x}_{step_y}', x=x, y=step_y)
                y = int(step_y + k * segment_sizes)
                g.add_node(f'{step_x}_{y}', x=step_x, y=y)

    return g


'''
primary: '#0091ea',
accent: '#64c1ff',
info: '#0064b7',
test: '#ff6659',
secondary: '#d32f2f',
warning: '#9a0007',
error: '#ffffff',
success: '#2e7d32'
'''


# process
def plotter(_ax,
            _iters,
            _xs,
            _res_factor=1,
            _plot_maps=(),
            _plot_colours=('#555555', '#d32f2f', '#0091ea', '#2e7d32', '#64c1ff', '#9a0007'),
            _plot_scales=(1, 1, 1, 1, 1, 1)):
    if not _plot_maps:
        raise AttributeError('No plot maps provided')

    if len(_plot_maps) > 6:
        raise AttributeError('Too many plot maps provided. Max=6')

    n_maps = len(_plot_maps) + 1
    vh = int(_iters * n_maps / _res_factor)

    # opacities = [0.4 if c == '#2e2e2e' else 0.8 for c in colours]
    for n, arr in enumerate(_plot_maps):
        _arr = np.copy(arr)
        global_counter = n
        theme_counter = 0
        s_max = np.nanmax(_arr)  # avoid issues with array changing in place
        c = _plot_colours[n]
        size = _plot_scales[n]
        # a = opacities[n]
        while theme_counter < _iters:
            if _arr is not None:
                s = _arr[theme_counter]
                s /= s_max
                s *= 5 * size
                # plot from top to bottom
                ys = np.full(len(_xs), vh - global_counter)
                _ax.scatter(_xs, ys, s=s, c=c, edgecolors='none', alpha=1)

            theme_counter += _res_factor
            global_counter += n_maps

    # set extents
    _ax.set_ylim(bottom=0, top=vh)
    _ax.set_xlim(left=_xs[0], right=_xs[-1])


@njit
def pressure_reallocation(_pressure_grad, _node_map, _data_map, _iters, _max,
                          _filter_arr=None, _filter_key=None, _random_prob=0.5):
    randomised = False
    if np.random.randint(0, 10) < _random_prob * 10:
        randomised = True

    counter = 0
    exit_counter = 0
    while counter < _iters:
        exit_counter += 1
        if exit_counter > 100:
            break
        # randomly pick two units from the data layer
        l1 = np.random.randint(0, _data_map.shape[0])
        l2 = np.random.randint(0, _data_map.shape[0])
        # get the currently assigned network nodes
        l1_assigned = int(_data_map[l1][2])
        l2_assigned = int(_data_map[l2][2])
        # if randomised, move regardless
        # check which of the current pressure gradient points (on network) is lower or higher
        # reassign accordingly
        if randomised or _pressure_grad[l1_assigned] < _pressure_grad[l2_assigned]:
            # check filter, if provided
            if _filter_arr is not None:
                # can only relocate where filter shows current landuse exists
                if _filter_arr[l1] != _filter_key:
                    continue
            # check that the max has not been exceeded
            num_assigned = np.nansum(_data_map[:, 2] == l2_assigned)
            if num_assigned == _max:
                continue
            # otherwise, go ahead and reassign from l1 to l2
            _data_map[l1][:2] = _node_map[l2_assigned][:2]
            _data_map[l1][2:] = [l2_assigned, np.nan]
        elif _pressure_grad[l2_assigned] < _pressure_grad[l1_assigned]:
            if _filter_arr is not None:
                # can only relocate where filter shows current landuse exists
                if _filter_arr[l2] != _filter_key:
                    continue
            # check that the max has not been exceeded
            num_assigned = np.nansum(_data_map[:, 2] == l1_assigned)
            if num_assigned == _max:
                continue
            # otherwise, go ahead and reassign from l1 to l2
            _data_map[l2][:2] = _node_map[l1_assigned][:2]
            _data_map[l2][2:] = [l1_assigned, np.nan]
        counter += 1
    return _data_map


@njit
def set_current_num(_data_map, _node_map):
    agg = np.full(_node_map.shape[0], 0.0)
    node_assignment = _data_map[:, 2]
    for node_idx in node_assignment:
        agg[int(node_idx)] += 1.0
    return agg / np.nanmax(agg)


def generate_data_layer(_spans, _intensity, _Network_Layer, _randomised=True):
    '''
    Generates a data layer randomly assigned to nodes in the Network Layer
    Number of data points is determined by _spans * _intensity
    Randomised sets whether assignments are random or sequential
    Useful for creating a data layer based on the number and location of nodes (for sims)
    '''
    # generate the population data
    data_arr = np.full((int(_spans * _intensity), 4), 0.0)
    if _randomised:
        # randomly assign each population unit to a node
        for data_idx in range(data_arr.shape[0]):
            node_idx = np.random.randint(0, len(_Network_Layer.uids))
            # first two indices are x, y - copy from Network Layer
            data_arr[data_idx][:2] = _Network_Layer._node_data[node_idx][:2]
            # third and fourth are nearest and next nearest
            data_arr[data_idx][2:] = [node_idx, np.nan]
    else:
        data_idx = 0
        for node_idx in range(_spans):
            for n in range(_intensity):
                # first two indices are x, y - copy from Network Layer
                data_arr[data_idx][:2] = _Network_Layer._node_data[node_idx][:2]
                # third and fourth are nearest and next nearest
                data_arr[data_idx][2:] = [node_idx, np.nan]
                data_idx += 1
    # generate the population layer
    # no need to assign to network as this has already been done
    D_Layer = layers.Data_Layer(tuple(range(data_arr.shape[0])), data_arr)
    # set network manually instead of calling assign_to_network()
    D_Layer._Network = _Network_Layer
    #
    return D_Layer
