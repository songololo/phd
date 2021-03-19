# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import phd_util

os.environ['CITYSEER_QUIET_MODE'] = '1'
from cityseer.util import mock
from cityseer.metrics import networks, layers

from src.explore.MMM._blocks import generate_graph, generate_data_layer, set_current_num, plotter, pressure_reallocation

# %%
'''
1 - density agglomeration

A) randomly instanced population
B) calculate density (normalise)
C) calculate density weighted centrality
D) randomly select two population units and relocate from lower to higher (enforce min and max constraint)
'''

iters = 400
spans = 200
spine_nx = generate_graph(_spans=spans)
spine_nx_stepped = generate_graph(_spans=spans, _stepped=True)

phd_util.plt_setup()
fig, axes = plt.subplots(1, 2, figsize=(12, 20))

for ax_n, graph in enumerate([spine_nx, spine_nx_stepped]):
    # generate the Network Layer
    Netw_Layer = networks.Network_Layer_From_nX(graph, distances=[400, 800, np.inf])
    # generate population layer
    Pop_Layer = generate_data_layer(_spans=spans, _intensity=20, _Network_Layer=Netw_Layer)
    # population state and map
    pop_state = np.full(len(Pop_Layer.uids), 1.0)  # use floats!
    pop_map = np.full((iters, spans), 0.0)
    # iterate
    for n in tqdm(range(iters)):
        # POPULATION
        # record current state
        pop_map[n] = set_current_num(Pop_Layer._data, Netw_Layer._nodes)
        # calculate the effective density
        # each population point is a single unit
        # the state technically remains the same, it is the x, y and assignments that change!
        Pop_Layer.compute_stats_single('density', pop_state)
        dens = Netw_Layer.metrics['stats']['density']['sum_weighted'][800]
        dens[np.isnan(dens)] = 0
        # set the centrality weights accordingly
        Netw_Layer.weights = dens / np.nanmax(dens)  # normalised
        # calculate the density weighted centrality
        Netw_Layer.gravity_index()
        cent_pop_wt = Netw_Layer.metrics['centrality']['gravity_index'][400]
        # reassign people from low to high pressure
        pressure_reallocation(cent_pop_wt, Netw_Layer._nodes, Pop_Layer._data, _iters=50, _max=40)

    plotter(axes[ax_n], iters, Netw_Layer.x_arr, _res_factor=1, _grey_map=pop_map)

theme = 'density_agglomeration'
fig.suptitle(theme)
plt.savefig(f'./src/explore/H_mmm/exploratory_plots/{theme}.png', dpi=300)
plt.show()

#  %%
'''
2 - density vis-a-vis mixed-uses

Scenarios: sequential, random, random with topo stretch
1) People pursue mixed uses
2) Centrality is aggregate weighted by people, and destination weighted by respective R, G, B flows
3) Each R, G, B entity relocates from lower to higher flow environment.
4) Constraints introduced by max cap per location

Note that there is now a ratio between rate of population change and rate of mixed-uses change

'''

iters = 400
spans = 200
spine_nx = generate_graph(_spans=spans)
spine_nx_stepped = generate_graph(_spans=spans, _stepped=True)

phd_util.plt_setup()
fig, axes = plt.subplots(1, 3, figsize=(18, 20))

for ax_n, (graph, randomised) in enumerate(zip([spine_nx, spine_nx, spine_nx_stepped],
                                               [False, True, True])):
    # generate the Network Layer
    Netw_Layer = networks.Network_Layer_From_nX(graph, distances=[400, 800, np.inf])
    # generate population layer
    Pop_Layer = generate_data_layer(spans, 20, Netw_Layer, _randomised=randomised)
    # population state and map - note that the state doesn't change (changes occur via assignments)
    pop_state = np.full(len(Pop_Layer.uids), 1.0)  # use floats!
    pop_map = np.full((iters, spans), 0.0)
    # generate the landuse layer
    Landuse_Layer = generate_data_layer(spans, 1, Netw_Layer, _randomised=randomised)
    lu_flow_a = np.full((iters, spans), 0.0)
    lu_flow_b = np.full((iters, spans), 0.0)
    lu_flow_c = np.full((iters, spans), 0.0)
    # get the landuse encodings - note that the labels don't change (changes occur via assignments)
    landuse_labels = mock.mock_categorical_data(length=len(Landuse_Layer.uids), num_classes=3)
    if not randomised:
        l = len(Netw_Layer.uids)
        l_1 = int(l / 3)
        l_2 = l_1 * 2
        for d_idx, assigned_idx in enumerate(Landuse_Layer._data[:, 2]):
            if assigned_idx < l_1:
                landuse_labels[d_idx] = 'a'
            elif assigned_idx < l_2:
                landuse_labels[d_idx] = 'b'
            else:
                landuse_labels[d_idx] = 'c'
    landuse_classes, landuse_encodings = layers.encode_categorical(landuse_labels)

    # iterate
    for n in tqdm(range(iters)):
        # POPULATION
        # record current assignment state
        pop_map[n] = set_current_num(Pop_Layer._data, Netw_Layer._nodes)
        # calculate the effective density
        # each population point is a single unit
        # the state technically remains the same, it is the x, y and assignments that change!
        Pop_Layer.compute_stats_single('density', pop_state)
        dens = Netw_Layer.metrics['stats']['density']['sum_weighted'][800]
        dens[np.isnan(dens)] = 0
        # set the centrality weights accordingly
        Netw_Layer.weights = dens
        # calculate the density weighted centrality
        Netw_Layer.gravity_index()
        cent_pop_wt = Netw_Layer.metrics['centrality']['gravity_index'][800]
        # reset weights
        Netw_Layer.weights = np.full(len(Netw_Layer.uids), 1)

        # MIXED USES
        # as above, state effectively remains the same but the x, y and assignments change over time
        Landuse_Layer.compute_aggregated(
            landuse_labels=landuse_labels,
            mixed_use_keys=['hill_branch_wt'],
            qs=[0],
            accessibility_keys=['a', 'b', 'c'])

        # gather metrics
        mu = Netw_Layer.metrics['mixed_uses']['hill_branch_wt'][0][400]

        access_a = Netw_Layer.metrics['accessibility']['weighted']['a'][400]
        lu_flow_a[n] = cent_pop_wt * access_a

        access_b = Netw_Layer.metrics['accessibility']['weighted']['b'][400]
        lu_flow_b[n] = cent_pop_wt * access_b

        access_c = Netw_Layer.metrics['accessibility']['weighted']['c'][400]
        lu_flow_c[n] = cent_pop_wt * access_c

        # reassign from low to high pressure
        # people move to mixed-uses
        pressure_reallocation(mu, Netw_Layer._nodes, Pop_Layer._data, _iters=20, _max=40)
        # landuses move to respective flows - repeat for each flow map
        pressure_reallocation(lu_flow_a[n], Netw_Layer._nodes, Landuse_Layer._data,
                              _iters=10, _max=5, _filter_arr=landuse_encodings, _filter_key=0)
        pressure_reallocation(lu_flow_b[n], Netw_Layer._nodes, Landuse_Layer._data,
                              _iters=10, _max=5, _filter_arr=landuse_encodings, _filter_key=1)
        pressure_reallocation(lu_flow_c[n], Netw_Layer._nodes, Landuse_Layer._data,
                              _iters=10, _max=5, _filter_arr=landuse_encodings, _filter_key=2)

    plotter(axes[ax_n],
            iters,
            Netw_Layer.x_arr,
            _res_factor=1,
            _grey_map=pop_map,
            _red_map=lu_flow_a,
            _green_map=lu_flow_b,
            _blue_map=lu_flow_c)

theme = 'density_vs_mixed_uses'
fig.suptitle(theme)
plt.savefig(f'./src/explore/H_mmm/exploratory_plots/{theme}.png', dpi=300)
plt.show()
