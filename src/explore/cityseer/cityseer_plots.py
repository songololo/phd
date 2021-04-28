#%% copied from cityseer-api

import utm
from matplotlib import colors

from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, plot

from src import util_funcs

###
util_funcs.plt_setup()
# INTRO PLOT
G = mock.mock_graph()
plot.plot_nX(G,
             labels=True,
             node_size=80,
             path='../phd-admin/PhD/part_1/images/cityseer/graph.pdf')


# INTRO EXAMPLE PLOTS
G = graphs.nX_simple_geoms(G)
G = graphs.nX_decompose(G, 20)

N = networks.NetworkLayerFromNX(G, distances=[400, 800])
N.segment_centrality(measures=['segment_harmonic'])

data_dict = mock.mock_data_dict(G, random_seed=25)
D = layers.DataLayerFromDict(data_dict)
D.assign_to_network(N, max_dist=400)
landuse_labels = mock.mock_categorical_data(len(data_dict), random_seed=25)
D.hill_branch_wt_diversity(landuse_labels, qs=[0])
G_metrics = N.to_networkX()

segment_harmonic_vals = []
mixed_uses_vals = []
for node, data in G_metrics.nodes(data=True):
    segment_harmonic_vals.append(data['metrics']['centrality']['segment_harmonic'][800])
    mixed_uses_vals.append(data['metrics']['mixed_uses']['hill_branch_wt'][0][400])

# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])
segment_harmonic_vals = colors.Normalize()(segment_harmonic_vals)
segment_harmonic_cols = cmap(segment_harmonic_vals)
plot.plot_nX(G_metrics,
             plot_geoms=True,
             node_colour=segment_harmonic_cols,
             path='../phd-admin/PhD/part_1/images/cityseer/intro_segment_harmonic.pdf')

# plot hill mixed uses
mixed_uses_vals = colors.Normalize()(mixed_uses_vals)
mixed_uses_cols = cmap(mixed_uses_vals)

plot.plot_assignment(N,
                     D,
                     node_colour=mixed_uses_cols,
                     data_labels=landuse_labels,
                     path='../phd-admin/PhD/part_1/images/cityseer/intro_mixed_uses.pdf')

#
#
# MOCK MODULE
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
plot.plot_nX(G,
             plot_geoms=True,
             labels=True,
             node_size=80,
             path='../phd-admin/PhD/part_1/images/cityseer/graph_example.pdf')  # WITH LABELS

G_simple = graphs.nX_simple_geoms(G)
G_decomposed = graphs.nX_decompose(G_simple, 100)
plot.plot_nX(G_decomposed,
             plot_geoms=True,
             path='../phd-admin/PhD/part_1/images/cityseer/graph_decomposed.pdf')

G_dual = graphs.nX_to_dual(G_simple)
plot.plot_nX_primal_or_dual(G_simple,
                            G_dual,
                            plot_geoms=True,
                            path='../phd-admin/PhD/part_1/images/cityseer/graph_dual.pdf')

# graph cleanup examples
lng, lat = -0.13396079424572427, 51.51371088849723
G_utm = mock.make_buffered_osm_graph(lng, lat, 1250)
easting, northing, _zone, _letter = utm.from_latlon(lat, lng)
buffer = 750
min_x = easting - buffer
max_x = easting + buffer
min_y = northing - buffer
max_y = northing + buffer


# reusable plot function
def simple_plot(_G, _path, plot_geoms=True):
    # plot using the selected extents
    plot.plot_nX(_G,
                 labels=False,
                 plot_geoms=plot_geoms,
                 node_size=10,
                 edge_width=1,
                 x_lim=(min_x, max_x),
                 y_lim=(min_y, max_y),
                 path=_path)


G = graphs.nX_simple_geoms(G_utm)
simple_plot(G, '../phd-admin/PhD/part_1/images/cityseer/graph_cleaning_1.pdf', plot_geoms=False)

G = graphs.nX_remove_filler_nodes(G)
G = graphs.nX_remove_dangling_nodes(G, despine=20, remove_disconnected=True)
G = graphs.nX_remove_filler_nodes(G)
simple_plot(G, '../phd-admin/PhD/part_1/images/cityseer/graph_cleaning_2.pdf')

# first pass of consolidation
G1 = graphs.nX_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
simple_plot(G1, '../phd-admin/PhD/part_1/images/cityseer/graph_cleaning_3.pdf')

# split opposing line geoms to facilitate parallel merging
G2 = graphs.nX_split_opposing_geoms(G1, buffer_dist=15)
simple_plot(G2, '../phd-admin/PhD/part_1/images/cityseer/graph_cleaning_4.pdf')

# second pass of consolidation
G3 = graphs.nX_consolidate_nodes(G2,
                                 buffer_dist=15,
                                 crawl=False,
                                 min_node_degree=2,
                                 cent_min_degree=4)
simple_plot(G3, '../phd-admin/PhD/part_1/images/cityseer/graph_cleaning_5.pdf')

#
#
# LAYERS MODULE
# show assignment to network
# random seed 25
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])

data_dict = mock.mock_data_dict(G, random_seed=25)
L = layers.DataLayerFromDict(data_dict)
L.assign_to_network(N, max_dist=500)
plot.plot_assignment(N,
                     L,
                     path='../phd-admin/PhD/part_1/images/cityseer/assignment.pdf')

G_decomposed = graphs.nX_decompose(G, 50)
N_decomposed = networks.NetworkLayerFromNX(G_decomposed, distances=[200, 400, 800, 1600])

L = layers.DataLayerFromDict(data_dict)
L.assign_to_network(N_decomposed, max_dist=500)
plot.plot_assignment(N_decomposed,
                     L,
                     path='../phd-admin/PhD/part_1/images/cityseer/assignment_decomposed.pdf')

#
#
# PLOT MODULE
from cityseer.tools import mock, graphs, plot

G = mock.mock_graph()
G_simple = graphs.nX_simple_geoms(G)
G_dual = graphs.nX_to_dual(G_simple)
plot.plot_nX_primal_or_dual(G_simple,
                            G_dual,
                            plot_geoms=False,
                            path='../phd-admin/PhD/part_1/images/cityseer/graph_dual.pdf')

# INTRO EXAMPLE PLOTS
G = graphs.nX_simple_geoms(G)
G = graphs.nX_decompose(G, 20)

N = networks.NetworkLayerFromNX(G, distances=[400, 800])
N.segment_centrality(measures=['segment_harmonic'])

data_dict = mock.mock_data_dict(G, random_seed=25)
D = layers.DataLayerFromDict(data_dict)
D.assign_to_network(N, max_dist=400)
landuse_labels = mock.mock_categorical_data(len(data_dict), random_seed=25)
D.hill_branch_wt_diversity(landuse_labels, qs=[0])
G_metrics = N.to_networkX()

segment_harmonic_vals = []
mixed_uses_vals = []
for node, data in G_metrics.nodes(data=True):
    segment_harmonic_vals.append(data['metrics']['centrality']['segment_harmonic'][800])
    mixed_uses_vals.append(data['metrics']['mixed_uses']['hill_branch_wt'][0][400])

# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])
segment_harmonic_vals = colors.Normalize()(segment_harmonic_vals)
segment_harmonic_cols = cmap(segment_harmonic_vals)
plot.plot_nX(G_metrics,
             plot_geoms=True,
             node_colour=segment_harmonic_cols,
             path='../phd-admin/PhD/part_1/images/cityseer/intro_segment_harmonic.pdf')

# plot hill mixed uses
mixed_uses_vals = colors.Normalize()(mixed_uses_vals)
mixed_uses_cols = cmap(mixed_uses_vals)

plot.plot_assignment(N,
                     D,
                     node_colour=mixed_uses_cols,
                     data_labels=landuse_labels,
                     path='../phd-admin/PhD/part_1/images/cityseer/intro_mixed_uses.pdf')
