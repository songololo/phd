# coding=utf-8
__author__ = 'Gareth Simons'
"""
NB - Written while at Woods Bagot - Superspace. Give credit accordingly.

Loads a network table layer from PostGIS and performs a series of network analysis measures.
Writes the results to a new table in the database with the provided postfix.

Only runs on ubuntu, thus requires docker if using on Windows.

Several versions were attempted (see experimental versions directory).
This version uses graph-tool (which wraps the C Boost Graph alogs) and numba (for fast loop iteration).
"""
import os, sys, inspect
dir_current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_parent = os.path.dirname(dir_current)
sys.path.insert(0, dir_parent)

import logging
import math
import time
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import asyncpg
import numpy as np
from graph_tool.all import *
import multiprocessing
from shapely import geometry, wkb, speedups
if speedups.available:
    speedups.enable()
from numba import jit

from utilities.toolbox import make_dsn_string, \
    find_geom_column, \
    wgs_bounds_from_postGIS, \
    wgs_bounds_from_range, \
    convert_wgs_to_utm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

graph_tool.openmp_enabled()
graph_tool.openmp_set_num_threads(multiprocessing.cpu_count())
logger.info('...set openmp to {0} threads'.format(graph_tool.openmp_get_num_threads()))


# used for processing the result matrices and assigning values to closeness and betweenness matrices
@jit(nopython=True, nogil=True, cache=True)
def process_paths(
        i,
        j_verts,
        distance_map,  # distance_map length = masked graph
        max_distance,
        pred_map,  # pred_map length = full graph (same with property maps)
        gd_vert_betweenness_int,
        gd_vert_betw_weighted_float,
        gd_vert_primary_distance_float,
        gd_vert_closeness_count_float,
        gd_vert_closeness_sum_float,
        gd_vert_closeness_sum_weighted_float,
        angular=False):

    # it is necessary to track j_count separately from j because the distance_map tracks the masked graph,
    # whereas the predecessor map and property maps track the full graph...
    for j_count, j in enumerate(j_verts):

        if i != j:

            network_distance = distance_map[j_count]

            # don't process disconnected verts
            if network_distance == np.inf:
                continue

            # don't process verts farther away than network cutoff, this only works for metric
            # not a huge issue for angular because crow flies cutoff still applies
            if not angular:
                if network_distance > max_distance:
                    continue

            # increment closeness count
            gd_vert_closeness_count_float[i] += 1

            # increment regular closeness
            gd_vert_closeness_sum_float[i] += network_distance

            # increment weighted closeness
            gd_vert_closeness_sum_weighted_float[i] += (network_distance * gd_vert_primary_distance_float[j])

            # calculate path for betweenness
            intermediary = j

            # find the reverse path from j to i
            while True:

                # this is at the top of loop to effectively move on from j immediately (so as not to count j)
                intermediary = pred_map[intermediary]

                # if not yet arrived at i, add the betweenness to the intermediary edge
                if not intermediary == i:

                    # increment betweeness
                    gd_vert_betweenness_int[intermediary] += 1

                    # increment weighted betweenness (weighted by source and target weights)
                    gd_vert_betw_weighted_float[intermediary] += \
                        (1 * gd_vert_primary_distance_float[i] * gd_vert_primary_distance_float[j])

                    continue

                break

# used for filtering out vertices based on max distance
@jit(nopython=True, nogil=True, cache=True)
def set_dg_mask(i, gd_vert_idx, gd_vert_mask_bool, x_arr, y_arr, max_distance):
    # reset the mask where 1 means not masked
    gd_vert_mask_bool[...] = 1
    # mask verts farther than max_distance from i
    i_x = x_arr[i]
    i_y = y_arr[i]
    # mask items farther away than max distance
    gd_vert_mask_bool[np.sqrt(np.power(x_arr - i_x, 2) + np.power(y_arr - i_y, 2)) > max_distance] = 0
    return gd_vert_mask_bool

class Graph_Analyse:

    def __init__(self):
        '''
        Initiates the graph analyse class.
        Establishes the variables for primary graph, dual graph, and associated property maps required for analysis.
        :return:
        '''
        logger.info('...instancing graph object')
        self.graph_primary = Graph(directed=False)  # the graph
        self.g_vert_xy_vect_float = self.graph_primary.new_vertex_property('vector<float>')
        self.g_edge_uid_str = self.graph_primary.new_edge_property('string')
        self.g_edge_distance_float = self.graph_primary.new_edge_property('float')

        self.graph_dual = Graph(directed=False)  # the dual graph for calculating edge centrality
        self.gd_vert_uid_str = self.graph_dual.new_vertex_property('string')
        self.gd_vert_mask_bool = self.graph_dual.new_vertex_property('bool')
        self.gd_vert_idx_int = self.graph_dual.new_vertex_property('int')
        self.gd_vert_x_float = self.graph_dual.new_vertex_property('float')  # don't use vector re numba function
        self.gd_vert_y_float = self.graph_dual.new_vertex_property('float')  # don't use vector re numba function
        self.gd_vert_xy_vect_float = self.graph_dual.new_vertex_property('vector<float>')  # for plotting maps
        self.gd_vert_primary_distance_float = self.graph_dual.new_vertex_property('float')
        # property maps for storing the measures
        self.gd_vert_betweenness_int = self.graph_dual.new_vertex_property('int')
        self.gd_vert_betw_weighted_float = self.graph_dual.new_vertex_property('float')

        self.gd_vert_closeness_count_float = self.graph_dual.new_vertex_property('float')
        self.gd_vert_closeness_sum_float = self.graph_dual.new_vertex_property('float')
        self.gd_vert_closeness_float = self.graph_dual.new_vertex_property('float')
        self.gd_vert_closeness_sum_weighted_float = self.graph_dual.new_vertex_property('float')
        self.gd_vert_closeness_weighted_float = self.graph_dual.new_vertex_property('float')

        # edge maps for distance and float weights
        self.gd_edge_distance_float = self.graph_dual.new_edge_property('float')
        self.gd_edge_angle_float = self.graph_dual.new_edge_property('float')

        self.max_distance = None
        self.selected_measure_str = None

    def build_graph(self, *args, **kwargs):
        '''
        Override build_graph from child classes.
        This method must provide a graph suitable for use by analyse_graph().
        e.g. Graph_Analyse_PostGIS will read information from a PostGIS database and creates a graph for analysis.
        :return:
        '''
        logger.info('...building graph.')

    def return_graph(self, *args, **kwargs):
        '''
        Override set_graph from child classes.
        This method is for extracting information from the analysed graph in a manner compatible with the end-use.
        e.g. GraphAnalyse_PostGIS will write the information back to a PostGIS database.
        :return:
        '''
        logger.info('...exporting graph information.')

    def analyse_graph(self, max_distance:int, angular=False):
        '''
        This method executes the actual graph analysis using the graph prepared by build_graph.
        :param distance_thresholds: A tuple of distance thresholds at which to run the measures.
        :param angular: Whether to perform an angular analysis. If set to false, a metric and a compound (metric * angular)
        analysis will be performed. Note that the compound method doesn't create a full split dual graph.
        :param length_weighted: Whether to weight the output using the length of the segments.
        :return:
        '''

        logger.info('...commencing graph analysis for distance: {0}'.format(max_distance))

        logger.info('...building the dual graph')
        self.build_dual_graph()
        logger.info('...dual graph: {0}'.format(self.graph_dual))
        #graph_draw(self.graph_dual, pos=self.gd_vert_xy_vect_float)

        # store parameters in class -> used for setting outputs, e.g. to PostGIS database
        self.max_distance = max_distance
        max_distance_shortest_path = max_distance
        weight = self.gd_edge_distance_float
        self.selected_measure_str = 'metric'
        # by default, parameters are set to metric
        # override if using angular
        if angular:
            weight = self.gd_edge_angle_float
            self.selected_measure_str = 'angular'
            max_distance_shortest_path = None

        time_start = time.time()
        time_now = time_start

        update_interval = 1000
        count = 0

        # set up vertex index property map
        for v in self.graph_dual.vertices():
            self.gd_vert_idx_int[v] = int(v)

        logger.info('...starting graph analysis')
        for i_v in self.graph_dual.vertices():

            count += 1
            if count % update_interval == 0 and count != 0:
                time_elapsed = round(time.time() - time_now, 2)
                time_now = time.time()
                # not using the average because time decreases as the iteration progresses due to j>i filter
                time_remaining = round((time_elapsed * ((self.graph_dual.num_vertices() - count) / update_interval) / 60), 2)
                logger.info('...presently at index {0}, last batch: {1}s, ~ time remaining: {2}m'.format(count, time_elapsed, time_remaining))

            # set mask
            # NB -> use actual max distance, not max_distance_m_vs_a in the angular case...
            self.gd_vert_mask_bool.a = set_dg_mask(int(i_v), self.gd_vert_idx_int.a, self.gd_vert_mask_bool.a,
                                                   self.gd_vert_x_float.a, self.gd_vert_y_float.a, max_distance)

            # set graph filter -> DON'T FILTER OUT i -> WILL CRASH SHORTEST PATH METHOD...
            self.graph_dual.set_vertex_filter(self.gd_vert_mask_bool)
            #graph_draw(self.graph_dual, pos=self.gd_vert_xy_vect_float)

            # get active vertices
            j_verts = [j_v for j_v in self.graph_dual.vertices()]
            j_ids = self.gd_vert_idx_int.fa

            # nodes connected to only one other node throw errors in numba and are likely orphaned segments
            # these don't have closeness or betweenness values, so it is safe to ignore
            if len(j_ids) > 1:
                distance_map, pred_map = shortest_distance(
                    self.graph_dual,
                    source=i_v,
                    target=j_verts,
                    weights=weight,
                    max_dist=max_distance_shortest_path,
                    pred_map=True)

                # using jitted function to speed up loops and assignments:
                process_paths(
                    int(i_v),
                    np.array(j_ids),
                    distance_map,  # distance_map length = masked graph
                    max_distance,
                    pred_map.a,  # pred_map length = full graph
                    self.gd_vert_betweenness_int.a,
                    self.gd_vert_betw_weighted_float.a,
                    self.gd_vert_primary_distance_float.a,
                    self.gd_vert_closeness_count_float.a,
                    self.gd_vert_closeness_sum_float.a,
                    self.gd_vert_closeness_sum_weighted_float.a,
                    angular=angular)

            # remove mask filter
            self.graph_dual.set_vertex_filter(None)

        # calculate closeness -> dividing by average rather than nodes
        # sum of distances / count of edges
        # i.e. don't count self node when calculating average
        # LATER NOTE -> This is actually more along the lines of: Allen, Liu, Singer, 1993
        # OR -> maybe a form of efficiency measure per
        closeness_avg = self.gd_vert_closeness_sum_float.a / self.gd_vert_closeness_count_float.a

        # regular closeness
        self.gd_vert_closeness_float.a = self.gd_vert_closeness_sum_float.a / closeness_avg

        # weighted closeness
        # NB -> can't divide by average weighted closeness, otherwise the result ends up the same as non-weighted
        self.gd_vert_closeness_weighted_float.a = self.gd_vert_closeness_sum_weighted_float.a / closeness_avg

        # reduce the size of the weighted betweenness numbers
        self.gd_vert_betw_weighted_float.a = self.gd_vert_betw_weighted_float.a / 100000

        logger.info('...completed graph analysis for distance: {0}, time elapsed: {1}m'.format(max_distance, round((time.time() - time_start) / 60, 2)))

    def build_dual_graph(self):
        '''
        Creates a dual graph based on the primary graph and inserts compound (distance * angular) weights.

        :return: A dual representation of the original planar graph with metric distance and compound angular /
        metric measures.
        '''
        # create the dual graph from the original
        # note that the base dual graph is originally instanced in the __init__ method, don't reinstance here...

        # An index is necessary for referencing the dual graph ids based on the original segment's start and end vertices
        # This enables matching up to the neighbouring vertices in the next step
        # The id can't simply be the edge value because it has to be deduced for neighbours...
        # in which case the order of the two numbers isn't necessarily known
        self.dual_vert_idx = {}

        # double check that duplicate edges are removed, otherwise errors may ensue
        remove_parallel_edges(self.graph_primary)

        # iterate all edges and convert to the dual vertex for the new graph
        for edge in self.graph_primary.edges():

            # get the uid and distance from the primary graph's edges
            uid = self.g_edge_uid_str[edge]
            distance = self.g_edge_distance_float[edge]

            # add the node
            node_id = '{0}:{1}'.format(*sorted(edge))
            vert_dual = self.graph_dual.add_vertex()

            # keep the vertex in the dictionary for later reference
            self.dual_vert_idx[node_id] = vert_dual

            # set the dual graph's vertex uid and index based on the primary edge's values
            self.gd_vert_uid_str[vert_dual] = uid
            self.gd_vert_primary_distance_float[vert_dual] = distance

        # iterate the edges again, but this time build the links
        for source_v, target_v in self.graph_primary.edges():

            # calculate psuedo xy
            start_xy = geometry.Point(self.g_vert_xy_vect_float[source_v])
            end_xy = geometry.Point(self.g_vert_xy_vect_float[target_v])
            centroid_xy = geometry.LineString((start_xy, end_xy)).centroid

            vert_idx = '{0}:{1}'.format(*sorted((source_v, target_v)))
            vert_dual = self.dual_vert_idx[vert_idx]

            self.gd_vert_x_float[vert_dual] = centroid_xy.x
            self.gd_vert_y_float[vert_dual] = centroid_xy.y
            self.gd_vert_xy_vect_float[vert_dual] = [centroid_xy.x, centroid_xy.y]

            # generate the links
            self.build_dual_graph_links(source_v, target_v)
            # TODO: presently doubling links...convert to split dual?? Remember to adapt i < j approach accordingly?
            self.build_dual_graph_links(target_v, source_v)


    def build_dual_graph_links(self, A_v, B_v):

        # iterate the neighbours of the original edge, then connect the new node to the corresponding new neighbours
        # note that the underlying boost library has only out-neighbours for undirected graphs
        for neighbour_vertex in B_v.all_neighbours():

            # first check if the neighbouring vertex is A vertex
            if neighbour_vertex != A_v:

                # get the corresponding dual vertex idx key (the original edge)
                dual_vertex_id = '{0}:{1}'.format(*sorted((A_v, B_v)))
                # use the idx key to get the dual vertex id
                vert_dual = self.dual_vert_idx[dual_vertex_id]
                # get the corresponding dual vertex idx key for the neighbour (the original neighbouring edge)
                dual_neighbor_id = '{0}:{1}'.format(*sorted((B_v, neighbour_vertex)))
                # use this idx key to get the dual neighbour id
                vert_neighbour = self.dual_vert_idx[dual_neighbor_id]

                # calculate the angle difference between the two segments in the original primary graph
                angle = self.fetch_angle(A_v, B_v, neighbour_vertex)
                # calculate the new metric distance by halving then adding each original segment
                distance = (self.gd_vert_primary_distance_float[vert_dual] / 2) + (self.gd_vert_primary_distance_float[vert_neighbour] / 2)

                # normalise angles to 2 (where 0 = no turn, and 1 = 90 degree turn, and 2 = 180 degrees)
                e = self.graph_dual.add_edge(vert_dual, vert_neighbour)

                self.gd_edge_distance_float[e] = distance
                self.gd_edge_angle_float[e] = angle / 90

    def fetch_angle(self, root_vertex, out_vertex, neighbour_vertex):
        '''
        helper function for returning the angular change between two segments
        :param graph:
        :param root_vertex:
        :param out_vertex:
        :param neighbour_vertex:
        :return:
        '''

        root_x, root_y = self.g_vert_xy_vect_float[root_vertex]
        out_x, out_y = self.g_vert_xy_vect_float[out_vertex]
        nb_x, nb_y = self.g_vert_xy_vect_float[neighbour_vertex]

        # non numpy method
        base_segment_angle = math.degrees(math.atan2(out_y - root_y, out_x - root_x))
        neighbour_segment_angle = math.degrees(math.atan2(nb_y - out_y, nb_x - out_x))
        angle = math.fabs(neighbour_segment_angle - base_segment_angle) % 360
        if angle > 180:
            angle = 360 - angle

        # normalise to 2
        return angle

class Graph_Analyse_PostGIS(Graph_Analyse):
    '''
    PostGIS variant of the Graph Analyse class.
    '''

    def __init__(self):
        super().__init__()
        self.dsn_string = None
        self.source_schema = None
        self.source_table = None
        self.id_column = None
        self.geom_column = None

    async def build_graph(self, dsn_string: str, source_schema: str, source_table: str, wgs_bounds: tuple=None, id_column: str='id'):
        '''
        Prepares a graph for network_analyse()
        :param dsn_string: The database connection dsn string.
        :param source_schema: The source schema.
        :param source_table: The source road segment network table from the postGIS database.
        :param wgs_bounds: The bounds for the network analysis, in wgs_lon_min, wgs_lat_min, wgs_lon_max, wgs_lat_max.
        :param id_column: The column name for unique row identifiers.
        :return: Nothing is returned. Outputs are written to the database.
        '''

        super().build_graph()

        if not wgs_bounds:
            lon_min, lat_min, lon_max, lat_max = await wgs_bounds_from_postGIS(dsn_string, source_schema, source_table)
        else:
            lon_min, lat_min, lon_max, lat_max = wgs_bounds

        # save schema, table, and id column for later writing back to the database
        self.dsn_string = dsn_string
        self.source_schema = source_schema
        self.source_table = source_table
        self.id_column = id_column
        self.geom_column = await find_geom_column(dsn_string, source_schema, source_table)
        logger.info('...identified geom column as {0}'.format(self.geom_column))

        # fetch the utm epsg code
        utm_epsg_code = convert_wgs_to_utm((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)
        logger.info('...selected EPSG local UTM code: {0}'.format(utm_epsg_code))

        # connect to db
        db_connect = await asyncpg.connect(dsn=dsn_string)

        # setup the table names
        source_schema_and_table = '"{0}"."{1}"'.format(self.source_schema, self.source_table)
        destination_table = source_schema_and_table

        # check if the resulting columns exist (fail before computations if not)
        # FIXME ensure columns exist
        # await db_connect.execute('''
        #     UPDATE {0} AS t SET
        #         closeness__{1}__{2} = null,
        #         closeness__{1}__{2}__weighted = null,
        #         betweenness__{1}__{2} = null,
        #         betweenness__{1}__{2}__weighted = null
        #     WHERE false
        #     '''.format(destination_table, self.selected_measure_str, self.max_distance))

        # fetch the data
        logger.info('...querying network table {0}.{1} for {2}, {3}'.format(
            source_schema, source_table, id_column, self.geom_column))
        async with db_connect.transaction():

            nodes_coords = {}  # temporary dictionary for gathering node coordinates
            count = 0

            # Dump the geometries in case the individual streets are multi-line geoms
            async for row in db_connect.cursor('''
                SELECT {0}, (ST_Dump(ST_Transform({1}, {2}))).geom
                    FROM {3}.{4}
                    WHERE ST_Transform({1}, 4326) &&
                        ST_SetSRID(ST_MakeBox2D(ST_Point({5}, {6}), ST_Point({7}, {8})), 4326)
                    '''.format(id_column, self.geom_column, utm_epsg_code, source_schema, source_table,
                               lon_min, lat_min, lon_max, lat_max)):

                count += 1

                uid = row[0]
                line = wkb.loads(row[1], hex=True)

                start = geometry.Point(line.coords[0])
                start_node = (start.x, start.y)

                end = geometry.Point(line.coords[-1])
                end_node = (end.x, end.y)

                if start_node not in nodes_coords:
                    s_v = self.graph_primary.add_vertex()
                    nodes_coords[start_node] = s_v
                    self.g_vert_xy_vect_float[s_v] = [start.x, start.y]
                else:
                    s_v = nodes_coords[start_node]

                if end_node not in nodes_coords:
                    e_v = self.graph_primary.add_vertex()
                    nodes_coords[end_node] = e_v
                    self.g_vert_xy_vect_float[e_v] = [end.x, end.y]
                else:
                    e_v = nodes_coords[end_node]

                # add the edge to the graph
                e = self.graph_primary.add_edge(s_v, e_v)  # e = tuple, e.g. (0, 1)
                self.g_edge_uid_str[e] = uid
                self.g_edge_distance_float[e] = line.length

                if count % 1000 == 0:
                    logger.info('...added {0} edges to graph'.format(count))

        # remove duplicate edges -> NB
        remove_parallel_edges(self.graph_primary)

        logger.info('...built graph: {0}'.format(self.graph_primary))

        #graph_draw(self.graph_primary, pos=self.g_vert_xy_vect_float)

    async def return_graph(self, postfix):
        '''
        Prepares a graph for network_analyse()
        :param dsn_string: The database connection dsn string.
        :param source_schema: The source schema.
        :param source_schema_and_table: The source road segment network table from the postGIS database.
        :param id_column: The column name for unique row identifiers.
        :param postfix: The results will be written to a new table named source_table_postfix.
        :return: Nothing is returned. Outputs are written to the database.
        '''

        logger.info('...sending data to DB for threshold distance: {0}'.format(self.max_distance))

        # connect to the database
        db_connect = await asyncpg.connect(dsn=self.dsn_string)

        # setup the table names
        source_schema_and_table = '"{0}"."{1}"'.format(self.source_schema, self.source_table)
        destination_table = source_schema_and_table

        # create a hash index for equality checking later-on (hash is best for equality)
        await db_connect.execute('CREATE INDEX IF NOT EXISTS "{0}_{1}_id_hash_index" ON {2} USING HASH (id)'.format(
            self.source_table, postfix, destination_table))

        # hash indexes can be prone to issues if crashes etc.
        await db_connect.execute('REINDEX INDEX "{0}"."{1}_{2}_id_hash_index"'.format(
            self.source_schema, self.source_table, postfix))

        logger.info('......creating the db update query')
        prepared_statement = await db_connect.prepare('''
            UPDATE {0} AS t SET
                closeness__{1}__{2} = $1,
                closeness__{1}__{2}__weighted = $2,
                betweenness__{1}__{2} = $3,
                betweenness__{1}__{2}__weighted = $4
                WHERE t.id::text = $5::text
            '''.format(destination_table, self.selected_measure_str, self.max_distance))

        # convert the NaN and '0' values to None values ('0' for the sake of visualisation of betweenness)
        logger.info('......converting NaN to None')
        closen = np.where(np.isnan(self.gd_vert_closeness_float.a), None, self.gd_vert_closeness_float.a)
        closen = np.where(closen == 0, None, closen)
        closen_weighted = np.where(np.isnan(self.gd_vert_closeness_weighted_float.a), None, self.gd_vert_closeness_weighted_float.a)
        closen_weighted = np.where(closen_weighted == 0, None, closen_weighted)

        between = np.where(np.isnan(self.gd_vert_betweenness_int.a), None, self.gd_vert_betweenness_int.a)
        between = np.where(between == 0, None, between)
        between_weighted = np.where(np.isnan(self.gd_vert_betw_weighted_float.a), None, self.gd_vert_betw_weighted_float.a)
        between_weighted = np.where(between_weighted == 0, None, between_weighted)

        # execute the queries in batches
        logger.info('......sending data to database')
        batch_size = 1000
        for count, i in enumerate(self.graph_dual.vertices()):
            await prepared_statement.fetchval(
                closen[i],
                closen_weighted[i],
                between[i],
                between_weighted[i],
                self.gd_vert_uid_str[i]
            )
            if count != 0 and count % batch_size == 0:
                logger.info('......sent {0} queries to db'.format(count))

        await db_connect.execute('VACUUM ANALYZE {0}'.format(destination_table))

        # cleanup remaining query values since last modulo update:
        logger.info('...done sending data to DB for threshold distance: {0}'.format(self.max_distance))


if __name__ == '__main__':

    import os

    if 'DB_IP_ADDRESS' in os.environ:
        db_ip = os.environ['DB_IP_ADDRESS']
    else:
        db_ip = '10.160.0.145'

    if 'DB_PORT' in os.environ:
        db_port = os.environ['DB_PORT']
    else:
        db_port = '5432'

    if 'DB_NAME' in os.environ:
        database_name = os.environ['DB_NAME']
    else:
        database_name = 'datasets'

    if 'DB_USERNAME' in os.environ:
        db_username = os.environ['DB_USERNAME']
    else:
        db_username = 'dataset_contributor'

    angular = False
    if 'ANGULAR' in os.environ:
        angular_str = os.environ['ANGULAR']
        if angular_str in ('True', 'TRUE', 'true', 't', 'yes', '1'):
            angular = True

    distances = (2000, 800, 400)
    if 'DISTANCES' in os.environ:
        distances = os.environ['DISTANCES']
        distances = distances.strip('()')
        distances = distances.split(',')
        distances = [int(d) for d in distances]

    postfix = ''
    if 'TABLE_POSTFIX' in os.environ:
        postfix = os.environ['TABLE_POSTFIX']

    schema_name = os.environ['SCHEMA_NAME']

    table_name = os.environ['TABLE_NAME']

    db_pw = os.environ['DB_PW']

    dsn = make_dsn_string(db_ip, db_port, database_name, db_username, db_pw)

    async def process_graph(dsn, schema_name, table_name, distances:tuple, angular:bool, postfix):
        bounds = await wgs_bounds_from_postGIS(dsn, schema_name, table_name)
        graph = Graph_Analyse_PostGIS()
        await graph.build_graph(dsn, schema_name, table_name, bounds, id_column='id')
        for distance in distances:
            graph.analyse_graph(max_distance=distance, angular=angular)
            await graph.return_graph(postfix=postfix)

    loop = asyncio.get_event_loop()

    loop.run_until_complete(
        process_graph(dsn, schema_name, table_name, distances, angular, postfix)
    )
