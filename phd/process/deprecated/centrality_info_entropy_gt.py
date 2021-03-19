"""
Graph-tool version of information Entropy based centrality index.
DEPRECATED -- see numba version

It uses the graph-tool breadth-first-visitor class to evenly weight route choices in an outwards and distance sensitive manner.
Because the network is split into smaller chunks (20m, 50m, 100m) it avoids issues with artefacts due to length or network structure.
This also avoids potential issues where longer segments more rapidly claim route options than shorter segments.

This method is slower than the numba method, though if using a cheat
-- stopping the graph search at nanmax rather than nanmean -- then it is slightly faster, albeit less accurate.
For city 100 = Execution time: 375s vs. 700s for threshold distance 600
"""
import os
import logging
import time
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisitorProb(BFSVisitor):

    def __init__(self, source, threshold, edge_dist_map, probability, agg_dist, neighbours):
        self.threshold = threshold
        self.active_parent = int(source)  # keep track of current active parent
        self.active_distance = 0  # for saving state from examining tree_edge to setting agg distance at discover_vertex
        self.nb = neighbours  # for keeping track of new neighbours -> this is reset for each examine_vertex event
        self.prob = probability
        self.prob[:] = 0  # reset probability
        self.prob[int(source)] = 1  # source is 1
        self.edge_dist_map = edge_dist_map
        self.agg_dist = agg_dist
        self.agg_dist[:] = np.nan  # set to nan -> nanmean() is used for keeping track of average search distance
        self.agg_dist[int(source)] = 0  # source is 0m

    # STEP 1
    def examine_vertex(self, u):
        self.active_parent = int(u)  # set new active parent
        self.nb[:] = False  # reset neighbours

    # STEP 2
    def tree_edge(self, e):
        self.active_distance = self.edge_dist_map[e]  # keep track of edge distance

    # STEP 3
    def discover_vertex(self, u):
        self.nb[int(u)] = True  # add new white vertices to new neighbours
        self.agg_dist[int(u)] = self.agg_dist[self.active_parent] + self.active_distance  # aggregate distance

    # STEP 4
    def finish_vertex(self, e):
        if self.nb.any():  # if new neighbours, distribute the resultant probabilities
            self.prob[self.nb] = self.prob[self.active_parent] / np.nansum(self.nb)
            self.prob[self.active_parent] = 0  # reset the parent node's probability
            self.agg_dist[self.active_parent] = np.nan  # reset the parent agg distance
        # NOTE -> nanmean (slower but more accurate) vs. nanmax (faster) substantially affects performance...
        if np.nanmean(self.agg_dist) > self.threshold:  # check whether whether mean distance has exceeded the threshold
            raise StopSearch


async def information_centrality(db_config, network_schema, verts_table, links_table, boundary_schema, boundary_table,
                                 threshold_distance, city_id):

    # the graph
    graph = Graph(directed=False)

    # property maps
    g_edge_id = graph.new_edge_property('int')
    g_edge_dist = graph.new_edge_property('float')
    g_vert_id = graph.new_vertex_property('int')

    # open database connection
    logger.info(f'loading data from {network_schema}.{verts_table}')
    vertex_dict = {}
    db_con = await asyncpg.connect(**db_config)
    async with db_con.transaction():
        async for record in db_con.cursor(f'''
            SELECT links.id, links.start_node, links.end_node, ST_Length(links.geom) as dist
                FROM {network_schema}.{links_table} as links,
                    (SELECT geom FROM {boundary_schema}.{boundary_table} WHERE id = {city_id}::text) as boundary
                WHERE ST_Contains(boundary.geom, links.geom)'''):
            # add start vertex if not present
            start = record['start_node']
            if (start not in vertex_dict):
                vertex_dict[start] = graph.add_vertex()
            g_vert_id[vertex_dict[start]] = start
            # add end vertex if not present
            end = record['end_node']
            if (end not in vertex_dict):
                vertex_dict[end] = graph.add_vertex()
            g_vert_id[vertex_dict[end]] = end
            # add edge
            e = graph.add_edge(vertex_dict[start], vertex_dict[end])
            g_edge_id[e] = record['id']
            g_edge_dist[e] = record['dist']
    await db_con.close()
    logger.info(f'edges: {graph.num_edges()}, vertices: {graph.num_vertices()}')

    # arrays for computations
    prob = np.zeros(graph.num_vertices())
    neighbours = np.zeros(graph.num_vertices(), dtype=bool)
    agg_dist = np.zeros(graph.num_vertices())

    start_time = time.time()
    # iterate the 'i' verts
    resultSet = []
    for v in graph.vertices():
        if int(v) % 1000 == 0:
            completion = round((int(v) / graph.num_vertices()) * 100, 2)
            logger.info(f'processing vertex: {v}, {completion}% completed')
        # breadth first search from vertex
        bfs_search(graph, source=v, visitor=VisitorProb(v, threshold_distance, g_edge_dist, prob, agg_dist, neighbours))
        # append entropy of probabilities to result set
        resultSet.append((-np.nansum(prob * np.log2(prob)), g_vert_id[v]))

    end_time = time.time()

    logger.info(f'Execution time: {end_time - start_time}s for threshold distance {threshold_distance}')

    logger.info('Writing data back to database')

    db_con = await asyncpg.connect(**db_config)
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{verts_table}
            ADD COLUMN IF NOT EXISTS centrality_entropy_{threshold_distance} real;
    ''')
    await db_con.executemany(f'''
        UPDATE {network_schema}.{verts_table}
            SET centrality_entropy_{threshold_distance} = $1
            WHERE id = $2
    ''', resultSet)


if __name__ == '__main__':

    logging.getLogger('asyncio').setLevel(logging.WARNING)
    loop = asyncio.get_event_loop()
    # loop.set_debug(True)

    db_config = {
        'host': 'localhost',
        'port': 5434,
        'user': 'gareth',
        'database': 'os',
        'password': os.environ['CITYSEERDB_PW']
    }

    network_schema = 'analysis'
    verts_table = 'roadnodes_test_20'
    links_table = 'roadlinks_20'
    boundary_schema = 'boundaries'
    boundary_table = 'city_boundaries_320'
    threshold_distances = [600]

    for threshold_distance in threshold_distances:
        for city_id in range(100, 99, -1):
            logger.info(f'Starting execution of entropy centrality for city id: {city_id} '
                        f'on table {network_schema}.{verts_table} at threshold distance {threshold_distance}')
            loop.run_until_complete(information_centrality(db_config, network_schema, verts_table, links_table,
                                                           boundary_schema, boundary_table, threshold_distance, city_id))
