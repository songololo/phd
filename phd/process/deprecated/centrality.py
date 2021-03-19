"""
Closeness and betweenness.

For city 100 = Execution time: 42s for threshold distance 600
"""
import os
import logging
import time
from collections import deque
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *
from numba import jit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@jit(nopython=True, nogil=True)
def centrality_numba(i_v, offset, dist_map, pred_map, betweenness, closeness_count, closeness_sum):
    for dist_idx in np.where(dist_map != np.inf)[0]:
        #closeness
        closeness_count[i_v] += 1
        closeness_count[dist_idx + offset] += 1
        closeness_sum[i_v] += dist_map[dist_idx]
        closeness_sum[dist_idx + offset] += dist_map[dist_idx]
        # betweenness
        intermediary = dist_idx + offset
        while True:
            betweenness[intermediary] += 1
            if not intermediary == i_v:
                intermediary = pred_map[intermediary]
                continue
            break

async def centrality(db_config, network_schema, nodes_table, links_table, boundary_schema, boundary_table,
                     threshold_dist, city_id):

    # the graph
    graph = Graph(directed=False)

    # property maps
    g_edge_id = graph.new_edge_property('int')
    g_edge_dist = graph.new_edge_property('float')
    g_node_id = graph.new_vertex_property('int')

    # open database connection
    logger.info(f'loading data from {network_schema}.{links_table}')
    vertex_dict = {}
    db_con = await asyncpg.connect(**db_config)
    # create a temporary boundary buffer

    async with db_con.transaction():
        # need a spatial index to speed up subsequent operation
        logger.info('NOTE -> creating temporary buffered city geom')
        await db_con.execute(f'''
            INSERT INTO {boundary_schema}.{boundary_table} (id, geom)
                VALUES (9999999999, (SELECT ST_Buffer(geom, {threshold_dist})
                    FROM {boundary_schema}.{boundary_table} WHERE id = {city_id}::text))
        ''')
        logger.info('NOTE -> fetching links geometry')
        async for record in db_con.cursor(f'''
            SELECT links.id, links.start_node, links.end_node, ST_Length(links.geom) as dist
                FROM {network_schema}.{links_table} as links,
                    (SELECT geom FROM {boundary_schema}.{boundary_table} WHERE id = 9999999999::text) as boundary
                WHERE ST_Contains(boundary.geom, links.geom)'''):
            # add start vertex if not present
            start = record['start_node']
            if (start not in vertex_dict):
                vertex_dict[start] = graph.add_vertex()
            g_node_id[vertex_dict[start]] = start
            # add end vertex if not present
            end = record['end_node']
            if (end not in vertex_dict):
                vertex_dict[end] = graph.add_vertex()
            g_node_id[vertex_dict[end]] = end
            # add edge
            e = graph.add_edge(vertex_dict[start], vertex_dict[end])
            g_edge_id[e] = record['id']
            g_edge_dist[e] = record['dist']
        logger.info('NOTE -> removing temporary buffered city geom')
        await db_con.execute(f'''DELETE FROM {boundary_schema}.{boundary_table} WHERE id = 9999999999::text''')
        
    await db_con.close()
    logger.info(f'edges: {graph.num_edges()}, vertices: {graph.num_vertices()}')

    # arrays for computations
    betweenness = np.zeros(graph.num_vertices())
    closness_count = np.zeros(graph.num_vertices(), dtype=int)
    closeness_sum = np.zeros(graph.num_vertices())

    start_time = time.time()

    counter = 0
    vert_pool = deque(tuple(sorted([int(i) for i in graph.vertices()])))
    # iterate pool while more than 1 (no distance / predecessor maps to compute for last vertex)
    while len(vert_pool) > 1:
        counter += 1
        if counter % 1000 == 0:
            completion = round((counter / graph.num_vertices()) * 100, 2)
            logger.info(f'processing node: {counter}, {completion}% completed')
        v = vert_pool.popleft()
        # NOTE -> distance map is returned as array -> therefore only as long as remaining verts in pool
        # use an offset for fetching the corresponding distance index in the optimised function
        distance_map, pred_map = shortest_distance(graph, source=v, target=vert_pool, weights=g_edge_dist,
                                                   max_dist=threshold_dist, pred_map=True)
        # type checking necessary because last single-value distance map is returned as np.float instead of array
        # TODO: this is apparently resolved in pending versions of graph-tool
        if isinstance(distance_map, np.float):
            distance_map = np.array([float(distance_map)])
        # perform centrality using an optimised numba function -> much faster than pure python
        centrality_numba(int(v), counter, distance_map, pred_map.a, betweenness, closness_count, closeness_sum)

    # calculate 'improved' closeness
    closeness_avg = closeness_sum / closness_count
    closeness = closeness_sum / closeness_avg

    # iterate the 'i' verts
    resultSet = []
    for v in graph.vertices():
        resultSet.append((closeness[int(v)], betweenness[int(v)], g_node_id[v]))

    end_time = time.time()

    logger.info(f'Execution time: {end_time - start_time}s for threshold distance {threshold_dist}')

    logger.info('Writing data back to database')

    db_con = await asyncpg.connect(**db_config)
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{nodes_table}
            ADD COLUMN IF NOT EXISTS closeness_{threshold_dist} real,
            ADD COLUMN IF NOT EXISTS betweenness_{threshold_dist} real;
    ''')
    await db_con.executemany(f'''
        UPDATE {network_schema}.{nodes_table}
            SET closeness_{threshold_dist} = $1,
                betweenness_{threshold_dist} = $2
            WHERE id = $3
    ''', resultSet)
    await db_con.close()

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
    nodes_table = 'roadnodes_20'
    links_table = 'roadlinks_20'
    boundary_schema = 'boundaries'
    boundary_table = 'city_boundaries_320'
    threshold_distances = [2000]

    for threshold_distance in threshold_distances:
        for city_id in range(100, 101):
            logger.info(f'Starting execution of closness and betweenness centrality for city id: {city_id} '
                        f'on table {network_schema}.{nodes_table} at threshold distance {threshold_distance}')
            loop.run_until_complete(centrality(db_config, network_schema, nodes_table, links_table,
                                                        boundary_schema, boundary_table, threshold_distance, city_id))
