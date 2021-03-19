"""
Closeness and betweenness.

For city 100 = Execution time: 42s for threshold distance 600
"""
from collections import deque
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *
from numba import jit

@jit(nopython=True, nogil=True)
def centrality_numba(from_idx, offset, dist_map, pred_map, vertex_degrees,
                        betweenness_50,
                        betweenness_100,
                        betweenness_150,
                        betweenness_200,
                        betweenness_300,
                        betweenness_400,
                        betweenness_600,
                        betweenness_800,
                        betweenness_1200,
                        betweenness_1600,
                        betweenness_weighted_50,
                        betweenness_weighted_100,
                        betweenness_weighted_150,
                        betweenness_weighted_200,
                        betweenness_weighted_300,
                        betweenness_weighted_400,
                        betweenness_weighted_600,
                        betweenness_weighted_800,
                        betweenness_weighted_1200,
                        betweenness_weighted_1600,
                        gravity_50,
                        gravity_100,
                        gravity_150,
                        gravity_200,
                        gravity_300,
                        gravity_400,
                        gravity_600,
                        gravity_800,
                        gravity_1200,
                        gravity_1600,
                        node_count_50,
                        node_count_100,
                        node_count_150,
                        node_count_200,
                        node_count_300,
                        node_count_400,
                        node_count_600,
                        node_count_800,
                        node_count_1200,
                        node_count_1600,
                        farness_50,
                        farness_100,
                        farness_150,
                        farness_200,
                        farness_300,
                        farness_400,
                        farness_600,
                        farness_800,
                        farness_1200,
                        farness_1600,
                        route_complexity_50,
                        route_complexity_100,
                        route_complexity_150,
                        route_complexity_200,
                        route_complexity_300,
                        route_complexity_400,
                        route_complexity_600,
                        route_complexity_800,
                        route_complexity_1200,
                        route_complexity_1600):
    '''
    1 - the target vertices pool is continually shrinking as the verts are processed (and popped)
    Because the centrality path contains an iterable of to vertices, it returns an array of corresponding length
    This is the reason for the offset -> it maps the distance map index back to the corresponding indices of the other arrays
    (the master arrays still contain the full original number of vertices)
    2 - the shrinking pool method is used for efficiency,
    however, this also requires that both the source and target vertices get iterated at the same time
    '''
    beta_50 = -0.08
    beta_100 = -0.04
    beta_150 = -0.02666666666666667
    beta_200 = -0.02
    beta_300 = -0.013333333333333334
    beta_400 = -0.01
    beta_600 = -0.006666666666666667
    beta_800 = -0.005
    beta_1200 = -0.0033333333333333335
    beta_1600 = -0.0025

    for to_dist_map_idx in np.where(dist_map != np.inf)[0]:
        # map the to distance and to vert index
        to_distance = dist_map[to_dist_map_idx]
        to_vert_idx = to_dist_map_idx + offset
        # the strength of the weight is based on the start and end vertices, not the intermediate locations
        weight_50 = np.exp(beta_50 * to_distance)
        weight_100 = np.exp(beta_100 * to_distance)
        weight_150 = np.exp(beta_150 * to_distance)
        weight_200 = np.exp(beta_200 * to_distance)
        weight_300 = np.exp(beta_300 * to_distance)
        weight_400 = np.exp(beta_400 * to_distance)
        weight_600 = np.exp(beta_600 * to_distance)
        weight_800 = np.exp(beta_800 * to_distance)
        weight_1200 = np.exp(beta_1200 * to_distance)
        weight_1600 = np.exp(beta_1600 * to_distance)
        # route complexity -> simpler option to information centrality - distance weighted node degrees greater than 2
        if to_distance <= 50:
            if vertex_degrees[from_idx] > 2:
                route_complexity_50[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_50
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_50[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_50
            # gravity -> an accessibility measure, or effectively a closeness consisting of inverse distance weighted node count
            gravity_50[from_idx] += weight_50
            gravity_50[to_vert_idx] += weight_50
            # closeness -> using 'improved' formulation which requires averages, hence only counting and summing at this stage
            # final values are calculated later-on once averages can be computed
            node_count_50[from_idx] += 1
            node_count_50[to_vert_idx] += 1
            farness_50[from_idx] += to_distance
            farness_50[to_vert_idx] += to_distance
        if to_distance <= 100:
            if vertex_degrees[from_idx] > 2:
                route_complexity_100[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_100
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_100[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_100
            gravity_100[from_idx] += weight_100
            gravity_100[to_vert_idx] += weight_100
            node_count_100[from_idx] += 1
            node_count_100[to_vert_idx] += 1
            farness_100[from_idx] += to_distance
            farness_100[to_vert_idx] += to_distance
        if to_distance <= 150:
            if vertex_degrees[from_idx] > 2:
                route_complexity_150[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_150
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_150[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_150
            gravity_150[from_idx] += weight_150
            gravity_150[to_vert_idx] += weight_150
            node_count_150[from_idx] += 1
            node_count_150[to_vert_idx] += 1
            farness_150[from_idx] += to_distance
            farness_150[to_vert_idx] += to_distance
        if to_distance <= 200:
            if vertex_degrees[from_idx] > 2:
                route_complexity_200[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_200
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_200[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_200
            gravity_200[from_idx] += weight_200
            gravity_200[to_vert_idx] += weight_200
            node_count_200[from_idx] += 1
            node_count_200[to_vert_idx] += 1
            farness_200[from_idx] += to_distance
            farness_200[to_vert_idx] += to_distance
        if to_distance <= 300:
            if vertex_degrees[from_idx] > 2:
                route_complexity_300[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_300
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_300[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_300
            gravity_300[from_idx] += weight_300
            gravity_300[to_vert_idx] += weight_300
            node_count_300[from_idx] += 1
            node_count_300[to_vert_idx] += 1
            farness_300[from_idx] += to_distance
            farness_300[to_vert_idx] += to_distance
        if to_distance <= 400:
            if vertex_degrees[from_idx] > 2:
                route_complexity_400[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_400
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_400[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_400
            gravity_400[from_idx] += weight_400
            gravity_400[to_vert_idx] += weight_400
            node_count_400[from_idx] += 1
            node_count_400[to_vert_idx] += 1
            farness_400[from_idx] += to_distance
            farness_400[to_vert_idx] += to_distance
        if to_distance <= 600:
            if vertex_degrees[from_idx] > 2:
                route_complexity_600[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_600
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_600[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_600
            gravity_600[from_idx] += weight_600
            gravity_600[to_vert_idx] += weight_600
            node_count_600[from_idx] += 1
            node_count_600[to_vert_idx] += 1
            farness_600[from_idx] += to_distance
            farness_600[to_vert_idx] += to_distance
        if to_distance <= 800:
            if vertex_degrees[from_idx] > 2:
                route_complexity_800[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_800
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_800[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_800
            gravity_800[from_idx] += weight_800
            gravity_800[to_vert_idx] += weight_800
            node_count_800[from_idx] += 1
            node_count_800[to_vert_idx] += 1
            farness_800[from_idx] += to_distance
            farness_800[to_vert_idx] += to_distance
        if to_distance <= 1200:
            if vertex_degrees[from_idx] > 2:
                route_complexity_1200[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_1200
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_1200[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_1200
            gravity_1200[from_idx] += weight_1200
            gravity_1200[to_vert_idx] += weight_1200
            node_count_1200[from_idx] += 1
            node_count_1200[to_vert_idx] += 1
            farness_1200[from_idx] += to_distance
            farness_1200[to_vert_idx] += to_distance
        if to_distance <= 1600:
            if vertex_degrees[from_idx] > 2:
                route_complexity_1600[to_vert_idx] += (vertex_degrees[from_idx] - 2) * weight_1600
            if vertex_degrees[to_vert_idx] > 2:
                route_complexity_1600[from_idx] += (vertex_degrees[to_vert_idx] - 2) * weight_1600
            gravity_1600[from_idx] += weight_1600
            gravity_1600[to_vert_idx] += weight_1600
            node_count_1600[from_idx] += 1
            node_count_1600[to_vert_idx] += 1
            farness_1600[from_idx] += to_distance
            farness_1600[to_vert_idx] += to_distance

        # betweenness - only counting truly between vertices, not starting and ending verts
        intermediary_idx = pred_map[to_vert_idx]
        # only counting betweenness in one 'direction' since the graph is symmetrical (non-directed)
        while True:
            # break out of while loop if the intermediary has reached the source node
            if intermediary_idx == from_idx:
                break
            if to_distance <= 50:
                betweenness_50[intermediary_idx] += 1
                betweenness_weighted_50[intermediary_idx] += weight_50
            if to_distance <= 100:
                betweenness_100[intermediary_idx] += 1
                betweenness_weighted_100[intermediary_idx] += weight_100
            if to_distance <= 150:
                betweenness_150[intermediary_idx] += 1
                betweenness_weighted_150[intermediary_idx] += weight_150
            if to_distance <= 200:
                betweenness_200[intermediary_idx] += 1
                betweenness_weighted_200[intermediary_idx] += weight_200
            if to_distance <= 300:
                betweenness_300[intermediary_idx] += 1
                betweenness_weighted_300[intermediary_idx] += weight_300
            if to_distance <= 400:
                betweenness_400[intermediary_idx] += 1
                betweenness_weighted_400[intermediary_idx] += weight_400
            if to_distance <= 600:
                betweenness_600[intermediary_idx] += 1
                betweenness_weighted_600[intermediary_idx] += weight_600
            if to_distance <= 800:
                betweenness_800[intermediary_idx] += 1
                betweenness_weighted_800[intermediary_idx] += weight_800
            if to_distance <= 1200:
                betweenness_1200[intermediary_idx] += 1
                betweenness_weighted_1200[intermediary_idx] += weight_1200
            if to_distance <= 1600:
                betweenness_1600[intermediary_idx] += 1
                betweenness_weighted_1600[intermediary_idx] += weight_1600
            # unlike the dist_map the pred_map contains all vertices, so no offset required
            intermediary_idx = pred_map[intermediary_idx]


async def centrality(db_config, network_schema, nodes_table, links_table,
                     city_pop_id=None, boundary_schema=None, boundary_table=None):

    # the graph
    graph = Graph(directed=False)

    # property maps
    g_edge_id = graph.new_edge_property('int')
    g_edge_dist = graph.new_edge_property('float')
    g_node_id = graph.new_vertex_property('int')

    # open database connection
    vertex_dict = {}
    db_con = await asyncpg.connect(**db_config)
    # create a temporary boundary buffer

    # default query
    if city_pop_id is None:
        logger.info(f'calculating centrality for {network_schema}.{nodes_table}')
        q = f'''
            SELECT links.id, links.start_node, links.end_node, ST_Length(links.geom) as dist
                FROM {network_schema}.{links_table} as links;'''

    # override default query if city id provided
    else:
        logger.info(f'calculating centrality on {network_schema}.{nodes_table} for city {city_pop_id}')
        # setup temporary buffered boundary based on max threshold distance - prevents centrality falloff
        logger.info('NOTE -> creating temporary buffered city geom')
        # convex hull prevents potential issues with multipolygons deriving from buffer...
        temp_id = await db_con.fetchval(f'''
            INSERT INTO {boundary_schema}.{boundary_table} (geom)
                VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, 1600))
                    FROM {boundary_schema}.{boundary_table} WHERE pop_id = {city_pop_id}))
                RETURNING id;''')
        # override the query
        q = f'''
            SELECT links.id, links.start_node, links.end_node, ST_Length(links.geom) as dist
                FROM {network_schema}.{links_table} as links,
                    (SELECT geom FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}) as boundary
                WHERE ST_Contains(boundary.geom, links.geom);'''

    async with db_con.transaction():
        # get the geometry
        logger.info('NOTE -> fetching links geometry')
        async for record in db_con.cursor(q):
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

    # remove the temporary boundary if city id
    if city_pop_id is not None:
        logger.info('NOTE -> removing temporary buffered city geom')
        await db_con.execute(f'''DELETE FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id};''')

    await db_con.close()
    logger.info(f'edges: {graph.num_edges()}, vertices: {graph.num_vertices()}')

    # arrays for computations
    betweenness_50 = np.zeros(graph.num_vertices())
    betweenness_100 = np.zeros(graph.num_vertices())
    betweenness_150 = np.zeros(graph.num_vertices())
    betweenness_200 = np.zeros(graph.num_vertices())
    betweenness_300 = np.zeros(graph.num_vertices())
    betweenness_400 = np.zeros(graph.num_vertices())
    betweenness_600 = np.zeros(graph.num_vertices())
    betweenness_800 = np.zeros(graph.num_vertices())
    betweenness_1200 = np.zeros(graph.num_vertices())
    betweenness_1600 = np.zeros(graph.num_vertices())
    betweenness_weighted_50 = np.zeros(graph.num_vertices())
    betweenness_weighted_100 = np.zeros(graph.num_vertices())
    betweenness_weighted_150 = np.zeros(graph.num_vertices())
    betweenness_weighted_200 = np.zeros(graph.num_vertices())
    betweenness_weighted_300 = np.zeros(graph.num_vertices())
    betweenness_weighted_400 = np.zeros(graph.num_vertices())
    betweenness_weighted_600 = np.zeros(graph.num_vertices())
    betweenness_weighted_800 = np.zeros(graph.num_vertices())
    betweenness_weighted_1200 = np.zeros(graph.num_vertices())
    betweenness_weighted_1600 = np.zeros(graph.num_vertices())
    gravity_50 = np.zeros(graph.num_vertices())
    gravity_100 = np.zeros(graph.num_vertices())
    gravity_150 = np.zeros(graph.num_vertices())
    gravity_200 = np.zeros(graph.num_vertices())
    gravity_300 = np.zeros(graph.num_vertices())
    gravity_400 = np.zeros(graph.num_vertices())
    gravity_600 = np.zeros(graph.num_vertices())
    gravity_800 = np.zeros(graph.num_vertices())
    gravity_1200 = np.zeros(graph.num_vertices())
    gravity_1600 = np.zeros(graph.num_vertices())
    node_count_50 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_100 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_150 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_200 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_300 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_400 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_600 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_800 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_1200 = np.zeros(graph.num_vertices(), dtype=int)
    node_count_1600 = np.zeros(graph.num_vertices(), dtype=int)
    farness_50 = np.zeros(graph.num_vertices())
    farness_100 = np.zeros(graph.num_vertices())
    farness_150 = np.zeros(graph.num_vertices())
    farness_200 = np.zeros(graph.num_vertices())
    farness_300 = np.zeros(graph.num_vertices())
    farness_400 = np.zeros(graph.num_vertices())
    farness_600 = np.zeros(graph.num_vertices())
    farness_800 = np.zeros(graph.num_vertices())
    farness_1200 = np.zeros(graph.num_vertices())
    farness_1600 = np.zeros(graph.num_vertices())
    route_complexity_50 = np.zeros(graph.num_vertices())
    route_complexity_100 = np.zeros(graph.num_vertices())
    route_complexity_150 = np.zeros(graph.num_vertices())
    route_complexity_200 = np.zeros(graph.num_vertices())
    route_complexity_300 = np.zeros(graph.num_vertices())
    route_complexity_400 = np.zeros(graph.num_vertices())
    route_complexity_600 = np.zeros(graph.num_vertices())
    route_complexity_800 = np.zeros(graph.num_vertices())
    route_complexity_1200 = np.zeros(graph.num_vertices())
    route_complexity_1600 = np.zeros(graph.num_vertices())

    # get the vertex degrees
    vertex_degrees = graph.get_out_degrees(graph.get_vertices())

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
                                                   max_dist=1600, pred_map=True)
        # perform centrality using an optimised numba function -> much faster than pure python
        centrality_numba(int(v), counter, distance_map, pred_map.a, vertex_degrees,
            betweenness_50,
            betweenness_100,
            betweenness_150,
            betweenness_200,
            betweenness_300,
            betweenness_400,
            betweenness_600,
            betweenness_800,
            betweenness_1200,
            betweenness_1600,
            betweenness_weighted_50,
            betweenness_weighted_100,
            betweenness_weighted_150,
            betweenness_weighted_200,
            betweenness_weighted_300,
            betweenness_weighted_400,
            betweenness_weighted_600,
            betweenness_weighted_800,
            betweenness_weighted_1200,
            betweenness_weighted_1600,
            gravity_50,
            gravity_100,
            gravity_150,
            gravity_200,
            gravity_300,
            gravity_400,
            gravity_600,
            gravity_800,
            gravity_1200,
            gravity_1600,
            node_count_50,
            node_count_100,
            node_count_150,
            node_count_200,
            node_count_300,
            node_count_400,
            node_count_600,
            node_count_800,
            node_count_1200,
            node_count_1600,
            farness_50,
            farness_100,
            farness_150,
            farness_200,
            farness_300,
            farness_400,
            farness_600,
            farness_800,
            farness_1200,
            farness_1600,
            route_complexity_50,
            route_complexity_100,
            route_complexity_150,
            route_complexity_200,
            route_complexity_300,
            route_complexity_400,
            route_complexity_600,
            route_complexity_800,
            route_complexity_1200,
            route_complexity_1600)

    # iterate the 'i' verts
    resultSet = []
    for v in graph.vertices():
        v_idx = int(v)
        resultSet.append(
            (betweenness_50[v_idx],
             betweenness_100[v_idx],
             betweenness_150[v_idx],
             betweenness_200[v_idx],
             betweenness_300[v_idx],
             betweenness_400[v_idx],
             betweenness_600[v_idx],
             betweenness_800[v_idx],
             betweenness_1200[v_idx],
             betweenness_1600[v_idx],
             betweenness_weighted_50[v_idx],
             betweenness_weighted_100[v_idx],
             betweenness_weighted_150[v_idx],
             betweenness_weighted_200[v_idx],
             betweenness_weighted_300[v_idx],
             betweenness_weighted_400[v_idx],
             betweenness_weighted_600[v_idx],
             betweenness_weighted_800[v_idx],
             betweenness_weighted_1200[v_idx],
             betweenness_weighted_1600[v_idx],
             gravity_50[v_idx],
             gravity_100[v_idx],
             gravity_150[v_idx],
             gravity_200[v_idx],
             gravity_300[v_idx],
             gravity_400[v_idx],
             gravity_600[v_idx],
             gravity_800[v_idx],
             gravity_1200[v_idx],
             gravity_1600[v_idx],
             node_count_50[v_idx],
             node_count_100[v_idx],
             node_count_150[v_idx],
             node_count_200[v_idx],
             node_count_300[v_idx],
             node_count_400[v_idx],
             node_count_600[v_idx],
             node_count_800[v_idx],
             node_count_1200[v_idx],
             node_count_1600[v_idx],
             farness_50[v_idx],
             farness_100[v_idx],
             farness_150[v_idx],
             farness_200[v_idx],
             farness_300[v_idx],
             farness_400[v_idx],
             farness_600[v_idx],
             farness_800[v_idx],
             farness_1200[v_idx],
             farness_1600[v_idx],
             route_complexity_50[v_idx],
             route_complexity_100[v_idx],
             route_complexity_150[v_idx],
             route_complexity_200[v_idx],
             route_complexity_300[v_idx],
             route_complexity_400[v_idx],
             route_complexity_600[v_idx],
             route_complexity_800[v_idx],
             route_complexity_1200[v_idx],
             route_complexity_1600[v_idx],
             g_node_id[v]))

    logger.info('Writing data back to database')

    db_con = await asyncpg.connect(**db_config)
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{nodes_table}
            ADD COLUMN IF NOT EXISTS met_betw_50 real,
            ADD COLUMN IF NOT EXISTS met_betw_100 real,
            ADD COLUMN IF NOT EXISTS met_betw_150 real,
            ADD COLUMN IF NOT EXISTS met_betw_200 real,
            ADD COLUMN IF NOT EXISTS met_betw_300 real,
            ADD COLUMN IF NOT EXISTS met_betw_400 real,
            ADD COLUMN IF NOT EXISTS met_betw_600 real,
            ADD COLUMN IF NOT EXISTS met_betw_800 real,
            ADD COLUMN IF NOT EXISTS met_betw_1200 real,
            ADD COLUMN IF NOT EXISTS met_betw_1600 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_50 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_100 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_150 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_200 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_300 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_400 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_600 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_800 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_1200 real,
            ADD COLUMN IF NOT EXISTS met_betw_w_1600 real,
            ADD COLUMN IF NOT EXISTS met_gravity_50 real,
            ADD COLUMN IF NOT EXISTS met_gravity_100 real,
            ADD COLUMN IF NOT EXISTS met_gravity_150 real,
            ADD COLUMN IF NOT EXISTS met_gravity_200 real,
            ADD COLUMN IF NOT EXISTS met_gravity_300 real,
            ADD COLUMN IF NOT EXISTS met_gravity_400 real,
            ADD COLUMN IF NOT EXISTS met_gravity_600 real,
            ADD COLUMN IF NOT EXISTS met_gravity_800 real,
            ADD COLUMN IF NOT EXISTS met_gravity_1200 real,
            ADD COLUMN IF NOT EXISTS met_gravity_1600 real,
            ADD COLUMN IF NOT EXISTS met_node_count_50 real,
            ADD COLUMN IF NOT EXISTS met_node_count_100 real,
            ADD COLUMN IF NOT EXISTS met_node_count_150 real,
            ADD COLUMN IF NOT EXISTS met_node_count_200 real,
            ADD COLUMN IF NOT EXISTS met_node_count_300 real,
            ADD COLUMN IF NOT EXISTS met_node_count_400 real,
            ADD COLUMN IF NOT EXISTS met_node_count_600 real,
            ADD COLUMN IF NOT EXISTS met_node_count_800 real,
            ADD COLUMN IF NOT EXISTS met_node_count_1200 real,
            ADD COLUMN IF NOT EXISTS met_node_count_1600 real,
            ADD COLUMN IF NOT EXISTS met_farness_50 real,
            ADD COLUMN IF NOT EXISTS met_farness_100 real,
            ADD COLUMN IF NOT EXISTS met_farness_150 real,
            ADD COLUMN IF NOT EXISTS met_farness_200 real,
            ADD COLUMN IF NOT EXISTS met_farness_300 real,
            ADD COLUMN IF NOT EXISTS met_farness_400 real,
            ADD COLUMN IF NOT EXISTS met_farness_600 real,
            ADD COLUMN IF NOT EXISTS met_farness_800 real,
            ADD COLUMN IF NOT EXISTS met_farness_1200 real,
            ADD COLUMN IF NOT EXISTS met_farness_1600 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_50 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_100 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_150 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_200 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_300 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_400 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_600 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_800 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_1200 real,
            ADD COLUMN IF NOT EXISTS met_rt_complex_1600 real;
    ''')
    await db_con.executemany(f'''
        UPDATE {network_schema}.{nodes_table}
            SET
                met_betw_50 = $1,
                met_betw_100 = $2,
                met_betw_150 = $3,
                met_betw_200 = $4,
                met_betw_300 = $5,
                met_betw_400 = $6,
                met_betw_600 = $7,
                met_betw_800 = $8,
                met_betw_1200 = $9,
                met_betw_1600 = $10,
                met_betw_w_50 = $11,
                met_betw_w_100 = $12,
                met_betw_w_150 = $13,
                met_betw_w_200 = $14,
                met_betw_w_300 = $15,
                met_betw_w_400 = $16,
                met_betw_w_600 = $17,
                met_betw_w_800 = $18,
                met_betw_w_1200 = $19,
                met_betw_w_1600 = $20,
                met_gravity_50 = $21,
                met_gravity_100 = $22,
                met_gravity_150 = $23,
                met_gravity_200 = $24,
                met_gravity_300 = $25,
                met_gravity_400 = $26,
                met_gravity_600 = $27,
                met_gravity_800 = $28,
                met_gravity_1200 = $29,
                met_gravity_1600 = $30,
                met_node_count_50 = $31,
                met_node_count_100 = $32,
                met_node_count_150 = $33,
                met_node_count_200 = $34,
                met_node_count_300 = $35,
                met_node_count_400 = $36,
                met_node_count_600 = $37,
                met_node_count_800 = $38,
                met_node_count_1200 = $39,
                met_node_count_1600 = $40,
                met_farness_50 = $41,
                met_farness_100 = $42,
                met_farness_150 = $43,
                met_farness_200 = $44,
                met_farness_300 = $45,
                met_farness_400 = $46,
                met_farness_600 = $47,
                met_farness_800 = $48,
                met_farness_1200 = $49,
                met_farness_1600 = $50,
                met_rt_complex_50 = $51,
                met_rt_complex_100 = $52,
                met_rt_complex_150 = $53,
                met_rt_complex_200 = $54,
                met_rt_complex_300 = $55,
                met_rt_complex_400 = $56,
                met_rt_complex_600 = $57,
                met_rt_complex_800 = $58,
                met_rt_complex_1200 = $59,
                met_rt_complex_1600 = $60
            WHERE id = $61
    ''', resultSet)
    await db_con.close()


if __name__ == '__main__':

    import time
    import datetime
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logging.getLogger('asyncio').setLevel(logging.WARNING)
    loop = asyncio.get_event_loop()
    # loop.set_debug(True)

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    network_schema = 'analysis'

    # global
    #loop.run_until_complete(centrality(db_config, network_schema, nodes_table, links_table))

    boundary_schema = 'analysis'
    boundary_table = 'city_boundaries_150'

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    nodes_table = 'roadnodes_20'
    links_table = 'roadlinks_20'
    for city_pop_id in range(1, 959):
       loop.run_until_complete(
           centrality(db_config, network_schema, nodes_table, links_table, city_pop_id, boundary_schema, boundary_table))
       logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')