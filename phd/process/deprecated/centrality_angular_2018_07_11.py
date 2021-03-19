"""
ANGULAR IMPLEMENTATION

Complete rewrite though initial research performed at WOODS BAGOT per network_analyse_wb.py script

IMPORTANT

This script is different from centrality_angular.py:
- It compares more distances
- It computes both metric and angular
- It only records information back to dual -> does not average back to primary as this process may introduce artefacts

Used for exploring issues raised in prior analysis using distance and angular methods

NOTE:

Generally the dual values will scale differently than the primary values.
This is because of more node counts and betweenness counts due to larger numbers of nodes on the dual.

"""
import logging
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *
from numba import jit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#@jit(nopython=True, nogil=True)
def centrality_numba(from_idx,  # single value
                     targets,  # len targets
                     distances_m,  # len graph_dual
                     pred_map_m,  # len graph_dual
                     distances_a,  # len targets
                     pred_map_a,  # len graph_dual
                     vert_distances_m,  # len graph_dual
                     # arrays
                     met_betweenness_50,
                     met_betweenness_100,
                     met_betweenness_150,
                     met_betweenness_200,
                     met_betweenness_300,
                     met_betweenness_400,
                     met_betweenness_600,
                     met_betweenness_800,
                     met_betweenness_1200,
                     met_betweenness_1600,
                     ang_betweenness_50,
                     ang_betweenness_100,
                     ang_betweenness_150,
                     ang_betweenness_200,
                     ang_betweenness_300,
                     ang_betweenness_400,
                     ang_betweenness_600,
                     ang_betweenness_800,
                     ang_betweenness_1200,
                     ang_betweenness_1600,
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
                     met_farness_50,
                     met_farness_100,
                     met_farness_150,
                     met_farness_200,
                     met_farness_300,
                     met_farness_400,
                     met_farness_600,
                     met_farness_800,
                     met_farness_1200,
                     met_farness_1600,
                     ang_farness_50,
                     ang_farness_100,
                     ang_farness_150,
                     ang_farness_200,
                     ang_farness_300,
                     ang_farness_400,
                     ang_farness_600,
                     ang_farness_800,
                     ang_farness_1200,
                     ang_farness_1600,
                     ang_farness_m_50,
                     ang_farness_m_100,
                     ang_farness_m_150,
                     ang_farness_m_200,
                     ang_farness_m_300,
                     ang_farness_m_400,
                     ang_farness_m_600,
                     ang_farness_m_800,
                     ang_farness_m_1200,
                     ang_farness_m_1600):

    # unlike the metric centrality measures, the angular implementation only calculates one (directed) way at a time
    for to_idx, dist_a in zip(targets, distances_a):
        if to_idx == from_idx:
            continue
        dist_m = distances_m[to_idx]

        # only generating counts and farness so that various implementations can be experimented with on DB
        if dist_m <= 50:
            node_count_50[from_idx] += 1
            met_farness_50[from_idx] += dist_m
            ang_farness_50[from_idx] += dist_a
        if dist_m <= 100:
            node_count_100[from_idx] += 1
            met_farness_100[from_idx] += dist_m
            ang_farness_100[from_idx] += dist_a
        if dist_m <= 150:
            node_count_150[from_idx] += 1
            met_farness_150[from_idx] += dist_m
            ang_farness_150[from_idx] += dist_a
        if dist_m <= 200:
            node_count_200[from_idx] += 1
            met_farness_200[from_idx] += dist_m
            ang_farness_200[from_idx] += dist_a
        if dist_m <= 300:
            node_count_300[from_idx] += 1
            met_farness_300[from_idx] += dist_m
            ang_farness_300[from_idx] += dist_a
        if dist_m <= 400:
            node_count_400[from_idx] += 1
            met_farness_400[from_idx] += dist_m
            ang_farness_400[from_idx] += dist_a
        if dist_m <= 600:
            node_count_600[from_idx] += 1
            met_farness_600[from_idx] += dist_m
            ang_farness_600[from_idx] += dist_a
        if dist_m <= 800:
            node_count_800[from_idx] += 1
            met_farness_800[from_idx] += dist_m
            ang_farness_800[from_idx] += dist_a
        if dist_m <= 1200:
            node_count_1200[from_idx] += 1
            met_farness_1200[from_idx] += dist_m
            ang_farness_1200[from_idx] += dist_a
        if dist_m <= 1600:
            node_count_1600[from_idx] += 1
            met_farness_1600[from_idx] += dist_m
            ang_farness_1600[from_idx] += dist_a


        # angular betweenness - only counting truly between vertices, not starting and ending verts
        # NOTE -> ang_farness_m (metres) has to be computed manually
        # tested against also manually computed distance farness using same method which gives same as above from graph-tool
        intermediary_idx = pred_map_a[to_idx]
        # start and end vert counts at half each (corresponding to primary edge distance)
        if dist_m <= 50:
            ang_farness_m_50[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_50[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 100:
            ang_farness_m_100[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_100[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 150:
            ang_farness_m_150[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_150[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 200:
            ang_farness_m_200[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_200[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 300:
            ang_farness_m_300[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_300[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 400:
            ang_farness_m_400[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_400[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 600:
            ang_farness_m_600[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_600[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 800:
            ang_farness_m_800[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_800[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 1200:
            ang_farness_m_1200[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_1200[from_idx] += vert_distances_m[to_idx] * 0.5
        if dist_m <= 1600:
            ang_farness_m_1600[from_idx] += vert_distances_m[from_idx] * 0.5
            ang_farness_m_1600[from_idx] += vert_distances_m[to_idx] * 0.5
        while True:
            # break out of while loop if the intermediary is the source node
            # sum distance en route
            if intermediary_idx == from_idx:
                break
            if dist_m <= 50:
                ang_farness_m_50[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_50[intermediary_idx] += 1
            if dist_m <= 100:
                ang_farness_m_100[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_100[intermediary_idx] += 1
            if dist_m <= 150:
                ang_farness_m_150[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_150[intermediary_idx] += 1
            if dist_m <= 200:
                ang_farness_m_200[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_200[intermediary_idx] += 1
            if dist_m <= 300:
                ang_farness_m_300[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_300[intermediary_idx] += 1
            if dist_m <= 400:
                ang_farness_m_400[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_400[intermediary_idx] += 1
            if dist_m <= 600:
                ang_farness_m_600[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_600[intermediary_idx] += 1
            if dist_m <= 800:
                ang_farness_m_800[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_800[intermediary_idx] += 1
            if dist_m <= 1200:
                ang_farness_m_1200[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_1200[intermediary_idx] += 1
            if dist_m <= 1600:
                ang_farness_m_1600[from_idx] += vert_distances_m[intermediary_idx]
                ang_betweenness_1600[intermediary_idx] += 1
            # follow the next intermediary
            intermediary_idx = pred_map_a[intermediary_idx]

        # distance betweenness - only counting truly between vertices, not starting and ending verts
        intermediary_idx = pred_map_m[to_idx]
        while True:
            # break out of while loop if the intermediary is the source node
            if intermediary_idx == from_idx:
                break
            if dist_m <= 50:
                met_betweenness_50[intermediary_idx] += 1
            if dist_m <= 100:
                met_betweenness_100[intermediary_idx] += 1
            if dist_m <= 150:
                met_betweenness_150[intermediary_idx] += 1
            if dist_m <= 200:
                met_betweenness_200[intermediary_idx] += 1
            if dist_m <= 300:
                met_betweenness_300[intermediary_idx] += 1
            if dist_m <= 400:
                met_betweenness_400[intermediary_idx] += 1
            if dist_m <= 600:
                met_betweenness_600[intermediary_idx] += 1
            if dist_m <= 800:
                met_betweenness_800[intermediary_idx] += 1
            if dist_m <= 1200:
                met_betweenness_1200[intermediary_idx] += 1
            if dist_m <= 1600:
                met_betweenness_1600[intermediary_idx] += 1
            # follow the next intermediary
            intermediary_idx = pred_map_m[intermediary_idx]


async def centrality_angular(db_config, network_schema, nodes_table, links_table,
                             city_pop_id=None, boundary_schema=None, boundary_table=None):
    graph_dual = Graph(directed=True)

    # dual vert maps
    gd_vert_id = graph_dual.new_vertex_property('int')
    gd_vert_geom = graph_dual.new_vertex_property('string')
    gd_vert_distance = graph_dual.new_vertex_property('float')

    # dual edge maps for distance and float weights on dual implementation
    gd_edge_distance = graph_dual.new_edge_property('float')
    gd_edge_angle = graph_dual.new_edge_property('float')
    gd_edge_to_vert_map = graph_dual.new_edge_property('int')

    db_con = await asyncpg.connect(**db_config)

    # create indices if they don't exist
    await db_con.execute(f'''
      CREATE INDEX IF NOT EXISTS start_node_idx_{links_table} ON {network_schema}.{links_table} (start_node);
      CREATE INDEX IF NOT EXISTS end_node_idx_{links_table} ON {network_schema}.{links_table} (end_node);
    ''')

    # NOTE:
    # self-loops sometimes occur in the event of lollypop dead-ends or when two lines touch at both ends
    # in these cases one or the other end will effectively be chosen for the link
    # these situations are rare with the 20m network
    # they can be found by using the agg count columns in QGIS - they have irregular agg counts
    query_base = f'''
        SELECT edge_id_a, ST_Length(geom_a) as edge_a_len,
                edge_id_b, ST_Length(geom_b) as edge_b_len,
                vert_common, vert_b_out, ST_Length(geom) as length, angle,
                ST_Force2D(ST_LineInterpolatePoint(geom_a, 0.5)) as midpoint_a,
                ST_Force2D(ST_LineInterpolatePoint(geom_b, 0.5)) as midpoint_b FROM
            -- find all touching edges (see WHERE clause below
            (SELECT id as edge_id_a, start_node, end_node, geom as geom_a FROM {network_schema}.{links_table}) as links_a
            CROSS JOIN LATERAL
            (SELECT id as edge_id_b, start_node, end_node, geom as geom_b FROM {network_schema}.{links_table}) as links_b
            CROSS JOIN LATERAL
            -- use matching node as basis for common and out ids
            (SELECT CASE
                WHEN links_a.start_node = links_b.start_node THEN links_a.start_node
                WHEN links_a.start_node = links_b.end_node THEN links_a.start_node
                WHEN links_a.end_node = links_b.start_node THEN links_a.end_node
                WHEN links_a.end_node = links_b.end_node THEN links_a.end_node
            END as vert_common) as n_c
            CROSS JOIN LATERAL
            (SELECT CASE
              WHEN links_a.start_node = links_b.start_node THEN links_b.end_node
              WHEN links_a.start_node = links_b.end_node THEN links_b.start_node
              WHEN links_a.end_node = links_b.start_node THEN links_b.end_node
              WHEN links_a.end_node = links_b.end_node THEN links_b.start_node
            END as vert_b_out) as n_b_o
            CROSS JOIN LATERAL
            -- create a new line geom for the dual edge
            (SELECT CASE
              WHEN links_a.start_node = links_b.start_node
                THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(geom_a, 0, 0.5), ST_LineSubString(geom_b, 0, 0.5))))
              WHEN links_a.start_node = links_b.end_node
                THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(geom_a, 0, 0.5), ST_LineSubString(geom_b, 0.5, 1))))
              WHEN links_a.end_node = links_b.start_node
                THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(geom_a, 0.5, 1), ST_LineSubString(geom_b, 0, 0.5))))
              WHEN links_a.end_node = links_b.end_node
                THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(geom_a, 0.5, 1), ST_LineSubString(geom_b, 0.5, 1))))
            END as geom) as g
            CROSS JOIN LATERAL
            -- get the angle between the start and end of the line
            (SELECT
               degrees(ST_Azimuth(ST_StartPoint(geom), ST_LineInterpolatePoint(geom, 0.01)))::int as az_a,
               degrees(ST_Azimuth(ST_LineInterpolatePoint(geom, 0.99), ST_EndPoint(geom)))::int as az_b) as az
            CROSS JOIN LATERAL
            -- reduce the angle in cases where computed through the long direction
            (SELECT abs((abs(az_b - az_a) + 180) % 360 - 180) as angle) as a
    '''

    # default query
    if city_pop_id is None:
        logger.info(f'calculating angular centrality for {network_schema}.{nodes_table}')

        query_where = '''
            WHERE edge_id_a != edge_id_b
                -- catch problematic line merges
                AND ST_GeometryType(geom) = 'ST_LineString'
                AND (links_a.start_node = links_b.start_node
                    OR links_a.start_node = links_b.end_node
                    OR links_a.end_node = links_b.start_node
                    OR links_a.end_node = links_b.end_node
            );
          '''

    # override default query if city id provided
    else:
        logger.info(f'calculating angular centrality on {network_schema}.{nodes_table} for city {city_pop_id}')
        # setup temporary buffered boundary based on threshold distance - prevents centrality falloff
        logger.info('NOTE -> creating temporary buffered city geom')
        # convex hull prevents potential issues with multipolygons deriving from buffer...
        temp_id = await db_con.fetchval(f'''
            INSERT INTO {boundary_schema}.{boundary_table} (geom)
                VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, {1600}))
                    FROM {boundary_schema}.{boundary_table} WHERE pop_id = {city_pop_id}))
                RETURNING id;''')
        # override the query
        query_where = f'''
            CROSS JOIN LATERAL
            (SELECT geom as geom_bounds FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}) as bounds
            WHERE edge_id_a != edge_id_b
                -- catch problematic line merges
                AND ST_GeometryType(geom) = 'ST_LineString'
                AND ST_Intersects(geom_bounds, geom_a)
                AND ST_Intersects(geom_bounds, geom_b)
                AND (links_a.start_node = links_b.start_node
                    OR links_a.start_node = links_b.end_node
                    OR links_a.end_node = links_b.start_node
                    OR links_a.end_node = links_b.end_node
                );
        '''

    # open database connection
    vertex_dict = {}
    async with db_con.transaction():
        # get the geometry
        logger.info('NOTE -> fetching links geometry')
        async for record in db_con.cursor(query_base + query_where):

            # 1 - The graph has to be transposed to its dual to embody angular change information
            # 2 - The graph has to be directed to prevent shortcutting
            # 3 - Each node (primary edge) needs to be split so that forward and backward directions are distinct
            # 4 - Each split node accordingly needs to face one of two edge (primary node) directions
            # 5 - The split nodes later need to be consolidated
            # 6 - The dual nodes then need to be condensed back to their corresponding primary nodes

            # this builds "dual" contra-directional
            # the SQL query returns the same new edge_ids twice, once from "A" to "B" and once from "B" to "A"

            # get the primary graph's "a" and "b" edge ids corresponding to the new dual vertices
            primary_edge_id_a = record['edge_id_a']
            primary_edge_id_b = record['edge_id_b']

            # get the primary graphs' common and out vertices corresponding to the new dual edges
            # note that because vert_a corresponds to the in-facing direction, there is no need for its out facing node
            # (this is covered by the opposite-facing split instance in a subsequent SQL record)
            primary_vert_common = record['vert_common']
            primary_vert_b_out = record['vert_b_out']

            # assign the new dual vertex ids consisting of their corresponding primary edge id
            # and their corresponding facing direction
            # the first should face in to the common primary vertex
            dual_start_vert_instance_id = f'{primary_edge_id_a}facing{primary_vert_common}'
            # therefore the second must face away from the primary vertex
            dual_end_vert_instance_id = f'{primary_edge_id_b}facing{primary_vert_b_out}'

            # add new source and target vertices for the first new edge
            # whereas the edge is new, the vertex may already have been added by a neighbouring node, check:
            if dual_start_vert_instance_id not in vertex_dict:
                # if not, then add it
                vertex_dict[dual_start_vert_instance_id] = graph_dual.add_vertex()
                # the id and geom is set from the primary graph's edge id and edge midpoint respectively
                # this is used for outputing the dual graph representation of the data
                gd_vert_id[vertex_dict[dual_start_vert_instance_id]] = primary_edge_id_a
                gd_vert_geom[vertex_dict[dual_start_vert_instance_id]] = record['midpoint_a']
                gd_vert_distance[vertex_dict[dual_start_vert_instance_id]] = record['edge_a_len']

            # likewise for the opposite facing vertex
            if dual_end_vert_instance_id not in vertex_dict:
                vertex_dict[dual_end_vert_instance_id] = graph_dual.add_vertex()
                gd_vert_id[vertex_dict[dual_end_vert_instance_id]] = primary_edge_id_b
                gd_vert_geom[vertex_dict[dual_end_vert_instance_id]] = record['midpoint_b']
                gd_vert_distance[vertex_dict[dual_end_vert_instance_id]] = record['edge_b_len']

            # the directionality can be assumed since this is the first copy of the edge
            # hence, wire up the directional edge, in this case a -> b
            # the b -> a will be wired up later (returned separately in SQL query)
            e = graph_dual.add_edge(vertex_dict[dual_start_vert_instance_id], vertex_dict[dual_end_vert_instance_id])

            # add the corresponding edge data, this is common to either direction
            # this is for mapping data back to the corresponding primary vert
            gd_edge_to_vert_map[e] = primary_vert_common
            gd_edge_distance[e] = record['length']
            gd_edge_angle[e] = record['angle']

    # remove the temporary boundary if city id
    if city_pop_id is not None:
        logger.info('NOTE -> removing temporary buffered city geom')
        await db_con.execute(f'''DELETE FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}''')

    await db_con.close()

    num_verts = graph_dual.num_vertices()
    remove_parallel_edges(graph_dual)
    assert graph_dual.num_vertices() == num_verts
    logger.info(f'edges: {graph_dual.num_edges()}, vertices: {graph_dual.num_vertices()}')

    # arrays for computations
    met_betweenness_50 = np.zeros(graph_dual.num_vertices())
    met_betweenness_100 = np.zeros(graph_dual.num_vertices())
    met_betweenness_150 = np.zeros(graph_dual.num_vertices())
    met_betweenness_200 = np.zeros(graph_dual.num_vertices())
    met_betweenness_300 = np.zeros(graph_dual.num_vertices())
    met_betweenness_400 = np.zeros(graph_dual.num_vertices())
    met_betweenness_600 = np.zeros(graph_dual.num_vertices())
    met_betweenness_800 = np.zeros(graph_dual.num_vertices())
    met_betweenness_1200 = np.zeros(graph_dual.num_vertices())
    met_betweenness_1600 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_50 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_100 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_150 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_200 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_300 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_400 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_600 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_800 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_1200 = np.zeros(graph_dual.num_vertices())
    ang_betweenness_1600 = np.zeros(graph_dual.num_vertices())
    node_count_50 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_100 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_150 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_200 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_300 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_400 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_600 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_800 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_1200 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_1600 = np.zeros(graph_dual.num_vertices(), dtype=int)
    met_farness_50 = np.zeros(graph_dual.num_vertices())
    met_farness_100 = np.zeros(graph_dual.num_vertices())
    met_farness_150 = np.zeros(graph_dual.num_vertices())
    met_farness_200 = np.zeros(graph_dual.num_vertices())
    met_farness_300 = np.zeros(graph_dual.num_vertices())
    met_farness_400 = np.zeros(graph_dual.num_vertices())
    met_farness_600 = np.zeros(graph_dual.num_vertices())
    met_farness_800 = np.zeros(graph_dual.num_vertices())
    met_farness_1200 = np.zeros(graph_dual.num_vertices())
    met_farness_1600 = np.zeros(graph_dual.num_vertices())
    ang_farness_50 = np.zeros(graph_dual.num_vertices())
    ang_farness_100 = np.zeros(graph_dual.num_vertices())
    ang_farness_150 = np.zeros(graph_dual.num_vertices())
    ang_farness_200 = np.zeros(graph_dual.num_vertices())
    ang_farness_300 = np.zeros(graph_dual.num_vertices())
    ang_farness_400 = np.zeros(graph_dual.num_vertices())
    ang_farness_600 = np.zeros(graph_dual.num_vertices())
    ang_farness_800 = np.zeros(graph_dual.num_vertices())
    ang_farness_1200 = np.zeros(graph_dual.num_vertices())
    ang_farness_1600 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_50 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_100 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_150 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_200 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_300 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_400 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_600 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_800 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_1200 = np.zeros(graph_dual.num_vertices())
    ang_farness_m_1600 = np.zeros(graph_dual.num_vertices())

    # because this is a split (parallel) graph, each copy is iterated independently
    for i, v in enumerate(graph_dual.vertices()):

        i += 1
        if i % 1000 == 0:
            completion = round((i / graph_dual.num_vertices()) * 100, 2)
            logger.info(f'processing node: {i}, {completion}% completed')

        # first filter all vertices reachable within max threshold distance
        distance_map_m, pred_map_m = shortest_distance(graph_dual, source=v, weights=gd_edge_distance, max_dist=1600,
                                           directed=True, pred_map=True)

        # set targets manually before performing angular centrality distance
        target_verts = np.where(distance_map_m.a != np.inf)[0]

        # perform centrality path search on angular
        # NOTE -> distance map is returned as array because a specific list of targets is provided
        # The length of the distance map therefore matches the list of targets
        distance_map_a, pred_map_a = shortest_distance(graph_dual, source=v, target=target_verts,
                                            weights=gd_edge_angle, directed=True, pred_map=True)

        # perform centrality using an optimised numba function -> faster than pure python
        centrality_numba(int(v),
                            target_verts,
                            distance_map_m.a,
                            pred_map_m.a,
                            distance_map_a,
                            pred_map_a.a,
                            gd_vert_distance.a,
                            met_betweenness_50,
                            met_betweenness_100,
                            met_betweenness_150,
                            met_betweenness_200,
                            met_betweenness_300,
                            met_betweenness_400,
                            met_betweenness_600,
                            met_betweenness_800,
                            met_betweenness_1200,
                            met_betweenness_1600,
                            ang_betweenness_50,
                            ang_betweenness_100,
                            ang_betweenness_150,
                            ang_betweenness_200,
                            ang_betweenness_300,
                            ang_betweenness_400,
                            ang_betweenness_600,
                            ang_betweenness_800,
                            ang_betweenness_1200,
                            ang_betweenness_1600,
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
                            met_farness_50,
                            met_farness_100,
                            met_farness_150,
                            met_farness_200,
                            met_farness_300,
                            met_farness_400,
                            met_farness_600,
                            met_farness_800,
                            met_farness_1200,
                            met_farness_1600,
                            ang_farness_50,
                            ang_farness_100,
                            ang_farness_150,
                            ang_farness_200,
                            ang_farness_300,
                            ang_farness_400,
                            ang_farness_600,
                            ang_farness_800,
                            ang_farness_1200,
                            ang_farness_1600,
                            ang_farness_m_50,
                            ang_farness_m_100,
                            ang_farness_m_150,
                            ang_farness_m_200,
                            ang_farness_m_300,
                            ang_farness_m_400,
                            ang_farness_m_600,
                            ang_farness_m_800,
                            ang_farness_m_1200,
                            ang_farness_m_1600)

        # AGGREGATE THE SPLIT NODES AND WRITE NEW DUAL GRAPH TO DB
    logger.info('Aggregating data')
    # ADD DUAL GRAPH LINKS TABLE
    db_con = await asyncpg.connect(**db_config)

    await db_con.execute(f'''
    -- add dual links table
    CREATE TABLE IF NOT EXISTS {network_schema}.{links_table}_dual (
      id text PRIMARY KEY,
      parent_primary_node bigint,
      start_node bigint,
      end_node bigint,
      angle real,
      distance real,
      geom geometry(Linestring, 27700)
    );
    CREATE INDEX IF NOT EXISTS geom_idx_{links_table}_dual ON {network_schema}.{links_table}_dual USING GIST (geom);
    ''')

    dual_edge_ids = set()
    dual_edge_data = []
    for e in graph_dual.edges():
        # use the primary common vertex id for the new edge id
        parent_primary_node = gd_edge_to_vert_map[e]
        # get the source and target verts
        source_v = e.source()
        source_v_id = gd_vert_id[source_v]
        target_v = e.target()
        target_v_id = gd_vert_id[target_v]
        # sort these so that you can find the duplicates (one directional edge in each direction)
        source, target = sorted([source_v_id, target_v_id])
        new_edge_id = f'{parent_primary_node}_{source}_{target}'
        # add only the first copy to the results (don't add duplicate since directionality not required)
        if new_edge_id in dual_edge_ids:
            continue
        # otherwise add it to the set
        dual_edge_ids.add(new_edge_id)
        source_v_geom = gd_vert_geom[source_v]
        target_v_geom = gd_vert_geom[target_v]
        # add to the result set
        dual_edge_data.append([new_edge_id, parent_primary_node, source, target, gd_edge_angle[e],
                               gd_edge_distance[e], source_v_geom, target_v_geom])

    await db_con.executemany(f'''
        INSERT INTO {network_schema}.{links_table}_dual (
            id,
            parent_primary_node,
            start_node,
            end_node,
            angle,
            distance,
            geom)
        VALUES ($1, $2, $3, $4, $5, $6, ST_MakeLine(ST_SetSRID($7::geometry, 27700), ST_SetSRID($8::geometry, 27700)))
        ON CONFLICT (id) DO UPDATE SET
            parent_primary_node = $2,
            start_node = $3,
            end_node = $4,
            angle = $5,
            distance = $6,
            geom = ST_MakeLine(ST_SetSRID($7::geometry, 27700), ST_SetSRID($8::geometry, 27700));
    ''', dual_edge_data)

    # ADD DUAL GRAPH'S NODES TABLE
    await db_con.execute(f'''
        -- add dual nodes table
        CREATE TABLE IF NOT EXISTS {network_schema}.{nodes_table}_dual (
            id bigint PRIMARY KEY,
            geom geometry(Point, 27700),
            city_pop_id int
        );
        CREATE INDEX IF NOT EXISTS geom_idx_{nodes_table}_dual ON {network_schema}.{nodes_table}_dual USING GIST (geom);
        ALTER TABLE {network_schema}.{nodes_table}_dual
          -- add the new columns for writing the results
          ADD COLUMN IF NOT EXISTS comp_agg_count_verify int,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_50 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_100 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_150 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_200 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_300 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_400 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_600 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_800 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_1200 real,
          ADD COLUMN IF NOT EXISTS comp_met_betweenness_1600 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_50 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_100 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_150 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_200 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_300 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_400 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_600 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_800 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_1200 real,
          ADD COLUMN IF NOT EXISTS comp_ang_betweenness_1600 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_50 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_100 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_150 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_200 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_300 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_400 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_600 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_800 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_1200 real,
          ADD COLUMN IF NOT EXISTS comp_node_count_1600 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_50 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_100 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_150 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_200 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_300 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_400 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_600 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_800 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_1200 real,
          ADD COLUMN IF NOT EXISTS comp_met_farness_1600 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_50 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_100 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_150 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_200 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_300 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_400 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_600 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_800 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_1200 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_1600 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_50 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_100 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_150 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_200 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_300 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_400 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_600 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_800 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_1200 real,
          ADD COLUMN IF NOT EXISTS comp_ang_farness_m_1600 real;
        ''')

    dual_result_dict = {}
    for v in graph_dual.vertices():
        # consolidate the results from the split verts to a single value
        # the matching happens on the original primary graph's edge id
        primary_edge_id = gd_vert_id[v]
        # proceed with the consolidation of the dual graph
        # these counts will be roughly double those of metric betweenness measures which single count instead (directed)
        v_int = int(v)
        if primary_edge_id not in dual_result_dict:
            dual_result_dict[primary_edge_id] = {
                'count': 1,
                'met_betweenness_50': met_betweenness_50[v_int],
                'met_betweenness_100': met_betweenness_100[v_int],
                'met_betweenness_150': met_betweenness_150[v_int],
                'met_betweenness_200': met_betweenness_200[v_int],
                'met_betweenness_300': met_betweenness_300[v_int],
                'met_betweenness_400': met_betweenness_400[v_int],
                'met_betweenness_600': met_betweenness_600[v_int],
                'met_betweenness_800': met_betweenness_800[v_int],
                'met_betweenness_1200': met_betweenness_1200[v_int],
                'met_betweenness_1600': met_betweenness_1600[v_int],
                'ang_betweenness_50': ang_betweenness_50[v_int],
                'ang_betweenness_100': ang_betweenness_100[v_int],
                'ang_betweenness_150': ang_betweenness_150[v_int],
                'ang_betweenness_200': ang_betweenness_200[v_int],
                'ang_betweenness_300': ang_betweenness_300[v_int],
                'ang_betweenness_400': ang_betweenness_400[v_int],
                'ang_betweenness_600': ang_betweenness_600[v_int],
                'ang_betweenness_800': ang_betweenness_800[v_int],
                'ang_betweenness_1200': ang_betweenness_1200[v_int],
                'ang_betweenness_1600': ang_betweenness_1600[v_int],
                'node_count_50': node_count_50[v_int],
                'node_count_100': node_count_100[v_int],
                'node_count_150': node_count_150[v_int],
                'node_count_200': node_count_200[v_int],
                'node_count_300': node_count_300[v_int],
                'node_count_400': node_count_400[v_int],
                'node_count_600': node_count_600[v_int],
                'node_count_800': node_count_800[v_int],
                'node_count_1200': node_count_1200[v_int],
                'node_count_1600': node_count_1600[v_int],
                'met_farness_50': met_farness_50[v_int],
                'met_farness_100': met_farness_100[v_int],
                'met_farness_150': met_farness_150[v_int],
                'met_farness_200': met_farness_200[v_int],
                'met_farness_300': met_farness_300[v_int],
                'met_farness_400': met_farness_400[v_int],
                'met_farness_600': met_farness_600[v_int],
                'met_farness_800': met_farness_800[v_int],
                'met_farness_1200': met_farness_1200[v_int],
                'met_farness_1600': met_farness_1600[v_int],
                'ang_farness_50': ang_farness_50[v_int],
                'ang_farness_100': ang_farness_100[v_int],
                'ang_farness_150': ang_farness_150[v_int],
                'ang_farness_200': ang_farness_200[v_int],
                'ang_farness_300': ang_farness_300[v_int],
                'ang_farness_400': ang_farness_400[v_int],
                'ang_farness_600': ang_farness_600[v_int],
                'ang_farness_800': ang_farness_800[v_int],
                'ang_farness_1200': ang_farness_1200[v_int],
                'ang_farness_1600': ang_farness_1600[v_int],
                'ang_farness_m_50': ang_farness_m_50[v_int],
                'ang_farness_m_100': ang_farness_m_100[v_int],
                'ang_farness_m_150': ang_farness_m_150[v_int],
                'ang_farness_m_200': ang_farness_m_200[v_int],
                'ang_farness_m_300': ang_farness_m_300[v_int],
                'ang_farness_m_400': ang_farness_m_400[v_int],
                'ang_farness_m_600': ang_farness_m_600[v_int],
                'ang_farness_m_800': ang_farness_m_800[v_int],
                'ang_farness_m_1200': ang_farness_m_1200[v_int],
                'ang_farness_m_1600': ang_farness_m_1600[v_int],
                'geom': gd_vert_geom[v]
            }
        else:
            dual_result_dict[primary_edge_id]['count'] += 1
            dual_result_dict[primary_edge_id]['met_betweenness_50'] += met_betweenness_50[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_100'] += met_betweenness_100[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_150'] += met_betweenness_150[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_200'] += met_betweenness_200[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_300'] += met_betweenness_300[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_400'] += met_betweenness_400[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_600'] += met_betweenness_600[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_800'] += met_betweenness_800[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_1200'] += met_betweenness_1200[v_int]
            dual_result_dict[primary_edge_id]['met_betweenness_1600'] += met_betweenness_1600[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_50'] += ang_betweenness_50[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_100'] += ang_betweenness_100[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_150'] += ang_betweenness_150[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_200'] += ang_betweenness_200[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_300'] += ang_betweenness_300[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_400'] += ang_betweenness_400[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_600'] += ang_betweenness_600[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_800'] += ang_betweenness_800[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_1200'] += ang_betweenness_1200[v_int]
            dual_result_dict[primary_edge_id]['ang_betweenness_1600'] += ang_betweenness_1600[v_int]
            dual_result_dict[primary_edge_id]['node_count_50'] += node_count_50[v_int]
            dual_result_dict[primary_edge_id]['node_count_100'] += node_count_100[v_int]
            dual_result_dict[primary_edge_id]['node_count_150'] += node_count_150[v_int]
            dual_result_dict[primary_edge_id]['node_count_200'] += node_count_200[v_int]
            dual_result_dict[primary_edge_id]['node_count_300'] += node_count_300[v_int]
            dual_result_dict[primary_edge_id]['node_count_400'] += node_count_400[v_int]
            dual_result_dict[primary_edge_id]['node_count_600'] += node_count_600[v_int]
            dual_result_dict[primary_edge_id]['node_count_800'] += node_count_800[v_int]
            dual_result_dict[primary_edge_id]['node_count_1200'] += node_count_1200[v_int]
            dual_result_dict[primary_edge_id]['node_count_1600'] += node_count_1600[v_int]
            dual_result_dict[primary_edge_id]['met_farness_50'] += met_farness_50[v_int]
            dual_result_dict[primary_edge_id]['met_farness_100'] += met_farness_100[v_int]
            dual_result_dict[primary_edge_id]['met_farness_150'] += met_farness_150[v_int]
            dual_result_dict[primary_edge_id]['met_farness_200'] += met_farness_200[v_int]
            dual_result_dict[primary_edge_id]['met_farness_300'] += met_farness_300[v_int]
            dual_result_dict[primary_edge_id]['met_farness_400'] += met_farness_400[v_int]
            dual_result_dict[primary_edge_id]['met_farness_600'] += met_farness_600[v_int]
            dual_result_dict[primary_edge_id]['met_farness_800'] += met_farness_800[v_int]
            dual_result_dict[primary_edge_id]['met_farness_1200'] += met_farness_1200[v_int]
            dual_result_dict[primary_edge_id]['met_farness_1600'] += met_farness_1600[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_50'] += ang_farness_50[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_100'] += ang_farness_100[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_150'] += ang_farness_150[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_200'] += ang_farness_200[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_300'] += ang_farness_300[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_400'] += ang_farness_400[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_600'] += ang_farness_600[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_800'] += ang_farness_800[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_1200'] += ang_farness_1200[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_1600'] += ang_farness_1600[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_50'] += ang_farness_m_50[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_100'] += ang_farness_m_100[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_150'] += ang_farness_m_150[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_200'] += ang_farness_m_200[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_300'] += ang_farness_m_300[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_400'] += ang_farness_m_400[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_600'] += ang_farness_m_600[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_800'] += ang_farness_m_800[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_1200'] += ang_farness_m_1200[v_int]
            dual_result_dict[primary_edge_id]['ang_farness_m_1600'] += ang_farness_m_1600[v_int]
            # as with the id, the geom is the same for both

    # NOTE -> these results are typically double of those on the primary graph
    # this is because each edge was split into two contra-directional dual-graph vertices
    # which have now been consolidated down to one dual-graph vertex
    dual_node_results = []
    # TODO: Should these values be divided by the counts?
    for k, v in dual_result_dict.items():
        dual_node_results.append([k, v['count'], v['geom'],
            v['met_betweenness_50'],
            v['met_betweenness_100'],
            v['met_betweenness_150'],
            v['met_betweenness_200'],
            v['met_betweenness_300'],
            v['met_betweenness_400'],
            v['met_betweenness_600'],
            v['met_betweenness_800'],
            v['met_betweenness_1200'],
            v['met_betweenness_1600'],
            v['ang_betweenness_50'],
            v['ang_betweenness_100'],
            v['ang_betweenness_150'],
            v['ang_betweenness_200'],
            v['ang_betweenness_300'],
            v['ang_betweenness_400'],
            v['ang_betweenness_600'],
            v['ang_betweenness_800'],
            v['ang_betweenness_1200'],
            v['ang_betweenness_1600'],
            v['node_count_50'],
            v['node_count_100'],
            v['node_count_150'],
            v['node_count_200'],
            v['node_count_300'],
            v['node_count_400'],
            v['node_count_600'],
            v['node_count_800'],
            v['node_count_1200'],
            v['node_count_1600'],
            v['met_farness_50'],
            v['met_farness_100'],
            v['met_farness_150'],
            v['met_farness_200'],
            v['met_farness_300'],
            v['met_farness_400'],
            v['met_farness_600'],
            v['met_farness_800'],
            v['met_farness_1200'],
            v['met_farness_1600'],
            v['ang_farness_50'],
            v['ang_farness_100'],
            v['ang_farness_150'],
            v['ang_farness_200'],
            v['ang_farness_300'],
            v['ang_farness_400'],
            v['ang_farness_600'],
            v['ang_farness_800'],
            v['ang_farness_1200'],
            v['ang_farness_1600'],
            v['ang_farness_m_50'],
            v['ang_farness_m_100'],
            v['ang_farness_m_150'],
            v['ang_farness_m_200'],
            v['ang_farness_m_300'],
            v['ang_farness_m_400'],
            v['ang_farness_m_600'],
            v['ang_farness_m_800'],
            v['ang_farness_m_1200'],
            v['ang_farness_m_1600'],
            city_pop_id])

    logger.info('Writing data to DB')
    await db_con.executemany(f'''
        INSERT INTO {network_schema}.{nodes_table}_dual (
                id,
                comp_agg_count_verify,
                geom,
                comp_met_betweenness_50,
                comp_met_betweenness_100,
                comp_met_betweenness_150,
                comp_met_betweenness_200,
                comp_met_betweenness_300,
                comp_met_betweenness_400,
                comp_met_betweenness_600,
                comp_met_betweenness_800,
                comp_met_betweenness_1200,
                comp_met_betweenness_1600,
                comp_ang_betweenness_50,
                comp_ang_betweenness_100,
                comp_ang_betweenness_150,
                comp_ang_betweenness_200,
                comp_ang_betweenness_300,
                comp_ang_betweenness_400,
                comp_ang_betweenness_600,
                comp_ang_betweenness_800,
                comp_ang_betweenness_1200,
                comp_ang_betweenness_1600,
                comp_node_count_50,
                comp_node_count_100,
                comp_node_count_150,
                comp_node_count_200,
                comp_node_count_300,
                comp_node_count_400,
                comp_node_count_600,
                comp_node_count_800,
                comp_node_count_1200,
                comp_node_count_1600,
                comp_met_farness_50,
                comp_met_farness_100,
                comp_met_farness_150,
                comp_met_farness_200,
                comp_met_farness_300,
                comp_met_farness_400,
                comp_met_farness_600,
                comp_met_farness_800,
                comp_met_farness_1200,
                comp_met_farness_1600,
                comp_ang_farness_50,
                comp_ang_farness_100,
                comp_ang_farness_150,
                comp_ang_farness_200,
                comp_ang_farness_300,
                comp_ang_farness_400,
                comp_ang_farness_600,
                comp_ang_farness_800,
                comp_ang_farness_1200,
                comp_ang_farness_1600,
                comp_ang_farness_m_50,
                comp_ang_farness_m_100,
                comp_ang_farness_m_150,
                comp_ang_farness_m_200,
                comp_ang_farness_m_300,
                comp_ang_farness_m_400,
                comp_ang_farness_m_600,
                comp_ang_farness_m_800,
                comp_ang_farness_m_1200,
                comp_ang_farness_m_1600,
                city_pop_id)
            VALUES ($1, $2, ST_SetSRID($3::geometry, 27700),
                $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
                $24, $25, $26, $27, $28, $29, $30, $31, $32, $33,
                $34, $35, $36, $37, $38, $39, $40, $41, $42, $43,
                $44, $45, $46, $47, $48, $49, $50, $51, $52, $53,
                $54, $55, $56, $57, $58, $59, $60, $61, $62, $63,
                $64)
            ON CONFLICT (id) DO UPDATE SET
                comp_agg_count_verify = $2,
                geom = ST_SetSRID($3::geometry, 27700),
                comp_met_betweenness_50 = $4,
                comp_met_betweenness_100 = $5,
                comp_met_betweenness_150 = $6,
                comp_met_betweenness_200 = $7,
                comp_met_betweenness_300 = $8,
                comp_met_betweenness_400 = $9,
                comp_met_betweenness_600 = $10,
                comp_met_betweenness_800 = $11,
                comp_met_betweenness_1200 = $12,
                comp_met_betweenness_1600 = $13,
                comp_ang_betweenness_50 = $14,
                comp_ang_betweenness_100 = $15,
                comp_ang_betweenness_150 = $16,
                comp_ang_betweenness_200 = $17,
                comp_ang_betweenness_300 = $18,
                comp_ang_betweenness_400 = $19,
                comp_ang_betweenness_600 = $20,
                comp_ang_betweenness_800 = $21,
                comp_ang_betweenness_1200 = $22,
                comp_ang_betweenness_1600 = $23,
                comp_node_count_50 = $24,
                comp_node_count_100 = $25,
                comp_node_count_150 = $26,
                comp_node_count_200 = $27,
                comp_node_count_300 = $28,
                comp_node_count_400 = $29,
                comp_node_count_600 = $30,
                comp_node_count_800 = $31,
                comp_node_count_1200 = $32,
                comp_node_count_1600 = $33,
                comp_met_farness_50 = $34,
                comp_met_farness_100 = $35,
                comp_met_farness_150 = $36,
                comp_met_farness_200 = $37,
                comp_met_farness_300 = $38,
                comp_met_farness_400 = $39,
                comp_met_farness_600 = $40,
                comp_met_farness_800 = $41,
                comp_met_farness_1200 = $42,
                comp_met_farness_1600 = $43,
                comp_ang_farness_50 = $44,
                comp_ang_farness_100 = $45,
                comp_ang_farness_150 = $46,
                comp_ang_farness_200 = $47,
                comp_ang_farness_300 = $48,
                comp_ang_farness_400 = $49,
                comp_ang_farness_600 = $50,
                comp_ang_farness_800 = $51,
                comp_ang_farness_1200 = $52,
                comp_ang_farness_1600 = $53,
                comp_ang_farness_m_50 = $54,
                comp_ang_farness_m_100 = $55,
                comp_ang_farness_m_150 = $56,
                comp_ang_farness_m_200 = $57,
                comp_ang_farness_m_300 = $58,
                comp_ang_farness_m_400 = $59,
                comp_ang_farness_m_600 = $60,
                comp_ang_farness_m_800 = $61,
                comp_ang_farness_m_1200 = $62,
                comp_ang_farness_m_1600 = $63,
                city_pop_id = $64;
    ''', dual_node_results)

    # calculate the closeness and ratio columns
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{nodes_table}_dual
            ADD COLUMN IF NOT EXISTS comp_ratio_50 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_100 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_150 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_200 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_300 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_400 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_600 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_800 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_1200 real,
            ADD COLUMN IF NOT EXISTS comp_ratio_1600 real;
          
        UPDATE {network_schema}.{nodes_table}_dual
            SET comp_ratio_50 = CASE comp_met_farness_50
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_50 / comp_met_farness_50)::numeric, 2)
                END,
            comp_ratio_100 = CASE comp_met_farness_100
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_100 / comp_met_farness_100)::numeric, 2)
                END,
            comp_ratio_150 = CASE comp_met_farness_150
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_150 / comp_met_farness_150)::numeric, 2)
                END,
            comp_ratio_200 = CASE comp_met_farness_200
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_200 / comp_met_farness_200)::numeric, 2)
                END,
              comp_ratio_300 = CASE comp_met_farness_300
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_300 / comp_met_farness_300)::numeric, 2)
                END,
              comp_ratio_400 = CASE comp_met_farness_400
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_400 / comp_met_farness_400)::numeric, 2)
                END,
              comp_ratio_600 = CASE comp_met_farness_600
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_600 / comp_met_farness_600)::numeric, 2)
                END,
              comp_ratio_800 = CASE comp_met_farness_800
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_800 / comp_met_farness_800)::numeric, 2)
                END,
              comp_ratio_1200 = CASE comp_met_farness_1200
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_1200 / comp_met_farness_1200)::numeric, 2)
                END,
              comp_ratio_1600 = CASE comp_met_farness_1600
                    WHEN 0 THEN NULL
                    ELSE round((comp_ang_farness_m_1600 / comp_met_farness_1600)::numeric, 2)
                END;
                ''')

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
    # loop.run_until_complete(centrality(db_config, network_schema, nodes_table, links_table))

    boundary_schema = 'analysis'
    boundary_table = 'city_boundaries_150'

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    nodes_table = 'roadnodes_100'
    links_table = 'roadlinks_100'
    for city_pop_id in range(100, 101):
        logger.info(f'calculating angular centralities for city {city_pop_id} derived from {nodes_table}')
        loop.run_until_complete(
            centrality_angular(db_config, network_schema, nodes_table, links_table,
                               city_pop_id, boundary_schema, boundary_table))
        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Started {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
