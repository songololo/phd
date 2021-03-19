"""
ANGULAR IMPLEMENTATION - DEPRECATED

Complete rewrite though initial research performed at WOODS BAGOT per network_analyse_wb.py script

NOTE: Use centrality_angular_compare.py for more detailed comparison.

This version also writes back to primary, though this is potentially problematic as artefacts and distortions could be introduced through the averaging process, making it questionable to compare betweenness and angular measures.

i.e. compare them directly to each other on the dual graph using the newer version of this script.

"""
import logging
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *
from numba import jit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@jit(nopython=True, nogil=True)
def centrality_numba(from_idx, targets, distances_m, distances_a, pred_map,
                     betweenness_200,
                     betweenness_400,
                     betweenness_800,
                     betweenness_1600,
                     node_count_200,
                     node_count_400,
                     node_count_800,
                     node_count_1600,
                     farness_200,
                     farness_400,
                     farness_800,
                     farness_1600):
    # unlike the metric centrality measures, the angular implementation only calculates one (directed) way at a time
    for to_idx, dist_a in zip(targets, distances_a):
        to_dist_m = distances_m[to_idx]
        # only generating counts and farness so that various implementations can be experimented with on DB
        if to_dist_m <= 200:
            node_count_200[from_idx] += 1
            farness_200[from_idx] += dist_a
        if to_dist_m <= 400:
            node_count_400[from_idx] += 1
            farness_400[from_idx] += dist_a
        if to_dist_m <= 800:
            node_count_800[from_idx] += 1
            farness_800[from_idx] += dist_a
        if to_dist_m <= 1600:
            node_count_1600[from_idx] += 1
            farness_1600[from_idx] += dist_a
        # betweenness - only counting truly between vertices, not starting and ending verts
        intermediary_idx = pred_map[to_idx]
        while True:
            # break out of while loop if the intermediary is the source node
            if intermediary_idx == from_idx:
                break
            if to_dist_m <= 200:
                betweenness_200[intermediary_idx] += 1
            if to_dist_m <= 400:
                betweenness_400[intermediary_idx] += 1
            if to_dist_m <= 800:
                betweenness_800[intermediary_idx] += 1
            if to_dist_m <= 1600:
                betweenness_1600[intermediary_idx] += 1
            # follow the next intermediary
            intermediary_idx = pred_map[intermediary_idx]


async def centrality_angular(db_config, network_schema, nodes_table, links_table,
                             city_pop_id=None, boundary_schema=None, boundary_table=None):
    graph_dual = Graph(directed=True)

    # dual vert maps
    gd_vert_id = graph_dual.new_vertex_property('int')
    gd_vert_geom = graph_dual.new_vertex_property('string')

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
        SELECT edge_id_a, edge_id_b, vert_common, vert_b_out, ST_Length(geom) as length, angle,
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

            # likewise for the opposite facing vertex
            if dual_end_vert_instance_id not in vertex_dict:
                vertex_dict[dual_end_vert_instance_id] = graph_dual.add_vertex()
                gd_vert_id[vertex_dict[dual_end_vert_instance_id]] = primary_edge_id_b
                gd_vert_geom[vertex_dict[dual_end_vert_instance_id]] = record['midpoint_b']

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
    betweenness_200 = np.zeros(graph_dual.num_vertices())
    betweenness_400 = np.zeros(graph_dual.num_vertices())
    betweenness_800 = np.zeros(graph_dual.num_vertices())
    betweenness_1600 = np.zeros(graph_dual.num_vertices())
    node_count_200 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_400 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_800 = np.zeros(graph_dual.num_vertices(), dtype=int)
    node_count_1600 = np.zeros(graph_dual.num_vertices(), dtype=int)
    farness_200 = np.zeros(graph_dual.num_vertices())
    farness_400 = np.zeros(graph_dual.num_vertices())
    farness_800 = np.zeros(graph_dual.num_vertices())
    farness_1600 = np.zeros(graph_dual.num_vertices())

    # because this is a split (parallel) graph, each copy is iterated independently
    for i, v in enumerate(graph_dual.vertices()):

        i += 1
        if i % 1000 == 0:
            completion = round((i / graph_dual.num_vertices()) * 100, 2)
            logger.info(f'processing node: {i}, {completion}% completed')

        # first filter all vertices reachable within max threshold distance
        distance_map_m = shortest_distance(graph_dual, source=v, weights=gd_edge_distance, max_dist=1600)

        # set targets manually before performing angular centrality distance
        target_verts = np.where(distance_map_m.a != np.inf)[0]

        # perform centrality path search on angular
        # NOTE -> distance map is returned as array because a specific list of targets is provided
        # The length of the distance map therefore matches the list of targets
        distance_map_a, pred_map = shortest_distance(graph_dual, source=v, target=target_verts,
                                                     weights=gd_edge_angle, pred_map=True)

        # perform centrality using an optimised numba function -> much faster than pure python
        centrality_numba(int(v), target_verts, distance_map_m.a, distance_map_a, pred_map.a,
                         betweenness_200,
                         betweenness_400,
                         betweenness_800,
                         betweenness_1600,
                         node_count_200,
                         node_count_400,
                         node_count_800,
                         node_count_1600,
                         farness_200,
                         farness_400,
                         farness_800,
                         farness_1600)

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
            geom geometry(Point, 27700)
        );
        CREATE INDEX IF NOT EXISTS geom_idx_{nodes_table}_dual ON {network_schema}.{nodes_table}_dual USING GIST (geom);
        ALTER TABLE {network_schema}.{nodes_table}_dual
          -- add the new columns for writing the results
          ADD COLUMN IF NOT EXISTS ang_agg_count_verify int,
          ADD COLUMN IF NOT EXISTS ang_betw_200 real,
          ADD COLUMN IF NOT EXISTS ang_betw_400 real,
          ADD COLUMN IF NOT EXISTS ang_betw_800 real,
          ADD COLUMN IF NOT EXISTS ang_betw_1600 real,
          ADD COLUMN IF NOT EXISTS ang_node_count_200 real,
          ADD COLUMN IF NOT EXISTS ang_node_count_400 real,
          ADD COLUMN IF NOT EXISTS ang_node_count_800 real,
          ADD COLUMN IF NOT EXISTS ang_node_count_1600 real,
          ADD COLUMN IF NOT EXISTS ang_farness_200 real,
          ADD COLUMN IF NOT EXISTS ang_farness_400 real,
          ADD COLUMN IF NOT EXISTS ang_farness_800 real,
          ADD COLUMN IF NOT EXISTS ang_farness_1600 real;
        ''')

    dual_primary_map = {}
    dual_result_dict = {}
    for v in graph_dual.vertices():
        # consolidate the results from the split verts to a single value
        # the matching happens on the primary edge id
        primary_edge_id = gd_vert_id[v]
        # map the primary edge id to the primary vert ids
        # this is used for averaging adjacent dual vert values back onto the primary vert (at a later stage)
        for out_edge in v.out_edges():
            primary_vert_id = gd_edge_to_vert_map[out_edge]
            if primary_vert_id not in dual_primary_map:
                dual_primary_map[primary_vert_id] = [primary_edge_id]
            # only one copy of each needs to be added because this serves as a reference to
            # dual_result_dict which will already have been consolidated...
            elif primary_edge_id not in dual_primary_map[primary_vert_id]:
                dual_primary_map[primary_vert_id].append(primary_edge_id)
        # proceed with the consolidation of the dual graph
        # these counts will be roughly double those of metric betweenness measures which single count instead (directed)
        v_int = int(v)
        if primary_edge_id not in dual_result_dict:
            dual_result_dict[primary_edge_id] = {
                'count': 1,
                'betw_200': betweenness_200[v_int],
                'betw_400': betweenness_400[v_int],
                'betw_800': betweenness_800[v_int],
                'betw_1600': betweenness_1600[v_int],
                'node_count_200': node_count_200[v_int],
                'node_count_400': node_count_400[v_int],
                'node_count_800': node_count_800[v_int],
                'node_count_1600': node_count_1600[v_int],
                'farness_200': farness_200[v_int],
                'farness_400': farness_400[v_int],
                'farness_800': farness_800[v_int],
                'farness_1600': farness_1600[v_int],
                'geom': gd_vert_geom[v]
            }
        else:
            dual_result_dict[primary_edge_id]['count'] += 1
            dual_result_dict[primary_edge_id]['betw_200'] += betweenness_200[v_int]
            dual_result_dict[primary_edge_id]['betw_400'] += betweenness_400[v_int]
            dual_result_dict[primary_edge_id]['betw_800'] += betweenness_800[v_int]
            dual_result_dict[primary_edge_id]['betw_1600'] += betweenness_1600[v_int]
            dual_result_dict[primary_edge_id]['node_count_200'] += node_count_200[v_int]
            dual_result_dict[primary_edge_id]['node_count_400'] += node_count_400[v_int]
            dual_result_dict[primary_edge_id]['node_count_800'] += node_count_800[v_int]
            dual_result_dict[primary_edge_id]['node_count_1600'] += node_count_1600[v_int]
            dual_result_dict[primary_edge_id]['farness_200'] += farness_200[v_int]
            dual_result_dict[primary_edge_id]['farness_400'] += farness_400[v_int]
            dual_result_dict[primary_edge_id]['farness_800'] += farness_800[v_int]
            dual_result_dict[primary_edge_id]['farness_1600'] += farness_1600[v_int]
            # as with the id, the geom is the same for both

    dual_node_results = []
    for k, v in dual_result_dict.items():
        dual_node_results.append([k, v['count'], v['geom'],
            v['betw_200'],
            v['betw_400'],
            v['betw_800'],
            v['betw_1600'],
            v['node_count_200'],
            v['node_count_400'],
            v['node_count_800'],
            v['node_count_1600'],
            v['farness_200'],
            v['farness_400'],
            v['farness_800'],
            v['farness_1600']])

    await db_con.executemany(f'''
        INSERT INTO {network_schema}.{nodes_table}_dual (
                id,
                ang_agg_count_verify,
                geom,
                ang_betw_200,
                ang_betw_400,
                ang_betw_800,
                ang_betw_1600,
                ang_node_count_200,
                ang_node_count_400,
                ang_node_count_800,
                ang_node_count_1600,
                ang_farness_200,
                ang_farness_400,
                ang_farness_800,
                ang_farness_1600)
            VALUES ($1, $2, ST_SetSRID($3::geometry, 27700), $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (id) DO UPDATE SET
                ang_agg_count_verify = $2,
                geom = ST_SetSRID($3::geometry, 27700),
                ang_betw_200 = $4,
                ang_betw_400 = $5,
                ang_betw_800 = $6,
                ang_betw_1600 = $7,
                ang_node_count_200 = $8,
                ang_node_count_400 = $9,
                ang_node_count_800 = $10,
                ang_node_count_1600 = $11,
                ang_farness_200 = $12,
                ang_farness_400 = $13,
                ang_farness_800 = $14,
                ang_farness_1600 = $15;
    ''', dual_node_results)

    # AVERAGE THE DUAL VERTICES BACK TO THE ADJACENT PRIMARY VERTICES
    primary_data = []
    for primary_vert_id, primary_edge_ids in dual_primary_map.items():
        count = 0
        betw_200 = 0
        betw_400 = 0
        betw_800 = 0
        betw_1600 = 0
        node_count_200 = 0
        node_count_400 = 0
        node_count_800 = 0
        node_count_1600 = 0
        farness_200 = 0
        farness_400 = 0
        farness_800 = 0
        farness_1600 = 0
        for primary_edge_id in primary_edge_ids:
            count += 1
            betw_200 += dual_result_dict[primary_edge_id]['betw_200']
            betw_400 += dual_result_dict[primary_edge_id]['betw_400']
            betw_800 += dual_result_dict[primary_edge_id]['betw_800']
            betw_1600 += dual_result_dict[primary_edge_id]['betw_1600']
            node_count_200 += dual_result_dict[primary_edge_id]['node_count_200']
            node_count_400 += dual_result_dict[primary_edge_id]['node_count_400']
            node_count_800 += dual_result_dict[primary_edge_id]['node_count_800']
            node_count_1600 += dual_result_dict[primary_edge_id]['node_count_1600']
            farness_200 += dual_result_dict[primary_edge_id]['farness_200']
            farness_400 += dual_result_dict[primary_edge_id]['farness_400']
            farness_800 += dual_result_dict[primary_edge_id]['farness_800']
            farness_1600 += dual_result_dict[primary_edge_id]['farness_1600']
        # dual graphs leave dangling dead-ends, catch these and fill data with nans
        if not count:
            dual_node_results.append([primary_vert_id, count] + [np.nan] * 12)
        else:
            primary_data.append((
                primary_vert_id,
                count,
                betw_200 / count,
                betw_400 / count,
                betw_800 / count,
                betw_1600 / count,
                node_count_200 / count,
                node_count_400 / count,
                node_count_800 / count,
                node_count_1600 / count,
                farness_200 / count,
                farness_400 / count,
                farness_800 / count,
                farness_1600 / count))

    logger.info('Writing data back to database')

    db_con = await asyncpg.connect(**db_config)
    await db_con.execute(f'''
            ALTER TABLE {network_schema}.{nodes_table}
                ADD COLUMN IF NOT EXISTS ang_agg_count_verify int,
                ADD COLUMN IF NOT EXISTS ang_betw_200 real,
                ADD COLUMN IF NOT EXISTS ang_betw_400 real,
                ADD COLUMN IF NOT EXISTS ang_betw_800 real,
                ADD COLUMN IF NOT EXISTS ang_betw_1600 real,
                ADD COLUMN IF NOT EXISTS ang_node_count_200 real,
                ADD COLUMN IF NOT EXISTS ang_node_count_400 real,
                ADD COLUMN IF NOT EXISTS ang_node_count_800 real,
                ADD COLUMN IF NOT EXISTS ang_node_count_1600 real,
                ADD COLUMN IF NOT EXISTS ang_farness_200 real,
                ADD COLUMN IF NOT EXISTS ang_farness_400 real,
                ADD COLUMN IF NOT EXISTS ang_farness_800 real,
                ADD COLUMN IF NOT EXISTS ang_farness_1600 real;
        ''')
    await db_con.executemany(f'''
            UPDATE {network_schema}.{nodes_table}
                SET 
                    ang_agg_count_verify = $2,
                    ang_betw_200 = $3,
                    ang_betw_400 = $4,
                    ang_betw_800 = $5,
                    ang_betw_1600 = $6,
                    ang_node_count_200 = $7,
                    ang_node_count_400 = $8,
                    ang_node_count_800 = $9,
                    ang_node_count_1600 = $10,
                    ang_farness_200 = $11,
                    ang_farness_400 = $12,
                    ang_farness_800 = $13,
                    ang_farness_1600 = $14
                WHERE id = $1
        ''', primary_data)

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

    nodes_table = 'roadnodes_full'
    links_table = 'roadlinks_full'
    for city_pop_id in range(1, 2):
        logger.info(f'calculating angular centralities for city {city_pop_id} derived from {nodes_table}')
        loop.run_until_complete(
            centrality_angular(db_config, network_schema, nodes_table, links_table,
                               city_pop_id, boundary_schema, boundary_table))
        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Started {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
