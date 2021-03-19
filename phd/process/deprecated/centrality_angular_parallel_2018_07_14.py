"""
ANGULAR IMPLEMENTATION

Complete rewrite though initial research performed at WOODS BAGOT per network_analyse_wb.py script

NEW NOTES:

Different approaches to avoiding shortcutting on angular graphs
1 - Match split pairs - this is very hard to extricate from database and to track - NO
2 - Select centrality of eight routes between split pairs - how to track and compare distances prior to summation? - NO
3 - Two fully split graphs - how to track and enforce opposing graph-cycles to allow access between all configurations - NO
4 - Manual - YES

OLD NOTES:

This script is different from older versions
- It compares more distances
- It computes both metric and angular
- It only records information back to dual -> does not average back to primary as this process may introduce artefacts

Used for exploring issues raised in prior analysis using distance and angular methods

Generally the dual values may scale differently than the primary values depending on methodologies

"""
import logging
import asyncio
import itertools
import numpy as np
import asyncpg
import numba

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# compute centralities on dual graph
# don't track master predecessor and distances matrices - this would be memory prohibitive
# i.e. calculate centralities as you go
# provide arrays of data for each vert - nbs, lens, angles -> the maximum cardinality on database table is 8
# provide 1d arrays for storing incremental data

@numba.njit  # parallel and fastmath don't apply
def shortest_path_tree(nbs_arr, dist_arr, source_idx, max_dist, active, dist_map_m, pred_map_m):
    '''
    This is the no-frills all centrality paths to max dist from source vertex
    '''

    # set starting node
    dist_map_m[source_idx] = 0
    active[source_idx] = source_idx  # store actual index number instead of booleans, easier for iteration below:

    # search to max distance threshold to determine reachable verts
    while np.any(np.isfinite(active)):
        # get the index for the min of currently active vert distances
        # note, this index corresponds only to currently active vertices
        # min_idx = np.argmin(dist_map_m[np.isfinite(active)])
        # map the min index back to the vertices array to get the corresponding vert idx
        # v = active[np.isfinite(active)][min_idx]
        # v_idx = np.int(v)  # cast to int
        # manually iterating definitely faster
        min_idx = None
        min_dist = np.inf
        for i, d in enumerate(dist_map_m):
            if d < min_dist and np.isfinite(active[i]):
                min_dist = d
                min_idx = i
        v_idx = np.int(min_idx)  # cast to int
        # set current vertex to visited
        active[v_idx] = np.inf
        # visit neighbours
        # for n, dist in zip(nbs_arr[v_idx], dist_arr[v_idx]):
        # manually iterating a tad faster
        for i, n in enumerate(nbs_arr[v_idx]):
            # exit once all neighbours visited
            if np.isnan(n):
                break
            n_idx = np.int(n)  # cast to int for indexing
            # distance is previous distance plus new distance
            dist = dist_arr[v_idx][i]
            d = dist_map_m[v_idx] + dist
            # only pursue if less than max and less than prior assigned distances
            if d < max_dist and d < dist_map_m[n_idx]:
                dist_map_m[n_idx] = d
                pred_map_m[n_idx] = v_idx
                active[n_idx] = n_idx  # using actual vert idx instead of boolean to simplify finding indices


@numba.njit  # parallel and fastmath don't apply
def shortest_path_angular_tree(nbs_arr, ang_dist_arr, dist_arr, source_idx, max_dist, dist_map_m, active, dist_map_a, dist_map_a_m, pred_map_a):
    '''
    This is the angular variant which has more complexity:
    - using source and target version of dijkstra because there are situations where angular routes exceed max dist
    - i.e. algo searches for and quits once target reached
    - returns both angular and corresponding euclidean distances
    - checks that centrality path algorithm doesn't back-step
    '''

    # set starting node
    dist_map_a[source_idx] = 0
    dist_map_a_m[source_idx] = 0
    active[source_idx] = source_idx  # store actual index number instead of booleans, easier for iteration below:

    # search to max distance threshold to determine reachable verts
    while np.any(np.isfinite(active)):
        # get the index for the min of currently active vert distances
        # note, this index corresponds only to currently active vertices
        # min_idx = np.argmin(dist_map_a[np.isfinite(active)])
        # map the min index back to the vertices array to get the corresponding vert idx
        # v = active[np.isfinite(active)][min_idx]
        # v_idx = np.int(v)  # cast to int
        # manually iterating definitely faster
        min_idx = None
        min_ang = np.inf
        for i, a in enumerate(dist_map_a):
            if a < min_ang and np.isfinite(active[i]):
                min_ang = a
                min_idx = i
        v_idx = np.int(min_idx)  # cast to int
        # set current vertex to visited
        active[v_idx] = np.inf
        # visit neighbours
        # for n, degrees, meters in zip(nbs_arr[v_idx], ang_dist_arr[v_idx], dist_arr[v_idx]):
        # manually iterating a tad faster
        for i, n in enumerate(nbs_arr[v_idx]):
            # exit once all neighbours visited
            if np.isnan(n):
                break
            n_idx = np.int(n)  # cast to int for indexing
            # check that the neighbour node does not exceed the euclidean distance threshold
            if dist_map_m[n_idx] > max_dist:
                continue
            # check that the neighbour was not directly accessible from the prior node
            # this prevents angular backtrack-shortcutting
            # first get the previous vert's id
            pred_idx = pred_map_a[v_idx]
            # need to check for nan in case of first vertex
            if np.isfinite(pred_idx):
                # could check that previous is not equal to current neighbour but would automatically die-out...
                # don't proceed with this index if it could've been followed from predecessor
                # if np.any(nbs_arr[np.int(prev_nb_idx)] == n_idx):
                if np.any(nbs_arr[np.int(pred_idx)] == n_idx):
                    continue
            # distance is previous distance plus new distance
            degrees = ang_dist_arr[v_idx][i]
            d_a = dist_map_a[v_idx] + degrees
            meters = dist_arr[v_idx][i]
            d_m = dist_map_a_m[v_idx] + meters
            # only pursue if angular distance is less than prior assigned distance
            if d_a < dist_map_a[n_idx]:
                dist_map_a[n_idx] = d_a
                dist_map_a_m[n_idx] = d_m
                pred_map_a[n_idx] = v_idx
                active[n_idx] = n_idx

# as tested on full graph for city 100 - durations at different max search distances
# 1 max = 0:01:49
# 2 max = 0:02:19
# 3 max = 0:02:28
# 5 max = 0:02:31  /  Duration: 0:02:35
# 5 max with internal arrays = 0:02:36
# 5 max with fill instead of index 0:02:27
# 5 max with internal arrays and fill instead of index 0:02:29
# no max = 0:02:33 - max dist searched: 7660 = 4.78

# as tested on full graph for city 100 - based on actual algo time
# 1 max = ~0:00:15 (after optimisations)
# 1 max = ~0:00:08 upper PRANGE
# no max = ~0:00:55

# as tested on full graph for city 500 - based on actual algo time
# 1 max = ~0:00:19 with iterative masking... slow
# 1 max = ~0:00:08 without iterative masking...
# 1 max = ~0:00:04 without angular
# 1 max = ~0:00:06 with manual iteration for metric instead of masking...
# 1 max = ~0:00:05 with manual iteration for metric and angular instead of masking...
# 1 max = ~0:00:04 after some optimisations...
# 1 max = ~0:00:09 with outer prange - not as much speed up as larger city
# 1 max no jit = 0:00:32
# 2 max = 0:00:12

# as tested on full graph for city 10 - based on actual algo time
# 1 max = ~0:06:51
# 1 max = ~0:05:44 PRANGE
# 1 max = ~0:06:10 PRANGE JUST FINITE
# 1 max = ~0:07:48 ??? PRANGE version... perhaps because CPU is hot, slightly throttled
# 1 max = ~0:07:48 ??? PRANGE version... nogil removed
# 1 max = ~0:07:39 ??? Non PRANGE version but still parallelised
# 1 max = ~0:02:17 !!! upper PRANGE version!
# 1 max = ~0:01:55 !!! checking upper PRANGE version (more CPUs available)
# 1 max = ~0:03:39 !!! trying upper PRANGE but back with
# 1 max = ~0:01:16 with outer prange and fastmath

# as tested on full graph for city 2 - based on actual algo time:
# 1 max = ~1:36:06

# as tested on full graph for city 5 - based on actual algo time:
# 1 max = ~0:11:05 after parallel improvements and fastmath

# vertices are implicit in index order, thus no direct verts_arr
@numba.njit(parallel=True, fastmath=True)
def custom_angular_centralities(nbs, lens, angles, city_ids, max_dist,
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

    total_count = len(nbs)
    print('...total vertices count:')
    print(total_count)

    interval = 10000
    print('...intervals:')
    print(round(total_count / interval, 2))

    discarded = 0
    total_visits = 0

    # iterate through each vert and calculated the centrality path tree
    for source_idx in numba.prange(total_count):

        if source_idx % interval == 0:
            print('...interval')
            print(round(source_idx / interval))

        # only compute for nodes in current city
        if not city_ids[source_idx]:
            continue

        # prepare the arrays - do this inside the parallel loop to allow privatised variables for parallel iterations
        # though if assigned outside the loop, then they are treated as private when assigned to from inside the loop...
        active = np.full(total_count, np.nan)
        dist_map_m = np.full(total_count, np.inf)
        pred_map_m = np.full(total_count, np.nan)
        dist_map_a = np.full(total_count, np.inf)
        dist_map_a_m = np.full(total_count, np.inf)
        pred_map_a = np.full(total_count, np.nan)

        shortest_path_tree(nbs, lens, source_idx, max_dist, active, dist_map_m, pred_map_m)

        # prepare arrays
        active.fill(np.nan)

        # get the angular centrality path
        # done one-by-one otherwise angular tree search doesn't know when to stop
        shortest_path_angular_tree(nbs, angles, lens, source_idx, max_dist, dist_map_m, active, dist_map_a, dist_map_a_m, pred_map_a)

        # use the metres distance map to filter reachable verts

        # use corresponding indices for reachable verts - parallelised regardless without explicit prange
        ind = np.where(np.isfinite(dist_map_m))[0]
        for to_idx in ind:

            if to_idx == source_idx:
                continue

            dist_m = dist_map_m[to_idx]
            if dist_m == np.inf:  # TODO: is this affected by fastmath?
                continue

            total_visits += 1

            dist_a = dist_map_a[to_idx]
            dist_a_metres = dist_map_a_m[to_idx]
            # TODO: it may be possible to drop this check with new methodology
            if dist_a == np.inf:  # TODO: is this affected by fastmath
                discarded += 1
                print('...NOTE -> no corresponding angular distance, skipping')
                continue

            if dist_m <= 50:
                node_count_50[source_idx] += 1
                met_farness_50[source_idx] += dist_m
                ang_farness_50[source_idx] += dist_a
                ang_farness_m_50[source_idx] += dist_a_metres
            if dist_m <= 100:
                node_count_100[source_idx] += 1
                met_farness_100[source_idx] += dist_m
                ang_farness_100[source_idx] += dist_a
                ang_farness_m_100[source_idx] += dist_a_metres
            if dist_m <= 150:
                node_count_150[source_idx] += 1
                met_farness_150[source_idx] += dist_m
                ang_farness_150[source_idx] += dist_a
                ang_farness_m_150[source_idx] += dist_a_metres
            if dist_m <= 200:
                node_count_200[source_idx] += 1
                met_farness_200[source_idx] += dist_m
                ang_farness_200[source_idx] += dist_a
                ang_farness_m_200[source_idx] += dist_a_metres
            if dist_m <= 300:
                node_count_300[source_idx] += 1
                met_farness_300[source_idx] += dist_m
                ang_farness_300[source_idx] += dist_a
                ang_farness_m_300[source_idx] += dist_a_metres
            if dist_m <= 400:
                node_count_400[source_idx] += 1
                met_farness_400[source_idx] += dist_m
                ang_farness_400[source_idx] += dist_a
                ang_farness_m_400[source_idx] += dist_a_metres
            if dist_m <= 600:
                node_count_600[source_idx] += 1
                met_farness_600[source_idx] += dist_m
                ang_farness_600[source_idx] += dist_a
                ang_farness_m_600[source_idx] += dist_a_metres
            if dist_m <= 800:
                node_count_800[source_idx] += 1
                met_farness_800[source_idx] += dist_m
                ang_farness_800[source_idx] += dist_a
                ang_farness_m_800[source_idx] += dist_a_metres
            if dist_m <= 1200:
                node_count_1200[source_idx] += 1
                met_farness_1200[source_idx] += dist_m
                ang_farness_1200[source_idx] += dist_a
                ang_farness_m_1200[source_idx] += dist_a_metres
            if dist_m <= 1600:
                node_count_1600[source_idx] += 1
                met_farness_1600[source_idx] += dist_m
                ang_farness_1600[source_idx] += dist_a
                ang_farness_m_1600[source_idx] += dist_a_metres

            # distance betweenness - only counting truly between vertices, not starting and ending verts
            intermediary_idx = np.int(pred_map_m[to_idx])  # cast to int
            while True:
                if intermediary_idx == source_idx:
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
                intermediary_idx = np.int(pred_map_m[intermediary_idx])

            # angular betweenness
            # angular all centrality paths algorithm searches up to 3 times the max distance
            # this prevents situations where angular aborts prior to finding a centrality path
            # which causes nans for when using
            intermediary_idx = np.int(pred_map_a[to_idx])  # cast to int
            while True:
                # break out of while loop if the intermediary is the source node
                # sum distance en route
                if intermediary_idx == source_idx:
                    break
                if dist_m <= 50:
                    ang_betweenness_50[intermediary_idx] += 1
                if dist_m <= 100:
                    ang_betweenness_100[intermediary_idx] += 1
                if dist_m <= 150:
                    ang_betweenness_150[intermediary_idx] += 1
                if dist_m <= 200:
                    ang_betweenness_200[intermediary_idx] += 1
                if dist_m <= 300:
                    ang_betweenness_300[intermediary_idx] += 1
                if dist_m <= 400:
                    ang_betweenness_400[intermediary_idx] += 1
                if dist_m <= 600:
                    ang_betweenness_600[intermediary_idx] += 1
                if dist_m <= 800:
                    ang_betweenness_800[intermediary_idx] += 1
                if dist_m <= 1200:
                    ang_betweenness_1200[intermediary_idx] += 1
                if dist_m <= 1600:
                    ang_betweenness_1600[intermediary_idx] += 1
                # follow the next intermediary
                intermediary_idx = np.int(pred_map_a[intermediary_idx])

    print('...discarded visits:')
    print(discarded)
    print(round(discarded / total_visits * 100, 2))


async def centrality_angular_custom(db_config, network_schema, nodes_table, links_table, city_pop_id, boundary_schema, boundary_table):

    logger.info('Generating dual graph from database')

    db_pool = await asyncpg.create_pool(**db_config, min_size=2, max_size=5)

    # load all vertices and corresponding neighbours and edges
    async with db_pool.acquire() as db_con_1:

        logger.info('NOTE -> creating temporary buffered city geom')
        # convex hull prevents potential issues with multipolygons deriving from buffer...
        temp_id = await db_con_1.fetchval(f'''
            INSERT INTO {boundary_schema}.{boundary_table} (geom)
                VALUES ((SELECT ST_ConvexHull(ST_Buffer(geom, 1600))
                    FROM {boundary_schema}.{boundary_table} WHERE pop_id = {city_pop_id}))
                RETURNING id;''')

        logger.info('Loading primary vertices data')
        db_data = await db_con_1.fetch(f'''
            SELECT nodes.id, nodes.edges, nodes.city_pop_id
                FROM {network_schema}.{nodes_table} as nodes,
                    (SELECT geom FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id}) as boundary
                WHERE ST_Contains(boundary.geom, nodes.geom);''')

        logger.info('NOTE -> removing temporary buffered city geom')
        await db_con_1.execute(f'''DELETE FROM {boundary_schema}.{boundary_table} WHERE id = {temp_id};''')

    # for each vert, create a new dual-vert midway to each adjacent neighbour, check for duplicates
    # use the original edge id for new dual-vert ids
    logger.info('Deriving dual graph data')
    dual_verts = {}
    dual_edges = {}
    idx_count = 0
    for count, r in enumerate(db_data):

        if count and count % 1000 == 0:
            logger.info(f'...loaded {count} items of {len(db_data)} from db: {round(count / len(db_data) * 100, 2)}%')

        # transpose the primary edges to dual vertices
        edges = r['edges']
        for e in edges:
            if e not in dual_verts:
                dual_verts[e] = {
                    'idx': idx_count,
                    'parent_node_idx': r['id'],
                    'city_pop_id': r['city_pop_id'],
                    'geom': None,
                    'nb': [],
                    'len': [],
                    'angle': []
                }
                idx_count += 1

        # only primary edge pairs (i.e. dual vertices) are here split and combined into a new interlinking dual-edges
        # i.e. primary vertices with a single neighbour / primary edge are automatically ignored
        for source, target in itertools.combinations(edges, 2):
            async with db_pool.acquire() as db_con_2:
                try:
                    e_data = await db_con_2.fetchrow(f'''
                        SELECT ST_Length(w.welded_geom) as len, a.angle, s.source_dual_vert_geom, t.target_dual_vert_geom,
                                ST_MakeLine(s.source_dual_vert_geom, t.target_dual_vert_geom) as dual_simple_edge_geom
                              FROM
                            (SELECT start_node, end_node, geom, ST_Force2D(ST_LineInterpolatePoint(geom, 0.5)) as source_dual_vert_geom
                              FROM {network_schema}.{links_table} WHERE id = {source}) as s
                            CROSS JOIN LATERAL
                            (SELECT start_node, end_node, geom, ST_Force2D(ST_LineInterpolatePoint(geom, 0.5)) as target_dual_vert_geom
                              FROM {network_schema}.{links_table} WHERE id = {target}) as t
                            CROSS JOIN LATERAL
                            (SELECT CASE
                                WHEN s.start_node = t.start_node
                                  THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0, 0.5), ST_LineSubString(t.geom, 0, 0.5))))
                                WHEN s.start_node = t.end_node
                                  THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0, 0.5), ST_LineSubString(t.geom, 0.5, 1))))
                                WHEN s.end_node = t.start_node
                                  THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0.5, 1), ST_LineSubString(t.geom, 0, 0.5))))
                                WHEN s.end_node = t.end_node
                                  THEN ST_Force2D(ST_LineMerge(ST_Union(ST_LineSubstring(s.geom, 0.5, 1), ST_LineSubString(t.geom, 0.5, 1))))
                              END as geom) as g
                            CROSS JOIN LATERAL
                            -- very rarely, a multilinestring is returned, in which cases just created a simplified linestring
                            (SELECT CASE
                              WHEN ST_GeometryType(g.geom) = 'ST_MultiLineString'
                                THEN ST_MakeLine(s.source_dual_vert_geom, t.target_dual_vert_geom)
                              ELSE g.geom
                              END as welded_geom) as w
                            CROSS JOIN LATERAL
                            -- get the angle between the start and end of the line
                            (SELECT
                               degrees(ST_Azimuth(ST_StartPoint(w.welded_geom), ST_LineInterpolatePoint(w.welded_geom, 0.1)))::int as az_a,
                               degrees(ST_Azimuth(ST_LineInterpolatePoint(w.welded_geom, 0.9), ST_EndPoint(w.welded_geom)))::int as az_b) as az
                            CROSS JOIN LATERAL
                            -- reduce the angle in cases where computed through the long direction
                            (SELECT abs((abs(az_b - az_a) + 180) % 360 - 180) as angle) as a;
                        ''')
                except asyncpg.exceptions.InternalServerError as e:
                    logger.error(e)
                    logger.error(f'Unable to load data for links {source} and {target}. See above error from asyncpg.')
                    pass

            # add the data to the corresponding dual vertices
            # add data from source to target
            dual_verts[source]['geom'] = e_data['source_dual_vert_geom']
            dual_verts[source]['nb'].append(target)
            dual_verts[source]['len'].append(e_data['len'])
            dual_verts[source]['angle'].append(e_data['angle'])

            # add data from target to source
            dual_verts[target]['geom'] = e_data['target_dual_vert_geom']
            dual_verts[target]['nb'].append(source)
            dual_verts[target]['len'].append(e_data['len'])
            dual_verts[target]['angle'].append(e_data['angle'])

            # create a corresponding dual edge in the dual edges dictionary
            s_t = sorted([source, target])
            edge_id = '{0}_{1}'.format(*s_t)
            if edge_id not in dual_edges:
                dual_edges[edge_id] = {
                    'parent_primary_node': r['id'],
                    'city_pop_id': r['city_pop_id'],
                    'source': source,
                    'target': target,
                    'angle': e_data['angle'],
                    'distance': e_data['len'],
                    'geom': e_data['dual_simple_edge_geom']
                }

    await db_pool.close()

    logger.info('Preparing arrays')

    # generate arrays for numba function
    graph_len = len(dual_verts)
    verts = np.full(graph_len, 0)  # int doesn't support nan
    nbs = np.full((graph_len, 10), np.nan)
    lens = np.full((graph_len, 10), np.nan)
    angles = np.full((graph_len, 10), np.nan)
    city_ids = np.full(graph_len, False)  # only compute for nodes in current city

    # iterate the dual_verts dict and insert into the verts array in the order corresponding to the index
    for k, v in dual_verts.items():
        idx = v['idx']
        verts[idx] = k
        city_ids[idx] = (v['city_pop_id'] == city_pop_id)  # only compute for nodes in current city
        for j, (nb, length, angle) in enumerate(zip(v['nb'], v['len'], v['angle'])):
            nbs[idx][j] = dual_verts[nb]['idx']  # retrieve the neighbours array index
            lens[idx][j] = length
            angles[idx][j] = angle

    """
    FOR TESTING
    
    verts = np.array([0, 1, 2, 3, 4, 5])
    nbs = np.array([[1, 3, np.nan, np.nan],
                    [0, 2, 3, 4],
                    [1, 4, np.nan, np.nan],
                    [0, 1, 5, np.nan],
                    [1, 2, 5, np.nan],
                    [3, 4, np.nan, np.nan]])
    lens = np.array([[500, 400, np.nan, np.nan],
                    [500, 1200, 400, 400],
                    [1200, 400, np.nan, np.nan],
                    [400, 400, 250, np.nan],
                    [400, 400, 250, np.nan],
                    [250, 250, np.nan, np.nan]])
    angles = np.array([[0, 90, np.nan, np.nan],
                     [0, 0, 90, 90],
                     [0, 90, np.nan, np.nan],
                     [90, 90, 90, np.nan],
                     [90, 90, 90, np.nan],
                     [90, 90, np.nan, np.nan]])
    city_ids = np.array([True, True, True, True, True, True])
    graph_len = 6
    """

    # arrays for computations
    met_betweenness_50 = np.zeros(graph_len)
    met_betweenness_100 = np.zeros(graph_len)
    met_betweenness_150 = np.zeros(graph_len)
    met_betweenness_200 = np.zeros(graph_len)
    met_betweenness_300 = np.zeros(graph_len)
    met_betweenness_400 = np.zeros(graph_len)
    met_betweenness_600 = np.zeros(graph_len)
    met_betweenness_800 = np.zeros(graph_len)
    met_betweenness_1200 = np.zeros(graph_len)
    met_betweenness_1600 = np.zeros(graph_len)
    ang_betweenness_50 = np.zeros(graph_len)
    ang_betweenness_100 = np.zeros(graph_len)
    ang_betweenness_150 = np.zeros(graph_len)
    ang_betweenness_200 = np.zeros(graph_len)
    ang_betweenness_300 = np.zeros(graph_len)
    ang_betweenness_400 = np.zeros(graph_len)
    ang_betweenness_600 = np.zeros(graph_len)
    ang_betweenness_800 = np.zeros(graph_len)
    ang_betweenness_1200 = np.zeros(graph_len)
    ang_betweenness_1600 = np.zeros(graph_len)
    node_count_50 = np.zeros(graph_len, dtype=int)
    node_count_100 = np.zeros(graph_len, dtype=int)
    node_count_150 = np.zeros(graph_len, dtype=int)
    node_count_200 = np.zeros(graph_len, dtype=int)
    node_count_300 = np.zeros(graph_len, dtype=int)
    node_count_400 = np.zeros(graph_len, dtype=int)
    node_count_600 = np.zeros(graph_len, dtype=int)
    node_count_800 = np.zeros(graph_len, dtype=int)
    node_count_1200 = np.zeros(graph_len, dtype=int)
    node_count_1600 = np.zeros(graph_len, dtype=int)
    met_farness_50 = np.zeros(graph_len)
    met_farness_100 = np.zeros(graph_len)
    met_farness_150 = np.zeros(graph_len)
    met_farness_200 = np.zeros(graph_len)
    met_farness_300 = np.zeros(graph_len)
    met_farness_400 = np.zeros(graph_len)
    met_farness_600 = np.zeros(graph_len)
    met_farness_800 = np.zeros(graph_len)
    met_farness_1200 = np.zeros(graph_len)
    met_farness_1600 = np.zeros(graph_len)
    ang_farness_50 = np.zeros(graph_len)
    ang_farness_100 = np.zeros(graph_len)
    ang_farness_150 = np.zeros(graph_len)
    ang_farness_200 = np.zeros(graph_len)
    ang_farness_300 = np.zeros(graph_len)
    ang_farness_400 = np.zeros(graph_len)
    ang_farness_600 = np.zeros(graph_len)
    ang_farness_800 = np.zeros(graph_len)
    ang_farness_1200 = np.zeros(graph_len)
    ang_farness_1600 = np.zeros(graph_len)
    ang_farness_m_50 = np.zeros(graph_len)
    ang_farness_m_100 = np.zeros(graph_len)
    ang_farness_m_150 = np.zeros(graph_len)
    ang_farness_m_200 = np.zeros(graph_len)
    ang_farness_m_300 = np.zeros(graph_len)
    ang_farness_m_400 = np.zeros(graph_len)
    ang_farness_m_600 = np.zeros(graph_len)
    ang_farness_m_800 = np.zeros(graph_len)
    ang_farness_m_1200 = np.zeros(graph_len)
    ang_farness_m_1600 = np.zeros(graph_len)

    logger.info('Calculating centrality paths and centralities')
    start = time.localtime()
    custom_angular_centralities(nbs, lens, angles, city_ids, 1600,
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

    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    logger.info('Aggregating data')

    db_con = await asyncpg.connect(**db_config)

    # ADD DUAL GRAPH'S EDGES TABLE
    await db_con.execute(f'''
        -- add dual links table
        CREATE TABLE IF NOT EXISTS {network_schema}.{links_table}_dual (
          id text PRIMARY KEY,
          parent_id bigint,
          city_pop_id int,
          node_a bigint,
          node_b bigint,
          angle int,
          distance real,
          geom geometry(Linestring, 27700)
        );
        CREATE INDEX IF NOT EXISTS geom_idx_{links_table}_dual ON {network_schema}.{links_table}_dual USING GIST (geom);
        ''')

    dual_edge_data = []
    for k, v in dual_edges.items():
        # only write data back if current city_pop_id - otherwise neighbouring towns can be obfuscated
        if v['city_pop_id'] == city_pop_id:
            dual_edge_data.append((k, v['parent_primary_node'], v['city_pop_id'], v['source'], v['target'], v['angle'], v['distance'], v['geom']))

    logger.info('Writing dual edges to DB')
    logger.info(dual_edge_data[int(len(dual_edge_data) / 2)])
    await db_con.executemany(f'''
            INSERT INTO {network_schema}.{links_table}_dual (
                id,
                parent_id,
                city_pop_id,
                node_a,
                node_b,
                angle,
                distance,
                geom)
            VALUES ($1, $2, $3, $4, $5, $6, $7, ST_SetSRID($8::geometry, 27700))
            ON CONFLICT (id) DO UPDATE SET
                parent_id = $2,
                city_pop_id = $3,
                node_a = $4,
                node_b = $5,
                angle = $6,
                distance = $7,
                geom = ST_SetSRID($8::geometry, 27700);
        ''', dual_edge_data)

    # ADD DUAL GRAPH'S VERTICES TABLE
    await db_con.execute(f'''
        -- add dual nodes table
        CREATE TABLE IF NOT EXISTS {network_schema}.{nodes_table}_dual (
            id bigint PRIMARY KEY,
            city_pop_id int,
            geom geometry(Point, 27700)
        );
        CREATE INDEX IF NOT EXISTS geom_idx_{nodes_table}_dual ON {network_schema}.{nodes_table}_dual USING GIST (geom);
        ALTER TABLE {network_schema}.{nodes_table}_dual
          -- add the new columns for writing the results
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_50 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_100 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_150 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_200 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_300 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_400 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_600 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_800 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_1200 real,
          ADD COLUMN IF NOT EXISTS dual_met_betweenness_1600 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_50 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_100 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_150 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_200 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_300 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_400 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_600 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_800 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_1200 real,
          ADD COLUMN IF NOT EXISTS dual_ang_betweenness_1600 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_50 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_100 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_150 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_200 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_300 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_400 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_600 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_800 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_1200 real,
          ADD COLUMN IF NOT EXISTS dual_node_count_1600 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_50 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_100 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_150 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_200 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_300 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_400 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_600 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_800 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_1200 real,
          ADD COLUMN IF NOT EXISTS dual_met_farness_1600 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_50 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_100 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_150 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_200 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_300 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_400 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_600 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_800 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_1200 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_1600 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_50 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_100 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_150 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_200 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_300 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_400 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_600 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_800 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_1200 real,
          ADD COLUMN IF NOT EXISTS dual_ang_farness_m_1600 real;
        ''')

    dual_verts_data = []
    for idx, v_id in enumerate(verts):
        # only write data back if current city_pop_id - otherwise neighbouring towns can be obfuscated
        if dual_verts[v_id]['city_pop_id'] == city_pop_id:
            dual_verts_data.append((
                v_id,
                dual_verts[v_id]['city_pop_id'],
                dual_verts[v_id]['geom'],
                met_betweenness_50[idx],
                met_betweenness_100[idx],
                met_betweenness_150[idx],
                met_betweenness_200[idx],
                met_betweenness_300[idx],
                met_betweenness_400[idx],
                met_betweenness_600[idx],
                met_betweenness_800[idx],
                met_betweenness_1200[idx],
                met_betweenness_1600[idx],
                ang_betweenness_50[idx],
                ang_betweenness_100[idx],
                ang_betweenness_150[idx],
                ang_betweenness_200[idx],
                ang_betweenness_300[idx],
                ang_betweenness_400[idx],
                ang_betweenness_600[idx],
                ang_betweenness_800[idx],
                ang_betweenness_1200[idx],
                ang_betweenness_1600[idx],
                node_count_50[idx],
                node_count_100[idx],
                node_count_150[idx],
                node_count_200[idx],
                node_count_300[idx],
                node_count_400[idx],
                node_count_600[idx],
                node_count_800[idx],
                node_count_1200[idx],
                node_count_1600[idx],
                met_farness_50[idx],
                met_farness_100[idx],
                met_farness_150[idx],
                met_farness_200[idx],
                met_farness_300[idx],
                met_farness_400[idx],
                met_farness_600[idx],
                met_farness_800[idx],
                met_farness_1200[idx],
                met_farness_1600[idx],
                ang_farness_50[idx],
                ang_farness_100[idx],
                ang_farness_150[idx],
                ang_farness_200[idx],
                ang_farness_300[idx],
                ang_farness_400[idx],
                ang_farness_600[idx],
                ang_farness_800[idx],
                ang_farness_1200[idx],
                ang_farness_1600[idx],
                ang_farness_m_50[idx],
                ang_farness_m_100[idx],
                ang_farness_m_150[idx],
                ang_farness_m_200[idx],
                ang_farness_m_300[idx],
                ang_farness_m_400[idx],
                ang_farness_m_600[idx],
                ang_farness_m_800[idx],
                ang_farness_m_1200[idx],
                ang_farness_m_1600[idx]))

    logger.info('Writing dual vertices to DB')
    logger.info(dual_verts_data[int(len(dual_verts_data) / 2)])
    await db_con.executemany(f'''
        INSERT INTO {network_schema}.{nodes_table}_dual (
                id,
                city_pop_id,
                geom,
                dual_met_betweenness_50,
                dual_met_betweenness_100,
                dual_met_betweenness_150,
                dual_met_betweenness_200,
                dual_met_betweenness_300,
                dual_met_betweenness_400,
                dual_met_betweenness_600,
                dual_met_betweenness_800,
                dual_met_betweenness_1200,
                dual_met_betweenness_1600,
                dual_ang_betweenness_50,
                dual_ang_betweenness_100,
                dual_ang_betweenness_150,
                dual_ang_betweenness_200,
                dual_ang_betweenness_300,
                dual_ang_betweenness_400,
                dual_ang_betweenness_600,
                dual_ang_betweenness_800,
                dual_ang_betweenness_1200,
                dual_ang_betweenness_1600,
                dual_node_count_50,
                dual_node_count_100,
                dual_node_count_150,
                dual_node_count_200,
                dual_node_count_300,
                dual_node_count_400,
                dual_node_count_600,
                dual_node_count_800,
                dual_node_count_1200,
                dual_node_count_1600,
                dual_met_farness_50,
                dual_met_farness_100,
                dual_met_farness_150,
                dual_met_farness_200,
                dual_met_farness_300,
                dual_met_farness_400,
                dual_met_farness_600,
                dual_met_farness_800,
                dual_met_farness_1200,
                dual_met_farness_1600,
                dual_ang_farness_50,
                dual_ang_farness_100,
                dual_ang_farness_150,
                dual_ang_farness_200,
                dual_ang_farness_300,
                dual_ang_farness_400,
                dual_ang_farness_600,
                dual_ang_farness_800,
                dual_ang_farness_1200,
                dual_ang_farness_1600,
                dual_ang_farness_m_50,
                dual_ang_farness_m_100,
                dual_ang_farness_m_150,
                dual_ang_farness_m_200,
                dual_ang_farness_m_300,
                dual_ang_farness_m_400,
                dual_ang_farness_m_600,
                dual_ang_farness_m_800,
                dual_ang_farness_m_1200,
                dual_ang_farness_m_1600)
            VALUES ($1, $2, ST_SetSRID($3::geometry, 27700),
                $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
                $24, $25, $26, $27, $28, $29, $30, $31, $32, $33,
                $34, $35, $36, $37, $38, $39, $40, $41, $42, $43,
                $44, $45, $46, $47, $48, $49, $50, $51, $52, $53,
                $54, $55, $56, $57, $58, $59, $60, $61, $62, $63)
            ON CONFLICT (id) DO UPDATE SET
                city_pop_id = $2,
                geom = ST_SetSRID($3::geometry, 27700),
                dual_met_betweenness_50 = $4,
                dual_met_betweenness_100 = $5,
                dual_met_betweenness_150 = $6,
                dual_met_betweenness_200 = $7,
                dual_met_betweenness_300 = $8,
                dual_met_betweenness_400 = $9,
                dual_met_betweenness_600 = $10,
                dual_met_betweenness_800 = $11,
                dual_met_betweenness_1200 = $12,
                dual_met_betweenness_1600 = $13,
                dual_ang_betweenness_50 = $14,
                dual_ang_betweenness_100 = $15,
                dual_ang_betweenness_150 = $16,
                dual_ang_betweenness_200 = $17,
                dual_ang_betweenness_300 = $18,
                dual_ang_betweenness_400 = $19,
                dual_ang_betweenness_600 = $20,
                dual_ang_betweenness_800 = $21,
                dual_ang_betweenness_1200 = $22,
                dual_ang_betweenness_1600 = $23,
                dual_node_count_50 = $24,
                dual_node_count_100 = $25,
                dual_node_count_150 = $26,
                dual_node_count_200 = $27,
                dual_node_count_300 = $28,
                dual_node_count_400 = $29,
                dual_node_count_600 = $30,
                dual_node_count_800 = $31,
                dual_node_count_1200 = $32,
                dual_node_count_1600 = $33,
                dual_met_farness_50 = $34,
                dual_met_farness_100 = $35,
                dual_met_farness_150 = $36,
                dual_met_farness_200 = $37,
                dual_met_farness_300 = $38,
                dual_met_farness_400 = $39,
                dual_met_farness_600 = $40,
                dual_met_farness_800 = $41,
                dual_met_farness_1200 = $42,
                dual_met_farness_1600 = $43,
                dual_ang_farness_50 = $44,
                dual_ang_farness_100 = $45,
                dual_ang_farness_150 = $46,
                dual_ang_farness_200 = $47,
                dual_ang_farness_300 = $48,
                dual_ang_farness_400 = $49,
                dual_ang_farness_600 = $50,
                dual_ang_farness_800 = $51,
                dual_ang_farness_1200 = $52,
                dual_ang_farness_1600 = $53,
                dual_ang_farness_m_50 = $54,
                dual_ang_farness_m_100 = $55,
                dual_ang_farness_m_150 = $56,
                dual_ang_farness_m_200 = $57,
                dual_ang_farness_m_300 = $58,
                dual_ang_farness_m_400 = $59,
                dual_ang_farness_m_600 = $60,
                dual_ang_farness_m_800 = $61,
                dual_ang_farness_m_1200 = $62,
                dual_ang_farness_m_1600 = $63;
    ''', dual_verts_data)

    # calculate the closeness and ratio columns
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{nodes_table}_dual
            ADD COLUMN IF NOT EXISTS dual_ratio_50 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_100 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_150 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_200 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_300 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_400 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_600 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_800 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_1200 real,
            ADD COLUMN IF NOT EXISTS dual_ratio_1600 real;

        UPDATE {network_schema}.{nodes_table}_dual
            SET dual_ratio_50 = CASE dual_met_farness_50
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_50 / dual_met_farness_50)::numeric, 2)
                END,
            dual_ratio_100 = CASE dual_met_farness_100
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_100 / dual_met_farness_100)::numeric, 2)
                END,
            dual_ratio_150 = CASE dual_met_farness_150
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_150 / dual_met_farness_150)::numeric, 2)
                END,
            dual_ratio_200 = CASE dual_met_farness_200
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_200 / dual_met_farness_200)::numeric, 2)
                END,
              dual_ratio_300 = CASE dual_met_farness_300
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_300 / dual_met_farness_300)::numeric, 2)
                END,
              dual_ratio_400 = CASE dual_met_farness_400
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_400 / dual_met_farness_400)::numeric, 2)
                END,
              dual_ratio_600 = CASE dual_met_farness_600
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_600 / dual_met_farness_600)::numeric, 2)
                END,
              dual_ratio_800 = CASE dual_met_farness_800
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_800 / dual_met_farness_800)::numeric, 2)
                END,
              dual_ratio_1200 = CASE dual_met_farness_1200
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_1200 / dual_met_farness_1200)::numeric, 2)
                END,
              dual_ratio_1600 = CASE dual_met_farness_1600
                    WHEN 0 THEN NULL
                    ELSE round((dual_ang_farness_m_1600 / dual_met_farness_1600)::numeric, 2)
                END;
                ''')

    await db_con.close()


if __name__ == '__main__':

    import time
    import os
    import datetime
    import logging

    os.environ['NUMBA_DEBUG_ARRAY_OPT_STATS'] = '1'

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

    boundary_schema = 'analysis'
    boundary_table = 'city_boundaries_150'
    schema = 'analysis'


    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    nodes_table = 'roadnodes_full'
    links_table = 'roadlinks_full'

    for city_pop_id in range(10, 11):
        logger.info(f'Calculating angular centralities for city {city_pop_id} derived from {nodes_table}')
        loop.run_until_complete(
            centrality_angular_custom(db_config, schema, nodes_table, links_table,
                                      city_pop_id, boundary_schema, boundary_table))
        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Started {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')