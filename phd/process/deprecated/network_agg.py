"""
Aggregates measures per node based on network distances
"""
import os
import logging
import time
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *
from numba import jit
import yagmail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'graph-tool version: {graph_tool.__version__}')

@jit(nopython=True, nogil=True)
def stirling_div(uses_unique, uses_probs, weight_1=0.25, weight_2=0.5, weight_3=1, alpha=1, beta=0.5, additive=False):
    mixedUses = 0  # variable for additive calculations of distance * p1 * p2
    for i in range(len(uses_unique) - 1):
        a = uses_unique[i]
        a_proportion = uses_probs[i]
        j = i + 1
        while j < len(uses_unique):
            b = uses_unique[j]
            b_proportion = uses_probs[j]
            # calculate 3rd level disparity
            a_rem = a % 1000000
            b_rem = b % 1000000
            if a - a_rem != b - b_rem:
                w = weight_3
            else:
                # else calculate 2ndd level disparity
                a_small_rem = a % 10000
                b_small_rem = b % 10000
                if a_rem - a_small_rem != b_rem - b_small_rem:
                    w = weight_2
                # else default used - which means the disparity is at level 1
                else:
                    w = weight_1
            # Stirling's distance weighted diversity index.
            # matching items have distance of 0 so don't compute for complete matches
            if not additive:
                mixedUses += w**alpha * (a_proportion * b_proportion)**beta
            else:
                mixedUses += w * a_proportion
                mixedUses += w * b_proportion
            j += 1
    return mixedUses

async def aggregator(db_config, networks_table, verts_table, links_table, threshold_distances, city_id):

    # the graph
    graph = Graph(directed=False)

    # edge property maps
    g_edge_id = graph.new_edge_property('int')
    g_edge_dist = graph.new_edge_property('float')
    # vertex property maps
    g_vert_id = graph.new_vertex_property('int')
    g_vert_list_bldgs = graph.new_vertex_property('int')
    g_vert_voa_count = graph.new_vertex_property('int')

    # open database connection
    logger.info(f'loading data from {networks_table}.{verts_table}')
    vertex_dict = {}
    db_con = await asyncpg.connect(**db_config)
    async with db_con.transaction():
        async for record in db_con.cursor(f'''
            SELECT id, start_node, end_node, ST_Length(geom) as dist
                FROM {networks_table}.{links_table}
                    WHERE city_id::int = {city_id}'''):
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

    logger.info(f'number of vertices: {graph.num_vertices()}')

    if graph.num_vertices():
        # vector property maps are very slow, use numpy arrays to store values instead
        # remember int arrays can't handle np.nan
        # City 1 at 100m nodes has a max of 819
        # but there are outliers for city 2 at 1475 (vs. 2nd at 318) and city 15 at 1746
        # at 50m there are outliers for city 2 at 1454 and city 15 at 1746
        # at 20m there are outliers for city 2 at 1453 and city 15 at 1746
        # these are likely due to lumped VOA locations located by postcode, possibly for city-wide area... ?
        #TODO: investigate the source of the lumping
        logger.info('setting maximum array size for VOA aggregation from DB array sizes')
        await db_con.execute(f'''
            CREATE INDEX IF NOT EXISTS voa_count_idx ON {networks_table}.{verts_table} (voa_count);
        ''')
        voa_size_cap = await db_con.fetchval(f'''
            SELECT voa_count
                FROM {networks_table}.{verts_table}
                WHERE city_id::int = {city_id}
                    AND voa_count IS NOT NULL
                ORDER BY voa_count DESC
                LIMIT 1''')
        max_cap = 1000
        if voa_size_cap and voa_size_cap > max_cap:
            logger.warning(f'Size cap of {voa_size_cap} being capped to {max_cap}')
            voa_size_cap = max_cap
        else:
            logger.info(f'Setting array size cap to {voa_size_cap}')

        if voa_size_cap:
            voa_area_arr = np.zeros((graph.num_vertices(), voa_size_cap))
            voa_area_arr.fill(np.nan)
            voa_val_arr = np.zeros((graph.num_vertices(), voa_size_cap))
            voa_val_arr.fill(np.nan)
            voa_rate_arr = np.zeros((graph.num_vertices(), voa_size_cap))
            voa_rate_arr.fill(np.nan)

        # fetch OS POI classification max length
        logger.info('setting maximum array size for OS POI aggregation from DB array sizes')
        class_size_cap = await db_con.fetchval(f'''
            SELECT cardinality(class_code_arr) AS c
                FROM {networks_table}.{verts_table}
                WHERE city_id::int = {city_id}
                    AND class_code_arr IS NOT NULL
                ORDER BY c DESC
                LIMIT 1''')
        max_cap = 1000
        if class_size_cap and class_size_cap > max_cap:
            logger.warning(f'Size cap of {class_size_cap} being capped to {max_cap}')
            class_size_cap = max_cap
        else:
            logger.info(f'Setting array size cap to {class_size_cap}')

        if class_size_cap:
            class_code_arr = np.zeros((graph.num_vertices(), class_size_cap), dtype=int)

        # load the node properties
        logger.info('building arrays')
        async with db_con.transaction():
            async for record in db_con.cursor(f'''
                SELECT id, class_code_arr, voa_count, voa_area_arr, voa_val_arr, voa_rate_arr,
                    listed_bldgs_grade_1 + listed_bldgs_grade_2_star + listed_bldgs_grade_2 as listed_bldgs
                    FROM {networks_table}.{verts_table}
                        WHERE city_id::int = {city_id}'''):
                # not all verts will have been loaded with the edges
                if record['id'] not in vertex_dict:
                    continue
                # populate node attributes
                vert_id = vertex_dict[record['id']]
                if record['listed_bldgs']:
                    g_vert_list_bldgs[vert_id] = record['listed_bldgs']
                if record['class_code_arr']:
                    arr = np.array([int(r) for r in record['class_code_arr']])
                    # skip if max cap exceeded for
                    arr = np.pad(arr, (0, class_size_cap - len(arr)), mode='constant', constant_values=0)  # np.nan doesn't work with int
                    class_code_arr[int(vert_id)] = arr
                if record['voa_count']:
                    g_vert_voa_count[vert_id] = record['voa_count']
                if record['voa_area_arr']:
                    arr = np.array(record['voa_area_arr'])
                    # check for outliers and skip VOA if max cap exceeded
                    if len(arr) > max_cap:
                        logger.error(f'Maximum array size of {max_cap} has been exceeded for VOA, skipping this node due to likely outlier')
                        continue
                    arr = np.pad(arr, (0, voa_size_cap - len(arr)), mode='constant', constant_values=np.nan)
                    voa_area_arr[int(vert_id)] = arr
                if record['voa_val_arr']:
                    arr = np.array(record['voa_val_arr'])
                    arr = np.pad(arr, (0, voa_size_cap - len(arr)), mode='constant', constant_values=np.nan)
                    voa_val_arr[int(vert_id)] = arr
                if record['voa_rate_arr']:
                    arr = np.array(record['voa_rate_arr'])
                    arr = np.pad(arr, (0, voa_size_cap - len(arr)), mode='constant', constant_values=np.nan)
                    voa_rate_arr[int(vert_id)] = arr
        await db_con.close()
        logger.info(f'edges: {graph.num_edges()}, vertices: {graph.num_vertices()}')

        for threshold_distance in threshold_distances:
            logger.info(f'starting {city_id} at threshold distance: {threshold_distance}')

            start_time = time.time()
            # iterate the 'i' verts
            results = []
            num_vertices = graph.num_vertices()
            for v in graph.vertices():
                if int(v) % 1000 == 0:
                    completion = round((int(v) / num_vertices) * 100, 2)
                    logger.info(f'processing vertex: {v}, {completion}% completed')
                distance_map = shortest_distance(graph, source=v, weights=g_edge_dist, max_dist=threshold_distance)
                # get vert ids
                verts = np.where(distance_map.a != np.inf)[0]
                # sum heritage
                heritage = np.nansum(g_vert_list_bldgs.a[verts])
                # sum voa
                voa_count = np.nansum(g_vert_voa_count.a[verts])
                # avg voa area
                voa_avg_area = voa_area_arr[verts].flatten()
                voa_avg_area = voa_avg_area[~np.isnan(voa_avg_area)]
                voa_area_mean = np.nanmean(voa_avg_area)
                # NOTE -> clean versions only work for arrays with more than one value otherwise standard deviation is zero
                z_scores = (voa_avg_area - voa_area_mean) / voa_avg_area.std()
                voa_avg_area_clean = voa_avg_area[(z_scores < 3) & (z_scores > -3)]
                voa_area_mean_clean = np.nanmean(voa_avg_area_clean)
                # voa valuations
                voa_avg_val = voa_val_arr[verts].flatten()
                voa_avg_val = voa_avg_val[~np.isnan(voa_avg_val)]
                voa_val_mean = np.nanmean(voa_avg_val)
                voa_cof_val = np.nanstd(voa_avg_val) / voa_val_mean
                z_scores = (voa_avg_val - voa_val_mean) / voa_avg_val.std()
                voa_avg_val_clean = voa_avg_val[(z_scores < 3) & (z_scores > -3)]
                voa_val_mean_clean = np.nanmean(voa_avg_val_clean)
                voa_cof_val_clean = np.nanstd(voa_avg_val_clean) / voa_val_mean_clean
                # voa rate
                voa_avg_rate = voa_rate_arr[verts].flatten()
                voa_avg_rate = voa_avg_rate[~np.isnan(voa_avg_rate)]
                voa_rate_mean = np.nanmean(voa_avg_rate)
                voa_cof_rate = np.nanstd(voa_avg_rate) / voa_rate_mean
                z_scores = (voa_avg_rate - voa_rate_mean) / voa_avg_rate.std()
                voa_avg_rate_clean = voa_avg_rate[(z_scores < 3) & (z_scores > -3)]
                voa_rate_mean_clean = np.nanmean(voa_avg_rate_clean)
                voa_cof_rate_clean = np.nanstd(voa_avg_rate_clean) / voa_rate_mean_clean
                # mixed uses
                uses_arr = class_code_arr[verts].flatten()
                uses_arr = uses_arr[uses_arr != 0]

                accommodation = np.count_nonzero(uses_arr[(uses_arr >= 1010000) & (uses_arr < 1020000)])
                eating = np.count_nonzero(uses_arr[(uses_arr >= 1020000) & (uses_arr < 2000000)])
                commercial = np.count_nonzero(uses_arr[(uses_arr >= 2000000) & (uses_arr < 3000000)])
                attractions = np.count_nonzero(uses_arr[(uses_arr >= 3000000) & (uses_arr < 4000000)])
                entertainment = np.count_nonzero(uses_arr[(uses_arr >= 4000000) & (uses_arr < 5000000)])
                manufacturing = np.count_nonzero(uses_arr[(uses_arr >= 7000000) & (uses_arr < 9000000)])
                retail = np.count_nonzero(uses_arr[(uses_arr >= 9000000) & (uses_arr < 10000000)])
                transport = np.count_nonzero(uses_arr[(uses_arr >= 10000000)])
                property = np.count_nonzero(uses_arr[(uses_arr == 2110192) | (uses_arr == 2110190)])
                health = np.count_nonzero(uses_arr[(uses_arr - (uses_arr % 10000) == 5260000) |
                       (uses_arr - (uses_arr % 10000) == 5280000) | (uses_arr - (uses_arr % 10000) == 5290000)])
                education = np.count_nonzero(uses_arr[(uses_arr - (uses_arr % 10000) == 5270000) |
                       (uses_arr - (uses_arr % 10000) == 5310000) | (uses_arr - (uses_arr % 10000) == 5320000)])
                parks = np.count_nonzero(uses_arr[(uses_arr - (uses_arr % 10000) == 3180000)])
                cultural = np.count_nonzero(uses_arr[(uses_arr - (uses_arr % 10000) == 3170000)])
                sports = np.count_nonzero(uses_arr[(uses_arr - (uses_arr % 10000) == 4240000)])

                uses_unique, uses_counts = np.unique(uses_arr, return_counts=True)
                uses_probs = uses_counts / len(uses_arr)
                mixed_uses_score = stirling_div(uses_unique, uses_probs)
                mixed_uses_primary_score = stirling_div(uses_unique, uses_probs, weight_1=0, weight_2=0, weight_3=1)
                mixed_uses_secondary_score = stirling_div(uses_unique, uses_probs, weight_1=0, weight_2=1, weight_3=0)
                mixed_uses_tertiary_score = stirling_div(uses_unique, uses_probs, weight_1=1, weight_2=0, weight_3=0)
                # add to result set
                res = [
                    heritage,
                    voa_count,
                    voa_area_mean,
                    voa_area_mean_clean,
                    voa_val_mean,
                    voa_val_mean_clean,
                    voa_cof_val,
                    voa_cof_val_clean,
                    voa_rate_mean,
                    voa_rate_mean_clean,
                    voa_cof_rate,
                    voa_cof_rate_clean,
                    mixed_uses_score,
                    mixed_uses_primary_score,
                    mixed_uses_secondary_score,
                    mixed_uses_tertiary_score,
                    accommodation,
                    eating,
                    commercial,
                    attractions,
                    entertainment,
                    manufacturing,
                    retail,
                    transport,
                    property,
                    health,
                    education,
                    parks,
                    cultural,
                    sports,
                    g_vert_id[v]
                ]
                res = [r if not np.isnan(r) else None for r in res]
                results.append(res)

            end_time = time.time()

            logger.info(f'Execution time: {end_time - start_time}s for threshold distance {threshold_distance}')

            logger.info('Writing data back to database')
            db_con = await asyncpg.connect(**db_config)
            await db_con.execute(f'''
                    ALTER TABLE {networks_table}.{verts_table}
                        ADD COLUMN IF NOT EXISTS listed_bldgs_count_{threshold_distance} int,
                        ADD COLUMN IF NOT EXISTS voa_count_{threshold_distance} int,
                        ADD COLUMN IF NOT EXISTS voa_area_mean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_area_mean_clean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_val_mean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_val_mean_clean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_cof_val_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_cof_val_clean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_rate_mean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_rate_mean_clean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_cof_rate_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS voa_cof_rate_clean_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS uses_score_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS uses_score_primary_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS uses_score_secondary_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS uses_score_tertiary_{threshold_distance} real,
                        ADD COLUMN IF NOT EXISTS uses_accommodation_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_eating_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_commercial_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_attractions_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_entertainment_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_manufacturing_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_retail_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_transport_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_property_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_health_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_education_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_parks_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_cultural_{threshold_distance} smallint,
                        ADD COLUMN IF NOT EXISTS uses_sports_{threshold_distance} smallint;
                ''')

            await db_con.executemany(f'''
                UPDATE {networks_table}.{verts_table}
                    SET listed_bldgs_count_{threshold_distance} = $1,
                        voa_count_{threshold_distance} = $2,
                        voa_area_mean_{threshold_distance} = $3,
                        voa_area_mean_clean_{threshold_distance} = $4,
                        voa_val_mean_{threshold_distance} = $5,
                        voa_val_mean_clean_{threshold_distance} = $6,
                        voa_cof_val_{threshold_distance} = $7,
                        voa_cof_val_clean_{threshold_distance} = $8,
                        voa_rate_mean_{threshold_distance} = $9,
                        voa_rate_mean_clean_{threshold_distance} = $10,
                        voa_cof_rate_{threshold_distance} = $11,
                        voa_cof_rate_clean_{threshold_distance} = $12,
                        uses_score_{threshold_distance} = $13,
                        uses_score_primary_{threshold_distance} = $14,
                        uses_score_secondary_{threshold_distance} = $15,
                        uses_score_tertiary_{threshold_distance} = $16,
                        uses_accommodation_{threshold_distance} = $17,
                        uses_eating_{threshold_distance} = $18,
                        uses_commercial_{threshold_distance} = $19,
                        uses_attractions_{threshold_distance} = $20,
                        uses_entertainment_{threshold_distance} = $21,
                        uses_manufacturing_{threshold_distance} = $22,
                        uses_retail_{threshold_distance} = $23,
                        uses_transport_{threshold_distance} = $24,
                        uses_property_{threshold_distance} = $25,
                        uses_health_{threshold_distance} = $26,
                        uses_education_{threshold_distance} = $27,
                        uses_parks_{threshold_distance} = $28,
                        uses_cultural_{threshold_distance} = $29,
                        uses_sports_{threshold_distance} = $30
                    WHERE id = $31
            ''', results)

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
    verts_tables = ['roadnodes_20', 'roadnodes_50', 'roadnodes_100']
    links_tables = ['roadlinks_20', 'roadlinks_50', 'roadlinks_100']
    threshold_distances = [100, 250, 500, 750, 1000, 2000]

    try:
        for verts_table, links_table in zip(verts_tables, links_tables):
            yag = yagmail.SMTP('garethsimons@gmail.com')
            yag.send(to='garethsimons@me.com', subject=f'starting {verts_table} at {threshold_distances}',
                     contents=f'starting network agg on  {verts_table} at {threshold_distances}')
            if verts_table == 'roadnodes_20':
                start = 3
            else:
                start = 1
            for city_id in range(start, 101):
                logger.info(f'Starting network aggregation of data for city id: {city_id} '
                            f'on table {network_schema}.{verts_table} at threshold distance {threshold_distances}')
                loop.run_until_complete(aggregator(db_config, network_schema, verts_table, links_table, threshold_distances, city_id))

    except Exception as e:
        yag = yagmail.SMTP('garethsimons@gmail.com')
        yag.send(to='garethsimons@me.com', subject=f'error {e}',
                 contents=f'error for network agg on  {verts_table} at {threshold_distances} with error: {e}')
        raise e
