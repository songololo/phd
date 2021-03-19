"""
Aggregates measures per node based on network distances

"""
import os
import math
import logging
import time
import asyncio
import numpy as np
import asyncpg
from graph_tool.all import *
from numba import jit
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hill(uses_probs, exp):
    '''
    Hill numbers - express actual diversity as opposed e.g. to entropy
    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index
    '''
    # exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    # see "Entropy and diversity" by Lou Jost
    if exp == 1:
        return math.exp(entropy(uses_probs))
    else:
        diversity = 0
        for i in uses_probs:
            diversity += i**exp
        return diversity ** (1 / (1 - exp))


@jit(nopython=True, nogil=True)
def stirling_div(uses_unique, uses_probs, weight_1=0.25, weight_2=0.5, weight_3=1, alpha=1, beta=1):
    mixedUses = 0  # variable for additive calculations of distance * p1 * p2
    #print(uses_unique)
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
            mixedUses += w**alpha * (a_proportion * b_proportion)**beta
            j += 1
    return mixedUses


@jit(nopython=True, nogil=True)
def quadratic_diversity_decomposed(uses_unique, uses_probs, num, weight_1=0.25, weight_2=0.5, weight_3=1):
    stirling = 0
    disparity = 0
    balance = 0
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
            stirling += w * (a_proportion * b_proportion)
            disparity += w
            balance += a_proportion * b_proportion
            j += 1

    simpson_index = 2 * balance
    n = 0
    if num > 1:
        n = 1/(num*(num-1)/2)
    species_distinctness = n * disparity
    d = n * balance
    balance_factor = 0
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
            balance_factor += (w - species_distinctness) * (a_proportion * b_proportion - d)
            j += 1
    balance_factor = 2 * balance_factor
    combined = simpson_index * species_distinctness + balance_factor
    return simpson_index, species_distinctness, balance_factor, combined


@jit(nopython=True, nogil=True)
def aggregate_uses(verts, class_code_arr, betas, uses_unique, uses_count):

    accommodation = 0
    eating = 0
    commercial = 0
    attractions = 0
    entertainment = 0
    manufacturing = 0
    retail = 0
    transport = 0
    property = 0
    health = 0
    education = 0
    parks = 0
    cultural = 0
    sports = 0

    # mixed uses - these are nested arrays
    for v_idx in verts:

        for use in class_code_arr[v_idx]:
            uses_count[np.where(uses_unique == use)] += 1 * betas[v_idx]

        accommodation += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 1010000) & (class_code_arr[v_idx] < 1020000)])) * betas[v_idx]
        eating += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 1020000) & (class_code_arr[v_idx] < 2000000)])) * betas[v_idx]
        commercial += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 2000000) & (class_code_arr[v_idx] < 3000000)])) * betas[v_idx]
        attractions += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 3000000) & (class_code_arr[v_idx] < 4000000)])) * betas[v_idx]
        entertainment += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 4000000) & (class_code_arr[v_idx] < 5000000)])) * betas[v_idx]
        manufacturing += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 7000000) & (class_code_arr[v_idx] < 9000000)])) * betas[v_idx]
        retail += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 9000000) & (class_code_arr[v_idx] < 10000000)])) * betas[v_idx]
        transport += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] >= 10000000)])) * betas[v_idx]
        property += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] == 2110192) | (class_code_arr[v_idx] == 2110190)])) * betas[v_idx]
        health += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 5260000) |
                    (class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 5280000) |
                    (class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 5290000)])) * betas[v_idx]
        education += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 5270000) |
                    (class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 5310000) |
                    (class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 5320000)])) * betas[v_idx]
        parks += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 3180000)])) * betas[v_idx]
        cultural += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 3170000)])) * betas[v_idx]
        sports += np.sum(np.isfinite(class_code_arr[v_idx]
                    [(class_code_arr[v_idx] - (class_code_arr[v_idx] % 10000) == 4240000)])) * betas[v_idx]

    return accommodation, eating, commercial, attractions, entertainment, manufacturing, retail, transport, property,\
           health, education, parks, cultural, sports, uses_count


async def aggregator(db_config, networks_table, verts_table, links_table, city_pop_id, threshold_distances):

    # the graph
    graph = Graph(directed=False)

    # edge property maps
    g_edge_dist = graph.new_edge_property('float')
    # vertex property maps
    g_vert_id = graph.new_vertex_property('int')

    # open database connection
    logger.info(f'loading data from {networks_table}.{verts_table}')
    vertex_dict = {}
    db_con = await asyncpg.connect(**db_config)
    async with db_con.transaction():
        # NB -> Can't use ST_Length on dual graph, use column instead
        async for record in db_con.cursor(f'''
            SELECT id, node_a, node_b, distance
                FROM {networks_table}.{links_table}
                    WHERE city_pop_id::int = {city_pop_id}'''):
            # add start vertex if not present
            start = record['node_a']
            if start not in vertex_dict:
                vertex_dict[start] = graph.add_vertex()
            g_vert_id[vertex_dict[start]] = start
            # add end vertex if not present
            end = record['node_b']
            if end not in vertex_dict:
                vertex_dict[end] = graph.add_vertex()
            g_vert_id[vertex_dict[end]] = end
            # add edge
            e = graph.add_edge(vertex_dict[start], vertex_dict[end])
            g_edge_dist[e] = record['distance']

    logger.info(f'number of vertices: {graph.num_vertices()}')

    if graph.num_vertices():

        # fetch OS POI classification max length
        logger.info('setting maximum array size for OS POI aggregation from DB array sizes')
        class_size_cap = await db_con.fetchval(f'''
            SELECT cardinality(class_code_arr) AS c
                FROM {networks_table}.{verts_table}
                WHERE city_pop_id::int = {city_pop_id}
                    AND class_code_arr IS NOT NULL
                ORDER BY c DESC
                LIMIT 1''')
        max_cap = 1000
        if class_size_cap and class_size_cap < max_cap:
            logger.info(f'Setting array size cap to {class_size_cap}')
        else:
            logger.warning(f'Size cap of {class_size_cap} being capped to {max_cap}')
            class_size_cap = max_cap

        # setup POI classification codes array
        class_code_arr = np.zeros((graph.num_vertices(), class_size_cap), dtype=int)

        # load the node properties
        logger.info('building arrays')
        async with db_con.transaction():
            async for record in db_con.cursor(f'''
                SELECT id, class_code_arr
                    FROM {networks_table}.{verts_table}
                        WHERE city_pop_id::int = {city_pop_id}'''):
                # not all verts will have been loaded with the edges
                if record['id'] not in vertex_dict:
                    continue
                # populate node attributes
                vert_id = vertex_dict[record['id']]
                if record['class_code_arr']:
                    arr = np.array([int(r) for r in record['class_code_arr']])
                    # skip if max cap exceeded for
                    arr = np.pad(arr, (0, class_size_cap - len(arr)), mode='constant', constant_values=0)  # np.nan doesn't work with int
                    class_code_arr[int(vert_id)] = arr
        await db_con.close()
        logger.info(f'edges: {graph.num_edges()}, vertices: {graph.num_vertices()}')

        for threshold_distance in threshold_distances:

            if threshold_distance == 50:
                beta = -0.08
            elif threshold_distance == 100:
                beta = -0.04
            elif threshold_distance == 200:
                beta = -0.02
            elif threshold_distance == 400:
                beta = -0.01
            elif threshold_distance == 800:
                beta = -0.005
            elif threshold_distance == 1600:
                beta = -0.0025
            else:
                raise NotImplementedError(f'NOTE -> no beta matching distance: {threshold_distance}')

            logger.info(f'processing threshold distance {threshold_distance}m at beta {beta}')

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
                # calculate distance weighted coefficients
                beta_weighted_distances = np.exp(beta * distance_map.a)

                # process uses counts and mixed uses
                uses_unique = np.unique(class_code_arr[verts].flatten())
                uses_counts = np.zeros_like(uses_unique, dtype=float)
                accommodation, eating, commercial, attractions, entertainment, manufacturing, retail, transport, \
                    property, health, education, parks, cultural, sports, uses_count = \
                        aggregate_uses(verts, class_code_arr, beta_weighted_distances, uses_unique, uses_counts)

                # the counts of the various types are here weighted by the diversity betas
                # which are distinct from the distance decay betas per above
                # not weighting as this would muddy diversity calcs?
                uses_probs = uses_counts / uses_counts.sum()
                mixed_uses_score_0 = stirling_div(uses_unique, uses_probs, beta=0)
                mixed_uses_score_5 = stirling_div(uses_unique, uses_probs, beta=0.5)
                mixed_uses_score_10 = stirling_div(uses_unique, uses_probs, beta=1)
                mixed_uses_primary_score = stirling_div(uses_unique, uses_probs, beta=1, weight_1=0, weight_2=0, weight_3=1)
                mixed_uses_secondary_score = stirling_div(uses_unique, uses_probs, beta=1, weight_1=0, weight_2=1, weight_3=0)
                mixed_uses_tertiary_score = stirling_div(uses_unique, uses_probs, beta=1, weight_1=1, weight_2=0, weight_3=0)
                mixed_uses_score_hill_0 = hill(uses_probs, 0)
                mixed_uses_score_hill_10 = hill(uses_probs, 1)
                mixed_uses_score_hill_20 = hill(uses_probs, 2)
                num = len(class_code_arr[verts].flatten())
                mixed_uses_d_simpson_index, mixed_uses_d_species_distinctness, mixed_uses_d_balance_factor, \
                mixed_uses_d_combined = quadratic_diversity_decomposed(uses_unique, uses_probs, num)

                # add to result set
                res = [
                    g_vert_id[v],
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
                    mixed_uses_score_0,
                    mixed_uses_score_5,
                    mixed_uses_score_10,
                    mixed_uses_primary_score,
                    mixed_uses_secondary_score,
                    mixed_uses_tertiary_score,
                    mixed_uses_score_hill_0,
                    mixed_uses_score_hill_10,
                    mixed_uses_score_hill_20,
                    mixed_uses_d_simpson_index,
                    mixed_uses_d_species_distinctness,
                    mixed_uses_d_balance_factor,
                    mixed_uses_d_combined
                ]
                res = [r if not np.isnan(r) else None for r in res]
                results.append(res)

            end_time = time.time()

            logger.info(f'Execution time: {end_time - start_time}s for threshold distance {threshold_distance}')

            logger.info('Writing data back to database')
            beta_str = str(beta).replace('-', '').replace('.', '')
            db_con = await asyncpg.connect(**db_config)
            append = f'{threshold_distance}_{beta_str}'
            await db_con.execute(f'''
                    ALTER TABLE {networks_table}.{verts_table}
                        ADD COLUMN IF NOT EXISTS uses_accommodation_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_eating_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_commercial_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_attractions_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_entertainment_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_manufacturing_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_retail_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_transport_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_property_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_health_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_education_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_parks_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_cultural_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_sports_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_score_0_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_score_5_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_score_10_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_score_primary_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_score_secondary_{append} real,
                        ADD COLUMN IF NOT EXISTS uses_score_tertiary_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_score_hill_0_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_score_hill_10_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_score_hill_20_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_d_simpson_index_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_d_species_distinctness_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_d_balance_factor_{append} real,
                        ADD COLUMN IF NOT EXISTS mixed_uses_d_combined_{append} real;
                ''')

            await db_con.executemany(f'''
                UPDATE {networks_table}.{verts_table}
                    SET 
                        uses_accommodation_{append} = $2,
                        uses_eating_{append} = $3,
                        uses_commercial_{append} = $4,
                        uses_attractions_{append} = $5,
                        uses_entertainment_{append} = $6,
                        uses_manufacturing_{append} = $7,
                        uses_retail_{append} = $8,
                        uses_transport_{append} = $9,
                        uses_property_{append} = $10,
                        uses_health_{append} = $11,
                        uses_education_{append} = $12,
                        uses_parks_{append} = $13,
                        uses_cultural_{append} = $14,
                        uses_sports_{append} = $15,
                        mixed_uses_score_0_{append} = $16,
                        mixed_uses_score_5_{append} = $17,
                        mixed_uses_score_10_{append} = $18,
                        uses_score_primary_{append} = $19,
                        uses_score_secondary_{append} = $20,
                        uses_score_tertiary_{append} = $21,
                        mixed_uses_score_hill_0_{append} = $22,
                        mixed_uses_score_hill_10_{append} = $23,
                        mixed_uses_score_hill_20_{append} = $24,
                        mixed_uses_d_simpson_index_{append} = $25,
                        mixed_uses_d_species_distinctness_{append} = $26,
                        mixed_uses_d_balance_factor_{append} = $27,
                        mixed_uses_d_combined_{append} = $28
                    WHERE id = $1
            ''', results)

            await db_con.close()


if __name__ == '__main__':

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
    threshold_distances = [50, 100, 200]

    verts_table = 'roadnodes_full_dual'
    links_table = 'roadlinks_full_dual'
    for city_pop_id in range(1, 2):
        logger.info(
            f'Starting network aggregation of data for city id: {city_pop_id} on table {network_schema}.{verts_table}')
        loop.run_until_complete(
            aggregator(db_config, network_schema, verts_table, links_table, city_pop_id, threshold_distances))

    verts_table = 'roadnodes_100_dual'
    links_table = 'roadlinks_100_dual'
    for city_pop_id in range(1, 2):
        logger.info(
            f'Starting network aggregation of data for city id: {city_pop_id} on table {network_schema}.{verts_table}')
        loop.run_until_complete(
            aggregator(db_config, network_schema, verts_table, links_table, city_pop_id, threshold_distances))

    verts_table = 'roadnodes_50_dual'
    links_table = 'roadlinks_50_dual'
    for city_pop_id in range(1, 2):
        logger.info(
            f'Starting network aggregation of data for city id: {city_pop_id} on table {network_schema}.{verts_table}')
        loop.run_until_complete(
            aggregator(db_config, network_schema, verts_table, links_table, city_pop_id, threshold_distances))

    verts_table = 'roadnodes_20_dual'
    links_table = 'roadlinks_20_dual'
    for city_pop_id in range(1, 2):
        logger.info(
            f'Starting network aggregation of data for city id: {city_pop_id} on table {network_schema}.{verts_table}')
        loop.run_until_complete(
            aggregator(db_config, network_schema, verts_table, links_table, city_pop_id, threshold_distances))

