"""
Abstract treatment of data relative to the network:
1. Loads a target class column from a target data table
2. Iterates all nodes in the source network table and deduces an array of reachable data classes
as well as their corresponding network distances - this means no preassignment is required
3. Precise mixed-use and class code specific workflows are done from the analysis end,
this gives complete flexibility without cluttering up the datbase

Earlier versions used pre-assignment:
i.e. each landuse was assigned to the closest adjacent road node on the respective road graphs
This required PostGIS pre-processing steps... not ideal because it is cludgy and workflow / error prone
Foregoing pre-assignment makes it easier to iterate and calculate on-the-fly with different land-use configs

Distance thresholds is an issue of beta falloff, not really cutoff distance per se...
B = -0.005 (rougly 800m) arguably better reflects our tendency to walk vs. -0.0025 (1600m) fall-off
It also has the benefit of substantially reduced computational complexity

"""

import logging
import time
import datetime
import asyncio
import numpy as np
import asyncpg
from cityseer import graphs
from process.loaders import postGIS_to_networkX, postGIS_to_landuses_dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def accessibility_calc(db_config, nodes_table, links_table, boundary_table, data_table, data_id_col, data_class_col, data_target_col, city_pop_id, max_dist, data_where=None, test_mode=False):

    logger.info(f'Calculating accessibilities for city {city_pop_id} derived from {nodes_table}')

    # load base network
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    G = graphs.networkX_edge_defaults(G)
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G)



    # load the landuses points and associated data
    data_len, data_ids, data_classes, data_x_arr, data_y_arr = \
        await postGIS_to_landuses_dict(db_config, data_table, data_id_col, data_class_col, boundary_table, city_pop_id, max_dist=max_dist, data_where=data_where)

    start = time.localtime()

    # assign each data point to the nearest corresponding network node
    # this produces two 1-d arrays, so can be done once for the whole graph without being memory prohibitive
    # node that this is a starting reference and that if the prior point on the centrality path is closer
    # then the distance will be calculated via the prior point instead
    logger.info('...assigning data points to nearest node points')
    data_assign_map, data_assign_dist = assign_accessibility_data(network_x_arr, network_y_arr, data_x_arr, data_y_arr, max_dist)

    # iterate through each vert and calculate
    logger.info('...iterating network and calculating accessibilities from each network node')
    db_pool = None
    if not test_mode:
        db_pool = await asyncpg.create_pool(**db_config, min_size=2, max_size=3)
        async with db_pool.acquire() as db_con:
            await db_con.execute(f'''
                ALTER TABLE {nodes_table}
                    ADD COLUMN IF NOT EXISTS {data_target_col}_classes_max_{max_dist} int[],
                    ADD COLUMN IF NOT EXISTS {data_target_col}_distances_max_{max_dist} real[];
                ''')

    for arr_idx, vert_idx in enumerate(verts):

        if arr_idx and arr_idx % 1000 == 0:
            logger.info('...progress')
            logger.info(round(arr_idx / graph_len * 100, 2))

        # only compute for nodes in current city
        if not city_ids[arr_idx]:
            continue

        # generate the reachable codes and their respective distances
        reachable_classes, reachable_classes_dist = accessibility_agg(arr_idx, max_dist, nbs, lens, data_classes,
                        network_x_arr, network_y_arr, data_x_arr, data_y_arr, data_assign_map, data_assign_dist)

        classes_pruned = [int(r_c) for r_c in reachable_classes if np.isfinite(r_c)]
        classes_dist_pruned = [r_d for r_d in reachable_classes_dist if np.isfinite(r_d)]

        assert len(classes_pruned) == len(classes_dist_pruned)

        # can be memory inhibitive so write directly instead of a batch write
        async with db_pool.acquire() as db_con:
            await db_con.execute(f'''
                UPDATE {nodes_table}
                    SET {data_target_col}_classes_max_{max_dist} = $2,
                        {data_target_col}_distances_max_{max_dist} = $3
                    WHERE id = $1;
                ''', int(vert_idx), classes_pruned, classes_dist_pruned)

    await db_pool.close()

    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')


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

    boundary_table = 'analysis.city_boundaries_150'
    data_table = 'os.poi'
    data_id_col = 'urn'
    data_class_col = 'class_code'
    data_target_col = 'poi'
    data_where = "date_last_updated = date('2018-06-01')"
    max_dist = 1600
    test_mode = False

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    nodes_table = 'analysis.roadnodes_full'
    links_table = 'analysis.roadlinks_full'
    city_pop_id = 900
    logger.info(
        f'Starting accessibility calc for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')
    loop.run_until_complete(
        accessibility_calc(db_config, nodes_table, links_table, boundary_table, data_table, data_id_col,
                           data_class_col, data_target_col, city_pop_id, max_dist, data_where=data_where))

    logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')