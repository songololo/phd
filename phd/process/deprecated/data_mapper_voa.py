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
import asyncpg
from process.loaders import load_network_graph, load_test_network_graph, postGIS_to_landuses_dict, load_test_data_graph
from process.util.shortest_paths import assign_accessibility_data, accessibility_agg
from process.util import mixed_uses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def accessibility_calc(db_config, nodes_table, links_table, boundary_table, data_table, data_id_col, city_pop_id, max_dist, data_where=None, test_mode=False):
    '''
    The VOA variant has some hacks below because multiple columns are required, i.e. the columns are specified manually below
    As a result, the data_cols parameter has been removed from the function signature
    '''

    # import the data
    # FOR TESTING
    if test_mode:
        logger.info('NOTE -> entering test mode')
        # network
        graph_len, verts, city_ids, nbs, lens, degs, network_x_arr, network_y_arr, angles = load_test_network_graph()
        # data
        data_len, data_ids, data_classes, data_x_arr, data_y_arr = load_test_data_graph()
    else:
        logger.info(f'Calculating accessibilities for city {city_pop_id} derived from {nodes_table}')
        # import network data
        graph_len, verts, city_ids, nbs, lens, degs, network_x_arr, network_y_arr, verts_dict = \
            await load_network_graph(db_config, nodes_table, links_table, boundary_table, city_pop_id)
        # import the landuses points and associated data
        data_len, data_ids, total_area, data_x_arr, data_y_arr = await postGIS_to_landuses_dict(db_config, data_table, data_id_col,
                    'total_area', boundary_table, city_pop_id, max_dist=max_dist, data_where=data_where)
        # ignore duplicated data
        _, _, rateable_value, _, _ = await postGIS_to_landuses_dict(db_config, data_table, data_id_col,
                    'adopted_rateable_value', boundary_table, city_pop_id, max_dist=max_dist, data_where=data_where)

    start = time.localtime()

    # assign each data point to the nearest corresponding network node
    # this produces two 1-d arrays, so can be done once for the whole graph without being memory prohibitive
    # node that this is a starting reference and that if the prior point on the centrality path is closer
    # then the distance will be calculated via the prior point instead
    data_assign_map, data_assign_dist = assign_accessibility_data(network_x_arr, network_y_arr, data_x_arr, data_y_arr, max_dist)

    # iterate through each vert and calculate
    logger.info('...iterating network and calculating accessibilities from each network node')

    # TODO: decided distances... 200, 400, 800, 1600???
    # TODO: use betas???
    distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]
    betas = [-0.08, -0.04, -0.02666666666666667, -0.02, -0.013333333333333334, -0.01, -0.006666666666666667, -0.005, -0.0033333333333333335, -0.0025]

    results = {}
    for d in distances:
        results[d] = []

    for arr_idx, vert_idx in enumerate(verts):

        if arr_idx and arr_idx % 1000 == 0:
            logger.info('...progress')
            logger.info(round(arr_idx / graph_len * 100, 2))

        # only compute for nodes in current city
        if not city_ids[arr_idx]:
            continue

        # generate the reachable classes and their respective distances
        reachable, reachable_dist = accessibility_agg(arr_idx, max_dist, nbs, lens, data_classes,
                        network_x_arr, network_y_arr, data_x_arr, data_y_arr, data_assign_map, data_assign_dist)

        # get unique classes, their counts, and nearest - use the default max distance of 1600m
        classes_unique, classes_counts, classes_nearest = mixed_uses.deduce_unique_species(reachable, reachable_dist)

        # iterate the distances and betas
        for dist, beta in zip(distances, betas):

            # set the beta weights
            class_weights = mixed_uses.beta_weights(classes_nearest, beta)

            # filter out the items not within the maximum distance
            cl_unique_trim, cl_counts_trim = mixed_uses.dist_filter(classes_unique, classes_counts, classes_nearest, dist)

            results[dist].append((
                # vert id
                int(vert_idx),
                # voa count

                # voa area

                # voa val

                # voa rate

            ))

    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    if test_mode:

        print(results)

    else:

        logger.info('Writing results to database')
        db_con = await asyncpg.connect(**db_config)
        # add the new columns
        for d_key in results.keys():
            await db_con.execute(f'''
                ALTER TABLE {nodes_table}
                    ADD COLUMN IF NOT EXISTS voa_count_{d_key} int,
                    ADD COLUMN IF NOT EXISTS voa_area_mu_{d_key} real,
                    ADD COLUMN IF NOT EXISTS voa_area_std_{d_key} real,
                    ADD COLUMN IF NOT EXISTS voa_val_mu_{d_key} real,
                    ADD COLUMN IF NOT EXISTS voa_val_std_{d_key} real,
                    ADD COLUMN IF NOT EXISTS voa_rate_mu_{d_key} real,
                    ADD COLUMN IF NOT EXISTS voa_rate_std_{d_key} real;
                ''')

        for d_key, d_vals in results.items():
            await db_con.executemany(f'''
                UPDATE {nodes_table}
                    SET
                        voa_count_{d_key} = $2,
                        voa_area_mu_{d_key} = $3,
                        voa_area_std_{d_key} = $4,
                        voa_val_mu_{d_key} = $5,
                        voa_val_std_{d_key} = $6,
                        voa_rate_mu_{d_key} = $7,
                        voa_rate_std_{d_key} = $8
                    WHERE id = $1;
                ''', d_vals)

        logger.info('DONE -> closing database pool')
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

    boundary_table = 'analysis.city_boundaries_150'
    data_table = 'voa.summary_data_2017'
    data_id_col = 'uarn'
    data_where = "date_last_updated = date('2018-06-01')"
    max_dist = 1600
    test_mode = False

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    nodes_table = 'analysis.roadnodes_20'
    links_table = 'analysis.roadlinks_20'
    city_pop_id = 1
    logger.info(
        f'Starting accessibility calc for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')
    loop.run_until_complete(
        accessibility_calc(db_config, nodes_table, links_table, boundary_table, data_table, data_id_col, city_pop_id,
                           max_dist, data_where=data_where, test_mode=test_mode))

    # testing
    # loop.run_until_complete(accessibility_calc(db_config, None, None, None, None, None, None, None, 400, None, True))

    logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
