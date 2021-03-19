import logging
import time
import datetime
import asyncio
import asyncpg

from cityseer.metrics import layers, networks
from process.loaders import postGIS_to_valuations_dict, postGIS_to_networkX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def valuations_calc(db_config,
                          nodes_table,
                          links_table,
                          boundary_table,
                          data_table,
                          city_pop_id,
                          distances,
                          data_where=None):

    logger.info(
        f'Starting shortest-path valuation calcs for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')

    logger.info(f'Loading network data')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    N = networks.Network_Layer_From_nX(G, distances)

    logger.info(f'Loading POI data')
    data_dict = await postGIS_to_valuations_dict(db_config,
                                               data_table,
                                               boundary_table,
                                               city_pop_id,
                                               max_dist=max_dist,
                                               data_where=data_where)

    # derive the landuse labels, classes, encodings
    voa_areas = [v['area'] for v in data_dict.values()]
    voa_vals = [v['val'] for v in data_dict.values()]
    voa_rates = [v['rate'] for v in data_dict.values()]

    logger.info('Creating data layer')
    D = layers.Data_Layer_From_Dict(data_dict)

    start = time.localtime()

    logger.info('Assigning data points to the network')
    D.assign_to_network(N, max_dist=400)

    # compute
    logger.info('Computing area and valuation stats')
    D.compute_stats_multiple(['areas', 'valuations', 'rates'], [voa_areas, voa_vals, voa_rates])
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    # add the new columns
    logger.info('Preparing database')
    db_con = await asyncpg.connect(**db_config)
    for theme in ['area_max_{dist}',
                  'area_min_{dist}',
                  'area_mean_{dist}',
                  'area_mean_wt_{dist}',
                  'area_variance_{dist}',
                  'area_variance_wt_{dist}',
                  'val_max_{dist}',
                  'val_min_{dist}',
                  'val_mean_{dist}',
                  'val_mean_wt_{dist}',
                  'val_variance_{dist}',
                  'val_variance_wt_{dist}',
                  'rate_max_{dist}',
                  'rate_min_{dist}',
                  'rate_mean_{dist}',
                  'rate_mean_wt_{dist}',
                  'rate_variance_{dist}',
                  'rate_variance_wt_{dist}']:
        for d_key in distances:
            await db_con.execute(f'''
                ALTER TABLE {nodes_table} ADD COLUMN IF NOT EXISTS {theme.format(dist=d_key)} real;
            ''')

    # aggregate the data
    logger.info('Aggregating results')

    bulk_data = []

    for idx, uid in enumerate(N.uids):

        # first check that this is a live node (i.e. within the original city boundary)
        if not N.live[idx]:
            continue

        node_data = [uid]

        for th_key in ['areas', 'valuations', 'rates']:

            for s_key in ['max',
                          'min',
                          'mean',
                          'mean_weighted',
                          'variance',
                          'variance_weighted']:

                for d_key in distances:
                    node_data.append(N.metrics['stats'][th_key][s_key][d_key][idx])

        bulk_data.append(tuple(node_data))

    logger.info('Writing results to database')
    await db_con.executemany(f'''
        UPDATE {nodes_table}
            SET
                area_max_50 = $2,
                area_max_100 = $3,
                area_max_200 = $4,
                area_max_400 = $5,
                area_max_800 = $6,
                area_max_1600 = $7,
                area_min_50 = $8,
                area_min_100 = $9,
                area_min_200 = $10,
                area_min_400 = $11,
                area_min_800 = $12,
                area_min_1600 = $13,
                area_mean_50 = $14,
                area_mean_100 = $15,
                area_mean_200 = $16,
                area_mean_400 = $17,
                area_mean_800 = $18,
                area_mean_1600 = $19,
                area_mean_wt_50 = $20,
                area_mean_wt_100 = $21,
                area_mean_wt_200 = $22,
                area_mean_wt_400 = $23,
                area_mean_wt_800 = $24,
                area_mean_wt_1600 = $25,
                area_variance_50 = $26,
                area_variance_100 = $27,
                area_variance_200 = $28,
                area_variance_400 = $29,
                area_variance_800 = $30,
                area_variance_1600 = $31,
                area_variance_wt_50 = $32,
                area_variance_wt_100 = $33,
                area_variance_wt_200 = $34,
                area_variance_wt_400 = $35,
                area_variance_wt_800 = $36,
                area_variance_wt_1600 = $37,
                val_max_50 = $38,
                val_max_100 = $39,
                val_max_200 = $40,
                val_max_400 = $41,
                val_max_800 = $42,
                val_max_1600 = $43,
                val_min_50 = $44,
                val_min_100 = $45,
                val_min_200 = $46,
                val_min_400 = $47,
                val_min_800 = $48,
                val_min_1600 = $49,
                val_mean_50 = $50,
                val_mean_100 = $51,
                val_mean_200 = $52,
                val_mean_400 = $53,
                val_mean_800 = $54,
                val_mean_1600 = $55,
                val_mean_wt_50 = $56,
                val_mean_wt_100 = $57,
                val_mean_wt_200 = $58,
                val_mean_wt_400 = $59,
                val_mean_wt_800 = $60,
                val_mean_wt_1600 = $61,
                val_variance_50 = $62,
                val_variance_100 = $63,
                val_variance_200 = $64,
                val_variance_400 = $65,
                val_variance_800 = $66,
                val_variance_1600 = $67,
                val_variance_wt_50 = $68,
                val_variance_wt_100 = $69,
                val_variance_wt_200 = $70,
                val_variance_wt_400 = $71,
                val_variance_wt_800 = $72,
                val_variance_wt_1600 = $73,
                rate_max_50 = $74,
                rate_max_100 = $75,
                rate_max_200 = $76,
                rate_max_400 = $77,
                rate_max_800 = $78,
                rate_max_1600 = $79,
                rate_min_50 = $80,
                rate_min_100 = $81,
                rate_min_200 = $82,
                rate_min_400 = $83,
                rate_min_800 = $84,
                rate_min_1600 = $85,
                rate_mean_50 = $86,
                rate_mean_100 = $87,
                rate_mean_200 = $88,
                rate_mean_400 = $89,
                rate_mean_800 = $90,
                rate_mean_1600 = $91,
                rate_mean_wt_50 = $92,
                rate_mean_wt_100 = $93,
                rate_mean_wt_200 = $94,
                rate_mean_wt_400 = $95,
                rate_mean_wt_800 = $96,
                rate_mean_wt_1600 = $97,
                rate_variance_50 = $98,
                rate_variance_100 = $99,
                rate_variance_200 = $100,
                rate_variance_400 = $101,
                rate_variance_800 = $102,
                rate_variance_1600 = $103,
                rate_variance_wt_50 = $104,
                rate_variance_wt_100 = $105,
                rate_variance_wt_200 = $106,
                rate_variance_wt_400 = $107,
                rate_variance_wt_800 = $108,
                rate_variance_wt_1600 = $109
            WHERE id = $1;
        ''', bulk_data)
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
    max_dist = 1600

    nodes_table = 'analysis.roadnodes_20'
    links_table = 'analysis.roadlinks_20'

    distances = [50, 100, 200, 400, 800, 1600]

    for city_pop_id in range(1, 959):

        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

        loop.run_until_complete(
            valuations_calc(db_config,
                            nodes_table,
                            links_table,
                            boundary_table,
                            data_table,
                            city_pop_id,
                            distances))

        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
