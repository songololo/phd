"""

"""
import asyncpg
from cityseer.util import graphs
from cityseer.metrics import networks
from process.loaders import postGIS_to_networkX

async def centrality_shortest(db_config,
    nodes_table,
    links_table,
    boundary_table,
    city_pop_id,
    distances):

    logger.info(f'Loading graph for city: {city_pop_id} derived from table: {nodes_table}')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    # generate default lengths and impedances based on geom lengths
    G = graphs.nX_auto_edge_params(G)
    # generate length weighted
    G = graphs.nX_m_weighted_nodes(G)

    logger.info(f'Generating node map and edge map')
    N = networks.Network_Layer_From_nX(G, distances=distances)

    logger.info(f'Calculating shortest paths and centralities for distances: {distances}')
    start = time.localtime()

    N.compute_centrality(close_metrics=['node_density',
                                        'farness_impedance',
                                        'harmonic',
                                        'improved',
                                        'gravity_index',
                                        'cycles'],
                         between_metrics=['betweenness',
                                          'betweenness_decay'])

    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    logger.info('Writing data back to database')
    db_con = await asyncpg.connect(**db_config)
    # check that the columns exist
    # do this separately to control the order in which the columns are added (by theme instead of distance)
    for theme in [
        'c_node_dens_{d}',
        'c_farness_{d}',
        'c_harm_close_{d}',
        'c_gravity_{d}',
        'c_between_{d}',
        'c_between_wt_{d}',
        'c_cycles_{d}',
        'c_imp_close_{d}']:
        for dist in distances:
            await db_con.execute(f'ALTER TABLE {nodes_table} ADD COLUMN IF NOT EXISTS {theme.format(d=dist)} real;')

    db_con.close()

    # Quite slow writing to database so do all distances at once
    logger.info('Prepping data for database')
    metrics = N.metrics_to_dict()
    bulk_data = []
    for k, v in metrics.items():
        # first check that this is a live node (i.e. within the original city boundary)
        if not v['live']:
            continue
        # start node data list - initialise with node label
        node_data = [k]
        for theme in [
            'node_density',
            'farness_impedance',
            'harmonic',
            'gravity_index',
            'betweenness',
            'betweenness_decay',
            'cycles',
            'improved']:
            for d in distances:
                node_data.append(v['centrality'][theme][d])

        bulk_data.append(node_data)

    db_con = await asyncpg.connect(**db_config)

    logger.info(f'Writing to database')
    await db_con.executemany(f'''
        UPDATE {nodes_table}
            SET
                c_node_dens_3200 = $2,
                c_node_dens_6400 = $3,
                c_node_dens_12800 = $4,
                c_farness_3200 = $5,
                c_farness_6400 = $6,
                c_farness_12800 = $7,
                c_harm_close_3200 = $8,
                c_harm_close_6400 = $9,
                c_harm_close_12800 = $10,
                c_gravity_3200 = $11,
                c_gravity_6400 = $12,
                c_gravity_12800 = $13,
                c_between_3200 = $14,
                c_between_6400 = $15,
                c_between_12800 = $16,
                c_between_wt_3200 = $17,
                c_between_wt_6400 = $18,
                c_between_wt_12800 = $19,
                c_cycles_3200 = $20,
                c_cycles_6400 = $21,
                c_cycles_12800 = $22,
                c_imp_close_3200 = $23,
                c_imp_close_6400 = $24,
                c_imp_close_12800 = $25
            WHERE id = $1
    ''', bulk_data)

    await db_con.close()


if __name__ == '__main__':

    '''
    v0.8.10 - 1:07:18 for city 10
    
    v0.8.10rc2 = 0:01:55 for city 100
    v0.8.10rc2 = 1:31:22 for city 10 - auto parallel - uses 600% - but seems slower...?
    
    v0.8.10rc3 = 1:04:38 for city 10
    
    50 => rc3 = 0:06:30
    50 => rc6 = 0:06:22
    50 => rc7 = 0:06:18
    
    100 => rc11 = 00:01:00
    10 => rc11 = 01:07:00
    1 => rc11 = 00
    
    3200, 6400, 12800m
    100 => rc11 = 00:00:49
    10 => rc11 = 0:50:49
    1 => rc11 = 0
    '''

    import time
    import asyncio
    import datetime
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db_config = {
        'host': 'localhost',
        'port': 5433,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    boundary_table = 'analysis.city_boundaries_150'
    nodes_table = 'analysis.roadnodes_full'
    links_table = 'analysis.roadlinks_full'

    distances = [3200, 6400, 12800]

    loop = asyncio.get_event_loop()
    city_pop_id = 100

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    loop.run_until_complete(
        centrality_shortest(db_config,
                            nodes_table,
                            links_table,
                            boundary_table,
                            city_pop_id,
                            distances))

    logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
