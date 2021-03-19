"""

"""
import asyncpg
import numpy as np
from cityseer.metrics import networks
from cityseer.util import graphs
from process.loaders import postGIS_to_networkX


async def centrality_simplest(db_config,
                              nodes_table,
                              links_table,
                              boundary_table,
                              city_pop_id,
                              distances):
    logger.info(f'Loading graph for city: {city_pop_id} derived from table: {nodes_table}')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    G = graphs.nX_to_dual(G)  # convert to dual

    start = time.localtime()

    # SIMPLEST PATH
    logger.info(f'Generating simplest paths network layer')
    N_simplest = networks.Network_Layer_From_nX(G, distances=distances, angular=True)
    logger.info(f'Calculating simplest path centralities for distances: {distances}')
    N_simplest.compute_centrality(close_metrics=['node_density', 'farness_impedance', 'farness_distance'],
                                  between_metrics=['betweenness'])

    # SHORTEST PATH
    logger.info(f'Generating centrality paths network layer')
    N_shortest = networks.Network_Layer_From_nX(G, distances=distances, angular=False)
    # override angular distances with normal distances
    assert not np.array_equal(N_shortest.edge_impedances, N_shortest.edge_lengths)
    # impedances are stored in index 3
    N_shortest._edges[:, 3] = N_shortest.edge_lengths
    assert np.array_equal(N_shortest.edge_impedances, N_shortest.edge_lengths)
    logger.info('Calculating centrality path centralities')
    N_shortest.compute_centrality(close_metrics=['node_density', 'farness_impedance', 'farness_distance'],
                                  between_metrics=['betweenness'])

    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    logger.info('Writing data back to database')
    db_con = await asyncpg.connect(**db_config)

    # check that the columns exist
    # do this separately to control the order in which the columns are added (by theme instead of distance)
    for theme in [
        'c_node_dens_simpl_{d}',
        'c_node_dens_short_{d}',
        'c_farness_simpl_{d}',
        'c_farness_short_{d}',
        'c_between_simpl_{d}',
        'c_between_short_{d}']:
        for dist in distances:
            await db_con.execute(
                f'ALTER TABLE {nodes_table}_dual ADD COLUMN IF NOT EXISTS {theme.format(d=dist)} real;')

    db_con.close()

    logger.info('Preparing data for dual nodes table')
    # Quite slow writing to database so do all distances at once
    dual_verts_data = []
    metrics_simplest = N_simplest.metrics_to_dict()
    metrics_shortest = N_shortest.metrics_to_dict()

    themes = [
        'node_density',
        'farness_impedance',
        'betweenness'
    ]

    for k, v in metrics_simplest.items():

        # first check that this is a live node (i.e. within the original city boundary)
        if not v['live']:
            continue

        node_data = [k]
        for theme in themes:
            for metrics in [metrics_simplest[k], metrics_shortest[k]]:
                for d in distances:
                    node_data.append(metrics['centrality'][theme][d])

        dual_verts_data.append(node_data)

    db_con = await asyncpg.connect(**db_config)

    logger.info('Writing dual nodes to DB')
    await db_con.executemany(f'''
        UPDATE {nodes_table}_dual
            SET
                c_node_dens_simpl_3200 = $2,
                c_node_dens_simpl_6400 = $3,
                c_node_dens_short_3200 = $4,
                c_node_dens_short_6400 = $5,
                c_farness_simpl_3200 = $6,
                c_farness_simpl_6400 = $7,
                c_farness_short_3200 = $8,
                c_farness_short_6400 = $9,
                c_between_simpl_3200 = $10,
                c_between_simpl_6400 = $11,
                c_between_short_3200 = $12,
                c_between_short_6400 = $13
            WHERE id = $1
        ''', dual_verts_data)

    await db_con.close()


if __name__ == '__main__':
    '''
    3200, 6400, 12800m
    100 => rc11 = 0:02:09
    10 => rc11 = 2:42:26
    1 => rc11 = 
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

    distances = [3200, 6400]

    loop = asyncio.get_event_loop()
    city_pop_id = 1

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    loop.run_until_complete(
        centrality_simplest(db_config,
                            nodes_table,
                            links_table,
                            boundary_table,
                            city_pop_id,
                            distances))

    logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Started {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
