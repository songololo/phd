import asyncpg
from cityseer.metrics import networks

from src.process.loaders import postGIS_to_networkX


async def centrality_shortest(db_config, nodes_table, links_table, city_pop_id, distances):
    logger.info(f'Loading graph for city: {city_pop_id} derived from table: {nodes_table}')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, city_pop_id)
    if len(G) == 0:
        return
    logger.info(f'Generating node map and edge map')
    N = networks.NetworkLayerFromNX(G, distances=distances)

    logger.info('Calculating shortest-path node centralities')
    start = time.localtime()
    node_measures = [
        'node_density',
        'node_farness',
        'node_cycles',
        'node_harmonic',
        'node_beta',
        'node_betweenness',
        'node_betweenness_beta'
    ]
    N.node_centrality(measures=node_measures)
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    logger.info('Calculating shortest-path segment centralities')
    start = time.localtime()
    segment_measures = [
        'segment_density',
        'segment_harmonic',
        'segment_beta',
        'segment_betweenness'
    ]
    N.segment_centrality(measures=segment_measures)
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    logger.info('Calculating simplest-path node centralities')
    start = time.localtime()
    angular_node_measures = [
        'node_harmonic_angular',
        'node_betweenness_angular'
    ]
    N.node_centrality(measures=angular_node_measures, angular=True)
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    logger.info('Calculating simplest-path segment centralities')
    start = time.localtime()
    angular_segment_measures = [
        'segment_harmonic_hybrid',
        'segment_betweeness_hybrid'
    ]
    N.segment_centrality(measures=angular_segment_measures, angular=True)
    logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

    # Quite slow writing to database so do all distances at once
    logger.info('Prepping data for database')
    metrics = N.metrics_to_dict()
    bulk_data = []
    #
    comb_measures = node_measures + segment_measures
    com_ang_measures = angular_node_measures + angular_segment_measures
    for k, v in metrics.items():
        # first check that this is a live node (i.e. within the original city boundary)
        if not v['live']:
            continue
        # start node data list - initialise with node label
        node_data = [k]

        # pack shortest path data
        for measure in comb_measures:
            inner_data = []
            for d in distances:
                inner_data.append(v['centrality'][measure][d])
            node_data.append(inner_data)

        # pack simplest path data
        for ang_measure in com_ang_measures:
            inner_ang_data = []
            for d in distances:
                inner_ang_data.append(v['centrality'][ang_measure][d])
            node_data.append(inner_ang_data)
        bulk_data.append(node_data)

    logger.info('Writing data back to database')
    db_con = await asyncpg.connect(**db_config)
    # check that the columns exist
    # do this separately to control the order in which the columns are added (by theme instead of distance)
    for measure in comb_measures:
        # prepend with "c_"
        c_measure = f'c_{measure}'
        await db_con.execute(f'ALTER TABLE {nodes_table} ADD COLUMN IF NOT EXISTS {c_measure} real[];')
    for ang_measure in com_ang_measures:
        c_ang_measure = f'c_{ang_measure}'
        await db_con.execute(f'ALTER TABLE {nodes_table} ADD COLUMN IF NOT EXISTS {c_ang_measure} real[];')
    await db_con.executemany(f'''
    UPDATE {nodes_table}
        SET
            c_node_density = $2,
            c_node_farness = $3,
            c_node_cycles = $4,
            c_node_harmonic = $5,
            c_node_beta = $6,
            c_node_betweenness = $7,
            c_node_betweenness_beta = $8,
            c_segment_density = $9,
            c_segment_harmonic = $10,
            c_segment_beta = $11,
            c_segment_betweenness = $12,
            c_node_harmonic_angular = $13,
            c_node_betweenness_angular = $14,
            c_segment_harmonic_hybrid = $15,
            c_segment_betweeness_hybrid = $16
        WHERE id = $1
    ''', bulk_data)
    await db_con.close()


if __name__ == '__main__':

    import time
    import asyncio
    import datetime
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    nodes_table = 'analysis.nodes_20'
    links_table = 'analysis.links_20'
    # distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 3200, 4800, 6400, 8000]
    distances = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

    # 931 for full and 20m
    # 1 only for 50m and 100m
    for city_pop_id in range(1, 2):
        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')
        asyncio.run(centrality_shortest(db_config,
                                        nodes_table,
                                        links_table,
                                        city_pop_id,
                                        distances))
        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')
        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
