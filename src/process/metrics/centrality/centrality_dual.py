import asyncpg
from cityseer.metrics import networks
from cityseer.tools import graphs

from src.process.loaders import postGIS_to_networkX


async def centrality_dual(db_config,
                          nodes_table,
                          links_table,
                          city_pop_id,
                          distances):
    logger.info(f'Loading graph for city: {city_pop_id} '
                f'derived from table: {nodes_table}')
    G = await postGIS_to_networkX(db_config,
                                  nodes_table,
                                  links_table,
                                  city_pop_id)
    if len(G) == 0:
        return
    logger.info('Casting to dual')
    G = graphs.nX_to_dual(G)  # convert to dual
    logger.info(f'Generating node map and edge map')
    N = networks.NetworkLayerFromNX(G, distances=distances)
    # the round trip graph is needed for the generated lengths, angles, etc.
    logger.info('Making round-trip graph')
    G_round_trip = N.to_networkX()

    db_con = await asyncpg.connect(**db_config)
    # ADD DUAL GRAPH'S VERTICES TABLE
    logger.info('Preparing dual nodes table')
    nodes_table_name_only = nodes_table
    if '.' in nodes_table_name_only:
        nodes_table_name_only = nodes_table_name_only.split('.')[-1]
    await db_con.execute(f'''
    -- add dual nodes table
    CREATE TABLE IF NOT EXISTS {nodes_table}_dual (
        id text PRIMARY KEY,
        city_pop_id int,
        within bool,
        geom geometry(Point, 27700)
    );
    CREATE INDEX IF NOT EXISTS geom_idx_{nodes_table_name_only}_dual
        ON {nodes_table}_dual USING GIST (geom);
    CREATE INDEX IF NOT EXISTS city_pop_idx_{nodes_table_name_only}_dual
        ON {nodes_table}_dual (city_pop_id);
    ''')

    logger.info('Preparing dual nodes data')
    dual_nodes_data = []
    for n, d in G_round_trip.nodes(data=True):
        dual_nodes_data.append([
            n,
            city_pop_id,
            d['live'],  # within
            d['x'],
            d['y']
        ])

    logger.info('Writing dual nodes to DB')
    await db_con.executemany(f'''
        INSERT INTO {nodes_table}_dual (id, city_pop_id, within, geom)
        VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($4, $5), 27700))
        ON CONFLICT DO NOTHING;
    ''', dual_nodes_data)

    logger.info('Preparing dual edges table')
    links_table_name_only = links_table
    if '.' in links_table_name_only:
        links_table_name_only = links_table_name_only.split('.')[-1]
    await db_con.execute(f'''
    -- add dual links table
    CREATE TABLE IF NOT EXISTS {links_table}_dual (
      id text PRIMARY KEY,
      parent_id text,
      city_pop_id int,
      node_a text,
      node_b text,
      distance real,
      angle real,
      impedance_factor real,
      geom geometry(Linestring, 27700)
    );
    CREATE INDEX IF NOT EXISTS city_pop_idx_{links_table_name_only}_dual 
        ON {links_table}_dual (city_pop_id);
    CREATE INDEX IF NOT EXISTS geom_idx_{links_table_name_only}_dual 
        ON {links_table}_dual USING GIST (geom);
    ''')

    # prepare the dual edges and nodes tables
    logger.info('Preparing data for dual edges table')
    dual_edge_data = []
    parent_primal_counter = {}
    for s, e, d in G_round_trip.edges(data=True):
        # number each of the new dual edges sequentially
        # based on the original parent primal node
        primal_parent = d['parent_primal_node']
        if primal_parent not in parent_primal_counter:
            parent_primal_counter[primal_parent] = 1
        else:
            parent_primal_counter[primal_parent] += 1
        label = f'{primal_parent}_{parent_primal_counter[primal_parent]}'
        # add the data tuple
        dual_edge_data.append((label,
                               primal_parent,
                               city_pop_id,
                               s,
                               e,
                               d['length'],
                               d['angle_sum'],
                               d['imp_factor'],
                               d['geom'].wkb_hex))

    logger.info('Writing dual edges to DB')
    await db_con.executemany(f'''
    INSERT INTO {links_table}_dual (
        id,
        parent_id,
        city_pop_id,
        node_a,
        node_b,
        distance,
        angle,
        impedance_factor,
        geom)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, ST_SetSRID($9::geometry, 27700))
    ON CONFLICT DO NOTHING;
    ''', dual_edge_data)
    await db_con.close()

    logger.info('Calculating centrality paths and centralities '
                'for centrality-path heuristics')
    start = time.localtime()
    measures = [
        'node_density',
        'node_harmonic',
        'node_beta',
        'node_betweenness',
        'node_betweenness_beta']
    N.node_centrality(measures=measures)
    time_duration = datetime.timedelta(
        seconds=time.mktime(time.localtime()) - time.mktime(start))
    logger.info(f'Algo duration: {time_duration}')
    logger.info('Calculating centrality paths and centralities '
                'for simplest-path heuristics')
    start = time.localtime()
    angular_measures = [
        'node_harmonic_angular',
        'node_betweenness_angular']
    N.node_centrality(measures=angular_measures, angular=True)
    time_duration = datetime.timedelta(
        seconds=time.mktime(time.localtime()) - time.mktime(start))
    logger.info(f'Algo duration: {time_duration}')
    db_con = await asyncpg.connect(**db_config)
    # check that the columns exist
    # do this separately to control the order in which the columns are added (by theme instead of distance)
    for measure in measures:
        # prepend with "c_"
        c_measure = f'c_{measure}'
        await db_con.execute(f'''
        ALTER TABLE {nodes_table}_dual 
            ADD COLUMN IF NOT EXISTS {c_measure} real[];
        ''')
    for ang_measure in angular_measures:
        c_ang_measure = f'c_{ang_measure}'
        await db_con.execute(f'''
        ALTER TABLE {nodes_table}_dual
            ADD COLUMN IF NOT EXISTS {c_ang_measure} real[];
        ''')
    # Quite slow writing to database so do all distances at once
    logger.info('Prepping data for database')
    metrics = N.metrics_to_dict()
    bulk_data = []
    for k, v in metrics.items():
        # first check that this is a live node
        # (i.e. within the original city boundary)
        if not v['live']:
            continue
        # start node data list - initialise with node label
        node_data = [k]
        # pack centrality path data
        for measure in measures:
            inner_data = []
            for d in distances:
                inner_data.append(v['centrality'][measure][d])
            node_data.append(inner_data)
        # pack simplest path data
        for ang_measure in angular_measures:
            inner_ang_data = []
            for d in distances:
                inner_ang_data.append(v['centrality'][ang_measure][d])
            node_data.append(inner_ang_data)
        bulk_data.append(node_data)
    logger.info('Writing data back to database')
    await db_con.executemany(f'''
     UPDATE {nodes_table}_dual
         SET
             c_node_density = $2,
             c_node_harmonic = $3,
             c_node_beta = $4,
             c_node_betweenness = $5,
             c_node_betweenness_beta = $6,
             c_node_harmonic_angular = $7,
             c_node_betweenness_angular = $8
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
    nodes_table = 'analysis.nodes_full'
    links_table = 'analysis.links_full'
    distances = [50, 100, 200, 300, 400, 500, 600, 700, 800,
                 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

    for city_pop_id in range(1, 2):
        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} '
                    f'at {start_time[3]}h:{start_time[4]}m')
        asyncio.run(centrality_dual(db_config,
                                    nodes_table,
                                    links_table,
                                    city_pop_id,
                                    distances))
        time_duration = datetime.timedelta(
            seconds=time.mktime(time.localtime()) - time.mktime(start_time))
        logger.info(f'Duration: {time_duration}')
        end_time = time.localtime()
        logger.info(f'Started {end_time[0]}-{end_time[1]}-{end_time[2]} '
                    f'at {end_time[3]}h:{end_time[4]}m')
