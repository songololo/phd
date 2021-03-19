import asyncpg
from process.loaders import postGIS_to_networkX
from cityseer.util import graphs
from cityseer.metrics import networks
from shapely import geometry
import numpy as np


async def centrality_simplest(db_config, nodes_table, links_table, boundary_table, city_pop_id, distances):
    logger.info(f'Loading graph for city: {city_pop_id} derived from table: {nodes_table}')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)

    G = graphs.nX_to_dual(G)  # convert to dual

    start = time.localtime()

    # SIMPLEST PATH
    logger.info(f'Generating simplest paths network layer')
    N_simplest = networks.Network_Layer_From_nX(G, distances=distances, angular=True)
    logger.info('Calculating simplest path centralities')
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

    # ADD DUAL GRAPH'S EDGES TABLE
    logger.info('Preparing dual edges table')
    links_table_name_only = links_table
    if '.' in links_table_name_only:
        links_table_name_only = links_table_name_only.split('.')[-1]
    await db_con.execute(f'''
        -- add dual links table
        CREATE TABLE IF NOT EXISTS {links_table}_dual (
          id text PRIMARY KEY,
          parent_id bigint,
          city_pop_id int,
          node_a text,
          node_b text,
          angle real,
          distance real,
          geom geometry(Linestring, 27700)
        );
        CREATE INDEX IF NOT EXISTS geom_idx_{links_table_name_only}_dual ON {links_table}_dual USING GIST (geom);
        ''')

    logger.info('Preparing data for dual edges table')
    dual_edge_data = []
    parent_primal_counter = {}
    for s, e, d in G.edges(data=True):
        # only write data back if current city_pop_id - otherwise neighbouring towns can be obfuscated
        if G.nodes[s]['live'] and G.nodes[e]['live']:
            # number each of the new dual edges sequentially based on the original parent primal node
            primal_parent = d['parent_primal_node']
            if primal_parent not in parent_primal_counter:
                parent_primal_counter[primal_parent] = 1
            else:
                parent_primal_counter[primal_parent] +=1
            label = f'{primal_parent}_{parent_primal_counter[primal_parent]}'
            # simple geoms

            # add the data tuple
            dual_edge_data.append((label, primal_parent, city_pop_id, s, e, d['impedance'], d['length'], d['geom'].wkb_hex))

    logger.info('Writing dual edges to DB')
    await db_con.executemany(f'''
        INSERT INTO {links_table}_dual (
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
    logger.info('Preparing dual nodes table')
    nodes_table_name_only = nodes_table
    if '.' in nodes_table_name_only:
        nodes_table_name_only = nodes_table_name_only.split('.')[-1]
    await db_con.execute(f'''
        -- add dual nodes table
        CREATE TABLE IF NOT EXISTS {nodes_table}_dual (
            id text PRIMARY KEY,
            city_pop_id int,
            geom geometry(Point, 27700)
        );
        CREATE INDEX IF NOT EXISTS geom_idx_{nodes_table_name_only}_dual ON {nodes_table}_dual USING GIST (geom);
        CREATE INDEX IF NOT EXISTS city_pop_idx_{nodes_table_name_only}_dual ON {nodes_table}_dual (city_pop_id);
        ''')

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
            await db_con.execute(f'ALTER TABLE {nodes_table}_dual ADD COLUMN IF NOT EXISTS {theme.format(d=dist)} real;')

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

        # start node data list - initialise with node label, city_pop_id, and geom
        node_data = [k, city_pop_id, geometry.Point(v['x'], v['y']).wkb_hex]

        for theme in themes:
            for metrics in [metrics_simplest[k], metrics_shortest[k]]:
                for d in distances:
                    node_data.append(metrics['centrality'][theme][d])

        dual_verts_data.append(node_data)

    logger.info('Writing dual nodes to DB')
    await db_con.executemany(f'''
        INSERT INTO {nodes_table}_dual (
                id,
                city_pop_id,
                geom,
                c_node_dens_simpl_50,
                c_node_dens_simpl_100,
                c_node_dens_simpl_150,
                c_node_dens_simpl_200,
                c_node_dens_simpl_300,
                c_node_dens_simpl_400,
                c_node_dens_simpl_600,
                c_node_dens_simpl_800,
                c_node_dens_simpl_1200,
                c_node_dens_simpl_1600,
                c_node_dens_short_50,
                c_node_dens_short_100,
                c_node_dens_short_150,
                c_node_dens_short_200,
                c_node_dens_short_300,
                c_node_dens_short_400,
                c_node_dens_short_600,
                c_node_dens_short_800,
                c_node_dens_short_1200,
                c_node_dens_short_1600,
                c_farness_simpl_50,
                c_farness_simpl_100,
                c_farness_simpl_150,
                c_farness_simpl_200,
                c_farness_simpl_300,
                c_farness_simpl_400,
                c_farness_simpl_600,
                c_farness_simpl_800,
                c_farness_simpl_1200,
                c_farness_simpl_1600,
                c_farness_short_50,
                c_farness_short_100,
                c_farness_short_150,
                c_farness_short_200,
                c_farness_short_300,
                c_farness_short_400,
                c_farness_short_600,
                c_farness_short_800,
                c_farness_short_1200,
                c_farness_short_1600,
                c_between_simpl_50,
                c_between_simpl_100,
                c_between_simpl_150,
                c_between_simpl_200,
                c_between_simpl_300,
                c_between_simpl_400,
                c_between_simpl_600,
                c_between_simpl_800,
                c_between_simpl_1200,
                c_between_simpl_1600,
                c_between_short_50,
                c_between_short_100,
                c_between_short_150,
                c_between_short_200,
                c_between_short_300,
                c_between_short_400,
                c_between_short_600,
                c_between_short_800,
                c_between_short_1200,
                c_between_short_1600)
            VALUES ($1, $2, ST_SetSRID($3::geometry, 27700), $4, $5, $6, $7, $8, $9, $10, 
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, 
                $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, 
                $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63)
            ON CONFLICT (id) DO UPDATE SET
                city_pop_id = $2,
                geom = ST_SetSRID($3::geometry, 27700),
                c_node_dens_simpl_50 = $4,
                c_node_dens_simpl_100 = $5,
                c_node_dens_simpl_150 = $6,
                c_node_dens_simpl_200 = $7,
                c_node_dens_simpl_300 = $8,
                c_node_dens_simpl_400 = $9,
                c_node_dens_simpl_600 = $10,
                c_node_dens_simpl_800 = $11,
                c_node_dens_simpl_1200 = $12,
                c_node_dens_simpl_1600 = $13,
                c_node_dens_short_50 = $14,
                c_node_dens_short_100 = $15,
                c_node_dens_short_150 = $16,
                c_node_dens_short_200 = $17,
                c_node_dens_short_300 = $18,
                c_node_dens_short_400 = $19,
                c_node_dens_short_600 = $20,
                c_node_dens_short_800 = $21,
                c_node_dens_short_1200 = $22,
                c_node_dens_short_1600 = $23,
                c_farness_simpl_50 = $24,
                c_farness_simpl_100 = $25,
                c_farness_simpl_150 = $26,
                c_farness_simpl_200 = $27,
                c_farness_simpl_300 = $28,
                c_farness_simpl_400 = $29,
                c_farness_simpl_600 = $30,
                c_farness_simpl_800 = $31,
                c_farness_simpl_1200 = $32,
                c_farness_simpl_1600 = $33,
                c_farness_short_50 = $34,
                c_farness_short_100 = $35,
                c_farness_short_150 = $36,
                c_farness_short_200 = $37,
                c_farness_short_300 = $38,
                c_farness_short_400 = $39,
                c_farness_short_600 = $40,
                c_farness_short_800 = $41,
                c_farness_short_1200 = $42,
                c_farness_short_1600 = $43,
                c_between_simpl_50 = $44,
                c_between_simpl_100 = $45,
                c_between_simpl_150 = $46,
                c_between_simpl_200 = $47,
                c_between_simpl_300 = $48,
                c_between_simpl_400 = $49,
                c_between_simpl_600 = $50,
                c_between_simpl_800 = $51,
                c_between_simpl_1200 = $52,
                c_between_simpl_1600 = $53,
                c_between_short_50 = $54,
                c_between_short_100 = $55,
                c_between_short_150 = $56,
                c_between_short_200 = $57,
                c_between_short_300 = $58,
                c_between_short_400 = $59,
                c_between_short_600 = $60,
                c_between_short_800 = $61,
                c_between_short_1200 = $62,
                c_between_short_1600 = $63;
        ''', dual_verts_data)
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

    boundary_table = 'analysis.city_boundaries_150'
    nodes_table = 'analysis.roadnodes_20'
    links_table = 'analysis.roadlinks_20'

    distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

    loop = asyncio.get_event_loop()
    for city_pop_id in range(5, 959):

        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

        loop.run_until_complete(
           centrality_simplest(db_config, nodes_table, links_table, boundary_table, city_pop_id, distances))

        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

        end_time = time.localtime()
        logger.info(f'Started {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
