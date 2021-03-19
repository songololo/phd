import asyncio
import logging
import time

import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def clone_tables(db_config, vacuum):
    db_con = await asyncpg.connect(**db_config)

    logger.info('Creating schema if not existing')
    await db_con.execute('CREATE SCHEMA IF NOT EXISTS ml;')

    b_exists = await db_con.fetchval(f'''
        SELECT EXISTS (
           SELECT 1
           FROM   information_schema.tables 
           WHERE  table_schema = 'ml'
           AND    table_name = 'city_boundaries_150'
        );
    ''')

    if b_exists:
        logger.info('Boundaries table already exists, skipping...')
    else:
        logger.info('Copying boundaries table')
        await db_con.execute(f'''
            CREATE TABLE ml.city_boundaries_150 AS SELECT * FROM analysis.city_boundaries_150;
            ALTER TABLE ml.city_boundaries_150 DROP COLUMN id;
            ALTER TABLE ml.city_boundaries_150 ADD COLUMN id SERIAL PRIMARY KEY;
            CREATE INDEX geom_idx_boundaries ON ml.city_boundaries_150 USING GIST (geom);
            CREATE INDEX geom_petite_idx_boundaries ON ml.city_boundaries_150 USING GIST (geom_petite);
            CREATE INDEX pop_idx_boundaries ON ml.city_boundaries_150 (pop_id);
        ''')

    rl_tables = ['roadlinks_full', 'roadlinks_100', 'roadlinks_50', 'roadlinks_20']

    for rl_table in rl_tables:

        rl_exists = await db_con.fetchval(f'''
                SELECT EXISTS (
                   SELECT 1
                   FROM   information_schema.tables 
                   WHERE  table_schema = 'ml'
                   AND    table_name = '{rl_table}'
                );
            ''')

        if rl_exists:
            logger.info(f'Roadlinks table {rl_table} already exists, skipping...')

        else:
            logger.info(f'Copying roadlinks table: {rl_table}')

            await db_con.execute(f'''
                CREATE TABLE ml.{rl_table} AS SELECT * FROM analysis.{rl_table};
                ALTER TABLE ml.{rl_table} ADD PRIMARY KEY (id);
                CREATE INDEX geom_idx_{rl_table} ON ml.{rl_table} USING GIST (geom);
                CREATE INDEX start_node_idx_{rl_table} ON ml.{rl_table} (start_node);
                CREATE INDEX end_node_idx_{rl_table} ON ml.{rl_table} (end_node);
                CREATE INDEX pop_idx_{rl_table} ON ml.{rl_table} (city_pop_id);
            ''')

    # CREATE BASE roadnodes tables
    rn_tables = ['roadnodes_full', 'roadnodes_100', 'roadnodes_50', 'roadnodes_20']

    roadnode_columns = [
        'id',
        'neighbour_nodes',
        'edges',
        'parent_edges',
        'geom',
        'city_pop_id'
    ]

    for rn_table in rn_tables:

        rn_exists = await db_con.fetchval(f'''
                        SELECT EXISTS (
                           SELECT 1
                           FROM   information_schema.tables 
                           WHERE  table_schema = 'ml'
                           AND    table_name = '{rn_table}'
                        );
                    ''')

        if rn_exists:
            logger.info(f'Roadnodes table {rn_table} already exists, skipping...')

        else:
            logger.info(f'Copying roadnodes table: {rn_table}')

            await db_con.execute(f'''
                    CREATE TABLE ml.{rn_table} AS SELECT {', '.join(roadnode_columns)} FROM analysis.{rn_table};
                    ALTER TABLE ml.{rn_table} ADD PRIMARY KEY (id);
                    CREATE INDEX geom_idx_{rn_table} ON ml.{rn_table} USING GIST (geom);
                    CREATE INDEX pop_idx_{rn_table} ON ml.{rn_table} (city_pop_id);
                ''')

    # CREATE METRICS roanodes tables
    rn_metrics_tables = ['roadnodes_full_metrics', 'roadnodes_100_metrics', 'roadnodes_50_metrics',
                         'roadnodes_20_metrics']

    roadnode_metrics_columns = ['id', 'city_pop_id']

    for theme in [
        'c_gravity_{d}',
        'c_between_wt_{d}',
        'c_cycles_{d}',
        'mu_hill_branch_wt_0_{d}',
        'ac_accommodation_{d}',
        'ac_eating_{d}',
        'ac_commercial_{d}',
        'ac_tourism_{d}',
        'ac_entertainment_{d}',
        'ac_manufacturing_{d}',
        'ac_retail_{d}',
        'ac_transport_{d}',
        'ac_health_{d}',
        'ac_education_{d}',
        'ac_parks_{d}',
        'ac_cultural_{d}',
        'ac_sports_{d}']:
        for d in [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
            roadnode_metrics_columns.append(theme.format(d=d))

    for theme in [
        'area_mean_wt_{d}',
        'area_variance_wt_{d}',
        'val_mean_wt_{d}',
        'val_variance_wt_{d}',
        'rate_mean_wt_{d}',
        'rate_variance_wt_{d}']:
        for d in [50, 100, 200, 400, 800, 1600]:
            roadnode_metrics_columns.append(theme.format(d=d))

    for theme in [
        'cens_tot_pop_{d}',
        'cens_employed_{d}',
        'cens_dwellings_{d}']:
        for d in [200, 400, 800, 1600]:
            roadnode_metrics_columns.append(theme.format(d=d))

    roadnode_metrics_columns += [
        'cens_nocars_interp',
        'cens_cars_interp',
        'cens_ttw_peds_interp',
        'cens_ttw_bike_interp',
        'cens_ttw_motors_interp',
        'cens_ttw_pubtrans_interp',
        'cens_ttw_home_interp',
        'cens_density_interp'
    ]

    for rn_table, rn_m_table in zip(rn_tables, rn_metrics_tables):

        rn_m_exists = await db_con.fetchval(f'''
                        SELECT EXISTS (
                           SELECT 1
                           FROM   information_schema.tables 
                           WHERE  table_schema = 'ml'
                           AND    table_name = '{rn_m_table}'
                        );
                    ''')

        if rn_m_exists:
            logger.info(f'Roadnodes table {rn_m_table} already exists, skipping...')

        else:
            logger.info(f'Copying roadnodes table: {rn_m_table}')

            await db_con.execute(f'''
                    CREATE TABLE ml.{rn_m_table} AS SELECT {', '.join(roadnode_metrics_columns)} FROM analysis.{rn_table};
                    ALTER TABLE ml.{rn_m_table} ADD PRIMARY KEY (id);
                    CREATE INDEX pop_idx_{rn_m_table} ON ml.{rn_m_table} (city_pop_id);
                ''')

    logger.info('Vacuuming')
    if vacuum:
        await db_con.execute('VACUUM FULL')

    await db_con.close()


if __name__ == '__main__':
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    vacuum = False

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    loop = asyncio.get_event_loop()
    loop.run_until_complete(clone_tables(db_config, vacuum))

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
