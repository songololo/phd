"""
Script version for assigning census stats based on interpolated values
DB versions can be hit-and-miss and take some time

This works for stats types that are percentage based
Use census_aggregation.py script for totals based stats, which are aggregated instead
"""
import asyncio
import logging

import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def stats_aggregator(db_config, network_schema, verts_table, census_schema, census_table, city_pop_id):
    db_con = await asyncpg.connect(**db_config)

    # create the columns
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{verts_table}
            ADD COLUMN IF NOT EXISTS cens_nocars_interp real,
            ADD COLUMN IF NOT EXISTS cens_cars_interp real,
            ADD COLUMN IF NOT EXISTS cens_ttw_peds_interp real,
            ADD COLUMN IF NOT EXISTS cens_ttw_bike_interp real,
            ADD COLUMN IF NOT EXISTS cens_ttw_motors_interp real,
            ADD COLUMN IF NOT EXISTS cens_ttw_pubtrans_interp real,
            ADD COLUMN IF NOT EXISTS cens_ttw_home_interp real;
    ''')

    # iterate the nodes and assign the new values to a list
    logger.info(f'fetching all ids for city {city_pop_id}')
    ids = []
    records = await db_con.fetch(f'''
        SELECT id
            FROM {network_schema}.{verts_table}
            WHERE city_pop_id::int = {city_pop_id}
    ''')
    for r in records:
        ids.append(r['id'])

    max_dist = 800
    max_nodes = 3
    results_agg = []

    logger.info(f'processing stats for {len(ids)} ids')
    count = 0
    for uid in ids:

        count += 1
        if count % 10000 == 0:
            completion = round((count / len(ids)) * 100, 2)
            logger.info(f'{completion}% completed')

        # ttwbike is not the same as ttwmbike!
        stats = await db_con.fetch(f'''
          SELECT * FROM
            (SELECT geom FROM {network_schema}.{verts_table} WHERE id = $1) as nodes
            CROSS JOIN LATERAL
            (SELECT
              totpop,
              nocars,
              car1 + car2 + car3 + car4plus as cars,
              ttwfoot,
              ttwbike,
              ttwcar + ttwcarpass + ttwmbike + ttwtaxi as ttwmotor,
              ttwbus + ttwtube + ttwtrain as ttwpubtrans,
              ttwhome,
              geom <-> nodes.geom as dist
              FROM {census_schema}.{census_table}
              WHERE ST_DWithin(nodes.geom, geom, {max_dist})
              ORDER BY geom <-> nodes.geom
              LIMIT {max_nodes}) as stats
        ''', uid)

        dist = 0
        no_cars = cars = ttw_foot = ttw_bike = ttw_motor = ttw_pubtrans = ttw_home = 0

        for s in stats:
            weight = s['dist'] ** -2  # vs ? math.exp(s['dist'] * beta)
            dist += weight
            no_cars += s['nocars'] * weight
            cars += s['cars'] * weight
            ttw_foot += s['ttwfoot'] * weight
            ttw_bike += s['ttwbike'] * weight
            ttw_motor += s['ttwmotor'] * weight
            ttw_pubtrans += s['ttwpubtrans'] * weight
            ttw_home += s['ttwhome'] * weight

            results_agg.append((
                uid,
                no_cars / dist,
                cars / dist,
                ttw_foot / dist,
                ttw_bike / dist,
                ttw_motor / dist,
                ttw_pubtrans / dist,
                ttw_home / dist
            ))

    await db_con.executemany(f'''
        UPDATE {network_schema}.{verts_table}
            SET cens_nocars_interp = $2,
                cens_cars_interp = $3,
                cens_ttw_peds_interp = $4,
                cens_ttw_bike_interp = $5,
                cens_ttw_motors_interp = $6,
                cens_ttw_pubtrans_interp = $7,
                cens_ttw_home_interp = $8
            WHERE id = $1
        ''', results_agg)

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
    census_schema = 'census_2011'
    census_table = 'census_centroids'

    verts_table = 'roadnodes_full_dual'
    for city_pop_id in range(1, 2):
        logger.info(f'Starting stats interpolation for city id: {city_pop_id} on table {network_schema}.{verts_table}')
        loop.run_until_complete(
            stats_aggregator(db_config, network_schema, verts_table, census_schema, census_table, city_pop_id))

    logger.info('completed')
