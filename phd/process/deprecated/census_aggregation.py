"""
Script version for assigning census population
DB versions can be hit-and-miss and take some time

This works for stats types that are totals (not percentage) based
Use census_interpolation.py script for percentage based stats, which are interpolated instead
"""
import logging
import asyncio
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def population_aggregator(db_config, network_schema, verts_table, census_schema, census_table, city_pop_id):

    db_con = await asyncpg.connect(**db_config)

    # create the columns
    await db_con.execute(f'''
        ALTER TABLE {network_schema}.{verts_table}
            ADD COLUMN IF NOT EXISTS cens_tot_pop_200 real,
            ADD COLUMN IF NOT EXISTS cens_tot_pop_400 real,
            ADD COLUMN IF NOT EXISTS cens_tot_pop_800 real,
            ADD COLUMN IF NOT EXISTS cens_tot_pop_1600 real,
            ADD COLUMN IF NOT EXISTS cens_adults_200 real,
            ADD COLUMN IF NOT EXISTS cens_adults_400 real,
            ADD COLUMN IF NOT EXISTS cens_adults_800 real,
            ADD COLUMN IF NOT EXISTS cens_adults_1600 real,
            ADD COLUMN IF NOT EXISTS cens_employed_200 real,
            ADD COLUMN IF NOT EXISTS cens_employed_400 real,
            ADD COLUMN IF NOT EXISTS cens_employed_800 real,
            ADD COLUMN IF NOT EXISTS cens_employed_1600 real,
            ADD COLUMN IF NOT EXISTS cens_dwellings_200 real,
            ADD COLUMN IF NOT EXISTS cens_dwellings_400 real,
            ADD COLUMN IF NOT EXISTS cens_dwellings_800 real,
            ADD COLUMN IF NOT EXISTS cens_dwellings_1600 real,
            ADD COLUMN IF NOT EXISTS cens_students_200 real,
            ADD COLUMN IF NOT EXISTS cens_students_400 real,
            ADD COLUMN IF NOT EXISTS cens_students_800 real,
            ADD COLUMN IF NOT EXISTS cens_students_1600 real;
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

    logger.info(f'processing population for {len(ids)} ids')
    agg_results = []
    count = 0
    for uid in ids:

        count += 1
        if count % 10000 == 0:
            completion = round((count / len(ids)) * 100, 2)
            logger.info(f'{completion}% completed')

        # NB -> Use ST_DWithin (uses index) AND NOT ST_Distance which calculates everything from scratch
        pop_200, adults_200, employed_200, dwellings_200, students_200 = await db_con.fetchrow(f'''
                SELECT sum(d.totpop_w) as population, sum(d.totadult_w) as adults, sum(d.totemploy_w) as employed, sum(d.dwelling_w) as dwellings, sum(d.stud18plus_w) as students FROM (
                    (SELECT geom FROM {network_schema}.{verts_table} WHERE id = $1) AS nodes
                    CROSS JOIN LATERAL
                    (SELECT totpop, totadult, totemploy, dwelling, stud18plus, geom <-> nodes.geom as dist FROM {census_schema}.{census_table} WHERE ST_DWithin(geom, nodes.geom, $2)) AS c
                    CROSS JOIN LATERAL
                    (SELECT exp($3 * c.dist) as weight) as w
                    CROSS JOIN LATERAL
                    (SELECT c.totpop * w.weight as totpop_w,
                       c.totadult * w.weight as totadult_w,
                       c.totemploy * w.weight as totemploy_w,
                       c.dwelling * w.weight as dwelling_w,
                       c.stud18plus * w.weight as stud18plus_w) as w_c
                ) AS d;
                ''', uid, 200, -0.02)

        # NB -> Use ST_DWithin (uses index) AND NOT ST_Distance which calculates everything from scratch
        pop_400, adults_400, employed_400, dwellings_400, students_400 = await db_con.fetchrow(f'''
            SELECT sum(d.totpop_w) as population, sum(d.totadult_w) as adults, sum(d.totemploy_w) as employed, sum(d.dwelling_w) as dwellings, sum(d.stud18plus_w) as students FROM (
                (SELECT geom FROM {network_schema}.{verts_table} WHERE id = $1) AS nodes
                CROSS JOIN LATERAL
                (SELECT totpop, totadult, totemploy, dwelling, stud18plus, geom <-> nodes.geom as dist FROM {census_schema}.{census_table} WHERE ST_DWithin(geom, nodes.geom, $2)) AS c
                CROSS JOIN LATERAL
                (SELECT exp($3 * c.dist) as weight) as w
                CROSS JOIN LATERAL
                (SELECT c.totpop * w.weight as totpop_w,
                   c.totadult * w.weight as totadult_w,
                   c.totemploy * w.weight as totemploy_w,
                   c.dwelling * w.weight as dwelling_w,
                   c.stud18plus * w.weight as stud18plus_w) as w_c
            ) AS d;
            ''', uid, 400, -0.01)

        pop_800, adults_800, employed_800, dwellings_800, students_800 = await db_con.fetchrow(f'''
            SELECT sum(d.totpop_w) as population, sum(d.totadult_w) as adults, sum(d.totemploy_w) as employed, sum(d.dwelling_w) as dwellings, sum(d.stud18plus_w) as students FROM (
                (SELECT geom FROM {network_schema}.{verts_table} WHERE id = $1) AS nodes
                CROSS JOIN LATERAL
                (SELECT totpop, totadult, totemploy, dwelling, stud18plus, geom <-> nodes.geom as dist FROM {census_schema}.{census_table} WHERE ST_DWithin(geom, nodes.geom, $2)) AS c
                CROSS JOIN LATERAL
                (SELECT exp($3 * c.dist) as weight) as w
                CROSS JOIN LATERAL
                (SELECT c.totpop * w.weight as totpop_w,
                   c.totadult * w.weight as totadult_w,
                   c.totemploy * w.weight as totemploy_w,
                   c.dwelling * w.weight as dwelling_w,
                   c.stud18plus * w.weight as stud18plus_w) as w_c
            ) AS d;
            ''', uid, 800, -0.005)

        pop_1600, adults_1600, employed_1600, dwellings_1600, students_1600 = await db_con.fetchrow(f'''
            SELECT sum(d.totpop_w) as population, sum(d.totadult_w) as adults, sum(d.totemploy_w) as employed, sum(d.dwelling_w) as dwellings, sum(d.stud18plus_w) as students FROM (
                (SELECT geom FROM {network_schema}.{verts_table} WHERE id = $1) AS nodes
                CROSS JOIN LATERAL
                (SELECT totpop, totadult, totemploy, dwelling, stud18plus, geom <-> nodes.geom as dist FROM {census_schema}.{census_table} WHERE ST_DWithin(geom, nodes.geom, $2)) AS c
                CROSS JOIN LATERAL
                (SELECT exp($3 * c.dist) as weight) as w
                CROSS JOIN LATERAL
                (SELECT c.totpop * w.weight as totpop_w,
                   c.totadult * w.weight as totadult_w,
                   c.totemploy * w.weight as totemploy_w,
                   c.dwelling * w.weight as dwelling_w,
                   c.stud18plus * w.weight as stud18plus_w) as w_c
            ) AS d;
            ''', uid, 1600, -0.0025)

        agg_results.append((uid,
            pop_200, pop_400, pop_800, pop_1600,
            adults_200, adults_400, adults_800, adults_1600,
            employed_200, employed_400, employed_800, employed_1600,
            dwellings_200, dwellings_400, dwellings_800, dwellings_1600,
            students_200, students_400, students_800, students_1600))

    # write back to db
    await db_con.executemany(f'''
        UPDATE {network_schema}.{verts_table}
            SET 
                cens_tot_pop_200 = $2,
                cens_tot_pop_400 = $3,
                cens_tot_pop_800 = $4,
                cens_tot_pop_1600 = $5,
                cens_adults_200 = $6,
                cens_adults_400 = $7,
                cens_adults_800 = $8,
                cens_adults_1600 = $9,
                cens_employed_200 = $10,
                cens_employed_400 = $11,
                cens_employed_800 = $12,
                cens_employed_1600 = $13,
                cens_dwellings_200 = $14,
                cens_dwellings_400 = $15,
                cens_dwellings_800 = $16,
                cens_dwellings_1600 = $17,
                cens_students_200 = $18,
                cens_students_400 = $19,
                cens_students_800 = $20,
                cens_students_1600 = $21
            WHERE id = $1
        ''', agg_results)

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
        logger.info(
            f'Starting population aggregation for city id: {city_pop_id} on table {network_schema}.{verts_table}')
        loop.run_until_complete(
            population_aggregator(db_config, network_schema, verts_table, census_schema, census_table, city_pop_id))

    logger.info('completed')
