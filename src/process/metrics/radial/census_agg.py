"""
Script version for aggregating census data
DB versions can be hit-and-miss and take some time

Using a weighted aggregation instead of cliff-edge aggregation
-> smoother result, though smaller distances may be questionable...
-> and counts are effective...

"""
import asyncio
import logging

import asyncpg
from cityseer.metrics import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def population_aggregator(db_config, nodes_table, census_table, city_pop_id):
    db_con = await asyncpg.connect(**db_config)

    distances = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    betas = networks.beta_from_distance(distances)

    # create the columns
    await db_con.execute(f'''
    ALTER TABLE {nodes_table}
        ADD COLUMN IF NOT EXISTS cens_tot_pop real[],
        ADD COLUMN IF NOT EXISTS cens_adults real[],
        ADD COLUMN IF NOT EXISTS cens_employed real[],
        ADD COLUMN IF NOT EXISTS cens_dwellings real[],
        ADD COLUMN IF NOT EXISTS cens_students real[];
    ''')

    # iterate the nodes and assign the new values to a list
    logger.info(f'fetching all ids for city {city_pop_id}')
    uids = []
    records = await db_con.fetch(f'''
    SELECT id
        FROM {nodes_table}
        WHERE city_pop_id::int = {city_pop_id} and within = true;
    ''')
    for r in records:
        uids.append(r['id'])

    logger.info(f'processing population for {len(uids)} ids')
    agg_results = []
    count = 0
    for uid in uids:
        count += 1
        if count % 10000 == 0:
            completion = round((count / len(uids)) * 100, 2)
            logger.info(f'{completion}% completed')
        tot_pop = []
        adults = []
        employed = []
        dwellings = []
        students = []
        for dist, beta in zip(distances, betas):
            # NB -> Use ST_DWithin (uses index) AND NOT ST_Distance which calculates everything from scratch
            # data = tot_pop, adults, employed, dwellings, students
            record = await db_con.fetchrow(f'''
            SELECT
                    sum(d.totpop_w) as tot_pop,
                    sum(d.adults_w) as adults,
                    sum(d.employed_w) as employed,
                    sum(d.dwellings_w) as dwellings,
                    sum(d.students_w) as students
                FROM (
                    (SELECT geom FROM {nodes_table} WHERE id = $1) AS node
                    CROSS JOIN LATERAL
                    (SELECT
                        totpop,
                        totadult,
                        totemploy,
                        dwelling,
                        stud18plus,
                        geom <-> node.geom as dist
                        FROM {census_table}
                        WHERE ST_DWithin(geom, node.geom, $2)) AS c
                    CROSS JOIN LATERAL
                    (SELECT exp($3 * c.dist) as weight)  as w
                    CROSS JOIN LATERAL
                    (SELECT
                        c.totpop * w.weight as totpop_w,
                        c.totadult * w.weight as adults_w,
                        c.totemploy * w.weight as employed_w,
                        c.dwelling * w.weight as dwellings_w,
                        c.stud18plus * w.weight as students_w
                    ) as w_c
                ) AS d;
            ''', uid, dist, beta)
            # convert to list
            data = [d for d in record]
            # change None values to 0
            for i in range(len(data)):
                if data[i] is None:
                    data[i] = 0
            # data = tot_pop, adults, employed, dwellings, students
            for data_point, arr in zip(data, [tot_pop, adults, employed, dwellings, students]):
                arr.append(data_point)
        # add to main agg
        agg_results.append((uid, tot_pop, adults, employed, dwellings, students))
    assert len(agg_results) == len(uids)

    # write back to db
    await db_con.executemany(f'''
    UPDATE {nodes_table}
        SET 
            cens_tot_pop = $2,
            cens_adults = $3,
            cens_employed = $4,
            cens_dwellings = $5,
            cens_students = $6
        WHERE id = $1
    ''', agg_results)

    await db_con.close()


if __name__ == '__main__':

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }
    census_table = 'census_2011.census_centroids'
    nodes_table = 'analysis.nodes_20'
    for city_pop_id in range(669, 932):
        logger.info(
            f'Starting population aggregation for city id: {city_pop_id} on table {nodes_table}')
        asyncio.run(population_aggregator(db_config, nodes_table, census_table, city_pop_id))
    logger.info('completed')
