"""
Script version for assigning census stats based on interpolated values
DB versions can be hit-and-miss and take some time

This works for stats types that are percentage based
Use census_aggregation.py script for totals based stats, which are aggregated instead
"""
import numpy as np
import logging
import asyncio
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def stats_aggregator(db_config, nodes_table, census_table, city_pop_id):
    db_con = await asyncpg.connect(**db_config)
    # create the columns
    await db_con.execute(f'''
        ALTER TABLE {nodes_table}
            ADD COLUMN IF NOT EXISTS cens_density_interp real,
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
    uids = []
    records = await db_con.fetch(f'''
        SELECT id
            FROM {nodes_table}
            WHERE city_pop_id::int = {city_pop_id} and within = true;
    ''')
    for r in records:
        uids.append(r['id'])
    max_dist = 800  # max search distance - not necessarily in sync to beta
    beta = -0.04  # need fairly aggressive decay
    max_nodes = 3
    results_agg = []
    non_matches = 0
    logger.info(f'processing stats for {len(uids)} ids')
    count = 0
    for uid in uids:
        count += 1
        if count % 10000 == 0:
            completion = round((count / len(uids)) * 100, 2)
            logger.info(f'{completion}% completed')
        # ttwbike is not the same as ttwmbike!
        # values are raw counts, so convert to rates
        stats = await db_con.fetch(f'''
          SELECT * FROM
            (SELECT geom FROM {nodes_table} WHERE id = $1) as node
            CROSS JOIN LATERAL
            (SELECT
              id,
              totpop,
              area,
              nocars,
              car1 + car2 + car3 + car4plus as cars,
              ttwfoot,
              ttwbike,
              ttwcar + ttwcarpass + ttwmbike + ttwtaxi as ttw_motor,
              ttwbus + ttwtube + ttwtrain as ttw_pubtrans,
              ttwhome,
              geom <-> node.geom as dist
              FROM {census_table}
              WHERE ST_DWithin(node.geom, geom, {max_dist})
              ORDER BY geom <-> node.geom
              LIMIT {max_nodes}) as stats
        ''', uid)
        dist = 0
        dens = no_cars = cars = ttw_foot = ttw_bicycle = ttw_motor = ttw_pubtrans = ttw_home = 0
        err = False
        if len(stats) == 0:
            err = True
        for s in stats:
            pop = s['totpop']
            weight = np.exp(s['dist'] * beta)
            dist += weight
            if s['area'] is not None:
                dens += s['totpop'] / s['area'] * weight
            else:
                err = True
            no_cars += s['nocars'] / pop * weight
            cars += s['cars'] / pop * weight
            ttw_foot += s['ttwfoot'] / pop * weight
            ttw_bicycle += s['ttwbike'] / pop * weight
            ttw_motor += s['ttw_motor'] / pop * weight
            ttw_pubtrans += s['ttw_pubtrans'] / pop * weight
            ttw_home += s['ttwhome'] / pop * weight
        if err:
            non_matches += 1
            continue
        results_agg.append((
            uid,
            dens / dist,
            no_cars / dist,
            cars / dist,
            ttw_foot / dist,
            ttw_bicycle / dist,
            ttw_motor / dist,
            ttw_pubtrans / dist,
            ttw_home / dist
        ))
    assert len(results_agg) + non_matches == len(uids)
    await db_con.executemany(f'''
        UPDATE {nodes_table}
            SET cens_density_interp = $2,
                cens_nocars_interp = $3,
                cens_cars_interp = $4,
                cens_ttw_peds_interp = $5,
                cens_ttw_bike_interp = $6,
                cens_ttw_motors_interp = $7,
                cens_ttw_pubtrans_interp = $8,
                cens_ttw_home_interp = $9
            WHERE id = $1
        ''', results_agg)
    await db_con.close()
    logger.warning(f'{non_matches} roadnodes had no corresponding census data within the {max_dist} max distance')


if __name__ == '__main__':
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }
    """
    # for building the area column, only needs to be done once if column doesn't exist
    async def prep_areas():
        db_con = await asyncpg.connect(**db_config)
        # if necessary, add an area column to the centroid
        await db_con.execute('''
            ALTER TABLE census_2011.census_centroids
                ADD COLUMN IF NOT EXISTS area real;
            
            UPDATE census_2011.census_centroids AS cent
                SET area = ST_Area(enclose.geom)
                FROM (SELECT geom FROM census_2011.census_boundaries) as enclose
                WHERE ST_Contains(enclose.geom, cent.geom);
                
            -- some centroids aren't contained by their respective polygons
            -- these are here fixed manually for london
            UPDATE census_2011.census_centroids AS cent
                SET area = ST_Area(ST_Union(geom_a.a, geom_b.b))
                    FROM
                        (SELECT geom as a FROM census_2011.census_boundaries WHERE id = 100186) as geom_a,
                        (SELECT geom as b FROM census_2011.census_boundaries WHERE id = 100187) as geom_b
                    WHERE id = 4293;
            
            UPDATE census_2011.census_centroids AS cent
                SET area = ST_Area(ST_Union(geom_a.a, geom_b.b))
                    FROM
                        (SELECT geom as a FROM census_2011.census_boundaries WHERE id = 98088) as geom_a,
                        (SELECT geom as b FROM census_2011.census_boundaries WHERE id = 98089) as geom_b
                    WHERE id = 119887;
            
            UPDATE census_2011.census_centroids AS cent
                SET area = ST_Area(geom.a)
                    FROM
                        (SELECT geom as a FROM census_2011.census_boundaries WHERE id = 76938) as geom
                    WHERE id = 119123;
            
            UPDATE census_2011.census_centroids AS cent
                SET area = ST_Area(geom.a)
                    FROM
                        (SELECT geom as a FROM census_2011.census_boundaries WHERE id = 97848) as geom
                    WHERE id = 113160;

        ''')
    db_con = asyncio.run(prep_areas())
    """
    # set the census centroid areas to that of the enclosing geoms
    census_table = 'census_2011.census_centroids'
    nodes_table = 'analysis.nodes_20'
    for city_pop_id in range(619, 932):
        logger.info(f'Starting stats interpolation for city id: {city_pop_id} on table {nodes_table}')
        asyncio.run(stats_aggregator(db_config, nodes_table, census_table, city_pop_id))
    logger.info('completed')
