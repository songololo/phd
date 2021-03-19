import asyncio
import logging

import asyncpg
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# geoms randomly reassigned within the bounds of the city
# note that this only runs once (across all distances)
# so let one copy finish prior to starting other distances
async def randomise_poi(db_config,
                        boundary_table,
                        data_table,
                        city_pop_id,
                        data_where=None):
    data_table_original = data_table
    data_table += '_randomised'
    data_schema, data_table_only = data_table.split('.')
    # check if randomised POI table exists
    db_con = await asyncpg.connect(**db_config)
    poi_randomised_exists = await db_con.fetchval(f'''
    SELECT EXISTS (
       SELECT 1
       FROM   information_schema.tables 
       WHERE  table_schema = '{data_schema}'
       AND    table_name = '{data_table_only}'
    );
    ''')

    if poi_randomised_exists:
        logger.warning(f'Randomised POI table {data_table} already exists, skipping...')
    else:
        logger.info(f'Copying base columns from data table: {data_table_original} to new table: {data_table}')
        base_cols = [
            'urn',
            'class_code',
            'geom',
            'date_last_updated'
        ]
        # select only the most recently updated data
        await db_con.execute(f'''
        CREATE TABLE {data_table}
            AS SELECT {', '.join(base_cols)}
            FROM {data_table_original}
            WHERE {data_where};
        ALTER TABLE {data_table} ADD PRIMARY KEY (urn);
        CREATE INDEX geom_idx_{data_table_only} ON {data_table} USING GIST (geom);
        ''')
        # load all vertices and corresponding neighbours and edges
        logger.info('Creating temporary buffered geom to filter data points')
        # convex hull prevents potential issues with multipolygons deriving from buffer...
        temp_id = await db_con.fetchval(f'''
        INSERT INTO {boundary_table} (geom)
            VALUES ((SELECT ST_Buffer(geom, 1600)
                FROM {boundary_table} WHERE pop_id = {city_pop_id}))
            RETURNING id;''')
        logger.info('Getting x and y extents for new POI geom locations')
        min_x, max_x, min_y, max_y = await db_con.fetchrow(f'''
        SELECT ST_XMin(geom), ST_XMax(geom), ST_YMin(geom), ST_YMax(geom)
            FROM {boundary_table} WHERE id = {temp_id};
        ''')
        logger.info('Counting number of POIs to iterate')
        num_rows = await db_con.fetchval(f'''
        SELECT count(dt.*) FROM 
            {data_table} as dt,
            (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as bd
            WHERE ST_Contains(bd.geom, dt.geom);
        ''')
        logger.info('Randomly scrambling POI geom locations')
        async with db_con.transaction():
            with tqdm(total=num_rows) as pbar:
                async for record in db_con.cursor(f'''
                SELECT dt.urn FROM
                    {data_table} as dt,
                    (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) as bd
                    WHERE ST_Contains(bd.geom, dt.geom);
                '''):
                    while True:
                        rd_x = np.random.randint(min_x, max_x)
                        rd_y = np.random.randint(min_y, max_y)
                        update_status = await db_con.execute(f'''
                        UPDATE {data_table}
                            SET geom = pt.p
                            FROM
                                (SELECT geom FROM {boundary_table} WHERE id = {temp_id}) AS bd,
                                (SELECT ST_SetSRID(ST_MakePoint($1, $2), 27700) as p) as pt
                            WHERE urn = $3 AND ST_Intersects(bd.geom, pt.p);
                        ''', rd_x, rd_y, record['urn'])
                        if update_status == 'UPDATE 1':
                            break
                    pbar.update(1)
        logger.info('NOTE -> removing temporary buffered city geom')
        await db_con.execute(f'DELETE FROM {boundary_table} WHERE id = $1;', temp_id)
    await db_con.close()


if __name__ == '__main__':
    ######################################################################################
    # NOTE -> if newer data in table than date last updated then won't find correct data #
    ######################################################################################
    async def fetch_date_last_updated(db_config):
        db_con = await asyncpg.connect(**db_config)
        d_u = await db_con.fetchval(f'''
            select date_last_updated from os.poi order by date_last_updated desc limit 1; 
        ''')
        await db_con.close()
        logger.info(f'Using {d_u} as the DATE_LAST_UPDATED parameter.')
        return d_u


    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    boundary_table = 'analysis.city_boundaries_150'
    data_table = 'os.poi'
    date_last_updated = asyncio.run(fetch_date_last_updated(db_config))
    data_where = f"date_last_updated = date('{date_last_updated}')"
    city_pop_id = 1
    asyncio.run(randomise_poi(db_config,
                              boundary_table,
                              data_table,
                              city_pop_id,
                              data_where=data_where))
