import os
import pandas as pd
import asyncpg
import numpy as np
from numba import jit
import psycopg2
import logging
from scipy import stats
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def foursquare_daily_stats(dsn_string, db_schema, db_table):

    # open database connection
    logger.info(f'loading data from {db_schema}.{db_table}')
    with psycopg2.connect(dsn_string) as db_connection:
        # load nodes
        table_query = f'''
          SELECT id, total_checkins, total_unique_visitors FROM {db_schema}.{db_table}
        '''
        logger.info(f'query: {table_query}')
        foursquare_df = pd.read_sql(
            sql=table_query,
            con=db_connection,
            index_col='id',
            coerce_float=True,
            params=None
        )
        logger.info(f'Total number of venues: {len(foursquare_df)}')
        logger.info(f'All time checkins: {foursquare_df.total_checkins.values.sum()}')
        logger.info(f'All time visitors: {foursquare_df.total_unique_visitors.values.sum()}')

        logger.info('computing entropy')
        counts_df = pd.read_sql(
            sql=f'''
                SELECT category_name_primary as categories, count(category_name_primary) as totals
                  FROM {db_schema}.{db_table}
                  GROUP BY categories
                  ORDER BY totals DESC
            ''',
            con=db_connection,
            index_col='categories'
        )
        logger.info(f'Most common category types: {counts_df.categories.head}')
        logger.info(f'Total unique category types: {len(counts_df)}')
        probs = counts_df.totals.values / len(foursquare_df)
        entropy = stats.entropy(probs)
        logger.info(f'Entropy of category types: {entropy}')

if __name__ == '__main__':
    dsn_string = f"dbname='foursquare' user='gareth' host='localhost' port=5435 password={os.environ['CITYSEERDB_PW']}"
    db_schema = 'foursquare'
    db_tables = ['venues_london', 'venues_sfc', 'venues_nyc', 'venues_istanbul']

    for db_table in db_tables:
        foursquare_daily_stats(dsn_string, db_schema, db_table)
