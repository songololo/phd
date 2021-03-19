import logging
import time
import datetime
import asyncio

from phd.process.metrics.landuses.landuses_poi import accessibility_calc, fetch_date_last_updated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    boundary_table = 'analysis.city_boundaries_150'
    data_table = 'os.poi_randomised'
    date_last_updated = asyncio.run(fetch_date_last_updated(db_config))
    data_where = f"date_last_updated = date('{date_last_updated}')"
    distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

    nodes_table = 'analysis.roadnodes_full'
    links_table = 'analysis.roadlinks_full'
    city_pop_id = 1

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')
    # NOTE the rdm_flag flag for randomised version
    asyncio.run(accessibility_calc(db_config,
                                   nodes_table,
                                   links_table,
                                   boundary_table,
                                   data_table,
                                   city_pop_id,
                                   distances,
                                   data_where=data_where,
                                   rdm_flag=True))
    logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')
    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
