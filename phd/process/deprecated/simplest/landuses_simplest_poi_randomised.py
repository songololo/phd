import logging
import time
import datetime
import asyncio

from process.deprecated.simplest import accessibility_calc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    from process.metrics.shortest.landuses_shortest_poi import fetch_date_last_updated

    loop = asyncio.get_event_loop()

    db_config = {
        'host': 'localhost',
        'port': 5433,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    boundary_table = 'analysis.city_boundaries_150'
    ########################################################################################
    # NOTE -> run after the centrality-path versions - which create the randomised POI table #
    ########################################################################################
    data_table = 'os.poi_randomised'
    date_last_updated = loop.run_until_complete(fetch_date_last_updated(db_config))
    data_where = f"date_last_updated = date('{date_last_updated}')"

    nodes_table = 'analysis.roadnodes_100'
    links_table = 'analysis.roadlinks_100'

    distances = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

    city_pop_id = 1

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    loop.run_until_complete(
        # NOTE the random flag for randomised version
        accessibility_calc(db_config,
                           nodes_table,
                           links_table,
                           boundary_table,
                           data_table,
                           city_pop_id,
                           distances,
                           data_where=data_where,
                           random=True))

    logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
