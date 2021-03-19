import logging
import time
import datetime
import asyncio
import asyncpg
import numpy as np

from cityseer.util import graphs
from cityseer.metrics import layers, networks
from process.loaders import postGIS_to_landuses_dict, postGIS_to_networkX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Accessibility_Codes():

    def __init__(self, landuse_classes, n_elements, distances):

        self._class_labels = landuse_classes
        self._n_elements = n_elements
        self._distances = distances

        # generate the class categories by filtering ranges per OS POI categories
        self.class_categories = {
            'accommodation': [cl for cl in landuse_classes if int(cl) >= 1010000 and int(cl) < 1020000],
            'eating': [cl for cl in landuse_classes if int(cl) >= 1020000 and int(cl) < 2000000],
            'commercial': [cl for cl in landuse_classes if int(cl) >= 2000000 and int(cl) < 3000000],
            'tourism': [cl for cl in landuse_classes if int(cl) >= 3200000 and int(cl) < 4000000],
            'entertainment': [cl for cl in landuse_classes if int(cl) >= 4250000 and int(cl) < 5000000],
            'manufacturing': [cl for cl in landuse_classes if int(cl) >= 7000000 and int(cl) < 8000000],
            'retail': [cl for cl in landuse_classes if int(cl) >= 9000000 and int(cl) < 10000000],
            'transport': [cl for cl in landuse_classes if int(cl) >= 10000000 and int(cl) < 11000000],
            'health': [cl for cl in landuse_classes if int(cl) >= 5280000 and int(cl) < 5310000],
            'education': [cl for cl in landuse_classes if int(cl) >= 5310000 and int(cl) < 6000000],
            'parks': [cl for cl in landuse_classes if int(cl) >= 3180000 and int(cl) < 3190000],
            'cultural': [cl for cl in landuse_classes if int(cl) >= 3170000 and int(cl) < 3180000],
            'sports': [cl for cl in landuse_classes if int(cl) >= 4000000 and int(cl) < 4250000]
        }

        # prepare data structure for unpacking results
        self.metrics = {}
        for wt in ['weighted', 'non_weighted']:
            self.metrics[wt] = {}
            for cat in self.class_categories.keys():
                self.metrics[wt][cat] = {}
                for dist in distances:
                    self.metrics[wt][cat][dist] = np.full(n_elements, 0.0)

    @property
    def categories(self):
        return [k for k in self.class_categories.keys()]

    @property
    def all_codes(self):
        class_codes = []
        for v in self.class_categories.values():
            class_codes += v
        return class_codes

    def set_metrics(self, access_metrics_dict):
        for wt in self.metrics.keys():
            for cat in self.metrics[wt].keys():
                for dist in self.metrics[wt][cat].keys():
                    for class_code in self.class_categories[cat]:
                        # array-wise
                        self.metrics[wt][cat][dist] += access_metrics_dict[wt][class_code][dist]
        # add totals
        # pre-total categories
        for wt in self.metrics.keys():
            self.metrics[wt]['total'] = {}
            for dist in self._distances:
                self.metrics[wt]['total'][dist] = np.full(self._n_elements, 0.0)
                for cat in self.categories:
                    # array-wise
                    self.metrics[wt]['total'][dist] += self.metrics[wt][cat][dist]


async def accessibility_calc(db_config,
                             nodes_table,
                             links_table,
                             boundary_table,
                             data_table,
                             data_id_col,
                             data_class_col,
                             city_pop_id,
                             epochs):

    themes = [
        'mu_epochs_hill_branch_wt_0_{dist}',
        'ac_epochs_accommodation_{dist}',
        'ac_epochs_eating_{dist}',
        'ac_epochs_commercial_{dist}',
        'ac_epochs_tourism_{dist}',
        'ac_epochs_entertainment_{dist}',
        'ac_epochs_manufacturing_{dist}',
        'ac_epochs_retail_{dist}',
        'ac_epochs_transport_{dist}',
        'ac_epochs_health_{dist}',
        'ac_epochs_education_{dist}',
        'ac_epochs_parks_{dist}',
        'ac_epochs_cultural_{dist}',
        'ac_epochs_sports_{dist}',
        'ac_epochs_total_{dist}'
    ]
    distances = [50, 100, 200, 400, 800, 1600]

    logger.info('Creating columns, if necessary')

    logger.info(f'Loading network data')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    G = graphs.nX_auto_edge_params(G)  # generate default lengths and impedances based on geom lengths
    N = networks.Network_Layer_From_nX(G, distances)

    epochs_table = f'{nodes_table}_epochs'
    epochs_table_stripped = epochs_table.split('.')[-1]

    db_con = await asyncpg.connect(**db_config)
    et_exists = await db_con.fetchval(f'''
            SELECT EXISTS (
               SELECT 1
               FROM   information_schema.tables 
               WHERE  table_schema = 'ml'
               AND    table_name = '{epochs_table_stripped}'
            );
        ''')
    if et_exists:
        logger.info(f'{epochs_table} table already exists, skipping...')
    else:
        logger.info(f'Creating {epochs_table} table from {nodes_table}...')
        await db_con.execute(f'''
            CREATE TABLE {epochs_table} AS SELECT id, city_pop_id FROM {nodes_table};
            ALTER TABLE {epochs_table} ADD PRIMARY KEY (id);
            CREATE INDEX pop_idx_{epochs_table_stripped} ON {epochs_table} (city_pop_id);
        ''')

    # use fillfactor - this means that the table will keep some spare space for HOT updating values
    # otherwise postgres will delete and rewrite rows when updating
    # this chews-up hard drive space very quickly
    await db_con.execute(f'ALTER TABLE {epochs_table} SET (fillfactor=80);')

    # add columns
    for theme in themes:
        for dist in distances:
            d = ['0.0'] * 40
            d = 'ARRAY[' + ', '.join(d) + ']'
            # default value
            await db_con.execute(f'''
                ALTER TABLE {epochs_table} ADD COLUMN IF NOT EXISTS {theme.format(dist=dist)} real[40] DEFAULT {d};
            ''')

    await db_con.close()

    logger.info(
        f'Starting all epochs calcs for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')

    for epoch_idx, epoch in enumerate(epochs):

        logger.info(f'Presently processing epoch {epoch_idx + 1} of {len(epochs)}: {epoch}')

        # reset metrics - otherwise a memory leak of sorts, haven't yet figured out...
        N.metrics = {
            'centrality': {},
            'mixed_uses': {},
            'accessibility': {
                'non_weighted': {},
                'weighted': {}
            },
            'stats': {}
        }

        logger.info(f'Loading POI data')
        data_dict = await postGIS_to_landuses_dict(db_config,
                                                   data_table,
                                                   data_id_col,
                                                   data_class_col,
                                                   boundary_table,
                                                   city_pop_id,
                                                   max_dist=max_dist,
                                                   data_where=f"date_first_updated <= date('{epoch}') AND date_last_updated >= date('{epoch}')")
        data_uids, data_map = layers.data_map_from_dict(data_dict)
        # derive the landuse labels, classes, encodings
        landuse_labels = [v['class'] for v in data_dict.values()]
        landuse_classes, landuse_encodings = layers.encode_categorical(landuse_labels)

        logger.info('Creating data layer')
        D = layers.Data_Layer(data_uids, data_map)

        start = time.localtime()

        logger.info('Assigning data points to the network')
        D.assign_to_network(N, max_dist=400)

        # generate the accessibility codes Class - this deduces codes and squashes results into categories
        logger.info('Generating POI accessibility codes')
        Acc_codes = Accessibility_Codes(landuse_classes, len(N.uids), distances)

        mixed_use_metrics = ['hill_branch_wt']

        # compute
        logger.info('Computing landuses')
        D.compute_aggregated(landuse_labels=landuse_labels,
                             mixed_use_keys=mixed_use_metrics,
                             accessibility_keys=Acc_codes.all_codes,
                             qs=[0])

        logger.info(f'Algo duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start))}')

        # squash the accessibility data
        logger.info('Squashing accessibility data')
        Acc_codes.set_metrics(N.metrics['accessibility'])

        # aggregate the data
        logger.info('Aggregating results')

        bulk_data = []

        for n_idx, uid in enumerate(N.uids):

            if N.live[n_idx]:

                data_pack = [uid]

                for dist in distances:

                    # mixed uses
                    data_pack.append(N.metrics['mixed_uses']['hill_branch_wt'][0][dist][n_idx])

                # accessibility keys
                for theme in [
                    'accommodation',
                    'eating',
                    'commercial',
                    'tourism',
                    'entertainment',
                    'manufacturing',
                    'retail',
                    'transport',
                    'health',
                    'education',
                    'parks',
                    'cultural',
                    'sports',
                    'total']:
                    for dist in distances:
                        data_pack.append(Acc_codes.metrics['weighted'][theme][dist][n_idx])

                bulk_data.append(data_pack)

        logger.info(f'Writing results to database for epoch {epoch}')

        db_con = await asyncpg.connect(**db_config)
        arr_idx = epoch_idx + 1
        await db_con.executemany(f"""
            UPDATE {epochs_table}
                SET
                    mu_epochs_hill_branch_wt_0_50[{arr_idx}] = $2,
                    mu_epochs_hill_branch_wt_0_100[{arr_idx}] = $3,
                    mu_epochs_hill_branch_wt_0_200[{arr_idx}] = $4,
                    mu_epochs_hill_branch_wt_0_400[{arr_idx}] = $5,
                    mu_epochs_hill_branch_wt_0_800[{arr_idx}] = $6,
                    mu_epochs_hill_branch_wt_0_1600[{arr_idx}] = $7,
                    ac_epochs_accommodation_50[{arr_idx}] = $8,
                    ac_epochs_accommodation_100[{arr_idx}] = $9,
                    ac_epochs_accommodation_200[{arr_idx}] = $10,
                    ac_epochs_accommodation_400[{arr_idx}] = $11,
                    ac_epochs_accommodation_800[{arr_idx}] = $12,
                    ac_epochs_accommodation_1600[{arr_idx}] = $13,
                    ac_epochs_eating_50[{arr_idx}] = $14,
                    ac_epochs_eating_100[{arr_idx}] = $15,
                    ac_epochs_eating_200[{arr_idx}] = $16,
                    ac_epochs_eating_400[{arr_idx}] = $17,
                    ac_epochs_eating_800[{arr_idx}] = $18,
                    ac_epochs_eating_1600[{arr_idx}] = $19,
                    ac_epochs_commercial_50[{arr_idx}] = $20,
                    ac_epochs_commercial_100[{arr_idx}] = $21,
                    ac_epochs_commercial_200[{arr_idx}] = $22,
                    ac_epochs_commercial_400[{arr_idx}] = $23,
                    ac_epochs_commercial_800[{arr_idx}] = $24,
                    ac_epochs_commercial_1600[{arr_idx}] = $25,
                    ac_epochs_tourism_50[{arr_idx}] = $26,
                    ac_epochs_tourism_100[{arr_idx}] = $27,
                    ac_epochs_tourism_200[{arr_idx}] = $28,
                    ac_epochs_tourism_400[{arr_idx}] = $29,
                    ac_epochs_tourism_800[{arr_idx}] = $30,
                    ac_epochs_tourism_1600[{arr_idx}] = $31,
                    ac_epochs_entertainment_50[{arr_idx}] = $32,
                    ac_epochs_entertainment_100[{arr_idx}] = $33,
                    ac_epochs_entertainment_200[{arr_idx}] = $34,
                    ac_epochs_entertainment_400[{arr_idx}] = $35,
                    ac_epochs_entertainment_800[{arr_idx}] = $36,
                    ac_epochs_entertainment_1600[{arr_idx}] = $37,
                    ac_epochs_manufacturing_50[{arr_idx}] = $38,
                    ac_epochs_manufacturing_100[{arr_idx}] = $39,
                    ac_epochs_manufacturing_200[{arr_idx}] = $40,
                    ac_epochs_manufacturing_400[{arr_idx}] = $41,
                    ac_epochs_manufacturing_800[{arr_idx}] = $42,
                    ac_epochs_manufacturing_1600[{arr_idx}] = $43,
                    ac_epochs_retail_50[{arr_idx}] = $44,
                    ac_epochs_retail_100[{arr_idx}] = $45,
                    ac_epochs_retail_200[{arr_idx}] = $46,
                    ac_epochs_retail_400[{arr_idx}] = $47,
                    ac_epochs_retail_800[{arr_idx}] = $48,
                    ac_epochs_retail_1600[{arr_idx}] = $49,
                    ac_epochs_transport_50[{arr_idx}] = $50,
                    ac_epochs_transport_100[{arr_idx}] = $51,
                    ac_epochs_transport_200[{arr_idx}] = $52,
                    ac_epochs_transport_400[{arr_idx}] = $53,
                    ac_epochs_transport_800[{arr_idx}] = $54,
                    ac_epochs_transport_1600[{arr_idx}] = $55,
                    ac_epochs_health_50[{arr_idx}] = $56,
                    ac_epochs_health_100[{arr_idx}] = $57,
                    ac_epochs_health_200[{arr_idx}] = $58,
                    ac_epochs_health_400[{arr_idx}] = $59,
                    ac_epochs_health_800[{arr_idx}] = $60,
                    ac_epochs_health_1600[{arr_idx}] = $61,
                    ac_epochs_education_50[{arr_idx}] = $62,
                    ac_epochs_education_100[{arr_idx}] = $63,
                    ac_epochs_education_200[{arr_idx}] = $64,
                    ac_epochs_education_400[{arr_idx}] = $65,
                    ac_epochs_education_800[{arr_idx}] = $66,
                    ac_epochs_education_1600[{arr_idx}] = $67,
                    ac_epochs_parks_50[{arr_idx}] = $68,
                    ac_epochs_parks_100[{arr_idx}] = $69,
                    ac_epochs_parks_200[{arr_idx}] = $70,
                    ac_epochs_parks_400[{arr_idx}] = $71,
                    ac_epochs_parks_800[{arr_idx}] = $72,
                    ac_epochs_parks_1600[{arr_idx}] = $73,
                    ac_epochs_cultural_50[{arr_idx}] = $74,
                    ac_epochs_cultural_100[{arr_idx}] = $75,
                    ac_epochs_cultural_200[{arr_idx}] = $76,
                    ac_epochs_cultural_400[{arr_idx}] = $77,
                    ac_epochs_cultural_800[{arr_idx}] = $78,
                    ac_epochs_cultural_1600[{arr_idx}] = $79,
                    ac_epochs_sports_50[{arr_idx}] = $80,
                    ac_epochs_sports_100[{arr_idx}] = $81,
                    ac_epochs_sports_200[{arr_idx}] = $82,
                    ac_epochs_sports_400[{arr_idx}] = $83,
                    ac_epochs_sports_800[{arr_idx}] = $84,
                    ac_epochs_sports_1600[{arr_idx}] = $85,
                    ac_epochs_total_50[{arr_idx}] = $86,
                    ac_epochs_total_100[{arr_idx}] = $87,
                    ac_epochs_total_200[{arr_idx}] = $88,
                    ac_epochs_total_400[{arr_idx}] = $89,
                    ac_epochs_total_800[{arr_idx}] = $90,
                    ac_epochs_total_1600[{arr_idx}] = $91
                WHERE id = $1;
            """, bulk_data)
        await db_con.close()

    logger.info('Completed')


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

    boundary_table = 'ml.city_boundaries_150'
    data_table = 'os.poi'
    data_id_col = 'urn'
    data_class_col = 'class_code'
    max_dist = 1600

    epochs = [
            '2006-09-01',
            '2007-03-01',
            '2007-06-01',
            '2007-09-01',
            '2007-12-01',
            '2008-03-01',
            '2009-03-01',
            '2009-06-01',
            '2009-09-01',
            '2010-09-01',
            '2010-12-01',
            '2011-03-01',
            '2011-06-01',
            '2011-09-01',
            '2011-12-01',
            '2012-03-01',
            '2012-06-01',
            '2012-09-01',
            '2012-12-01',
            '2013-03-01',
            '2013-06-01',
            '2013-09-01',
            '2014-06-01',
            '2014-09-01',
            '2014-12-01',
            '2015-03-01',
            '2015-06-01',
            '2015-09-01',
            '2015-12-01',
            '2016-03-01',
            '2016-06-01',
            '2016-09-01',
            '2016-12-01',
            '2017-03-01',
            '2017-06-01',
            '2017-09-01',
            '2017-12-01',
            '2018-03-01',
            '2018-06-01',
            '2018-09-01']

    nodes_table = 'ml.roadnodes_100'
    links_table = 'ml.roadlinks_100'

    e_key = None
    if e_key is not None:
        logger.info(f'NOTE -> Using e_key {e_key}')
        if e_key == 0:
            epochs = epochs[:10]
        elif e_key == 1:
            epochs = epochs[10:20]
        elif e_key == 2:
            epochs = epochs[20:30]
        elif e_key == 3:
            epochs = epochs[30:]
        assert len(epochs) == 10

    for city_pop_id in range(100, 959):

        start_time = time.localtime()
        logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

        loop.run_until_complete(
            accessibility_calc(db_config,
                               nodes_table,
                               links_table,
                               boundary_table,
                               data_table,
                               data_id_col,
                               data_class_col,
                               city_pop_id,
                               epochs))

        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
