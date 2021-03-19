import asyncio
import datetime
import logging
import time

import numpy as np
import pandas as pd
from cityseer.metrics import layers, networks
from cityseer.util import graphs
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
            'eating': [cl for cl in landuse_classes if
                       (int(cl) >= 1020000 and int(cl) < 1020034) or
                       (int(cl) >= 1020043 and int(cl) < 2000000)],
            'drinking': [cl for cl in landuse_classes if int(cl) >= 1020034 and int(cl) < 1020043],
            'commercial': [cl for cl in landuse_classes if int(cl) >= 2000000 and int(cl) < 3000000],
            'tourism': [cl for cl in landuse_classes if int(cl) >= 3200000 and int(cl) < 4000000],
            'entertainment': [cl for cl in landuse_classes if int(cl) >= 4250000 and int(cl) < 5000000],
            'government': [cl for cl in landuse_classes if int(cl) >= 6000000 and int(cl) < 7000000],
            'manufacturing': [cl for cl in landuse_classes if int(cl) >= 7000000 and int(cl) < 8000000],
            'retail_food': [cl for cl in landuse_classes if int(cl) >= 9470000 and int(cl) < 9480000],
            'retail_other': [cl for cl in landuse_classes if
                             (int(cl) >= 9000000 and int(cl) < 9470000) or
                             (int(cl) >= 9480000 and int(cl) < 10000000)],
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
                             file_path,
                             epochs_idxs=None):
    themes = ['mixed_uses',
              'accommodation',
              'eating',
              'drinking',
              'commercial',
              'tourism',
              'entertainment',
              'government',
              'manufacturing',
              'retail_food',
              'retail_other',
              'transport',
              'health',
              'education',
              'parks',
              'cultural',
              'sports',
              'total']
    distances = [50, 100, 200, 400, 800, 1600]
    epochs = [
        # '2006-09-01',
        ## '2006-12-01' - missing
        # '2007-03-01',
        # '2007-06-01',
        # '2007-09-01',
        # '2007-12-01',
        # '2008-03-01',
        ## '2008-06-01' - missing
        ## '2008-09-01' - missing
        ## '2008-12-01' - missing
        # '2009-03-01',
        # '2009-06-01',
        # '2009-09-01',
        ## '2009-12-01' - missing
        ## '2010-03-01' - missing
        ## '2010-06-01' - missing
        # '2010-09-01',
        # '2010-12-01',
        # '2011-03-01',
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
        # '2013-12-01' - missing
        # '2014-03-01' - missing
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

    logger.info(f'Creating pandas dataset')

    # with pd.HDFStore(file_path) as data_store:
    # print(data_store)
    # data_store[store_key].loc[(1371767, 200, '2018-09-01', 900)]

    # if only computing certain indices
    if epochs_idxs is not None:
        logger.info(f'Indexing epochs with {epochs_idxs}')
        epochs = epochs[epochs_idxs]
        logger.info(f'Epochs n: {len(epochs)}, names: {epochs}')

    # cast to datetime
    epochs = pd.to_datetime(epochs, format='%Y-%m-%d')

    logger.info(f'Loading network data')
    G = await postGIS_to_networkX(db_config, nodes_table, links_table, boundary_table, city_pop_id)
    G = graphs.nX_auto_edge_params(G)  # generate default lengths and impedances based on geom lengths
    N = networks.Network_Layer_From_nX(G, distances)

    logger.info(
        f'Starting epoch calcs for city id: {city_pop_id} on network table {nodes_table} and data table {data_table}')

    for epoch_idx, epoch in enumerate(epochs):

        # if restarting from random location
        # if city_pop_id == 616 and epoch_idx != 27:
        #     continue

        logger.info(f'Presently processing epoch {epoch_idx + 1} of {len(epochs)}: {epoch}')

        logger.info(f'Loading POI data')
        data_dict = await postGIS_to_landuses_dict(db_config,
                                                   data_table,
                                                   data_id_col,
                                                   data_class_col,
                                                   boundary_table,
                                                   city_pop_id,
                                                   max_dist=1600,
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
        logger.info('Unpacking results to dataset')
        indices = []
        uid_count = 0
        for n_idx, uid in enumerate(N.uids):
            if N.live[n_idx]:
                uid_count += 1

        if not uid_count:
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
            continue

        # float 32 should be sufficient for mixed uses and accessibility info
        data = np.full((uid_count * len(distances), len(themes)), np.nan, dtype=np.float32)

        counter = 0
        for n_idx, uid in enumerate(N.uids):
            if N.live[n_idx]:
                for d_idx, dist in enumerate(distances):
                    # generate indices
                    indices.append((uid, dist, epoch, city_pop_id))
                    for theme_idx, theme in enumerate(themes):
                        if theme == 'mixed_uses':
                            # add mixed uses
                            data[counter][theme_idx] = N.metrics['mixed_uses']['hill_branch_wt'][0][dist][n_idx]
                        else:
                            # add landuses
                            data[counter][theme_idx] = Acc_codes.metrics['weighted'][theme][dist][n_idx]
                    # increment for each distance
                    counter += 1

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

        # create df and append to hdf5 store
        # create multi index
        # original attempt did not include names. Remove if patching from stopped attempt.
        multi_index = pd.MultiIndex.from_tuples(indices, names=['uid', 'dist', 'epoch', 'city_pop_id'])
        # float 32 should be sufficient for mixed uses and accessibility info
        df = pd.DataFrame(index=multi_index, data=data, columns=themes, dtype=np.float32)
        df = df.sort_index()

        # write to disk
        logger.info(f'Appending to HDF5 data store at: {file_path}')
        with pd.HDFStore(file_path) as data_store:
            data_store.append('df', df, format='table')


if __name__ == '__main__':

    logging.getLogger('asyncio').setLevel(logging.WARNING)
    loop = asyncio.get_event_loop()
    # loop.set_debug(True)

    db_config = {
        'host': 'localhost',
        'port': 5433,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    boundary_table = 'analysis.city_boundaries_150'
    data_table = 'os.poi'
    data_id_col = 'urn'
    data_class_col = 'class_code'

    decomp = '20'

    for city_pop_id in range(1, 959):
        nodes_table = f'analysis.roadnodes_{decomp}'
        links_table = f'analysis.roadlinks_{decomp}'
        file_path = f'/Users/gareth/Documents/Data/PhD/backups/df_epochs_{decomp}.nc'

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
                               file_path))

        logger.info(f'Duration: {datetime.timedelta(seconds=time.mktime(time.localtime()) - time.mktime(start_time))}')

        end_time = time.localtime()
        logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')
