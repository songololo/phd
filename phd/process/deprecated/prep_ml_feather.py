import logging
import time
import phd_util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# load data and write to HDF5
def create_feather(db_config, file_path):

    cols = [
        'id',
        'city_pop_id'
    ]

    for col in [
        'c_gravity_{d}',
        'c_between_wt_{d}',
        'c_cycles_{d}',
        'mu_hill_branch_wt_0_{d}',
        'ac_accommodation_{d}',
        'ac_eating_{d}',
        'ac_commercial_{d}',
        'ac_tourism_{d}',
        'ac_entertainment_{d}',
        'ac_manufacturing_{d}',
        'ac_retail_{d}',
        'ac_transport_{d}',
        'ac_health_{d}',
        'ac_education_{d}',
        'ac_parks_{d}',
        'ac_cultural_{d}',
        'ac_sports_{d}',
        'area_mean_wt_{d}',
        'area_variance_wt_{d}',
        'val_mean_wt_{d}',
        'val_variance_wt_{d}',
        'rate_mean_wt_{d}',
        'rate_variance_wt_{d}'
    ]:
        for d in [50, 100, 200, 400, 800, 1600]:
            cols.append(col.format(d=d))

    for col in [
        'cens_tot_pop_{d}',
        'cens_employed_{d}',
        'cens_dwellings_{d}'
    ]:
        for d in [200, 400, 800, 1600]:
            cols.append(col.format(d=d))

    cols += [
        'cens_nocars_interp',
        'cens_cars_interp',
        'cens_ttw_peds_interp',
        'cens_ttw_bike_interp',
        'cens_ttw_motors_interp',
        'cens_ttw_pubtrans_interp',
        'cens_ttw_home_interp',
        'cens_density_interp'
    ]

    for table in ['roadnodes_full', 'roadnodes_100', 'roadnodes_50', 'roadnodes_20']:
        path = f'{file_path}{table}_epochs_data.feather'
        # OK to filter by city pop id here, the data store only stores actual city nodes
        data = phd_util.load_data_as_pd_df(db_config, cols, f'analysis.{table}', 'WHERE city_pop_id IS NOT NULL')
        data.to_feather(path)


if __name__ == '__main__':

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'gareth',
        'database': 'gareth',
        'password': ''
    }

    file_path = '/Volumes/gareth 07783845654/epochs/'

    start_time = time.localtime()
    logger.info(f'Started {start_time[0]}-{start_time[1]}-{start_time[2]} at {start_time[3]}h:{start_time[4]}m')

    create_feather(db_config, file_path)

    end_time = time.localtime()
    logger.info(f'Ended {end_time[0]}-{end_time[1]}-{end_time[2]} at {end_time[3]}h:{end_time[4]}m')