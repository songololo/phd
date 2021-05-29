import asyncio
import logging

import asyncpg
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plt_setup(dark=False):
    # clear existing plots
    plt.ioff()
    plt.close('all')
    plt.cla()
    plt.clf()
    mpl.rcdefaults()  # resets seaborn
    # load Open Sans - which is referred to from matplotlib.rc
    font_files = fm.findSystemFonts(fontpaths=['./fonts/Open_Sans'])
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    mpl_rc_path = './src/matplotlib.rc'
    mpl.rc_file(mpl_rc_path)
    if dark:
        dark_col = '#2e2e2e'
        text_col = 'lightgrey'
        mpl.rcParams['axes.facecolor'] = dark_col
        mpl.rcParams['figure.facecolor'] = dark_col
        mpl.rcParams['savefig.facecolor'] = dark_col
        mpl.rcParams['text.color'] = text_col
        mpl.rcParams['axes.edgecolor'] = text_col
        mpl.rcParams['axes.labelcolor'] = text_col
        mpl.rcParams['xtick.color'] = text_col
        mpl.rcParams['ytick.color'] = text_col
        mpl.rcParams['grid.color'] = text_col


# used by some global plots
class Style:

    def __init__(self):
        logger.info('setting colormaps and highlight colours')

        self.def_cmap = 'plasma_r'
        self.def_col_hl1 = 'green'
        self.def_col_hl1_a = 0.4
        self.def_col_hl2 = 'slategrey'
        self.def_col_hl2_a = 0.7
        self.alt_cmap = 'viridis_r'
        self.alt_col_hl1 = 'crimson'
        self.alt_col_hl1_a = 0.4
        self.alt_col_hl2 = 'indigo'
        self.alt_col_hl2_a = 0.4


def cityseer_cmap():
    return LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])


def cityseer_cmap_intense():
    return LinearSegmentedColormap.from_list('cityseer_intense', ['#0064b7', '#9a0007'])


# ordering of colours by darkest, lightest, mid is intentional for use with opacity or size gradients
def cityseer_cmap_red(dark=False):
    if dark:
        return LinearSegmentedColormap.from_list('cityseer_red', ['#FAFAFA', '#9a0007', '#ff6659', '#d32f2f'])
    else:
        return LinearSegmentedColormap.from_list('cityseer_red', ['#2E2E2E', '#9a0007', '#ff6659', '#d32f2f'])


# ordering of colours by darkest, lightest, mid is intentional for use with opacity or size gradients
def cityseer_cmap_blue(dark=False):
    if dark:
        return LinearSegmentedColormap.from_list('cityseer_blue', ['#FAFAFA', '#0064b7', '#64c1ff', '#0091ea'])
    else:
        return LinearSegmentedColormap.from_list('cityseer_blue', ['#2E2E2E', '#0064b7', '#64c1ff', '#0091ea'])


def cityseer_diverging_cmap(dark=False):
    if dark:
        return LinearSegmentedColormap.from_list('cityseer',
                                                 ['#0064b7', '#0091ea', '#64c1ff', '#2E2E2E', '#ff6659', '#d32f2f',
                                                  '#9a0007'])
    else:
        return LinearSegmentedColormap.from_list('cityseer',
                                                 ['#0064b7', '#0091ea', '#64c1ff', '#FAFAFA', '#ff6659', '#d32f2f',
                                                  '#9a0007'])


async def _fetch_cols(db_config, column_names, table_name, conditions):
    cols = ', '.join(column_names)
    q = f'SELECT {cols} FROM {table_name} {conditions};'
    logger.info(f'Example query: {q}')

    db_con = await asyncpg.connect(**db_config)

    data_dict = {}
    col_keys = []
    for col in column_names:
        if 'as' in col.lower():
            key = col.lower().split(' as ')[-1]
        else:
            key = col
        col_keys.append(key)
        data_dict[key] = []

    # get count of columns matching where clause
    simple_conditions = conditions.lower().split('order')[0]
    total = await db_con.fetchval(f'SELECT count(*) FROM {table_name} {simple_conditions};')
    logger.info(f'{total} total rows to fetch')

    with tqdm(total=total, miniters=total / 100) as pbar:
        async with db_con.transaction():
            async for record in db_con.cursor(q):
                for key, col in zip(col_keys, column_names):
                    data_dict[key].append(record[key])
                pbar.update(1)

    return data_dict


def load_data_as_pd_df(db_config, column_names: list, table_name: str, conditions: str):
    logger.info('loading data from DB')
    data_dict = asyncio.run(_fetch_cols(db_config, column_names, table_name, conditions))
    logger.info('generating data frame')
    df = pd.DataFrame(data=data_dict)
    logger.info(df.head())
    logger.info(df.dtypes)
    return df


async def _write_cols(db_config, table_name, data_arr, data_col, data_col_type, id_arr, id_col):
    db_con = await asyncpg.connect(**db_config)
    logger.info(f'writing column: {data_col} to {table_name}, matching to id col: {id_col}')
    await db_con.execute(f'''
            ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS {data_col} {data_col_type};
        ''')
    await db_con.executemany(f'''
            UPDATE {table_name}
            SET {data_col} = $1
            WHERE {id_col} = $2;
        ''', zip(data_arr, id_arr))
    await db_con.close()


def write_col_data(db_config, table_name, data_arr, data_col, data_col_type, id_arr, id_col):
    assert len(data_arr) == len(id_arr)
    asyncio.run(_write_cols(db_config,
                            table_name,
                            data_arr,
                            data_col,
                            data_col_type,
                            id_arr,
                            id_col))


def clean_pd(df, drop_na='all', fill_inf=np.nan):
    '''
    :param df:
    :param drop_na: 'all' or 'any'
    :return:
    '''
    d = df.copy(deep=True)  # otherwise changes are made in place
    # remove infinite and NaN values
    logger.info(f'Replacing inf with {fill_inf}')
    d = d.replace([np.inf, -np.inf], fill_inf)
    # axis 0 is index / row axis
    # only drop rows where all values are nan
    logger.info(f'Dropping rows where nan: {drop_na}')
    d = d.dropna(axis='index', how=drop_na)
    logger.info(f'Start rows: {len(df)}. End rows: {len(d)}.')
    return d


# the following plotting function per https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
def plot_ml_learning_curve(ax, train_scores, val_scores):
    # if multiple batches - use average of batch and fill areas between standard deviation
    if isinstance(train_scores, np.ndarray) and train_scores.ndim > 1:
        sample_size = np.array(range(1, len(train_scores.shape[1]) + 1))
        train_std = np.std(train_scores, axis=1)  # do before taking mean
        train_scores = np.mean(train_scores, axis=1)
        val_std = np.std(val_scores, axis=1)  # do before taking mean
        val_scores = np.mean(val_scores, axis=1)
        ax.fill_between(sample_size, train_scores + train_std, train_scores - train_std,
                        alpha=0.15, color='blue')
        ax.fill_between(sample_size, val_scores + val_std, val_scores - val_std,
                        alpha=0.15, color='green')
    else:
        sample_size = np.array(range(1, len(train_scores) + 1))

    ax.plot(sample_size - 0.5, train_scores, color='blue', marker='o', markersize=5,
            label='training accuracy')
    ax.plot(sample_size, val_scores, color='green', linestyle='--', marker='s', markersize=5,
            label='validation accuracy')

    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right', prop={'size': 5})


# the following plotting function per https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
def plot_ml_validation_curve(ax, param_name, param_range, train_scores, val_scores):
    # if multiple batches - use average of batch and fill areas between standard deviation
    if isinstance(train_scores, np.ndarray) and train_scores.ndim > 1:
        train_std = np.std(train_scores, axis=1)  # do before taking mean
        train_scores = np.mean(train_scores, axis=1)
        val_std = np.std(val_scores, axis=1)  # do before taking mean
        val_scores = np.mean(val_scores, axis=1)
        ax.fill_between(param_range, train_scores + train_std, train_scores - train_std,
                        alpha=0.15, color='blue')
        ax.fill_between(param_range, val_scores + val_std, val_scores - val_std,
                        alpha=0.15, color='green')

    ax.plot(param_range, train_scores, color='blue', marker='o', markersize=5,
            label='training accuracy')

    ax.plot(param_range, val_scores, color='green', linestyle='--', marker='s', markersize=5,
            label='validation accuracy')

    ax.set_xlabel(f'Parameter: {param_name}')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right', prop={'size': 5})


def train_test_idxs(df, mod):
    '''
    Modulo of 300 gives about 11%
    Whereas 200 gives about 25%
    '''
    xs = np.copy(df.x)
    xs /= 100
    xs = np.round(xs)
    xs *= 100
    xs_where = xs % mod == 0
    ys = np.copy(df.y)
    ys /= 100
    ys = np.round(ys)
    ys *= 100
    ys_where = ys % mod == 0
    xy_where = np.logical_and(xs_where, ys_where)
    print(f'Isolated {xy_where.sum() / len(df):.2%} test samples.')
    return xy_where
