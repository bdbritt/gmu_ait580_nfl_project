import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
PROJECT_DIR = Path(__file__).resolve().parents[2]
SB_DATA = 'data\\raw\\superbowl_data.csv'
TEAMS_STATS = 'data\\raw\\teams_stats.csv'
FINAL_DATA = 'data\\processed\\training_data.csv'
VALIDATION_DATA = 'data\\processed\\validation_data.csv'
VALIDATION_YRS = ['2020', '2021']

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    load data to be transformed
    """
    data_1 = pd.read_csv(os.path.join(PROJECT_DIR, SB_DATA))
    data_2 = pd.read_csv(os.path.join(PROJECT_DIR, TEAMS_STATS))

    logging.info(f'superbowl data loaded: {data_1.shape}')
    logging.info(f'team stats loaded: {data_2.shape}')

    return data_1, data_2


def prep_sb_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    prep superbowl data
    """
    # change roman numbers to numbers
    data.SB = [each[-3:-1:1] if len(each)>=3 else each for each in data.SB]
    data.SB = [each[1:] if each[0]=="(" else each for each in data.SB]

    # add year
    data['year'] = data.apply(lambda x:pd.to_datetime(x['Date']).strftime('%Y'), axis=1)
    data_subset = data[['Winner Abv', 'year']].copy()
    data_subset['superbowl_win'] = 1
    data_subset['season_year'] = data_subset.apply(lambda x: str(int(x['year']) - 1), axis=1)
    data_subset.drop('year', axis=1, inplace=True)
    data_subset = data_subset.rename(columns={'season_year': 'year', 'Winner Abv':'team'})
    data_subset.columns = data_subset.columns.str.strip()

    return data_subset


def wrangle_data(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
    """
    main wrangle function
    """
    sb_prepped = prep_sb_data(data_1)

    data_2['year'] = data_2['year'].astype(str)

    # combine into one df
    data_merged = pd.merge(data_2, sb_prepped, how="left", left_on=['team', 'year'],\
         right_on=['team', 'year'])

    # update NaN
    data_merged = data_merged.fillna({'superbowl_win':0})

    # drop not needed columns
    data_merged.drop(['totalGiveaways', 'totalTakeaways', 'totalDrives', \
        'totalYardsFromScrimmage', 'team_number'], axis=1, inplace=True)

    # rename target column
    data_merged = data_merged.rename(columns={'superbowl_win': 'target'})

    return data_merged


def split_data(data: pd.DataFrame) -> None:
    """
    split data for training and test
    """
    # going to use for testing later
    val_data = data.loc[data['year'].isin(VALIDATION_YRS)].copy()
    val_data.to_csv(os.path.join(PROJECT_DIR, VALIDATION_DATA), index=False)
    logging.info(f'validation written to file: {val_data.shape}')

    # use to explore and build the model
    training_data = data.loc[~data['year'].isin(VALIDATION_YRS)].copy()
    training_data.to_csv(os.path.join(PROJECT_DIR, FINAL_DATA), index=False)
    logging.info(f'training written to file: {training_data.shape}')

def main() -> None:
    """
    main function to
    run
    """

    logger = logging.getLogger(__name__)
    logger.info('getting data')

    sb_data, teams_data = load_data()
    data_wrangled = wrangle_data(sb_data, teams_data)
    split_data(data_wrangled)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    main()
