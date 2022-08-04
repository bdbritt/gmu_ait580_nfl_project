import argparse
import logging
import os
from pathlib import Path
from typing import Text, Tuple

import pandas as pd
import yaml

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_data(f_dir: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    load data to be transformed
    """
    data_sb = pd.read_csv(
        f'{os.path.join(PROJECT_DIR, f_dir["directory"]["raw_data"])}/superbowl_data.csv'
    )
    data_tm_stats = pd.read_csv(
        f'{os.path.join(PROJECT_DIR, f_dir["directory"]["raw_data"])}/teams_stats.csv'
    )

    logging.info(f"superbowl data loaded: {data_sb.shape}")
    logging.info(f"team stats loaded: {data_tm_stats.shape}")

    return data_sb, data_tm_stats


def prep_sb_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    prep superbowl data
    """
    # change roman numbers to numbers
    data.SB = [each[-3:-1:1] if len(each) >= 3 else each for each in data.SB]
    data.SB = [each[1:] if each[0] == "(" else each for each in data.SB]

    # add year
    data["year"] = data.apply(
        lambda x: pd.to_datetime(x["Date"]).strftime("%Y"), axis=1
    )
    data_subset = data[["Winner Abv", "year"]].copy()
    data_subset["superbowl_win"] = 1
    data_subset["season_year"] = data_subset.apply(
        lambda x: str(int(x["year"]) - 1), axis=1
    )
    data_subset.drop("year", axis=1, inplace=True)
    data_subset = data_subset.rename(
        columns={"season_year": "year", "Winner Abv": "team"}
    )
    data_subset.columns = data_subset.columns.str.strip()

    return data_subset


def wrangle_data(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
    """
    main wrangle function
    """
    sb_prepped = prep_sb_data(data_1)

    data_2["year"] = data_2["year"].astype(str)

    # combine into one df
    data_merged = pd.merge(
        data_2,
        sb_prepped,
        how="left",
        left_on=["team", "year"],
        right_on=["team", "year"],
    )

    # update NaN
    data_merged = data_merged.fillna({"superbowl_win": 0})

    # drop not needed columns
    data_merged.drop(
        [
            "totalDrives",
            "totalYardsFromScrimmage",
            "team_number",
        ],
        axis=1,
        inplace=True,
    )

    # rename target column
    data_merged = data_merged.rename(columns={"superbowl_win": "target"})

    return data_merged


def split_data(conf_params: dict, data: pd.DataFrame) -> None:
    """
    split data for training and test
    """
    # going to use for testing later
    val_data = data.loc[data["year"].isin(conf_params["validation_years"])].copy()
    val_data.to_csv(
        f'{os.path.join(PROJECT_DIR, conf_params["directory"]["processed_data"])}/validation_data.csv',
        index=False,
    )
    logging.info(f"validation written to file: {val_data.shape}")

    # use to explore and build the model
    training_data = data.loc[~data["year"].isin(conf_params["validation_years"])].copy()
    training_data.to_csv(
        f'{os.path.join(PROJECT_DIR, conf_params["directory"]["processed_data"])}/training_data.csv',
        index=False,
    )
    logging.info(f"training written to file: {training_data.shape}")


def prep_data(config_path: Text) -> None:
    """
    main function to
    run
    """

    logger = logging.getLogger(__name__)
    logger.info("getting data")

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    sb_data, teams_data = load_data(config)
    data_wrangled = wrangle_data(sb_data, teams_data)
    split_data(config, data_wrangled)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    prep_data(args.config)
