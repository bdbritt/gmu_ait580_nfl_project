import argparse
import logging
import os
from pathlib import Path
from typing import Text

import pandas as pd
import requests
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
# TEAMS_FILE = 'data\\raw\\teams.csv'
# TEAMS_STATS = 'data\\raw\\teams_stats.csv'


def load_data(f_dir: dict) -> pd.DataFrame:
    """
    Get local data
    """

    teams = pd.read_csv(
        f'{os.path.join(PROJECT_DIR, f_dir["directory"]["raw_data"])}/teams.csv'
    )
    logging.info(f"teams count: {teams.shape[0]}")

    return teams


def get_nfl_team_data(teams: dict, time_period: list) -> list:
    """
    get nfl team stats
    data from source
    """
    # send requests to get data
    data_list = []
    index_cntr = 0
    for team in teams:
        for year in time_period:
            url = f'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/2/teams/{team["team_number"]}/statistics'
            try:
                r_data = requests.get(url).json()["splits"]["categories"]

                stat_list = []
                for rec in r_data:
                    stats_data = rec["stats"]
                    for stats_rec in stats_data:
                        if stats_rec["name"].startswith(
                            ("passing", "rushing", "total")
                        ):
                            if (
                                "totalGiveaways" not in stats_rec["name"]
                                or "totalTakeaways" not in stats_rec["name"]
                            ):
                                stat_values = (stats_rec["name"], stats_rec["value"])
                                stat_list.append(stat_values)

                # convert to dataframe to get all values
                df = pd.DataFrame(stat_list)

                # transpose to get field values as columns
                df = df.T
                headers = df.iloc[0]
                new_df = pd.DataFrame(df.values[1:], columns=headers)
                new_df = new_df.reset_index(drop=True)

                # add team/year information
                new_df["team_number"] = team["team_number"]
                new_df["team"] = team["team_name"]
                new_df["year"] = year
                new_df["record_id"] = index_cntr
                new_df = new_df.set_index("record_id")

                # convert to dict
                data_dict = new_df.to_dict(orient="records")
                data_list.append(data_dict)
                index_cntr += 1

            except KeyError as err:
                logging.info(f"error: {err}")
                logging.info(f"error with team: {team}, year: {year}")
                # error will happen if data
                # does not exist for a teams year
                # skip past it for now
                continue

    return data_list


def get_season_stats(f_dir: dict, teams, start_year, end_year) -> None:
    """
    Get team stats
    """

    # convert teams df to dict for request
    teams_dict = teams.to_dict("records")

    # time period used to get
    # team stats
    time_period = list(
        pd.date_range(
            start=start_year, end=end_year, freq=pd.DateOffset(years=1)
        ).strftime("%Y")
    )

    # get data
    data = get_nfl_team_data(teams_dict, time_period)

    # convert data into one df
    stats_df = pd.concat([pd.DataFrame(rec) for rec in data]).reset_index(drop=True)

    logging.info(f"stats df: {stats_df.shape}")
    logging.info("writing records to file")
    stats_df.to_csv(
        f'{os.path.join(PROJECT_DIR, f_dir["directory"]["raw_data"])}/teams_stats.csv',
        index=False,
    )


def get_data(config_path: Text, start_date: Text, end_date: Text):
    """
    main help data function
    """

    logger = logging.getLogger(__name__)
    logger.info("getting data")

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    teams_data = load_data(config)
    get_season_stats(config, teams_data, start_date, end_date)


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--start", dest="start", required=True)
    args_parser.add_argument("--end", dest="end", required=True)
    args = args_parser.parse_args()

    get_data(args.config, args.start, args.end)
