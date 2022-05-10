import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Text, Tuple
import matplotlib

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PROJECT_DIR = Path(__file__).resolve().parents[2]
CLASSIFIER = "models\\logreg_clf.pkl"
VALIDATION_DATA = "data\\processed\\validation_data.csv"


def load_model() -> Tuple[
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.preprocessing._data.StandardScaler",
]:
    """
    load classifier
    and scaler
    """

    with (open(os.path.join(PROJECT_DIR, CLASSIFIER), "rb")) as f:
        clf_pipe = pickle.load(f)
        return clf_pipe[0], clf_pipe[1]


def load_val_data() -> pd.DataFrame:
    """
    load data
    """

    data = pd.read_csv(os.path.join(PROJECT_DIR, VALIDATION_DATA))
    logging.info(f"training data loaded: {data.shape}")

    return data


def save_predictions(data, nfl_year) -> None:
    """
    save to pdf
    """

    data = data.sort_values(by=["prediction"], ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=80)
    ax.axis("tight")
    ax.axis("off")
    ax.table(cellText=data.values, colLabels=data.columns, loc="center")

    pp = PdfPages(os.path.join(PROJECT_DIR, f"reports\\predictions_{nfl_year}.pdf"))
    pp.savefig(fig, bbox_inches="tight")
    pp.close()


def get_predictions(year, model, scaler, data):
    """ """
    data = data.loc[data["year"] == int(year)].copy()

    columns = data.columns[:-3]
    teams = data["team"].values

    data_scaled = pd.DataFrame(
        scaler.fit_transform(data.iloc[:, :-3].values),
        columns=columns,
        index=data.index,
    )
    data_scaled.loc[:, "team"] = teams
    data_scaled.loc[:, "prediction"] = model.predict_proba(
        data_scaled.iloc[:, :-1].values
    )[:, 1]

    results = data_scaled[["team", "prediction"]]
    logging.info("saving predictions to fle")
    save_predictions(results, year)
    logging.info(results.sort_values(by=["prediction"], ascending=False))


def main(nfl_year: Text) -> None:
    """
    main function to
    run
    """

    logger = logging.getLogger(__name__)
    logger.info("getting classifier")
    clf, scaler = load_model()
    data = load_val_data()
    get_predictions(nfl_year, clf, scaler, data)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--year", dest="year", required=True)
    args = args_parser.parse_args()
    main(args.year)
