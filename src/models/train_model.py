import argparse
import logging
import os
from collections import Counter
import pickle
from pathlib import Path
from typing import Text, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_data(f_path: dict) -> pd.DataFrame:
    """
    load data to be transformed
    """
    data = pd.read_csv(
        f'{os.path.join(PROJECT_DIR, f_path["directory"]["processed_data"])}/training_data.csv'
    )
    logging.info(f"training data loaded: {data.shape}")
    X, y = data.iloc[:, :-3].values, data["target"].values

    return X, y


def scale_data(data) -> Tuple[np.array, "sklearn.preprocessing._data.StandardScaler"]:
    """
    scale the data
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    return data_scaled, scaler


def get_class_scatter(save_dir, X, y, classes, title="Data", fig_fname="graph") -> None:
    """
    Helper function to
    plot graphs
    """
    plt.figure(figsize=(8, 8), dpi=80)
    for label, _ in classes.items():
        row_ix = np.where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.title(f"{title}")
    plt.legend()
    plt.savefig(
        f'{os.path.join(PROJECT_DIR,save_dir["directory"]["reports"])}/{fig_fname}.jpg'
    )


def get_ran_ovr_samp(ran_params, X, y) -> Tuple[np.array, np.array]:
    """
    Random oversampling works
    by picking samples at
    random with replacement
    of the minority class.
    """
    # fit and apply the transform
    over = RandomOverSampler(
        sampling_strategy=ran_params["rand_ovr_samp"]["sampling_size"], random_state=42
    )

    return over.fit_resample(X, y)


def get_ran_undr_samp(ran_params, X, y) -> Tuple[np.array, np.array]:
    """
    RandomUnderSampler is a fast
    and easy way to balance the
    data by randomly selecting
    a subset of data for the
    majority class.
    """
    under = RandomUnderSampler(
        sampling_strategy=ran_params["rand_undr_samp"]["sampling_size"], random_state=42
    )

    return under.fit_resample(X, y)


def sampling_helper(
    config_params: dict, X: np.array, y: np.array
) -> Tuple[np.array, np.array]:
    """
    Random over & under
    sampling helper function
    """
    logging.info(f"original class size: {Counter(y)}")
    org_class = Counter(y)
    get_class_scatter(config_params, X, y, org_class, "Original Data", "org_scatter")

    X_ovr, y_ovr = get_ran_ovr_samp(config_params, X, y)
    ovr_class = Counter(y_ovr)
    logging.info(f"random over sample class size: {ovr_class}")

    X_sampled, y_sampled = get_ran_undr_samp(config_params, X_ovr, y_ovr)
    sampled_class = Counter(y_sampled)
    logging.info(f"random over & under sampled class size: {sampled_class}")
    get_class_scatter(
        config_params,
        X_sampled,
        y_sampled,
        sampled_class,
        "Random Over & Under Sampled Data",
        "sampled_scatter",
    )

    return X_sampled, y_sampled


def train_model(
    lgr_params, X_train, y_train
) -> Tuple["sklearn.linear_model._logistic.LogisticRegression"]:
    """
    train logistic regression
    classifier
    """
    # now we'll create the model
    logreg = LogisticRegression(
        class_weight=lgr_params["logreg"]["class_weight"],
        C=lgr_params["logreg"]["c"],
        solver="liblinear",
        random_state=lgr_params["prepare"]["seed"],
        penalty=lgr_params["logreg"]["penalty"],
    )

    return logreg.fit(X_train, y_train)


def save_model(save_dir, model, scaler) -> None:
    """
    save classifier
    """

    clf_pkg = (model, scaler)
    with open(
        f'{os.path.join(PROJECT_DIR, save_dir["directory"]["models"])}/logreg_clf.pkl',
        "wb",
    ) as file:
        pickle.dump(clf_pkg, file)


def train_model_helper(train_params, X, y, scaler):
    """
    main helper
    functio
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_params["prepare"]["split"],
        random_state=train_params["prepare"]["seed"],
    )
    clf = train_model(train_params, X_train, y_train)

    logging.info("Saving model")
    save_model(train_params, clf, scaler)

    logging.info("writing split data to file")
    np.savetxt(
        f'{os.path.join(PROJECT_DIR, train_params["directory"]["processed_data"])}/x_train.txt',
        X_train,
        delimiter=",",
    )
    np.savetxt(
        f'{os.path.join(PROJECT_DIR, train_params["directory"]["processed_data"])}/x_test.txt',
        X_test,
        delimiter=",",
    )
    np.savetxt(
        f'{os.path.join(PROJECT_DIR, train_params["directory"]["processed_data"])}/y_train.txt',
        y_train,
        delimiter=",",
    )
    np.savetxt(
        f'{os.path.join(PROJECT_DIR, train_params["directory"]["processed_data"])}/y_test.txt',
        y_test,
        delimiter=",",
    )


def get_trained_model(config_path: Text) -> None:
    """
    main function to
    run
    """

    logger = logging.getLogger(__name__)
    logger.info("getting data")

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    X, y = load_data(config)
    X_scaled, scaler = scale_data(X)
    X_sampled, y_sampled = sampling_helper(config, X_scaled, y)
    train_model_helper(config, X_sampled, y_sampled, scaler)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    get_trained_model(args.config)
