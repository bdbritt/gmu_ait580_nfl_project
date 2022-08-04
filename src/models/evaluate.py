import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Tuple, Text
import yaml
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from matplotlib import pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]


def get_data(load_dir) -> Tuple[np.array, np.array]:
    """
    get data for training
    """
    return np.loadtxt(
        f"{os.path.join(PROJECT_DIR, load_dir['directory']['processed_data'])}\\x_test.txt",
        delimiter=",",
    ), np.loadtxt(
        f"{os.path.join(PROJECT_DIR, load_dir['directory']['processed_data'])}\\y_test.txt",
        delimiter=",",
    )


def get_model(load_dir) -> Tuple["sklearn.linear_model._logistic.LogisticRegression"]:
    """ """
    with (
        open(
            f'{os.path.join(PROJECT_DIR, load_dir["directory"]["models"])}\\logreg_clf.pkl',
            "rb",
        )
    ) as f:
        model = pickle.load(f)
        return model


def get_predictions(model, x_test) -> np.array:
    """ """
    return model.predict(x_test)


def get_model_metrics(model, x_test, y_test) -> Tuple[dict, np.array]:
    """ """

    predictions = get_predictions(model, x_test)
    roc = roc_auc_score(y_test, predictions)
    cr = classification_report(predictions, y_test, output_dict=True)
    cm = confusion_matrix(predictions, y_test)
    # tn,fp,fn,tp = cm.ravel()

    f1 = f1_score(y_true=y_test, y_pred=predictions, average="macro")
    print(f1)

    report = {
        "accuracy": cr["accuracy"],
        "f1": cr["macro avg"]["f1-score"],
        "class_0_precison": cr["0.0"]["precision"],
        "class_0_recall": cr["0.0"]["recall"],
        "class_1_precison": cr["1.0"]["precision"],
        "class_1_recall": cr["1.0"]["recall"],
        "roc": roc,
        "cm": cm.tolist(),
    }

    return report, cm


def evaluate_model(config_path: Text):
    """
    evaluate model on training data
    """
    logger = logging.getLogger(__name__)
    logger.info("getting model metrics")

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    x_test, y_test = get_data(config)

    model = get_model(config)[0]
    report, cm = get_model_metrics(model, x_test, y_test)

    print(report)
    print(cm)

    reports_folder = Path(config["evaluate"]["reports_dir"])
    metrics_path = reports_folder / config["evaluate"]["metrics_file"]

    json.dump(obj=report, fp=open(metrics_path, "w"))

    confusion_matrix_png_path = (
        reports_folder / config["evaluate"]["confusion_matrix_image"]
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.savefig(os.path.join(PROJECT_DIR, confusion_matrix_png_path))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
