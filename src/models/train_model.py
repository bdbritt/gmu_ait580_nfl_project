import json
import logging
import os
from collections import Counter
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PROJECT_DIR = Path(__file__).resolve().parents[2]
FINAL_DATA = "data\\processed\\training_data.csv"
VALIDATION_DATA = "data\\processed\\validation_data.csv"
TEST_SIZE = 0.20
LGREG_C = 0.5
LGREG_CLASS_WEIGHT = "balanced"
LGREG_PENALTY = "l1"


def load_data() -> pd.DataFrame:
    """
    load data to be transformed
    """
    data = pd.read_csv(os.path.join(PROJECT_DIR, FINAL_DATA))
    print(data.head())
    logging.info(f"training data loaded: {data.shape}")
    X, y = data.iloc[:, :-3].values, data["target"].values

    return X, y


def scale_data(data) -> Tuple[np.array, 'sklearn.preprocessing._data.StandardScaler']:
    """
    scale the data
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    return data_scaled, scaler


def get_class_scatter(X, y, classes, title="Data", fig_fname="graph") -> None:
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
    plt.savefig(os.path.join(PROJECT_DIR, f"reports\\{fig_fname}.jpg"))


def get_ran_ovr_samp(X, y, sampling_size=0.20) -> Tuple[np.array, np.array]:
    """
    Random oversampling works
    by picking samples at
    random with replacement
    of the minority class.
    """
    # fit and apply the transform
    over = RandomOverSampler(sampling_strategy=sampling_size, random_state=42)

    return over.fit_resample(X, y)


def get_ran_undr_samp(X, y, sampling_size=0.5) -> Tuple[np.array, np.array]:
    """
    RandomUnderSampler is a fast
    and easy way to balance the
    data by randomly selecting
    a subset of data for the
    majority class.
    """
    under = RandomUnderSampler(sampling_strategy=sampling_size, random_state=42)

    return under.fit_resample(X, y)


def sampling_helper(X, y) -> Tuple[np.array, np.array]:
    """
    Random over & under
    sampling helper function
    """
    logging.info(f"original class size: {Counter(y)}")
    org_class = Counter(y)
    get_class_scatter(X, y, org_class, "Original Data", "org_scatter")

    X_ovr, y_ovr = get_ran_ovr_samp(X, y)
    ovr_class = Counter(y_ovr)
    logging.info(f"random over sample class size: {ovr_class}")

    X_sampled, y_sampled = get_ran_undr_samp(X_ovr, y_ovr)
    sampled_class = Counter(y_sampled)
    logging.info(f"random over & under sampled class size: {sampled_class}")
    get_class_scatter(
        X_sampled,
        y_sampled,
        sampled_class,
        "Random Over & Under Sampled Data",
        "sampled_scatter",
    )

    return X_sampled, y_sampled


def train_model(
    X_train, y_train
) -> Tuple["sklearn.linear_model._logistic.LogisticRegression", np.array]:
    """
    train logistic regression
    classifier
    """
    # now we'll create the model
    logreg = LogisticRegression(
        class_weight=LGREG_CLASS_WEIGHT,
        C=LGREG_C,
        solver="liblinear",
        random_state=42,
        penalty=LGREG_PENALTY,
    )

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    clf = logreg.fit(X_train, y_train)
    n_scores = cross_val_score(
        clf, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise"
    )
    logging.info("Accuracy: %.3f (%.3f)" % (np.mean(n_scores), np.std(n_scores)))

    return clf, n_scores


def get_classification_report(clf, X_test, y_test) -> dict:
    """
    Write classification
    report to file
    """
    y_pred = clf.predict(X_test)
    report = {}
    report["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    return report


def get_confusion_matrix(clf, X_test, y_test) -> None:
    """
    Generate confusion
    matrix
    """
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="Blues")
    plt.savefig(os.path.join(PROJECT_DIR, f"reports\\confusion_matrix.jpg"))


def get_roc_graph(model, X_test, y_test, title="model") -> None:
    """
    generate roc
    graph
    """
    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=model_roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PROJECT_DIR, f"reports\\roc_graph.jpg"))


def save_model(model, scaler) -> None:
    """
    save classifier
    """

    clf_pkg = (model, scaler)

    with open(os.path.join(PROJECT_DIR, "models\\logreg_clf.pkl"), "wb") as file:
        pickle.dump(clf_pkg, file)


def save_model_metrics(scores, class_report) -> None:
    """
    save model metrics
    """
    report = {}
    report["accuracy"] = np.mean(scores)
    report["scores"] = list(scores)
    report["scores_std"] = np.std(scores)
    report.update(class_report)

    with open(os.path.join(PROJECT_DIR, "reports\\clf_results.json"), "w") as fout:
        json_dumps_str = json.dumps(report, indent=4)
        fout.write(json_dumps_str)


def train_model_helper(X, y, scaler):
    """
    main helper
    functio
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )
    clf, scores = train_model(X_train, y_train)
    logging.info("Generating classifier reports")
    clf_report = get_classification_report(clf, X_test, y_test)
    get_confusion_matrix(clf, X_test, y_test)
    get_roc_graph(clf, X_test, y_test)
    logging.info("Saving model metrics")
    save_model_metrics(scores, clf_report)
    logging.info("Saving model")
    save_model(clf, scaler)


def main() -> None:
    """
    main function to
    run
    """

    logger = logging.getLogger(__name__)
    logger.info("getting data")

    X, y = load_data()
    X_scaled, scaler = scale_data(X)
    X_sampled, y_sampled = sampling_helper(X_scaled, y)
    train_model_helper(X_sampled, y_sampled, scaler)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    main()
