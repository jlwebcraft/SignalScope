"""Tests for evaluation metrics and FPR calculation."""

import numpy as np
import pytest

from model.evaluate import calculate_metrics, find_threshold_for_target_fpr


def test_calculate_metrics_perfect_predictions():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.05, 0.10, 0.15, 0.20, 0.85, 0.90, 0.95, 0.99])

    metrics = calculate_metrics(y_true, y_scores, operating_threshold=0.50)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["confusion_matrix"]["tp"] == 4
    assert metrics["confusion_matrix"]["tn"] == 4
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_calculate_metrics_with_false_positives():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # One real image incorrectly scored high (false positive)
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.85, 0.9, 0.95])

    metrics = calculate_metrics(y_true, y_scores, operating_threshold=0.50)

    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["fpr"] == 0.25


def test_find_threshold_for_target_fpr():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    thr, fpr = find_threshold_for_target_fpr(y_true, y_scores, target_fpr=0.0)
    assert fpr <= 0.0
