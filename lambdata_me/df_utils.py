"""
About: Utility Functions for working with Dataframes
"""

import re
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


def GetConfusionMatrix(y_true, y_pred):
    """
    Example:
        table = get_confusion_matrix(y_true, y_pred)
        sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
    """
    labels = unique_labels(y_true)
    columns = [f'Predicted{label}' for label in labels]
    index = [f'Actual{label}' for label in labels]
    table = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=columns,
        index=index)

    return table


def PolynomialRegression(degree=2, **kwargs):
    from sklearn.preprocessing import PolynomialFeatures
    return make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression(
            **kwargs))

# web scraping, threads, launching multiple programs, task scheduling,
# https://automatetheboringstuff.com/chapter15/
# image scraping
# https://automatetheboringstuff.com/chapter17/

# df['columns'].str.contains('^my_regex')
