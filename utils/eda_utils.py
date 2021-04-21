import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_size(plot, size=20):
    """
        This method aims to set label, legend, title.... size of matplotlib.pyplot
    """
    plot.rc('font', size=size)  # controls default text sizes
    plot.rc('axes', titlesize=size)  # fontsize of the axes title
    plot.rc('axes', labelsize=size)  # fontsize of the x and y labels
    plot.rc('xtick', labelsize=size)  # fontsize of the tick labels
    plot.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plot.rc('legend', fontsize=size)  # legend fontsize
    plot.rc('figure', titlesize=size)


def top_frequency(df_column, plot, seaborn, x_label, y_label, title, figsize=(10, 8), top=30):
    """
        Plot top frequency of a pandas column. (Horizontal)
        Good if there is multiple values
    """
    plot.figure(figsize=figsize)
    df = df_column.value_counts().reset_index()
    df.columns = [y_label, x_label]
    df[y_label] = df[y_label].astype(str) + '-'
    df = df.sort_values([x_label], ascending=False)
    seaborn.barplot(x=x_label,
                    y=y_label,
                    data=df[:top],
                    orient='h')
    plot.title(title)


def countplot(df_column, plot, seaborn):
    """
        Plot top frequency of a pandas column. (Vertical)
        Good if there is a few values
    """
    plot.figure(dpi=100)
    seaborn.countplot(df_column)
    plot.show()

