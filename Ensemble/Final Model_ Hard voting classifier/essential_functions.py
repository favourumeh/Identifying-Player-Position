# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:49:59 2021
 There are some of the functions used in the tool 
 
@author: favou
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def min_max_scaler(df):
    df5 = pd.DataFrame()
    for column in df:
        if 'per' not in column :
            df5[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())
        
        if 'per' in column:
            df5[column] = df[column]
    return df5


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, annot=True, fmt='.1f', annot_kws={'size':12})
    else:
        sns.heatmap(cm)
    plt.title(title)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')