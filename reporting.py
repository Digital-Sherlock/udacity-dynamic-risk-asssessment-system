'''
This module generates and saves confusion
matrix.

Author: Vadim Polovnikov
Date: 2024-02-19
'''

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from diagnostics import model_predictions


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['test_data_path'])
model_dir = os.path.join(config['prod_deployment_path'])


def score_model(test_data_pth, model):
    '''
    Generates confusion matrix and saves
    to disk.

    Input:
        - test_data_pth: (str) test data location
        - model: (str) model location
    Output:
        - None
    '''
    # Loading test data
    test_data = pd.read_csv(os.path.join(test_data_pth, 'testdata.csv'))
    y = test_data['exited']
    
    # Making predictions
    yhat = model_predictions(model, test_data_pth)

    # Generating confusion matrix
    confusion_matrix = metrics.confusion_matrix(y, yhat)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=['Negative', 'Positive'])
    cm_display.plot()
    plt.savefig('confusionmatrix.png')
    plt.close()


if __name__ == '__main__':
    score_model(dataset_csv_path, model_dir)
