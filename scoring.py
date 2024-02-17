'''
This module accomplishes model scoring using
F1 score metric.

Author: Vadim Polovnikov
Date: 2024-02-16
'''

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Loading configuration file
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(model_path, 'model_scoring.log'),
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def score_model(model_path, testdata_path):
    '''
    This function scores the model and saves
    the resutls in the given directory.

    Input:
        - model_path: (str) model location
    Output:
        - None
    '''
    logger.info('Loading the model...')
    model = pickle.load(open(f'{model_path}/trainedmodel.pkl', 'rb'))
    
    logger.info("Loading test data...")
    dataset = pd.read_csv(f'{testdata_path}/testdata.csv')
    X_test = dataset.iloc[:, 1:].copy()
    y_test = X_test.pop('exited')

    logger.info("Making predictions...")
    yhat = model.predict(X_test)

    logger.info("Scoring the model...")
    f1_score = metrics.f1_score(yhat, y_test)
    logger.info("Model's F1 score - {}".format(f1_score))

    with open(f'{model_path}/latestscore.txt', 'w') as file:
        file.write(str(f1_score))


if __name__ == '__main__':
    score_model(model_path, test_data_path)