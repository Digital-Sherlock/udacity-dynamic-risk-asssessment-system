'''
Given module accomplishes model training and
consecutive model re-training.

Author: Vadim Polovnikov
Date: 2024-02-15
'''


from flask import Flask, session, jsonify, request
import logging
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Accessing configuration file
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

# Configuring logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(config['output_model_path'], 'model_training.log'),
    filemode='w'
)
logger = logging.getLogger()

def train_model(datapath, model_path):
    '''
    This function trains a model on a give dataset.
    
    Input:
        - datapath: (str) directory containing dataset
        - model_path: (str) directory to save final model to
    Output:
        - None
    '''
    
    filepath = os.path.join(datapath, "finaldata.csv")
    logger.info('Dataset {} has been collected'.format(filepath))
    df = pd.read_csv(filepath)

    logger.info("Extracting input and output variables...")
    X = df.iloc[:, 1:].copy()
    y = X.pop('exited')

    # Creating a model
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    logger.info("Fitting the model...")
    model.fit(X, y)

    model_path = os.path.join(model_path, 'trainedmodel.pkl')
    pickle.dump(model, open(model_path, 'wb'))
    logger.info("Model {} saved.".format(model_path))


if __name__ == "__main__":
    train_model(dataset_csv_path, model_path)