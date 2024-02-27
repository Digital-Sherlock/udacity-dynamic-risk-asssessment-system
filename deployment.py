'''
Model deployment - moves production-ready model,
its latest score and data sources into production
directory.

Author: Vadim Polovnikov
Date: 2024-02-16
'''

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Loading configuration file
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

def store_model_into_pickle(model, dst):
    '''
    Copies pickle file, the latestscore.txt value
    into the deployment directory.

    Input:
        - model: (str) path to the model
        - dst: (str) destination path
    Output:
        - None
    '''
    # Copying files to the deployment folder
    shutil.copytree(src=model, dst=dst, dirs_exist_ok=True)
    

if __name__ == '__main__':
    store_model_into_pickle(model_path, prod_deployment_path)
