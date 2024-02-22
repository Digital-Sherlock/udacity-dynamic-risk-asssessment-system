'''
This module configures an API for querying
ML project for various stats as well as
making predictions.

Author: Vadim Polovnikov
Date: 2024-02-19
'''

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import subprocess
import json
import os
from diagnostics import *


app = Flask(__name__)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_model_path = os.path.join(config['prod_deployment_path'])


@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    '''
    Takes dataset's file locattion and makes
    predictions based on the data.
    '''
    # User input
    dataset_path = request.args.get('datasetdir')
    
    # Predictions
    yhat = model_predictions(prod_model_path, dataset_path)
    
    return str(yhat) + '\n'


@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
   '''
   Returns a model's F1 score.
   '''
   # Runs the scoring script
   subprocess.run(['python', 'scoring.py'])

   # Reading the score
   f1_score = open(os.path.join('practicemodels', 'latestscore.txt')).read()

   return str(f1_score)[:4] + '\n'


@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    '''
    Returns key dataset statistics.
    '''
    stats = dataframe_summary(dataset_csv_path)

    return stats


@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diags():        
    '''
    Returns diagnostics data including
    timings, NA stats, and outdated
    dependencies.
    '''
    timings = execution_time('ingestion.py', 'training.py')
    na_perc_dict = missing_data(dataset_csv_path)
    outdated_packages = outdated_packages_list()

    return str([outdated_packages, timings, na_perc_dict])

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True)
