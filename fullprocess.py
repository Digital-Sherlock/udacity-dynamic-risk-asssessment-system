'''
This module automates model scoring, monitoring,
and re-deployment.

Author: Vadim Polovnikov
Date: 2024-02-21
'''

import ast
import logging
import os
import json
import subprocess
import pickle
import pandas as pd
from sklearn.metrics import f1_score


# Loading config file
with open('config.json') as file:
    config = json.load(file)

ingesteddata = config['output_folder_path']
sourcedata = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']


# Reading the already ingested data
ingested_files_path = os.path.join(ingesteddata, 'ingestedfiles.txt')
ingested_files = ast.literal_eval(open(ingested_files_path, 'r').read())
# Removing the absolute file path (sourcedata/)
ingested_files_cleaned = [file.split('/')[-1] for file in ingested_files]

# Checking if all source data was ingested
source_files = os.listdir(sourcedata)
all_ingested = set(source_files).issubset(set(ingested_files_cleaned))


if all_ingested == False:
    # Ingesting new data
    subprocess.run(['python', 'ingestion.py'])

    # Loading prod model and its score
    prod_model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))
    latestscore = open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r').read()
    
    # Loading newly ingested data (first col not used for preds)
    df = pd.read_csv(os.path.join(ingesteddata, 'finaldata.csv')).iloc[:, 1:]
    X = df.copy()
    y = X.pop('exited')

    # Making predictions on a new data
    yhat = prod_model.predict(X)
    current_score = f1_score(y, yhat)
else:
    pass

# Checking for model drift
try:
    model_drift = current_score < float(latestscore)
    if model_drift == True:
        # Re-training, scoring and re-deploying the model
        subprocess.run(['python', 'training.py'])
        subprocess.run(['python', 'scoring.py'])
        subprocess.run(['python', 'deployment.py'])
    else:
        pass
except NameError:
    print("No drift Detected")

# Diagnostics and reporting
subprocess.run(['python', 'reporting.py'])
subprocess.run(['python', 'apicalls.py'])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='fullprocess.log',
        filemode='a'
    )
    logger = logging.getLogger()