'''
Diagnostics module. Sets up a baseline of
expected model behavior.

Author: Vadim Polovnikov
Date: 2024-02-16
'''

import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle
import subprocess


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='diagnostics.log',
    filemode='w'
)
logger = logging.getLogger()


def model_predictions(model, data):
    '''
    Makes predictions with the give model.

    Input:
        - model: (str) model path
    Output:
        - yhat: (numpy.ndarray) predictions
    '''
    # Retrieving data and model
    model_path = os.path.join(model, 'trainedmodel.pkl')
    data_path = os.path.join(data, 'testdata.csv')

    # Setting up input/output variables
    data = pd.read_csv(data_path)
    X_test = data.iloc[:, 1:].copy()
    y_test = X_test.pop('exited')

    # Making predictions
    model = pickle.load(open(model_path, 'rb'))
    yhat = model.predict(X_test)
    logger.info(f'Predictions - {yhat}')

    return yhat


def dataframe_summary(data: str) -> list:
    '''
    Calculates statistics like mean,
    median and std for each column in the
    given dataset.

    Input:
        - data: (str) data path
    Output:
        - stats: (list) list of stats
    '''
    # Reading dataset
    df = pd.read_csv(os.path.join(data, 'finaldata.csv'))
    df = df.select_dtypes(include=[np.number])

    # Calculating the stats
    stats = [
        {
            'mean': np.mean(df[col]),
            'median': np.median(df[col]),
            'std': np.std(df[col])
        }
        for col in df.columns
            ]
    
    logging.info(f"Columns stats - {stats}")

    return stats


def execution_time(*args):
    '''
    Calculate timing of training.py and ingestion.py

    Input:
        - *args: (str) module to time
    Output:
        - timings: (dict) module execution timings
    '''
    timings = {}
    # Iterating over passed modules
    for module in args:
        # Setting up the timer
        start_time = timeit.default_timer()
        # Executing a module
        os.system(f'conda run python {module}')
        # Timing the difference
        timer = abs(start_time - timeit.default_timer())
        timings[module] = timer
        logger.info(f'{module} execution took {round(timer, 2)}s')

    return  timings


def missing_data(data: str) -> list:
    '''
    Calculates percentage of NA values
    in each column of the give dataset.

    Input:
        - data: (str) dataset path
    Output:
        - na_prctg: (list) list of NA % per column
    '''
    data_pth  = os.path.join(data, 'finaldata.csv')
    df = pd.read_csv(data_pth)

    # Calculating NA percentage per column
    na_perc_col = (pd.isna(df).sum() / len(df)) * 100 # pd.Series
    na_perc_dict = na_perc_col.to_dict()
    logger.info(f'NA percentage per column - {na_perc_dict}')

    return na_perc_dict


def outdated_packages_list():
    '''
    Checking version of installed
    dependencies
    
    Input:
        - None
    Output:
        - None
    '''
    outdated = subprocess.run(['pip', 'list', '--outdated'],
                              capture_output=True).stdout
    
    with open('outdated_packages.txt', 'wb') as file:
        file.write(outdated)
    
    logger.info('List of outdated packages generated in outdated_packages.txt')


if __name__ == '__main__':
    model_predictions(model_path, test_data_path)
    dataframe_summary(dataset_csv_path)
    execution_time('ingestion.py', 'training.py')
    missing_data(dataset_csv_path)
    outdated_packages_list()





    
