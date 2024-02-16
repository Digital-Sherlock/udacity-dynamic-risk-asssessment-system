'''
This is data ingestion module that reads files from
the source folder combines them in a single dataset
and outputs in a .csv format to a target directory.

Author: Vadim Polovnikov
Date: 2024-01-12
'''

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging


# Configuring input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    filename=os.path.join(output_folder_path, 'ingestedfiles.txt')
)
logger = logging.getLogger()

def merge_multiple_dataframe(src, dst):
    '''
    Locates datasets from the provided source,
    combines them into one and writes to 
    destination.

    Input:
        - src: (str) source directory
        - dst: (str) destination directory
    Output:
        - None
    '''
    # Locating .csv files in the src folder
    data_sources = []
    for file in os.listdir(src):
        if '.csv' in file:
            data_sources.append(os.path.join(src, file))

    logging.info(f'Located data sources - {data_sources}')
    
    # Converting data paths into Pandas DataFrames
    datasets = [pd.read_csv(dataset) for dataset in data_sources]

    # Concatenating datasets into a single dataframe
    concatenated = pd.concat(datasets, axis=0).drop_duplicates(ignore_index=True)
    
    # Checking if there are any duplicate 'corporation' entries
    assert concatenated.shape[0] == len(concatenated.corporation.unique())

    # Saving final dataset
    final_dataset = concatenated.to_csv(os.path.join(dst, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path)
