'''
This module calls API endpoints configured in
app.py and saves the output in a textfile.

Author: Vadim Polovnikov
Date: 2024-02-21
'''

import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Calling endpoints
response1 = requests.get(f'{URL}/prediction?datasetdir=testdata').content
response2 = requests.get(f'{URL}/scoring').content
response3 = requests.get(f'{URL}/summarystats').content
response4 = requests.get(f'{URL}/diagnostics').content


# Combining outputs to a text file
responses = [response1, response2, response3, response4]
with open('apireturns.txt', 'a') as file:
    for response in responses:
        file.write(str(response) + '\n')
