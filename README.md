# A Dynamic Risk Assessment System

The system is designed to predict attrition risk of each of the company's 10,000 clients.

## Technical Details

### Data Ingestion

- ingestion.py

Pulls data from the source directory and combines it in a single file.

### Training, Scoring, and Deploying

- training.py

Trains the model based on the ingested data.

- scoring.py

Calculates the F1 score the the trained model.

- deployment.py

Deploys the model into the production.

### Diagnostics

- diagnostics.py

Performs basic diagnostics including gathering key statistics for the given dataset, times the critical functions execution, and reports outdated packages.

### Reporting

- app.py

API configuration file.

- apicalls.py

Calls API endpoints to get reports, predictions, etc.

### Automation

- fullprocess.py

Automates the above processes including ingesting newly received data, re-training and re-deploying a model if there's a model drift and gathering key statistics via API reporting and diagnostics.
