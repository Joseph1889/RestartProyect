import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from collections import Counter


from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("DataModeling/Dataset.csv")

#Resampling
X = df.drop('Machine failure', axis=1)  # Drop the 'Machine failure' column from the features
y = df['Machine failure']  # Define the target variable 'Machine failure'

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train_before, X_test, y_train_before, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the SMOTETomek object for handling imbalanced data (oversampling + undersampling)
os_us = SMOTETomek()

# Resample the training data to balance the classes
X_train, y_train = os_us.fit_resample(X_train_before, y_train_before)

Rforest = RandomForestClassifier()
Rforest.fit(X_train, y_train)

y_pred = Rforest.predict(X_test)
#y_pred_train = Rforest.predict(X_train)

#alores_para_predecir = np.array([[298.1, 308.6, 1551, 42.8, 0, 0, 0, 1]])
valores_para_predecir = np.array([[298.9,309.1,2861,4.6,143,0,1,0]])
prediccion = Rforest.predict(valores_para_predecir)
if prediccion[0] == 1:
    print("El modelo predice que habrá una falla en la máquina.")
else:
    print("El modelo predice que no habrá una falla en la máquina.")

""""
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Set our tracking server uri for logging
#mlflow.set_tracking_uri(uri="http://172.23.73.213:30000")
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Milling-Experiment")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    #mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Random Forest model for the milling dataset")

    # Infer the model signature
    signature = infer_signature(X_train, Rforest.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=Rforest,
        artifact_path="RandomForest",
        signature=signature,
        input_example=X_train,
        registered_model_name="model",
    )
"""