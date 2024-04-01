import dash
from dash import html, dcc, ctx
from dash.dependencies import Input, Output
import pandas as pd
import requests
import io
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import os
from dotenv import load_dotenv
import numpy as np

import itertools

import pandas as pd
import requests
import io
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import multiprocessing
import os
from datetime import datetime, date, timedelta
import pickle
from boxsdk import Client, OAuth2
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ShuffleSplit, KFold
import math
from dateutil import parser
import seaborn as sns
import xgboost as xgb
import random
from itertools import combinations
from sklearn.neighbors import NearestNeighbors

import time
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_squared_error
from sklearn.utils.estimator_checks import check_estimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import io
import pandas as pd
import openpyxl
import re
import numpy as np
from pip._internal.cli.cmdoptions import pre
from scipy.spatial import cKDTree
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import builtins

import warnings
warnings.filterwarnings("ignore")


load_dotenv()

response = requests.get(os.getenv('url'), timeout=60)  # Timeout set to 30 seconds
response_2 = requests.get(os.getenv('url_2'), timeout=60)  # Timeout set to 30 seconds
response_3 = requests.get(os.getenv('url_3'), timeout=60)  # Timeout set to 30 seconds

# Set the IDE's Display to something really big so large DataFrames are visible
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 3000)


# Record the start time
start_time = time.time()

#Scaler
scaler = StandardScaler()

def load_csv_from_box(file_id):
    headers = {
        'Authorization': f'Bearer {os.getenv("DEVELOPER_TOKEN")}'
    }
    response = requests.get(f'https://api.box.com/2.0/files/{file_id}/content', headers=headers)
    response.raise_for_status()
    content = response.content
    # Try different encodings until successful
    encodings = ['utf-8', 'iso-8859-1', 'latin-1']
    for encoding in encodings:
        try:
            # Convert bytes to a file-like object
            file_like_object = io.BytesIO(content)
            # Read CSV content using specified encoding
            df = pd.read_csv(file_like_object, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to decode CSV content using any of the specified encodings")

#Load BYTE type data from box.com based on file ID
def load_pickled_data_from_box(file_id):
    # Box API endpoint for file content
    url = f"https://api.box.com/2.0/files/{file_id}/content"
    # Authorization header with the developer token
    headers = {
        "Authorization": f"Bearer {os.getenv('DEVELOPER_TOKEN')}"  # Use single quotes here
    }
    # Make GET request to retrieve file content
    response = requests.get(url, headers=headers)
    # Check if request was successful
    if response.status_code == 200:
        # Deserialize the byte content into Python objects
        loaded_data = pickle.loads(response.content)
        return loaded_data
    else:
        print(f"Failed to load data from Box. Status code: {response.status_code}")
        return None

#Function to load in Regimen data for patient information / result comparisons
def load_regimen_data():

    # Define File IDs
    MULTI_ID = '1465824013335'
    OTHER0_ID = '1465824153046'
    OTHER1_ID = '1465819719909'
    MET1_ID = '1465825873734'
    MET0_ID = '1465825057493'
    MET2_ID = '1468178601581'
    final_merged_id = '1468891677131'

    # Load in files using IDs

    MULTI = load_csv_from_box(MULTI_ID)
    OTHER0 = load_csv_from_box(OTHER0_ID)
    OTHER1 = load_csv_from_box(OTHER1_ID)
    MET1 = load_csv_from_box(MET1_ID)
    MET0 = load_csv_from_box(MET0_ID)
    MET2 = load_csv_from_box(MET2_ID)
    final_merged = load_csv_from_box(final_merged_id)

    # Define a dictionary to store the dataframes
    regimens_dict = {
        'OTHER0': OTHER0,
        'OTHER1': OTHER1,
        'MET1': MET1,
        'MET0': MET0,
        'MET2': MET2,
    }
    return regimens_dict, final_merged

#Function to load in math models from box
def load_models():
    # Box file IDs for the pickled files
    file_ids = {
        "OTHER0": "1469269595554",
        "OTHER1": "1469270888727",
        "MET1": "1469269479937",
        "MET0": "1469270872710",
        "MET2": "1469259769610",
    }

    # Load pickled data from Box
    loaded_models = {}
    for model_name, file_id in file_ids.items():
        loaded_model = load_pickled_data_from_box(file_id)
        if loaded_model is not None:
            loaded_models[model_name] = loaded_model

    return loaded_models


#Call function to load in matrix data
regimens_dict, final_merged = load_regimen_data()

#Call function to load in math model data
loaded_models = load_models()

#Patient ID recieved from front end
patient_id=56212830

#Function for outputs for the Front end, Similiar Patient Results (Panel B)
def create_similiar_data(patient_id, regimen_dict, final_merged):

    #Start with the Result Data, and filter by our patient
    patient_result_data = final_merged[final_merged['PatID'] == patient_id]
    patient_result_data = patient_result_data.sort_values(by='Result Date', ascending=False)
    patient_result_data = patient_result_data.head(1)
    patient_result_data = patient_result_data.reset_index()

    # Formulate patient data entry
    patient_data = patient_result_data.drop(columns=['Result Date'])
    patient_data = patient_data.drop(columns=['Component'])
    patient_data = patient_data.drop(columns=['Regimen'])
    patient_data = patient_data.drop(columns=['PatID'])
    patient_data = patient_data.drop(columns=['Date Encount'])
    patient_data = patient_data.drop(columns=['BP'])
    patient_data = patient_data.drop(columns=['Result Numeric Value'])
    patient_data = patient_data.drop(columns=['index'])
    patient_data = patient_data.drop(columns=['Regimen Date'])

    processed_regimen_data = {}

    for key, df in regimen_dict.items():
        processed_df = df.drop(columns=['Result Date','Component', 'Regimen', 'PatID', 'Date Encount', 'BP','Regimen Date'])
        processed_regimen_data[key] = processed_df

    regimen_dataframes = {}

    # Perform feature scaling
    scaled_features = scaler.fit_transform(patient_data)

    for key, df in processed_regimen_data.items():

        # kNN Model Setup
        k = int(math.sqrt(len(df)))
        knn_model = NearestNeighbors(n_neighbors=k)

        # Store the column 'results' separately for later use
        results_column = df['Result Numeric Value']

        # Separate features
        df = df.drop(columns=['Result Numeric Value'])

        # Perform patient single entry scaling for features
        patient_data_scaled = scaler.transform(df)
        knn_model.fit(patient_data_scaled)

        # Find the k-nearest neighbors for the patient data
        distances, indices = knn_model.kneighbors(scaled_features)

        # Extract the indices of the similar patient entries
        similar_patients_indices = indices.flatten()

        # Filter the merged data by the indices of similar patients
        similar_patients_results = results_column.iloc[similar_patients_indices]

        # Create a DataFrame with the unscaled results
        similar_patients_results_df = pd.DataFrame(similar_patients_results, columns=['Result Numeric Value'])

        # Store the DataFrame with unscaled 'results' for the current regimen in the dictionary
        regimen_dataframes[key] = similar_patients_results_df


    return regimen_dataframes

#Function for outputs for frontend, patient information (Panel ?)
def generate_patient_infos(patient_id, final_merged):

    # Starting with the encounter information, save the row the patients information is stored in
    patient_info = final_merged[final_merged['PatID'] == patient_id]

    # Sort the patients encounter information by date
    patient_info = patient_info.sort_values(by='Result Date', ascending=False)

    # Take the most recent encounter informaton available
    patient_info = patient_info.head(1)

    return patient_info

#Function for outputs for frontend, patient result / regimen history information
def create_patient_hisotory2(final_merged, patient_id):

    # Start with the patients Results, filter by our patient
    patient_results = final_merged[final_merged['PatID'] == patient_id]

    #Update
    patient_results = patient_results[['Regimen', 'Result Numeric Value','Result Date','Regimen Date']]

    return patient_results

#Function for outputs for frontend, apply math models to patient for best result / reigmen
def create_patient_recomendation(loaded_models, final_merged, patient_id):

    #set model parameters
    cutoff = 5
    tolerance = 1.5

    # Generate patient information
    patient_info = generate_patient_infos(patient_id, final_merged)

    # Extract the actual result
    actual_result = patient_info['Result Numeric Value'].iloc[0]

    # Define columns to keep
    columns_to_keep = ['ESRD Hx', 'IDDM Hx', 'GDM Hx', 'Other DM Hx', 'CFDM Hx', 'Pancreatectomy Hx', 'Insulin Pump Hx', 'CVD Hx', 'LVH Hx', 'Retinopathy Hx', 'HTN Hx', 'Tobacco Hx', 'Dyslipidemia Hx',
                       'CRF Hx', 'CHF Hx', 'Obesity Hx', 'Albuminuria Hx', 'Amputation Hx', 'GI Hx', 'Pancreatitis Hx', 'Hypoglycemia Hx', 'Hypotension Hx', 'Fracture Hx', 'UTI Hx', 'MEN Hx', 'MTC Hx',
                       'FLD Hx', 'Pregnancy Hx', 'Sex', 'Ethnicity', 'BMI', 'Weight', 'BSA', 'Height', 'Age', 'Previous Result']

    # Keep only the specified columns in patient_info DataFrame
    patient_info = patient_info[columns_to_keep]

    # Initialize variables to track best result and model
    best_result = float('inf')  # Initialize to positive infinity
    best_model_name = None
    lowest_predicted_above_cutoff = float('inf')

    # Loop over each model in the loaded_models dictionary
    for model_name, model in loaded_models.items():
        # Use the model to predict the HbA1C results
        predicted_result = model.predict(patient_info)

        # Apply cutoff condition
        if predicted_result < cutoff:
            continue

        # Track the lowest predicted result among models above cutoff but below tolerance
        if predicted_result < lowest_predicted_above_cutoff:
            lowest_predicted_above_cutoff = predicted_result

        # Check if the predicted result meets tolerance criteria
        if actual_result - predicted_result < tolerance:
            continue

        # Check if the predicted result is the best so far
        if predicted_result < best_result:
            best_result = predicted_result
            best_model_name = model_name

    # If no model satisfies cutoff and tolerance criteria, choose the model with the lowest predicted result
    if best_model_name is None:
        best_result = lowest_predicted_above_cutoff
        for model_name, model in loaded_models.items():
            predicted_result = model.predict(patient_info)
            if predicted_result == lowest_predicted_above_cutoff:
                best_model_name = model_name
                break

    best_regimen_name = best_model_name

    #Round precited HbA1c
    best_result = round(best_result, 2)

    return best_result, best_regimen_name


#Use the functions to generate data for front end

# Similiar data (for histograms) (returns DICTIONARY of dataframes labeld via Regimen name)
regimen_dataframes = create_similiar_data(patient_id, regimens_dict,final_merged)

# Patient panel information (for pat info) (Returns DATAFRAME with 1 row of patients most RECENT information)
patient_info = generate_patient_infos(patient_id, final_merged)

# Patient regimen / result history (for graphs) (Returns DATAFRAME with 4 columns: Regimen, Result Numeric Value, Result Date, Regimen Date)
patient_hist = create_patient_hisotory2(final_merged, patient_id)

# Patient regimen recomendation (for rec panel) (Returns 2 values best_result, best_regimen_name) (int, string)
best_result, best_regimen = create_patient_recomendation(loaded_models, final_merged, patient_id)


#View the data

patient_data_recommendation = create_patient_recomendation(loaded_models, final_merged, patient_id)
print(patient_data_recommendation)

# print(patient_info)
# print(patient_hist)
# print('Predicted best result:', best_result)
# print('Predicted best regimen:', best_regimen)

#histogram on regimen dataframes


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time:", elapsed_time, "seconds")

