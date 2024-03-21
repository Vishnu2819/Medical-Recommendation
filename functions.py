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

load_dotenv()

response = requests.get(os.getenv('url'), timeout=60)  # Timeout set to 30 seconds
response_2 = requests.get(os.getenv('url_2'), timeout=60)  # Timeout set to 30 seconds
response_3 = requests.get(os.getenv('url_3'), timeout=60)  # Timeout set to 30 seconds

# Set the IDE's Display to something really big so large DataFrames are visible
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 3000)

# def load_csv_from_box(file_id):
#     headers = {
#         'Authorization': f'Bearer {os.getenv("DEVELOPER_TOKEN")}'
#     }
#     response = requests.get(f'https://api.box.com/2.0/files/{file_id}/content', headers=headers)
#     response.raise_for_status()
#     content = response.content
#     encodings = ['utf-8', 'iso-8859-1', 'latin-1']
#     for encoding in encodings:
#         try:
#             file_like_object = io.BytesIO(content)
#             df = pd.read_csv(file_like_object, encoding=encoding)
#             return df
#         except UnicodeDecodeError:
#             continue
#     raise UnicodeDecodeError("Unable to decode CSV content using any of the specified encodings")

# def generate_clean_data():

#     feature_id = '1450001295856'
#     encounter_id = '1450965225437'
#     result1_id = '1450973176414'
#     result2_id = '1450972909692'
#     treatment_id = '1450975521323'

#     encounters = load_csv_from_box(encounter_id)
#     features = load_csv_from_box(feature_id)
#     result1 = load_csv_from_box(result1_id)
#     result2 = load_csv_from_box(result2_id)
#     treatments = load_csv_from_box(treatment_id)

#     results = pd.concat([result1, result2])

#     # Import a Patient info Data Frame, include Patient ID, and Previous Medical Illnesses
#     medical_features = features[
#         ['PatID', 'ESRD Hx', 'IDDM Hx', 'GDM Hx', 'Other DM Hx', 'CFDM Hx', 'Pancreatectomy Hx', 'Insulin Pump Hx', 'CVD Hx', 'LVH Hx', 'Retinopathy Hx', 'HTN Hx', 'Tobacco Hx', 'Dyslipidemia Hx',
#          'CRF Hx', 'CHF Hx', 'Obesity Hx', 'Albuminuria Hx', 'Amputation Hx', 'GI Hx', 'Pancreatitis Hx', 'Hypoglycemia Hx', 'Hypotension Hx', 'Fracture Hx', 'UTI Hx', 'MEN Hx', 'MTC Hx', 'FLD Hx',
#          'Pregnancy Hx', 'Sex', 'Ethnicity']].copy()
#     medical_features = medical_features.dropna(axis=0, how='any')

#     # Import a Patient info Data Frame, include Patient ID, Date, Treatment/Dosage
#     Treatment = treatments[['PatID', 'Order DTTM', 'Frequency', 'Prescription', 'Quantity']].copy()

#     # Rename Columns within matrix to Standard, drop any rows w an unfilled value
#     Treatment.rename(columns={'Prescription': 'Treatment and Dosage'}, inplace=True)
#     Treatment.rename(columns={'Order DTTM': 'Date Treat'}, inplace=True)

#     # Create Seperate Treatment and Dosage Columns
#     Treatment['Treatments'] = (Treatment['Treatment and Dosage'].str.split(r'(^[^\d]+)', expand=True)[1])
#     Treatment['Dosages'] = (Treatment['Treatment and Dosage'].str.split(r'(^[^\d]+)', expand=True)[2])

#     # Delete the Treatment and Dosage Column now that we have seperated it
#     Treatment = Treatment.drop('Treatment and Dosage', axis='columns')

#     # Standardize the Treatment Texts
#     Treatment['Treatments'] = Treatment['Treatments'].str.lower()
#     Treatment['Treatments'] = Treatment['Treatments'].str.replace('-', ' ', regex=True)
#     Treatment['Treatments'] = Treatment['Treatments'].str.replace(' ', '', regex=True)

#     # Standardize the Dosage Texts
#     Treatment['Dosages'] = Treatment['Dosages'].str.lower()
#     Treatment['Dosages'] = Treatment['Dosages'].str.replace('-', ' ', regex=True)
#     Treatment['Dosages'] = Treatment['Dosages'].str.replace(' ', '', regex=True)

#     # Import a Patient info Data Frame, include patient ID, Date, Age, BMI
#     Encounter = encounters[['PatID', 'Contact Date', 'BMI', 'Wt (kg)', 'BSA', 'Height (cm)', 'BP', 'Age at Encounter']].copy()

#     # Rename Columns within matrix to Standard, drop any rows w an unfilled value
#     Encounter.rename(columns={'Contact Date': 'Date Encount'}, inplace=True)
#     Encounter.rename(columns={'Wt (kg)': 'Weight'}, inplace=True)
#     Encounter.rename(columns={'Height (cm)': 'Height'}, inplace=True)
#     Encounter.rename(columns={'Age at Encounter': 'Age'}, inplace=True)
#     Encounter = Encounter.dropna(axis=0, how='any')

#     # Import a Patient info Data Frame, include Patient ID Component Result Numeric Value and Result Date
#     Result = results[['PatID', 'Component','Result Numeric Value', 'Result DTTM']]

#     # Rename Columns within matrix to Standard, drop any rows w an unfilled value
#     Result.rename(columns={'Result DTTM': 'Result Date'}, inplace=True)
#     Result = Result.dropna(axis=0, how='any')

#     return Encounter, Treatment, medical_features, Result

# def create_similar_data(patient_id, Treatment, Result, Encounter, medical_features):

#     Result = Result[Result['Component'] == 'HEMOGLOBIN A1C']

#     #Start with the Result Data, and filter by our patient
#     patient_result_data = Result[Result['PatID'] == patient_id]
#     patient_result_data = patient_result_data.sort_values(by='Result Date', ascending=False)
#     patient_result_data = patient_result_data.head(1)
#     patient_result_data = patient_result_data.reset_index()
#     patient_result_data = patient_result_data.drop(columns=['PatID'])

#     # Now sort teatment data
#     patient_treatment_data = Treatment[Treatment['PatID'] == patient_id]
#     patient_treatment_data = patient_treatment_data.sort_values(by='Date Treat', ascending=False)
#     patient_treatment_data = patient_treatment_data.head(1)
#     patient_treatment_data = patient_treatment_data.reset_index()
#     patient_treatment_data = patient_treatment_data.drop(columns=['PatID'])

#     # Now sort encounter data
#     patient_Encounter_data = Encounter[Encounter['PatID'] == patient_id]
#     patient_Encounter_data = patient_Encounter_data.sort_values(by='Date Encount', ascending=False)
#     patient_Encounter_data = patient_Encounter_data.head(1)
#     patient_Encounter_data = patient_Encounter_data.reset_index()
#     patient_Encounter_data = patient_Encounter_data.drop(columns=['PatID'])

#     # Now sort medical data
#     patient_medical_data = medical_features[medical_features['PatID'] == patient_id]
#     patient_medical_data = patient_medical_data.reset_index()
#     patient_medical_data = patient_medical_data.drop(columns=['PatID'])

#     # Formulate patient data entry
#     patient_data = pd.concat([patient_result_data, patient_Encounter_data, patient_treatment_data, patient_medical_data], axis = 1)
#     patient_data = patient_data.drop(columns=['index'])
#     patient_data = patient_data.drop(columns=['Result Date'])
#     patient_data = patient_data.drop(columns=['Date Treat'])
#     patient_data = patient_data.drop(columns=['Date Encount'])
#     patient_data = patient_data.drop(columns=['Component'])
#     patient_data = patient_data.drop(columns=['Frequency'])
#     patient_data = patient_data.drop(columns=['Quantity'])
#     #patient_data = patient_data.drop(columns=['Treatments'])
#     patient_data = patient_data.drop(columns=['Dosages'])

#     # Sort each DataFrame by date
#     Treatment.sort_values(by='Date Treat', ascending=False, inplace=True)
#     Result.sort_values(by='Result Date', ascending=False, inplace=True)
#     Encounter.sort_values(by='Date Encount', ascending=False, inplace=True)

#     # Filter the other DataFrames to include only patient IDs existing in Medical_Features
#     Treatment_filtered = Treatment[Treatment['PatID'].isin(medical_features['PatID'])]
#     Result_filtered = Result[Result['PatID'].isin(medical_features['PatID'])]
#     Encounter_filtered = Encounter[Encounter['PatID'].isin(medical_features['PatID'])]

#     # Merge filtered DataFrames on patient ID using inner merge
#     merged_df = pd.merge(Treatment_filtered, Result_filtered, on='PatID', how='inner')
#     merged_df = pd.merge(merged_df, Encounter_filtered, on='PatID', how='inner')

#     # Merge Medical_Features DataFrame into the merged DataFrame
#     merged_df = pd.merge(merged_df, medical_features, on='PatID', how='inner')

#     # Group by patient ID and select the last entry for each patient
#     recent_entries = merged_df.groupby('PatID').last().reset_index()

#     # Update BP to be ints
#     recent_entries[['Systolic_BP', 'Diastolic_BP']] = recent_entries['BP'].str.split('/', expand=True).astype(int)
#     recent_entries.drop(columns=['BP'], inplace=True)

#     patient_data[['Systolic_BP', 'Diastolic_BP']] = patient_data['BP'].str.split('/', expand=True).astype(int)
#     patient_data.drop(columns=['BP'], inplace=True)

#     # Get the column names of the first DataFrame
#     columns_to_keep = patient_data.columns

#     # Extract the 'Treatments' column
#     treatments = recent_entries['Treatments']

#     # Select only the columns present in the first DataFrame from the second DataFrame
#     recent_entries = recent_entries[columns_to_keep]

#     # Ensure the columns appear in the same order
#     recent_entries = recent_entries.reindex(columns=patient_data.columns)

#     # Assuming 'data' is your DataFrame
#     recent_entries.replace({'Y': 1, 'N': 0, 'Male': 1, 'Female': 0, 'Non-Hispanic or Non-Latino': 0, 'Hispanic or Latino': 1, 'Refused/Declined': 0}, inplace=True)
#     patient_data.replace({'Y': 1, 'N': 0, 'Male': 1, 'Female': 0, 'Non-Hispanic or Non-Latino': 0, 'Hispanic or Latino': 1, 'Refused/Declined': 0}, inplace=True)

#     # Get the top ten most popular treatments
#     top_treatments = recent_entries['Treatments'].value_counts().nlargest(10).index

#     # Filter the DataFrame to include only the data corresponding to the top ten treatments
#     data_top_treatments = recent_entries[recent_entries['Treatments'].isin(top_treatments)]

#     # Create an instance of StandardScaler for feature scaling
#     scaler = StandardScaler()

#     # Group by the 'Treatments' column for the top ten treatments
#     grouped_by_treatments = data_top_treatments.groupby('Treatments')

#     k = 25

#     treatment_dataframes = {}

#     # Iterate over each treatment group
#     for treatment, group in grouped_by_treatments:
#         # Store the column 'results' separately for later use
#         results_column = group['Result Numeric Value']

#         # Separate features
#         features = group.drop(columns=['Treatments', 'Result Numeric Value'])

#         # Perform feature scaling
#         scaled_features = scaler.fit_transform(features)

#         # Perform patient single entry scaling for features
#         patient_data_scaled = scaler.transform(patient_data.drop(columns=['Treatments','Result Numeric Value']))

#         # kNN Model Setup
#         knn_model = NearestNeighbors(n_neighbors=k)
#         knn_model.fit(scaled_features)

#         # Find the k-nearest neighbors for the patient data
#         distances, indices = knn_model.kneighbors(patient_data_scaled)

#         # Extract the indices of the similar patient entries
#         similar_patients_indices = indices.flatten()

#         # Filter the merged data by the indices of similar patients
#         similar_patients_results = results_column.iloc[similar_patients_indices]

#         # Create a DataFrame with the unscaled results
#         similar_patients_results_df = pd.DataFrame(similar_patients_results, columns=['Result Numeric Value'])

#         # Store the DataFrame with unscaled 'results' for the current treatment in the dictionary
#         treatment_dataframes[treatment] = similar_patients_results_df

#         # Print the DataFrame
#         print(f"DataFrame for Treatment: {treatment}")
#         print(similar_patients_results_df)
#         print("\n")

#     # Return the dictionary containing DataFrames for each treatment
#     return treatment_dataframes

# #Create outputs for the front end, Patient genreic info
# def generate_patient_info(patient_id, Treatment, Result, Encounter, medical_features):

#     # Starting with the encounter information, save the row the patients information is stored in
#     patient_info = Encounter[Encounter['PatID'] == patient_id]

#     # Sort the patients encounter information by date
#     patient_info = patient_info.sort_values(by='Date Encount', ascending=False)

#     # Take the most recent encounter informaton available
#     patient_info = patient_info.head(1)

#     # Fitler the Treatment information by our patient ID
#     patient_info_p2 = Treatment[Treatment['PatID'] == patient_id]

#     # Clean up the unneeded columns (treatment info)
#     columns_to_keep = ['PatID', 'Treatments']
#     patient_info_p2 = patient_info_p2[columns_to_keep]

#     # Filter the Result information by the patient ID
#     patient_info_p3 = Result[Result['PatID'] == patient_id]
#     patient_info_p3 = patient_info_p3[patient_info_p3['Component'] == 'HEMOGLOBIN A1C']

#     # Filter the medical features by patient ID
#     patient_info_p4 = medical_features[medical_features['PatID'] == patient_id]

#     # Clean up the unneeded columns (demographical info)
#     columns_to_keep_1 = ['PatID', 'Sex', 'Ethnicity']
#     patient_info_p4 = patient_info_p4[columns_to_keep_1]

#     # Reset indexs prior to compliling data
#     patient_info = patient_info.reset_index(drop=True)
#     patient_info_p2 = patient_info_p2.reset_index(drop=True)
#     patient_info_p3 = patient_info_p3.reset_index(drop=True)
#     patient_info_p4 = patient_info_p4.reset_index(drop=True)

#     # Compile filtered dataframes into a single entry
#     patient_info['Sex'] = patient_info_p4['Sex']
#     patient_info['Ethnicity'] = patient_info_p4['Ethnicity']
#     patient_info['Result Numeric Value'] = patient_info_p3['Result Numeric Value']
#     patient_info['Treatments'] = patient_info_p2['Treatments']
#     patient_info['Date Result'] = patient_info_p3['Result Date']

#     #Rename for standardized
#     patient_info.rename(columns={'Result Numeric Value': 'Recent Result'}, inplace=True)
#     patient_info.rename(columns={'Treatments': 'Recent Treatment'}, inplace=True)
#     patient_info.rename(columns={'Date Encount': 'Date Encounter'}, inplace=True)


#     return patient_info

# #Create outpus for the front end, Patient treatment history
# def generate_patient_history(Result, Treatment, patient_id):

#     #Start with the patients Results, filter by our patient
#     patient_results = Result[Result['PatID'] == patient_id]

#     #Filter by HbA1c Results
#     patient_results = patient_results[patient_results['Component'] == 'HEMOGLOBIN A1C']

#     #Finalize results by sorting by date and renaming to proper names
#     patient_results = patient_results.sort_values(by='Result Date', ascending=False)
#     patient_results.rename(columns={'Result Date': 'Date Result'}, inplace=True)
#     patient_results = patient_results.drop('Component', axis='columns')
#     patient_results = patient_results.drop('PatID', axis='columns')

#     #Then use patient treatments, filter by our patient
#     patient_treatments = Treatment[Treatment['PatID'] == patient_id]

#     # Update the Date of the Treatment/Dosage to a Datetime object
#     patient_treatments['Date Treat'] = pd.to_datetime(patient_treatments['Date Treat'])

#     # Delete the Treatment and Dosage Column now that we have seperated it
#     patient_treatments = patient_treatments.drop('Frequency', axis='columns')
#     patient_treatments = patient_treatments.drop('Quantity', axis='columns')

#     # Prepare for treatment row combinations, rename for standard names
#     patient_treatments = patient_treatments.sort_values(by='Date Treat')
#     patient_treatments.rename(columns={'Treatments': 'Treatment 1'}, inplace=True)
#     patient_treatments['Treatment 2'] = None
#     patient_treatments['Treatment 3'] = None

#     # Drop the time part from Treatment_Date
#     patient_treatments['Date Treat'] = patient_treatments['Date Treat'].dt.date

#     # Create a unique count for each Treatment_Date within each Patient_ID
#     patient_treatments['Treatment_Count'] = patient_treatments.groupby(['PatID', 'Date Treat']).cumcount() + 1

#     # Pivot the DataFrame to reshape it
#     df_pivot = patient_treatments.pivot(index=['PatID', 'Date Treat'], columns='Treatment_Count', values=['Treatment 1', 'Treatment 2', 'Treatment 3'])

#     # Flatten the MultiIndex columns
#     df_pivot.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in df_pivot.columns]

#     # Reset the index to make Patient_ID and Treatment_Date regular columns
#     df_pivot = df_pivot.reset_index()

#     # Clean up one more time after combinations
#     columns_to_drop_3 = [ 'Treatment 2_2', 'Treatment 3_1', 'Treatment 3_2']
#     df_pivot = df_pivot.drop(columns=columns_to_drop_3)

#     # Rename new for standard names
#     df_pivot.rename(columns={'Treatment 1_1': 'Treatment 1'}, inplace=True)
#     df_pivot.rename(columns={'Treatment 1_2': 'Treatment 2'}, inplace=True)
#     df_pivot.rename(columns={'Treatment 2_1': 'Treatment 3'}, inplace=True)
#     df_pivot = df_pivot.fillna(np.nan)

#     patient_treatments = df_pivot
#     patient_treatments.rename(columns={'Date Treat': 'Date Treatment'}, inplace=True)

#     return patient_treatments, patient_results



#**********************************************************************************************************

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

#Set the IDE's Display to something really big so I can see Large Dataframes as outputs
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 3000)

# Record the start time
start_time = time.time()

#Scaler
scaler = StandardScaler()

# Function to load CSV file from Box.com based on file ID
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


#Define File IDs
MULTI_ID = '1465824013335'
OTHER0_ID = '1465824153046'
OTHER1_ID = '1465819719909'
MET1_ID = '1465825873734'
MET0_ID = '1465825057493'
MET2_ID = '1468178601581'
final_merged_id = '1468891677131'


#Load in files using IDs

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

patient_id=56212830

#Create outputs for the Front end, Similiar Patient Results (Panel B)
def create_similar_data(patient_id, regimen_dict, final_merged):

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

    treatment_dataframes = {}

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

        # Store the DataFrame with unscaled 'results' for the current treatment in the dictionary
        treatment_dataframes[key] = similar_patients_results_df


    return treatment_dataframes

#Create outputs for frontend, patient information (Panel ?)
def generate_patient_infos(patient_id, final_merged):

    # Starting with the encounter information, save the row the patients information is stored in
    patient_info = final_merged[final_merged['PatID'] == patient_id]

    # Sort the patients encounter information by date
    patient_info = patient_info.sort_values(by='Result Date', ascending=False)

    # Take the most recent encounter informaton available
    patient_info = patient_info.head(1)

    return patient_info


def create_patient_hisotory2(final_merged, patient_id):

    # Start with the patients Results, filter by our patient
    patient_results = final_merged[final_merged['PatID'] == patient_id]

    #Update
    patient_results = patient_results[['Regimen', 'Result Numeric Value','Result Date','Regimen Date']]

    print(patient_results)

    return patient_results


#Use the functions
#treatment_dataframes = create_similiar_data(patient_id, regimens_dict,final_merged)

#patient_info = generate_patient_infos(patient_id, final_merged)

patient_hist = create_patient_hisotory2(final_merged, patient_id)


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time:", elapsed_time, "seconds")
