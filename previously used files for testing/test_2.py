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
import functions

load_dotenv()

response = requests.get(os.getenv('url'), timeout=60)  # Timeout set to 30 seconds
response_2 = requests.get(os.getenv('url_2'), timeout=60)  # Timeout set to 30 seconds
response_3 = requests.get(os.getenv('url_3'), timeout=60)  # Timeout set to 30 seconds

# Set the IDE's Display to something really big so large DataFrames are visible
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 3000)


# Main part of the script
if __name__ == '__main__':
    patient_id = 54758107

    Encounter, Treatment, medical_features, Result = functions.generate_clean_data()

    similar_data = functions.create_similar_data(patient_id, Treatment, Result, Encounter, medical_features)

    # Print or use the resulting dictionary of DataFrames
    for treatment, df in similar_data.items():
        # print(f"DataFrame for Treatment: {treatment}")
        print(df.to_string(index=False))  # Print the entire DataFrame without index
        print("\n")

