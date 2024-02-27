import dash
from dash import html, dcc, ctx
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import os
import requests
import io
import numpy as np

url = "https://api.box.com/2.0/files/1450972909692/content"
url_2 = "https://api.box.com/2.0/files/1450965225437/content"
url_3 = "https://api.box.com/2.0/files/1450975521323/content"
response = requests.get(url, timeout=60)  # Timeout set to 30 seconds
response_2 = requests.get(url_2, timeout=60)  # Timeout set to 30 seconds
response_3 = requests.get(url_3, timeout=60)  # Timeout set to 30 seconds


#Set the IDE's Display to something really big so I can see Large Dataframes as outputs
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 3000)

# Box.com developer token
DEVELOPER_TOKEN = 'jFwml2gyQyD0Xbtcd6GYwyiDFafz3zBV'


# Function to load CSV file from Box.com based on file ID
def load_csv_from_box(file_id):
    headers = {
        'Authorization': f'Bearer {DEVELOPER_TOKEN}'
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

#Function to generate Dataframes from Box
def generate_clean_data():

    feature_id = '1450001295856'
    encounter_id = '1450965225437'
    result1_id = '1450973176414'
    result2_id = '1450972909692'
    treatment_id = '1450975521323'

    encounters = load_csv_from_box(encounter_id)
    features = load_csv_from_box(feature_id)
    result1 = load_csv_from_box(result1_id)
    result2 = load_csv_from_box(result2_id)
    treatments = load_csv_from_box(treatment_id)

    results = pd.concat([result1, result2])

    # Import a Patient info Data Frame, include Patient ID, and Previous Medical Illnesses
    medical_features = features[
        ['PatID', 'ESRD Hx', 'IDDM Hx', 'GDM Hx', 'Other DM Hx', 'CFDM Hx', 'Pancreatectomy Hx', 'Insulin Pump Hx', 'CVD Hx', 'LVH Hx', 'Retinopathy Hx', 'HTN Hx', 'Tobacco Hx', 'Dyslipidemia Hx',
         'CRF Hx', 'CHF Hx', 'Obesity Hx', 'Albuminuria Hx', 'Amputation Hx', 'GI Hx', 'Pancreatitis Hx', 'Hypoglycemia Hx', 'Hypotension Hx', 'Fracture Hx', 'UTI Hx', 'MEN Hx', 'MTC Hx', 'FLD Hx',
         'Pregnancy Hx', 'Sex', 'Ethnicity']].copy()
    medical_features = medical_features.dropna(axis=0, how='any')

    # Import a Patient info Data Frame, include Patient ID, Date, Treatment/Dosage
    Treatment = treatments[['PatID', 'Order DTTM', 'Frequency', 'Prescription', 'Quantity']].copy()

    # Rename Columns within matrix to Standard, drop any rows w an unfilled value
    Treatment.rename(columns={'Prescription': 'Treatment and Dosage'}, inplace=True)
    Treatment.rename(columns={'Order DTTM': 'Date Treat'}, inplace=True)

    # Create Seperate Treatment and Dosage Columns
    Treatment['Treatments'] = (Treatment['Treatment and Dosage'].str.split(r'(^[^\d]+)', expand=True)[1])
    Treatment['Dosages'] = (Treatment['Treatment and Dosage'].str.split(r'(^[^\d]+)', expand=True)[2])

    # Delete the Treatment and Dosage Column now that we have seperated it
    Treatment = Treatment.drop('Treatment and Dosage', axis='columns')

    # Standardize the Treatment Texts
    Treatment['Treatments'] = Treatment['Treatments'].str.lower()
    Treatment['Treatments'] = Treatment['Treatments'].str.replace('-', ' ', regex=True)
    Treatment['Treatments'] = Treatment['Treatments'].str.replace(' ', '', regex=True)

    # Standardize the Dosage Texts
    Treatment['Dosages'] = Treatment['Dosages'].str.lower()
    Treatment['Dosages'] = Treatment['Dosages'].str.replace('-', ' ', regex=True)
    Treatment['Dosages'] = Treatment['Dosages'].str.replace(' ', '', regex=True)

    # Import a Patient info Data Frame, include patient ID, Date, Age, BMI
    Encounter = encounters[['PatID', 'Contact Date', 'BMI', 'Wt (kg)', 'BSA', 'Height (cm)', 'BP', 'Age at Encounter']].copy()

    # Rename Columns within matrix to Standard, drop any rows w an unfilled value
    Encounter.rename(columns={'Contact Date': 'Date Encount'}, inplace=True)
    Encounter.rename(columns={'Wt (kg)': 'Weight'}, inplace=True)
    Encounter.rename(columns={'Height (cm)': 'Height'}, inplace=True)
    Encounter.rename(columns={'Age at Encounter': 'Age'}, inplace=True)
    Encounter = Encounter.dropna(axis=0, how='any')

    # Import a Patient info Data Frame, include Patient ID Component Result Numeric Value and Result Date
    Result = results[['PatID', 'Component','Result Numeric Value', 'Result DTTM']]

    # Rename Columns within matrix to Standard, drop any rows w an unfilled value
    Result.rename(columns={'Result DTTM': 'Result Date'}, inplace=True)
    Result = Result.dropna(axis=0, how='any')

    return Encounter, Treatment, medical_features, Result

#Create outputs for the front end, Patient genreic info
def generate_patient_info(patient_id, Treatment, Result, Encounter, medical_features):

    # Starting with the encounter information, save the row the patients information is stored in
    patient_info = Encounter[Encounter['PatID'] == patient_id]

    # Sort the patients encounter information by date
    patient_info = patient_info.sort_values(by='Date Encount', ascending=False)

    # Take the most recent encounter informaton available
    patient_info = patient_info.head(1)

    # Fitler the Treatment information by our patient ID
    patient_info_p2 = Treatment[Treatment['PatID'] == patient_id]

    # Clean up the unneeded columns (treatment info)
    columns_to_keep = ['PatID', 'Treatments']
    patient_info_p2 = patient_info_p2[columns_to_keep]

    # Filter the Result information by the patient ID
    patient_info_p3 = Result[Result['PatID'] == patient_id]
    patient_info_p3 = patient_info_p3[patient_info_p3['Component'] == 'HEMOGLOBIN A1C']

    # Filter the medical features by patient ID
    patient_info_p4 = medical_features[medical_features['PatID'] == patient_id]

    # Clean up the unneeded columns (demographical info)
    columns_to_keep_1 = ['PatID', 'Sex', 'Ethnicity']
    patient_info_p4 = patient_info_p4[columns_to_keep_1]

    # Reset indexs prior to compliling data
    patient_info = patient_info.reset_index(drop=True)
    patient_info_p2 = patient_info_p2.reset_index(drop=True)
    patient_info_p3 = patient_info_p3.reset_index(drop=True)
    patient_info_p4 = patient_info_p4.reset_index(drop=True)

    # Compile filtered dataframes into a single entry
    patient_info['Sex'] = patient_info_p4['Sex']
    patient_info['Ethnicity'] = patient_info_p4['Ethnicity']
    patient_info['Result Numeric Value'] = patient_info_p3['Result Numeric Value']
    patient_info['Treatments'] = patient_info_p2['Treatments']
    patient_info['Date Result'] = patient_info_p3['Result Date']

    #Rename for standardized
    patient_info.rename(columns={'Result Numeric Value': 'Recent Result'}, inplace=True)
    patient_info.rename(columns={'Treatments': 'Recent Treatment'}, inplace=True)
    patient_info.rename(columns={'Date Encount': 'Date Encounter'}, inplace=True)


    return patient_info

#Create outpus for the front end, Patient treatment history
def generate_patient_history(Result, Treatment, patient_id):

    #Start with the patients Results, filter by our patient
    patient_results = Result[Result['PatID'] == patient_id]

    #Filter by HbA1c Results
    patient_results = patient_results[patient_results['Component'] == 'HEMOGLOBIN A1C']

    #Finalize results by sorting by date and renaming to proper names
    patient_results = patient_results.sort_values(by='Result Date', ascending=False)
    patient_results.rename(columns={'Result Date': 'Date Result'}, inplace=True)
    patient_results = patient_results.drop('Component', axis='columns')
    patient_results = patient_results.drop('PatID', axis='columns')

    #Then use patient treatments, filter by our patient
    patient_treatments = Treatment[Treatment['PatID'] == patient_id]

    # Update the Date of the Treatment/Dosage to a Datetime object
    patient_treatments['Date Treat'] = pd.to_datetime(patient_treatments['Date Treat'])

    # Delete the Treatment and Dosage Column now that we have seperated it
    patient_treatments = patient_treatments.drop('Frequency', axis='columns')
    patient_treatments = patient_treatments.drop('Quantity', axis='columns')

    # Prepare for treatment row combinations, rename for standard names
    patient_treatments = patient_treatments.sort_values(by='Date Treat')
    patient_treatments.rename(columns={'Treatments': 'Treatment 1'}, inplace=True)
    patient_treatments['Treatment 2'] = None
    patient_treatments['Treatment 3'] = None

    # Drop the time part from Treatment_Date
    patient_treatments['Date Treat'] = patient_treatments['Date Treat'].dt.date

    # Create a unique count for each Treatment_Date within each Patient_ID
    patient_treatments['Treatment_Count'] = patient_treatments.groupby(['PatID', 'Date Treat']).cumcount() + 1

    # Pivot the DataFrame to reshape it
    df_pivot = patient_treatments.pivot(index=['PatID', 'Date Treat'], columns='Treatment_Count', values=['Treatment 1', 'Treatment 2', 'Treatment 3'])

    # Flatten the MultiIndex columns
    df_pivot.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in df_pivot.columns]

    # Reset the index to make Patient_ID and Treatment_Date regular columns
    df_pivot = df_pivot.reset_index()

    # Clean up one more time after combinations
    columns_to_drop_3 = [ 'Treatment 2_2', 'Treatment 3_1', 'Treatment 3_2']
    df_pivot = df_pivot.drop(columns=columns_to_drop_3)

    # Rename new for standard names
    df_pivot.rename(columns={'Treatment 1_1': 'Treatment 1'}, inplace=True)
    df_pivot.rename(columns={'Treatment 1_2': 'Treatment 2'}, inplace=True)
    df_pivot.rename(columns={'Treatment 2_1': 'Treatment 3'}, inplace=True)
    df_pivot = df_pivot.fillna(np.nan)

    patient_treatments = df_pivot
    patient_treatments.rename(columns={'Date Treat': 'Date Treatment'}, inplace=True)

    return patient_treatments, patient_results

#Create  outputs for the Front end, Similiar Patient Results
def create_similiar_data(patient_id, Treatment, Result, Encounter, medical_features):

    Result = Result[Result['Component'] == 'HEMOGLOBIN A1C']

    #Start with the Result Data, and filter by our patient
    patient_result_data = Result[Result['PatID'] == patient_id]
    patient_result_data = patient_result_data.sort_values(by='Result Date', ascending=False)
    patient_result_data = patient_result_data.head(1)
    patient_result_data = patient_result_data.reset_index()
    patient_result_data = patient_result_data.drop(columns=['PatID'])

    # Now sort teatment data
    patient_treatment_data = Treatment[Treatment['PatID'] == patient_id]
    patient_treatment_data = patient_treatment_data.sort_values(by='Date Treat', ascending=False)
    patient_treatment_data = patient_treatment_data.head(1)
    patient_treatment_data = patient_treatment_data.reset_index()
    patient_treatment_data = patient_treatment_data.drop(columns=['PatID'])

    # Now sort encounter data
    patient_Encounter_data = Encounter[Encounter['PatID'] == patient_id]
    patient_Encounter_data = patient_Encounter_data.sort_values(by='Date Encount', ascending=False)
    patient_Encounter_data = patient_Encounter_data.head(1)
    patient_Encounter_data = patient_Encounter_data.reset_index()
    patient_Encounter_data = patient_Encounter_data.drop(columns=['PatID'])

    # Now sort medical data
    patient_medical_data = medical_features[medical_features['PatID'] == patient_id]
    patient_medical_data = patient_medical_data.reset_index()
    patient_medical_data = patient_medical_data.drop(columns=['PatID'])

    # Formulate patient data entry
    patient_data = pd.concat([patient_result_data, patient_Encounter_data, patient_treatment_data, patient_medical_data], axis = 1)
    patient_data = patient_data.drop(columns=['index'])
    patient_data = patient_data.drop(columns=['Result Date'])
    patient_data = patient_data.drop(columns=['Date Treat'])
    patient_data = patient_data.drop(columns=['Date Encount'])
    patient_data = patient_data.drop(columns=['Component'])
    patient_data = patient_data.drop(columns=['Frequency'])
    patient_data = patient_data.drop(columns=['Quantity'])
    #patient_data = patient_data.drop(columns=['Treatments'])
    patient_data = patient_data.drop(columns=['Dosages'])

    # Sort each DataFrame by date
    Treatment.sort_values(by='Date Treat', ascending=False, inplace=True)
    Result.sort_values(by='Result Date', ascending=False, inplace=True)
    Encounter.sort_values(by='Date Encount', ascending=False, inplace=True)

    # Filter the other DataFrames to include only patient IDs existing in Medical_Features
    Treatment_filtered = Treatment[Treatment['PatID'].isin(medical_features['PatID'])]
    Result_filtered = Result[Result['PatID'].isin(medical_features['PatID'])]
    Encounter_filtered = Encounter[Encounter['PatID'].isin(medical_features['PatID'])]

    # Merge filtered DataFrames on patient ID using inner merge
    merged_df = pd.merge(Treatment_filtered, Result_filtered, on='PatID', how='inner')
    merged_df = pd.merge(merged_df, Encounter_filtered, on='PatID', how='inner')

    # Merge Medical_Features DataFrame into the merged DataFrame
    merged_df = pd.merge(merged_df, medical_features, on='PatID', how='inner')

    # Group by patient ID and select the last entry for each patient
    recent_entries = merged_df.groupby('PatID').last().reset_index()

    # Update BP to be ints
    recent_entries[['Systolic_BP', 'Diastolic_BP']] = recent_entries['BP'].str.split('/', expand=True).astype(int)
    recent_entries.drop(columns=['BP'], inplace=True)

    patient_data[['Systolic_BP', 'Diastolic_BP']] = patient_data['BP'].str.split('/', expand=True).astype(int)
    patient_data.drop(columns=['BP'], inplace=True)

    # Get the column names of the first DataFrame
    columns_to_keep = patient_data.columns

    # Extract the 'Treatments' column
    treatments = recent_entries['Treatments']

    # Select only the columns present in the first DataFrame from the second DataFrame
    recent_entries = recent_entries[columns_to_keep]

    # Ensure the columns appear in the same order
    recent_entries = recent_entries.reindex(columns=patient_data.columns)

    # Assuming 'data' is your DataFrame
    recent_entries.replace({'Y': 1, 'N': 0, 'Male': 1, 'Female': 0, 'Non-Hispanic or Non-Latino': 0, 'Hispanic or Latino': 1, 'Refused/Declined': 0}, inplace=True)
    patient_data.replace({'Y': 1, 'N': 0, 'Male': 1, 'Female': 0, 'Non-Hispanic or Non-Latino': 0, 'Hispanic or Latino': 1, 'Refused/Declined': 0}, inplace=True)

    # Get the top ten most popular treatments
    top_treatments = recent_entries['Treatments'].value_counts().nlargest(10).index

    # Filter the DataFrame to include only the data corresponding to the top ten treatments
    data_top_treatments = recent_entries[recent_entries['Treatments'].isin(top_treatments)]

    # Create an instance of StandardScaler for feature scaling
    scaler = StandardScaler()

    # Group by the 'Treatments' column for the top ten treatments
    grouped_by_treatments = data_top_treatments.groupby('Treatments')

    k = 25

    treatment_dataframes = {}

    # Iterate over each treatment group
    for treatment, group in grouped_by_treatments:
        # Store the column 'results' separately for later use
        results_column = group['Result Numeric Value']

        # Separate features
        features = group.drop(columns=['Treatments', 'Result Numeric Value'])

        # Perform feature scaling
        scaled_features = scaler.fit_transform(features)

        # Perform patient single entry scaling for features
        patient_data_scaled = scaler.transform(patient_data.drop(columns=['Treatments','Result Numeric Value']))

        # kNN Model Setup
        knn_model = NearestNeighbors(n_neighbors=k)
        knn_model.fit(scaled_features)

        # Find the k-nearest neighbors for the patient data
        distances, indices = knn_model.kneighbors(patient_data_scaled)

        # Extract the indices of the similar patient entries
        similar_patients_indices = indices.flatten()

        # Filter the merged data by the indices of similar patients
        similar_patients_results = results_column.iloc[similar_patients_indices]

        # Create a DataFrame with the unscaled results
        similar_patients_results_df = pd.DataFrame(similar_patients_results, columns=['Result Numeric Value'])

        # Store the DataFrame with unscaled 'results' for the current treatment in the dictionary
        treatment_dataframes[treatment] = similar_patients_results_df

        # Print the DataFrame
        print(f"DataFrame for Treatment: {treatment}")
        print(similar_patients_results_df)
        print("\n")

    # Return the dictionary containing DataFrames for each treatment
    return treatment_dataframes

Encounter, Treatment, medical_features, Result = generate_clean_data()



treatment_dataframes = create_similiar_data(patient_id, Treatment, Result, Encounter, medical_features)



# Define the base directory
base_dir = r"C:\Users\swoos\Desktop\Shit"

# Read the Excel files from the new location
df_Dapagliflozin = pd.read_excel(os.path.join(base_dir, "Dapagliflozin.xlsx"))
df_Empagliflozin = pd.read_excel(os.path.join(base_dir, "Empagliflozin.xlsx"))
df_Glimepiride = pd.read_excel(os.path.join(base_dir, "Glimepiride.xlsx"))
df_Glipizide_XR = pd.read_excel(os.path.join(base_dir, "Glipizide XR.xlsx"))
df_Glipizide = pd.read_excel(os.path.join(base_dir, "Glipizide.xlsx"))
df_Metformin = pd.read_excel(os.path.join(base_dir, "Metformin (glucophage).xlsx"))
df_Pioglitazone = pd.read_excel(os.path.join(base_dir, "Pioglitazone.xlsx"))
df_Sitagliptin = pd.read_excel(os.path.join(base_dir, "Sitagliptin.xlsx"))
df_MetforminXR = pd.read_excel(os.path.join(base_dir, "Metformin (glucophage XR).xlsx"))

df_recommendation = pd.read_excel(os.path.join(base_dir, "Patient Treatment Recomendation Part A.xlsx"))
df_patient_info = pd.read_excel(os.path.join(base_dir, "Patient Generic Information Part C.xlsx"))
df_patient_history = pd.read_excel(os.path.join(base_dir, "Patient Treatment History Part D.xlsx"))
df_patient_history['Date Treatment'] = pd.to_datetime(df_patient_history['Date Treatment'])
df_sec_pred_res = pd.read_excel(os.path.join(base_dir, "Secondary predicted results.xlsx"))

print(df_Dapagliflozin)
print(df_Empagliflozin)




histograms = []

# Iterate over each treatment DataFrame
for treatment, df in treatment_dataframes.items():
    # Create histogram and append to the list
    histograms.append((treatment, px.histogram(df, x="Result Numeric Value", nbins=50, title=treatment)))

histogram_divs = [
    html.Div(
        className='slide',  # Set the class name for the div
        children=[
            dcc.Graph(
                id=f'graph{i}',
                figure=histogram
            )
        ]
    ) for i, (_, histogram) in enumerate(histograms, start=0)
]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the web app
app.layout = html.Div(
    children=[
        html.H1("Patient Treatment Recommendation"),

        html.Br(),
        html.Br(),

        # Input field for PatID
        dcc.Input(id='input-patient-id', type='text', placeholder='Enter PatID'),

        # "Go" button to trigger the search
        html.Button('Go', id='go-button', n_clicks=0),

        # Part B
        html.Div(id='output-recommendation', style={'padding': '0px'}),
        html.Div(
            className="layout",
            children=[
                html.Div(
                    id="slideshow-container",
                    className="slideshow-container",
                    children=[
                        html.Div(
                            id="histo_fig"
                        ),
                        html.A(className="prev", children="❮", n_clicks=0, id="prev-button"),
                        html.A(className="next", children="❯", n_clicks=0, id="next-button")
                    ]
                ),
                # Output for displaying recommendation information and patient details
                html.Div(
                    className="container2",
                    style={'width': '48%', 'float': 'left'},
                    children=[
                        html.Div(id='output-patient-details', style={'padding': '0px'}),
                        html.Div(
                            children=[
                                html.Div(
                                    id='point_graph',
                                    children=[
                                        dcc.Graph(
                                            id='graph',
                                            figure={
                                                'data': [],
                                                'layout': {
                                                    'title': 'Treatment History',
                                                    'xaxis': {'title': 'Timestamp', 'showgrid': True},
                                                    'yaxis': {'title': 'Values', 'showgrid': True},
                                                    'legend': {'x': 0, 'y': 1},  # Position the legend at the top-left corner
                                                    'font': {'family': 'Arial', 'size': 12},  # Customize font style
                                                    'margin': {'l': 200, 'r': -20, 't': 20, 'b': 20}  # Adjust the left margin
                                                },
                                            },
                                            config={'scrollZoom': True},
                                            style={'display': 'none'}  # Initially set display to 'none'
                                        )

                                    ])
                            ],
                            style={'marginLeft': 'auto', 'marginRight': 0}
                        ),
                        html.Div(
                            id='line_graph',
                            children=[
                                dcc.Graph(
                                    id='line-graph',
                                    figure={
                                        'data': [],
                                        'layout': {
                                            'title': 'HbA1c Over Time',
                                            'xaxis': {'title': 'Timestamp', 'showgrid': True},
                                            'yaxis': {'title': 'HbA1c Value', 'showgrid': True},
                                            'legend': {'x': 0, 'y': 1},  # Position the legend at the top-left corner
                                            'font': {'family': 'Arial', 'size': 12},  # Customize font style
                                        },
                                    },
                                    style={'display': 'none'}  # Initially set display to 'none'
                                )
                            ]
                        )
                    ])
            ]
        )
    ])


# Find the index of 'Dapagliflozin' in df_sec_pred_res
# dapagliflozin_index = df_sec_pred_res[df_sec_pred_res['Treatment'] == 'Dapagliflozin'].index[0]

# print(dapagliflozin_index)
# print(histogram_divs[dapagliflozin_index].children[0].figure)


# Callback to update the displayed slide when next or previous button is clicked
@app.callback(
    Output('histo_fig', 'children'),
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_slide(n_clicks_prev, n_clicks_next):
    slide_index = 0
    total_slides = len(histogram_divs)
    if "next-button" == ctx.triggered_id:
        slide_index = (n_clicks_next or 0) % total_slides
    elif "prev-button" == ctx.triggered_id:
        slide_index = (n_clicks_prev or 0) % total_slides
    slide_index = slide_index if slide_index >= 0 else total_slides - 1

    return histogram_divs[slide_index]


# Callback to update the output based on user input
@app.callback(
    [
        Output('output-recommendation', 'children'),
        Output('output-recommendation', 'style'),
        Output('output-patient-details', 'children'),
        Output('output-patient-details', 'style'),
        # Output("slides", "children"),
        Output('graph', 'figure'),
        Output('graph', 'style'),
        Output('line-graph', 'figure'),
        Output('line-graph', 'style')
    ],
    [Input('go-button', 'n_clicks')],
    [dash.dependencies.State('input-patient-id', 'value')]
    # [Input("prev-button", "n_clicks")],
    # [Input("next-button", "n_clicks")]
)
def update_output(n_clicks, patient_id):
    if n_clicks > 0:
        if patient_id is None or not patient_id.strip():
            return (
                'Please enter a valid PatID before clicking "Go"',
                {'padding': '20px'},
                [],
                {'padding': '0px'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'}
            )

        try:
            patient_id = int(patient_id)
        except ValueError:
            return (
                'Invalid input: Please enter a valid integer for PatID',
                {'padding': '20px'},
                [],
                {'padding': '0px'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'}
            )

        patient_data_recommendation = df_recommendation[df_recommendation['PatID'] == patient_id]
        patient_data_info = generate_patient_info(patient_id, Treatment, Result, Encounter,medical_features)
        df_patient_history_info = df_patient_history[df_patient_history['PatID'] == patient_id]

        if patient_data_recommendation.empty or patient_data_info.empty or df_patient_history_info.empty:
            return (
                f"No data found for PatID: {patient_id}",
                {'padding': '20px'},
                [],
                {'padding': '0px'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'}
            )

        current_treatment = patient_data_recommendation['Current Treatment'].values[0]
        recommended_treatment = patient_data_recommendation['Reccomended Treatment'].values[0]
        predicted_result = patient_data_recommendation['Predicted Result'].values[0]

        result_text = f"Recommendation: Switch from {current_treatment} to {recommended_treatment}\n"
        result_text += f"Predicted HbA1c (%): {predicted_result}"

        patient_details = pd.melt(patient_data_info, id_vars=['PatID'],
                                  value_vars=['PatID', 'Sex', 'Ethnicity', 'Age', 'BMI', 'Weight', 'Height',
                                               'Recent Treatment', 'Recent Result',
                                              'Date Encounter', 'Date Result'],
                                  var_name='Attribute', value_name='Value')

        table = html.Table(
            [html.Tr([html.Th(col) for col in ['Attribute', 'Value']])] +
            [html.Tr([html.Td(str(patient_details.iloc[i][col])) for col in ['Attribute', 'Value']]) for i in
             range(len(patient_details))]
        )

        df_patient_history_treatment, df_patient_history_result = generate_patient_history(Result, Treatment, patient_id)

        fig = {
            'data': [
                {'x': df_patient_history_treatment['Date Treatment'], 'y': df_patient_history_treatment['Treatment 1'], 'type': 'scatter', 'name': 'Treatment 1', 'mode': 'markers',
                 'marker': {'symbol': 'circle'}},
                {'x': df_patient_history_treatment['Date Treatment'], 'y': df_patient_history_treatment['Treatment 2'], 'type': 'scatter', 'name': 'Treatment 2', 'mode': 'markers',
                 'marker': {'symbol': 'circle'}},
                {'x': df_patient_history_treatment['Date Treatment'], 'y': df_patient_history_treatment['Treatment 3'], 'type': 'scatter', 'name': 'Treatment 3', 'mode': 'markers',
                 'marker': {'symbol': 'circle'}}
            ],
            'layout': {
                'title': 'Treatment History',
                'xaxis': {'title': 'Timestamp', 'showgrid': True},
                'yaxis': {'title': 'Values', 'showgrid': True},
                'legend': {'x': 0, 'y': 1},
                'font': {'family': 'Arial', 'size': 12},
            }
        }

        fig2 = {
            'data': [{'x': df_patient_history_result['Date Result'], 'y': df_patient_history_result['Result Numeric Value'], 'type': 'line', 'name': 'HbA1c'}],
            'layout': {
                'title': 'HbA1c Over Time',
                'xaxis': {'title': 'Timestamp', 'showgrid': True},
                'yaxis': {'title': 'HbA1c Value', 'showgrid': True},
                'legend': {'x': 0, 'y': 1},
                'font': {'family': 'Arial', 'size': 12},
            }
        }

        return result_text, {'padding': '20px'}, table, {'padding': '20px'}, fig, {'display': 'block'}, fig2, {'display': 'block'}

    return '', {'padding': '0px'}, [], {'padding': '0px'}, {'data': [], 'layout': {}}, {'display': 'none'}, {'data': [], 'layout': {}}, {'display': 'none'}  # Initial or no-click state


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)