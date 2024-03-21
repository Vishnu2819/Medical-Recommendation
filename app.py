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

# Define File IDs
MULTI_ID = '1465824013335'
OTHER0_ID = '1465824153046'
OTHER1_ID = '1465819719909'
MET1_ID = '1465825873734'
MET0_ID = '1465825057493'
MET2_ID = '1468178601581'
final_merged_id = '1468891677131'

# Load in files using IDs
MULTI = functions.load_csv_from_box(MULTI_ID)
OTHER0 = functions.load_csv_from_box(OTHER0_ID)
OTHER1 = functions.load_csv_from_box(OTHER1_ID)
MET1 = functions.load_csv_from_box(MET1_ID)
MET0 = functions.load_csv_from_box(MET0_ID)
MET2 = functions.load_csv_from_box(MET2_ID)
final_merged = functions.load_csv_from_box(final_merged_id)

# Define a dictionary to store the dataframes
regimens_dict = {
    'OTHER0': OTHER0,
    'OTHER1': OTHER1,
    'MET1': MET1,
    'MET0': MET0,
    'MET2': MET2,
}

histogram_divs = []
treatment_dataframes = {}  # Define treatment_dataframes here

# Read the CSV files
df_recommendation = pd.read_csv("OtherParts/Patient Treatment Recomendation Part A.csv")
# df_patient_info = pd.read_csv("OtherParts/Patient Generic Information Part C.csv")
# df_patient_history = pd.read_csv("OtherParts/Patient Treatment History Part D.csv")
# df_patient_history['Date Treatment'] = pd.to_datetime(df_patient_history['Date Treatment'])
# df_sec_pred_res = pd.read_csv("Part B/sec_pred_res/Secondary predicted results.csv")

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
                        id="histo_fig"  # This div will contain the histogram graphs
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
                                config={'scrollZoom': True},
                                style={'display': 'none'}  # Initially set display to 'none'
                            )
                        ] 
                    )
                ])
        ]
    )   
])

# Callback to update the output based on user input
@app.callback(
    [
        Output('output-recommendation', 'children'),
        Output('output-recommendation', 'style'),
        Output('output-patient-details', 'children'),
        Output('output-patient-details', 'style'),
        Output('graph', 'figure'),
        Output('graph', 'style'),
        Output('line-graph', 'figure'),
        Output('line-graph', 'style'),
        Output('histo_fig', 'children')
    ],
    [
        Input('go-button', 'n_clicks'),
        Input('prev-button', 'n_clicks'),
        Input('next-button', 'n_clicks')
    ],
    [dash.dependencies.State('input-patient-id', 'value')],
    prevent_initial_call=False
)
def update_output(n_clicks_go, n_clicks_prev, n_clicks_next, patient_id):
    global treatment_dataframes  # Access the global treatment_dataframes

    ctx_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if ctx_id == "go-button":
        n_clicks = n_clicks_go
    elif ctx_id == "prev-button":
        n_clicks = n_clicks_prev
    elif ctx_id == "next-button":
        n_clicks = n_clicks_next
    else:
        n_clicks = 0

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
                {'display': 'none'},
                []
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
                {'display': 'none'},
                []
            )

        final_merged_df = final_merged[final_merged['PatID'] == patient_id]
        patient_data_recommendation = df_recommendation[df_recommendation['PatID'] == patient_id]
        patient_data_recommendation_patient = patient_data_recommendation[patient_data_recommendation['PatID'] == patient_id]
        patient_data_info = functions.generate_patient_infos(patient_id, final_merged)

        # Check if any of the DataFrames are empty
        if final_merged_df.empty or patient_data_info.empty or patient_data_recommendation.empty:
            return (
                f"No data found for PatID: {patient_id}",
                {'padding': '20px'},
                [],
                {'padding': '0px'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                []
            )
        
        histograms = []

        # Iterate over each treatment DataFrame
        for treatment, df in regimens_dict.items():
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

        total_slides = len(histogram_divs)

        slide_index = 0
        if ctx_id in ["prev-button", "next-button"]:
            if ctx_id == "prev-button":
                slide_index = (n_clicks_prev or 0) % total_slides
            elif ctx_id == "next-button":
                slide_index = (n_clicks_next or 0) % total_slides

        treatment_dataframes = functions.create_similar_data(patient_id, regimens_dict, final_merged)

        current_treatment = patient_data_recommendation_patient['Current Treatment'].values[0]
        recommended_treatment = patient_data_recommendation_patient['Reccomended Treatment'].values[0]
        predicted_result = patient_data_recommendation_patient['Predicted Result'].values[0]

        result_text = f"Recommendation: Switch from {current_treatment} to {recommended_treatment}\n"
        result_text += f"Predicted HbA1c (%): {predicted_result}"

        patient_details = pd.melt(patient_data_info, id_vars=['PatID'],
                                  value_vars=['PatID', 'Sex', 'Ethnicity', 'Age', 'BMI', 'Weight', 'Height',
                                               'Regimen', 'Result Numeric Value',
                                              'Date Encount', 'Result Date', 'Regimen Date'],
                                  var_name='Attribute', value_name='Value')

        table = html.Table(
            [html.Tr([html.Th(col) for col in ['Attribute', 'Value']])] +
            [html.Tr([html.Td(str(patient_details.iloc[i][col])) for col in ['Attribute', 'Value']]) for i in
             range(len(patient_details))]
        )

        fig = px.scatter(final_merged_df, x='Regimen Date', y='Regimen', color='Regimen',
                         title='Treatment History', labels={'Regimen Date': 'Timestamp', 'Result Numeric Value': 'Values'},
                         template='plotly_white', range_x=[final_merged_df['Regimen Date'].min(), final_merged_df['Regimen Date'].max()])

        fig2 = px.line(final_merged_df, x='Result Date', y='Result Numeric Value', color='Regimen' , title='Treatment History',
                       labels={'Result Date': 'Date', 'Result Numeric Value': 'Values'}, 
                       template='plotly_white', range_x=[final_merged_df['Result Date'].min(), final_merged_df['Result Date'].max()])

        return (
            result_text, {'padding':'20px'}, table, {'padding':'20px'},
            fig, {'display': 'block'}, fig2, {'display': 'block'}, 
            histogram_divs[slide_index]
        )

    return '', {'padding':'0px'}, [], {'padding':'0px'}, {'data': [], 'layout': {}}, {'display': 'none'}, {'data': [], 'layout': {}}, {'display': 'none'}, []  # Initial or no-click state



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)



