import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Read the CSV files and store them in a dictionary
dataframes = {
    'Dapagliflozin': pd.read_csv("Part B/Dapagliflozin.csv"),
    'Empagliflozin': pd.read_csv("Part B/Empagliflozin.csv"),
    'Glimepiride': pd.read_csv("Part B/Glimepiride.csv"),
    'Glipizide_XR': pd.read_csv("Part B/Glipizide XR.csv"),
    'Glipizide': pd.read_csv("Part B/Glipizide.csv"),
    'Metformin': pd.read_csv("Part B/Metformin (glucophage).csv"),
    'Pioglitazone': pd.read_csv("Part B/Pioglitazone.csv"),
    'Sitagliptin': pd.read_csv("Part B/Sitagliptin.csv"),
    'MetforminXR': pd.read_csv("Part B/Metformin (glucophage XR).csv")
}

# Read other CSV files
df_recommendation = pd.read_csv("OtherParts/Patient Treatment Recomendation Part A.csv")
df_patient_info = pd.read_csv("OtherParts/Patient Generic Information Part C.csv")
df_patient_history = pd.read_csv("OtherParts/Patient Treatment History Part D.csv")
df_patient_history['Date Treatment'] = pd.to_datetime(df_patient_history['Date Treatment'])

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the web app
app.layout = html.Div(children=[
    html.H1("Patient Treatment Recommendation"),
    
    # Input field for PatID
    dcc.Input(id='input-patient-id', type='text', placeholder='Enter PatID'),
    
    # "Go" button to trigger the search
    html.Button('Go', id='go-button', n_clicks=0),

    #Part B
    html.H1("Treatments:"),
    html.Div(id='output-recommendation'),
    html.Div(
        className="layout",
        children=[
            html.Div(
                className='container',  # Set the class name for the container
                style={'width': '48%', 'float': 'left'},
                children=[
                    html.Div(
                        className='graphs',  # Set the class name for the graph
                        id=f'graph-{name.lower()}',
                        style={'display': 'none'},  # Initially hide the graph
                    ) for name in dataframes.keys()
                ]
            ),

            # Output for displaying recommendation information and patient details
            html.Div(
                className="container2",
                style={'width': '48%', 'float': 'left'},
                children=[
                    html.Div(id='output-patient-details'),
                    html.Div(children=[
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
                                },
                            },
                            style={'display': 'none'}  # Initially set display to 'none'
                        )
                    ]),
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
        ]
    )
])

# Callback to update the output based on user input
@app.callback(
    [
        Output(f'graph-{name.lower()}', 'children') for name in dataframes.keys()
    ] + [
        Output('output-recommendation', 'children'),
        Output('output-patient-details', 'children'),
        Output('graph', 'figure'),
        Output('graph', 'style'),
        Output('line-graph', 'figure'),
        Output('line-graph', 'style')
    ],
    [Input('go-button', 'n_clicks')],
    [dash.dependencies.State('input-patient-id', 'value')]
)
def update_output(n_clicks, patient_id):
    if n_clicks > 0:
        if patient_id is None or not patient_id.strip():
            return (
                ['Please enter a valid PatID before clicking "Go"'] * len(dataframes),
                [],
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'}
            )

        try:
            patient_id = int(patient_id)
        except ValueError:
            return (
                ['Invalid input: Please enter a valid integer for PatID'] * len(dataframes),
                [],
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'}
            )

        patient_data_recommendation = df_recommendation[df_recommendation['PatID'] == patient_id]
        patient_data_info = df_patient_info[df_patient_info['PatID'] == patient_id]
        df_patient_history_info = df_patient_history[df_patient_history['PatID']==patient_id]

        if patient_data_recommendation.empty or patient_data_info.empty or df_patient_history_info.empty:
            return (
                [f"No data found for PatID: {patient_id}"] * len(dataframes),
                [],
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
                                  value_vars=['PatID','Sex', 'Ethnicity ', 'DOB', 'BMI', 'Weight ', 'Height',
                                              'Systolic BP', 'Diastolic BP', 'Recent Treatment', 'Recent Result',
                                              'Date Encounter ', 'Date Result'],
                                  var_name='Attribute', value_name='Value')

        table = html.Table(
            [html.Tr([html.Th(col) for col in ['Attribute', 'Value']])] +
            [html.Tr([html.Td(str(patient_details.iloc[i][col])) for col in ['Attribute', 'Value']]) for i in
             range(len(patient_details))]
        )

        fig = {
            'data': [
                {'x': df_patient_history_info['Date Treatment'], 'y': df_patient_history_info['Treatment 1'], 'type': 'scatter', 'name': 'Treatment 1', 'mode': 'markers', 'marker': {'symbol': 'circle'}},
                {'x': df_patient_history_info['Date Treatment'], 'y': df_patient_history_info['Treatment 2'], 'type': 'scatter', 'name': 'Treatment 2', 'mode': 'markers', 'marker': {'symbol': 'circle'}},
                {'x': df_patient_history_info['Date Treatment'], 'y': df_patient_history_info['Treatment 3'], 'type': 'scatter', 'name': 'Treatment 3', 'mode': 'markers', 'marker': {'symbol': 'circle'}}
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
            'data': [{'x': df_patient_history_info['Date Result'], 'y': df_patient_history_info['Result Numeric Value'], 'type': 'line', 'name': 'HbA1c'}],
            'layout': {
                'title': 'HbA1c Over Time',
                'xaxis': {'title': 'Timestamp', 'showgrid': True},
                'yaxis': {'title': 'HbA1c Value', 'showgrid': True},
                'legend': {'x': 0, 'y': 1},
                'font': {'family': 'Arial', 'size': 12},
            }
        }
        
        return ([None] * len(dataframes)), result_text, table, fig, {'display': 'block'}, fig2, {'display': 'block'}

    return (
        [''] * len(dataframes),
        [],
        {'data': [], 'layout': {}},
        {'display': 'none'},
        {'data': [], 'layout': {}},
        {'display': 'none'}
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
