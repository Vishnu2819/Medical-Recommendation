import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Read the CSV files
df_Dapagliflozin = pd.read_csv("Part B/Dapagliflozin.csv")
df_Empagliflozin = pd.read_csv("Part B/Empagliflozin.csv")
df_Glimepiride = pd.read_csv("Part B/Glimepiride.csv")
df_Glipizide_XR = pd.read_csv("Part B/Glipizide XR.csv")
df_Glipizide = pd.read_csv("Part B/Glipizide.csv")
df_Metformin = pd.read_csv("Part B/Metformin (glucophage).csv")
df_Pioglitazone = pd.read_csv("Part B/Pioglitazone.csv")
df_Sitagliptin = pd.read_csv("Part B/Sitagliptin.csv")
df_MetforminXR = pd.read_csv("Part B/Metformin (glucophage XR).csv")
df_recommendation = pd.read_csv("OtherParts/Patient Treatment Recomendation Part A.csv")
df_patient_info = pd.read_csv("OtherParts/Patient Generic Information Part C.csv")
df_patient_history = pd.read_csv("OtherParts/Patient Treatment History Part D.csv")
df_patient_history['Date Treatment'] = pd.to_datetime(df_patient_history['Date Treatment'])

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the web app
app.layout = html.Div(
    children=[
    html.Script(src="/assets/your_script.js"),
    html.H1("Patient Treatment Recommendation"),
    
    # Input field for PatID
    dcc.Input(id='input-patient-id', type='text', placeholder='Enter PatID'),
    
    # "Go" button to trigger the search
    html.Button('Go', id='go-button', n_clicks=0),

    #Part B
    html.Div(id='output-recommendation' ,style={'padding': '0px'}),
    html.Div(
        className="layout",
        children=[
            html.Div(
                className="slideshow-container",
                children=[
                    html.Div(
                        className='slides',  # Set the class name for the container
                        style={'width': '48%', 'float': 'left'},
                        children=[
                            html.Div(
                                className='slide',  # Set the class name for the first graph
                                children=[
                                    dcc.Graph(
                                        id='graph1',
                                        figure=px.histogram(df_Dapagliflozin, x="Result Numeric Value", nbins=50, title="Dapagliflozin")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph2',
                                        figure=px.histogram(df_Empagliflozin, x="Result Numeric Value", nbins=50, title="Empagliflozin")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph3',
                                        figure=px.histogram(df_Glimepiride, x="Result Numeric Value", nbins=50, title="Glimepiride")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph4',
                                        figure=px.histogram(df_Glipizide_XR, x="Result Numeric Value", nbins=50, title="Glipizide XR")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph5',
                                        figure=px.histogram(df_Glipizide, x="Result Numeric Value", nbins=50, title="Glipizide")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph6',
                                        figure=px.histogram(df_Metformin, x="Result Numeric Value", nbins=50, title="Metformin (glucophage)")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph7',
                                        figure=px.histogram(df_Pioglitazone, x="Result Numeric Value", nbins=50, title="Pioglitazone")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph8',
                                        figure=px.histogram(df_Sitagliptin, x="Result Numeric Value", nbins=50, title="Sitagliptin")
                                    )
                                ]
                            ),
                            html.Div(
                                className='slide',  # Set the class name for the second graph
                                children=[
                                    dcc.Graph(
                                        id='graph9',
                                        figure=px.histogram(df_MetforminXR, x="Result Numeric Value", nbins=50, title="Metformin (glucophage XR)")
                                    )
                                ]
                            )
                        ]
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
                html.Div(id='output-patient-details' ,style={'padding': '0px'}),
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
        # Output("slides", "children"),
        # Output('graph', 'figure'),
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
        patient_data_info = df_patient_info[df_patient_info['PatID'] == patient_id]
        df_patient_history_info = df_patient_history[df_patient_history['PatID']==patient_id]

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
        
        return result_text, {'padding':'20px'}, table, {'padding':'20px'}, fig, {'display': 'block'}, fig2, {'display': 'block'}

    return '', {'padding':'0px'}, [], {'padding':'0px'}, {'data': [], 'layout': {}}, {'display': 'none'}, {'data': [], 'layout': {}}, {'display': 'none'}  # Initial or no-click state


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
