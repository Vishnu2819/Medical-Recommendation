import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import plotly.express as px
import functions
import plotly.graph_objs as go

load_dotenv()

response = requests.get(os.getenv('url'), timeout=60)
response_2 = requests.get(os.getenv('url_2'), timeout=60)
response_3 = requests.get(os.getenv('url_3'), timeout=60)

pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 3000)

MULTI_ID = '1465824013335'
OTHER0_ID = '1465824153046'
OTHER1_ID = '1465819719909'
MET1_ID = '1465825873734'
MET0_ID = '1465825057493'
MET2_ID = '1468178601581'
final_merged_id = '1468891677131'

MULTI = functions.load_csv_from_box(MULTI_ID)
OTHER0 = functions.load_csv_from_box(OTHER0_ID)
OTHER1 = functions.load_csv_from_box(OTHER1_ID)
MET1 = functions.load_csv_from_box(MET1_ID)
MET0 = functions.load_csv_from_box(MET0_ID)
MET2 = functions.load_csv_from_box(MET2_ID)
final_merged = functions.load_csv_from_box(final_merged_id)
loaded_models = functions.load_models()

regimens_dict = {
    'OTHER0': OTHER0,
    'OTHER1': OTHER1,
    'MET1': MET1,
    'MET0': MET0,
    'MET2': MET2,
}

regmimen_names = {
    'OTHER0': 'Non-Metformin monotherapy',
    'OTHER1': 'Non-Metformin dual therapy',
    'MET1': 'Metformin / Other0 dual therapy',
    'MET0': 'Metformin monotherapy',
    'MET2': 'Metformin / Other1 triple therapy',
}


histogram_divs = []
treatment_dataframes = {}

app = dash.Dash(__name__)

image_div = html.Div(
    className="logos",
    children=
        [
        html.Div([
            html.Img(src="./assets/curf.png", style={'height': '80px', 'object-fit': 'cover'})
        ]),
        html.Div([
            html.Img(src="./assets/prisma.png", style={'height': '80px', 'object-fit': 'cover'})
        ])
    ])

app.layout = html.Div(
    children=[
        image_div,
        html.H1("Patient Treatment Recommendation"),

        html.Br(),
        html.Br(),

        html.Div(
            id="search_bar",
            children=[
                dcc.Input(id='input-patient-id', type='text', placeholder='Enter PatID'),

                html.Button('Go', id='go-button', n_clicks=0)
            ]
        ),

        html.Div(id='output-recommendation', style={'padding': '0px'}),
        html.Div(
            className="layout",
            children=[
                 html.Div(
                     className="container1",
                     children=[
                         html.Div(
                            id="slideshow-container",
                            className="slideshow-container",
                            style={'display':'none'},
                            children=[
                                html.Div(
                                    id="histo_fig",  
                                    className="histo-fig-container",
                                ),
                                html.A(className="prev", children="❮", n_clicks=0, id="prev-button"),
                                html.A(className="next", children="❯", n_clicks=0, id="next-button")
                            ]
                        ),
                        html.Div(
                            className="patient_details_container",
                            children=[
                                html.Div(id='output-patient-details', style={'padding': '0px'})
                            ]
                        )
                     ]
                 ),
                html.Div(
                    className="container2",
                    style={'width': '48%', 'float': 'left'},
                    children=[
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
                                                    'legend': {'x': 0, 'y': 1},
                                                    'font': {'family': 'Arial', 'size': 12},
                                                    'margin': {'l': 200, 'r': -20, 't': 20, 'b': 20}
                                                },
                                            },
                                            style={'display': 'none'}
                                        )
                                    ]
                                ),
                            ],
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
                                            'legend': {'x': 0, 'y': 1},
                                            'font': {'family': 'Arial', 'size': 12},
                                        }
                                    },
                                    style={'display': 'none'}
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            id="tooltip",
            style={
                "position": "fixed",
                "bottom": "10px",
                "left": "10px",
                "padding": "5px",
                "background-color": "rgba(0, 0, 0, 0.8)",
                "color": "#fff",
                "border-radius": "5px"
            },
            children=[
                html.Span("Acknowledgments", title="Supported by CURF-HSC Innovation Maturation Fund granted by Clemson University Research Foundation and Prisma Health Science Center")
            ]
        )
])


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
        Output('histo_fig', 'children'),
        Output('slideshow-container', 'style')
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
    global treatment_dataframes 

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
                [],
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
                {'display': 'none'},
                [],
                {'display': 'none'}
            )

        final_merged_df = final_merged[final_merged['PatID'] == patient_id]
        best_result, best_regimen_name = functions.create_patient_recomendation(loaded_models, final_merged, patient_id)
        patient_data_info = functions.generate_patient_infos(patient_id, final_merged)

        # Define the replacement mappings
        sex_mapping = {1: 'Male', 0: 'Female'}
        ethnicity_mapping = {1: 'Hispanic or Latino', 0: 'Non-Hispanic or Non-Latino'}

        # Replace values in 'Sex' and 'Ethnicity' columns
        patient_data_info['Sex'] = patient_data_info['Sex'].replace(sex_mapping)
        patient_data_info['Ethnicity'] = patient_data_info['Ethnicity'].replace(ethnicity_mapping)

        # Check if any of the DataFrames are empty
        if final_merged_df.empty or patient_data_info.empty:
            return (
                f"No data found for PatID: {patient_id}",
                {'padding': '20px'},
                [],
                {'padding': '0px'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                {'data': [], 'layout': {}},
                {'display': 'none'},
                [],
                {'display': 'none'}
            )

        
        histograms = []

        current_treatment = patient_data_info['Regimen'].values[0]
        predicted_result = best_result

        vline_marker = {
            'type': 'line',
            'x0': best_result,
            'x1': best_result,
            'y0': 0,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {
                'color': 'red',
                'width': 2
            }
        }

        
        for treatment, df in regimens_dict.items():
            
            title = f"Similar Patients on {treatment}"
            if current_treatment == treatment or best_regimen_name == treatment:
                title += " *"

            histogram = px.histogram(df, x="Result Numeric Value", nbins=50, title=title)
            histograms.append((treatment, histogram))

            # If the condition matches, add the vline_marker to the layout
            if current_treatment == treatment:
                histogram.update_layout(shapes=[vline_marker])


        histogram_divs = [
            html.Div(
                className='slide',  
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

        if current_treatment == best_regimen_name:
            result_text = [
            html.Div(f"Continue current Regimen : {current_treatment}"),
            html.Br(),
            html.Div(f"Predicted HbA1c (%): {predicted_result}")
            ]
        else:
            result_text = [
            html.Div(f"Recommendation: Switch from {current_treatment} to {best_regimen_name}"),
            html.Br(),
            html.Div(f"Predicted HbA1c (%): {predicted_result}")
            ]


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

        traces = []
        for regimen in final_merged_df['Regimen'].unique():
            regimen_df = final_merged_df[final_merged_df['Regimen'] == regimen]
            trace = go.Scatter(
                x=regimen_df['Result Date'],
                y=regimen_df['Result Numeric Value'],
                mode='lines+markers',
                name=regimen
            )
            traces.append(trace)

        # Convert 'Regimen Date' column to datetime format
        final_merged_df['Regimen Date'] = pd.to_datetime(final_merged_df['Regimen Date'])

        start_date = final_merged_df['Regimen Date'].min() - pd.Timedelta(days=45)
        end_date = final_merged_df['Regimen Date'].max() + pd.Timedelta(days=45)

        fig = px.scatter(final_merged_df, x='Regimen Date', y='Regimen', color='Regimen',
                        title='Treatment History', labels={'Regimen Date': 'Timestamp', 'Result Numeric Value': 'Values'},
                        template='plotly_white', range_x=[start_date, end_date])

        fig2 = go.Figure(data=traces, layout=go.Layout(
                title='Result History',
                xaxis={'title': 'Date'},
                yaxis={'title': 'HbA1c'},
                template='plotly_white'
            ))

            # Update the layout of the figure
        fig2.update_layout(legend={'x': 0, 'y': 1}, margin={'l': 50, 'r': 20, 't': 80, 'b': 50})

        return (
            result_text, {'padding': '20px'}, table, {'padding': '20px'},
            fig, {'display': 'block', 'width': '47vw'}, fig2, {'display': 'block', 'width': '47vw'},
            histogram_divs[slide_index], {'display': 'flex'}
        )

    return '', {'padding': '0px'}, [], {'padding': '0px'}, {'data': [], 'layout': {}}, {'display': 'none'}, {'data': [], 'layout': {}}, {'display': 'none'}, [], {'display': 'none'}  # Initial or no-click state



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
