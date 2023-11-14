from collections import OrderedDict

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import os
from dash import dash_table
from sqlalchemy import create_engine
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets, url_base_pathname=os.environ["SAAGIE_BASE_PATH"]+"/")


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Environment variables
postgresql_host = os.environ["POSTGRESQL_IP"]
postgresql_port = os.environ["POSTGRESQL_PORT"]
postgresql_user = os.environ["POSTGRESQL_WRITER_USER"]
postgresql_pwd = os.environ["POSTGRESQL_WRITER_PWD"]
postgresql_db = os.environ["POSTGRESQL_DATABASE"]

postgresql_string_connecion = f'postgresql://{postgresql_user}:{postgresql_pwd}@{postgresql_host}:{postgresql_port}/{postgresql_db}?sslmode=require'
pg_engine = create_engine(postgresql_string_connecion)

# Define the postgresql_supplier_table component
df_supplier = pd.read_sql(f'SELECT * FROM test_supplier', pg_engine)
postgresql_supplier_table = dash_table.DataTable(id="postgresql_supplier_table",
                                                 data=df_supplier.to_dict('records'),
                                                 columns=[{"name": i, "id": i, "selectable": True} for i in df_supplier.columns],
                                                 cell_selectable=True,
                                                 style_cell={'textAlign': 'center'},
                                                 sort_action='native',
                                                 filter_action='native',
                                                 page_size=20,
                                                 row_selectable='multi',
                                                 column_selectable='multi',
                                                 selected_columns=[],
                                                 selected_rows=[],
                                                 style_table={'overflowX': 'scroll'},
                                                 )

df_cart = pd.read_sql(f'SELECT * FROM  test_cart', pg_engine)

#
df_additional_cols = pd.DataFrame(OrderedDict([
    ('nouvelle_colonne', ['nouveau_prix']),
    ('colonne1', ["Tarif1"]),
    ('operation', ["*"]),
    ('colonne2', ["Quantité de réception"]),
]))



@app.callback(Output(component_id='cart_data', component_property='children'),
              [Input(component_id='refresh', component_property='n_clicks')])
def populate_datatable(refresh):
    dt_cart = dash_table.DataTable(df_cart.to_dict('records'),
                                   [{"name": i, "id": i} for i in df_cart.columns],
                                   cell_selectable=True,
                                   style_cell={'textAlign': 'center'},
                                   sort_action='native',
                                   filter_action='native',
                                   page_size=20,
                                   style_table={'overflowX': 'scroll'}, )
    return dt_cart


@app.callback(Output(component_id='postgresql_supplier_table', component_property='data'),
              [Input(component_id='refresh', component_property='n_clicks')])
def populate_supplier_datatable(refresh):
    df_supplier_data = pd.read_sql(f'SELECT * FROM test_supplier', pg_engine)
    return df_supplier_data.to_dict('records')


@app.callback(
    Output(component_id="output_supplier_choices", component_property="children"),
    [Input(component_id='postgresql_supplier_table', component_property='selected_rows'),
     Input(component_id='postgresql_supplier_table', component_property='derived_virtual_indices'),
     Input(component_id='submit', component_property='n_clicks')],
)
def load_pages(rows, selected_row_indices, submit):
    selected_id_set = set(selected_row_indices or [])
    if rows:
        print("=====================================")
        print(selected_id_set)
        print(rows)
        print(df_supplier.loc[rows])
        return f"The selected files are: {df_supplier.loc[rows][['NomFichier']].values.tolist()}"
    else:
        return ""


@app.callback(
    Output('output_columns', 'children'),
    Input('postgresql_supplier_table', 'selected_columns')
)
def update_styles(selected_columns):
    if selected_columns:
        return "Selected columns are: {}".format(selected_columns)
    else:
        return ""


@app.callback(
    Output('additional_cols', 'children'),
    [Input('postgresql_supplier_table', 'selected_columns'),
     Input(component_id='postgresql_supplier_table', component_property='selected_rows'),
     Input(component_id='postgresql_supplier_table', component_property='derived_virtual_indices')]
)
def update_styles(selected_columns, rows, selected_row_indices):
    selected_id_set = set(selected_row_indices or [])
    if rows and selected_columns:
        print("=====================================")
        print(selected_id_set)
        print(rows)
        print(df_supplier.loc[rows])
        return f"The selected files are: {df_supplier.loc[rows][['NomFichier']].values.tolist()} " \
               f"and Selected columns are: {df_supplier.loc[rows][selected_columns].values.tolist()}"

    else:
        return ""



@app.callback(Output(component_id='output_schema', component_property='children'),
              [Input(component_id='refresh', component_property='n_clicks'),
               Input('postgresql_supplier_table', 'selected_columns'),
               Input(component_id='postgresql_supplier_table', component_property='selected_rows'),
               Input(component_id='postgresql_supplier_table', component_property='derived_virtual_indices'),
               Input(component_id='table-dropdown', component_property='data')
               ])
def populate_output_schema(refresh, selected_columns, rows, selected_row_indices, data):
    list_cols = df_cart.columns.tolist()
    print(data)

    if rows and selected_columns:
        print("============NEEEEEEWWWWWWW=========================")
        list_cols.extend(df_supplier.loc[rows][selected_columns].values.tolist()[0])
        print(list_cols)
        print("============FIIIIIINNNNNN=========================")

    return list_cols


@app.callback(
    Output('table-dropdown', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('table-dropdown', 'data'),
    State('table-dropdown', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(html.P('Données paniers')),
    dbc.Row(id="cart_data", style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.P('Données fournisseurs')),
    dbc.Row(postgresql_supplier_table, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    html.P("Ajouter des colonnes pour l'analyse"),
    dbc.Row(dash_table.DataTable(
        id='table-dropdown',
        data=df_additional_cols.to_dict('records'),
        columns=[
            {'id': 'nouvelle_colonne', 'name': 'nouvelle_colonne', },
            {'id': 'colonne1', 'name': 'colonne1', 'presentation': 'dropdown'},
            {'id': 'operation', 'name': 'operation', 'presentation': 'dropdown'},
            {'id': 'colonne2', 'name': 'colonne2', 'presentation': 'dropdown'},
        ],
        editable=True,
        dropdown={
            'colonne1': {
                'options': [
                    {'label': i, 'value': i}
                    for i in (df_cart.columns.tolist() + df_supplier.columns.tolist())
                ]
            },
            'operation': {
                'options': [
                    {'label': i, 'value': i}
                    for i in ['+', '-', '*', '/']
                ]
            },
            'colonne2': {
                'options': [
                    {'label': i, 'value': i}
                    for i in (df_cart.columns.tolist() + df_supplier.columns.tolist())
                ]
            }
        }
    ), style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    html.Button('Add Row', id='editing-rows-button', n_clicks=0),

    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dbc.Button("Submit", id='submit', color="primary", className="mr-1", n_clicks=0)),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dbc.Button("Refresh", id='refresh', color="primary", className="mr-1", n_clicks=0)),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown(id="output_supplier_choices"), ),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown(id="output_columns"), ),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown(id="additional_cols"), ),
    dbc.Row(dcc.Markdown(id="output_schema"), style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),

    dbc.Row(html.Br(), class_name=".mb-4"),

])

if __name__ == '__main__':
    print("Running second run_server")
    app.run_server(host='0.0.0.0', debug=True, port=8050)
