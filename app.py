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
import sys
import pandas as pd


# Define data type
def table_type(df_column):
    # Note - this only works with Pandas >= 1.0.0

    if sys.version_info < (3, 0):  # Pandas 1.0.0 does not support Python 2
        return 'any'

    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
          isinstance(df_column.dtype, pd.BooleanDtype) or
          isinstance(df_column.dtype, pd.CategoricalDtype) or
          isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif (isinstance(df_column.dtype, pd.SparseDtype) or
          isinstance(df_column.dtype, pd.IntervalDtype) or
          isinstance(df_column.dtype, pd.Int8Dtype) or
          isinstance(df_column.dtype, pd.Int16Dtype) or
          isinstance(df_column.dtype, pd.Int32Dtype) or
          isinstance(df_column.dtype, pd.Int64Dtype)):
        return 'numeric'
    else:
        return 'any'


operators = [['>='],
             ['<='],
             ['<'],
             ['>'],
             ['!='],
             ['='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets, url_base_pathname=os.environ["SAAGIE_BASE_PATH"]+"/")


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Environment variables
postgresql_host = os.environ["POSTGRESQL_IP"]
postgresql_port = os.environ["POSTGRESQL_PORT"]
postgresql_user = os.environ["POSTGRESQL_WRITER_USER"]
postgresql_pwd = os.environ["POSTGRESQL_WRITER_PWD"]
postgresql_db = os.environ["POSTGRESQL_DATABASE"]

# Connect to the postgresql database
postgresql_string_connecion = f'postgresql://{postgresql_user}:{postgresql_pwd}@{postgresql_host}:{postgresql_port}/{postgresql_db}?sslmode=require'
pg_engine = create_engine(postgresql_string_connecion)

# Define the postgresql_supplier_table component
df_supplier = pd.read_sql(f'SELECT * FROM test_supplier', pg_engine)
postgresql_supplier_table = dash_table.DataTable(id="postgresql_supplier_table",
                                                 data=df_supplier.to_dict('records'),
                                                 columns=[{"name": i, "id": i, "selectable": True,
                                                           'type': table_type(df_supplier[i])} for i in
                                                          df_supplier.columns],
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

# Get cart data
df_cart = pd.read_sql(f'SELECT * FROM  test_cart', pg_engine)
dt_cart = dash_table.DataTable(id="postgresql_cart_table",
                               data=df_cart.to_dict('records'),
                               columns=[{"name": i, "id": i, 'type': table_type(df_cart[i])} for i in df_cart.columns],
                               cell_selectable=True,
                               style_cell={'textAlign': 'center'},
                               sort_action='native',
                               filter_action='native',
                               page_size=20,
                               style_table={'overflowX': 'scroll'}, )

# Define the additional columns component
df_additional_cols = pd.DataFrame(OrderedDict([
    ('nouvelle_colonne', ['nouveau_prix']),
    ('colonne1', ["Tarif1"]),
    ('operation', ["*"]),
    ('colonne2', ["Quantité de réception"]),
]))

# Define final schema table
final_data = []
final_data_dict = {}
for col_name in df_cart.columns.tolist():
    final_data_dict[col_name] = ""
final_data.append(final_data_dict)

# Define the final schema table component
output_schema_table = dash_table.DataTable(id="output_schema_table",
                                           data=final_data,
                                           columns=[{"name": i, "id": i} for i in final_data[0].keys()],
                                           style_cell={'textAlign': 'center'},
                                           style_table={'overflowX': 'scroll'},
                                           )


@app.callback(Output(component_id='postgresql_cart_table', component_property='data'),
              [Input(component_id='refresh', component_property='n_clicks')])
def populate_cart_table(refresh):
    df_cart_data = pd.read_sql(f'SELECT * FROM test_cart', pg_engine)
    return df_cart_data.to_dict('records')


@app.callback(Output(component_id='postgresql_supplier_table', component_property='data'),
              [Input(component_id='refresh', component_property='n_clicks')])
def populate_supplier_table(refresh):
    df_supplier_data = pd.read_sql(f'SELECT * FROM test_supplier', pg_engine)
    return df_supplier_data.to_dict('records')


@app.callback(
    Output(component_id="output_supplier_choices", component_property="children"),
    [Input(component_id='postgresql_supplier_table', component_property='selected_rows')],
)
def show_selected_file_name(rows):
    if rows:
        return f"The selected files are: {df_supplier.loc[rows][['NomFichier']].values.tolist()}"
    else:
        return ""


@app.callback(
    Output('output_columns', 'children'),
    Input('postgresql_supplier_table', 'selected_columns')
)
def show_selected_cols(selected_columns):
    if selected_columns:
        return "Selected generic columns are: {}".format(selected_columns)
    else:
        return ""


@app.callback(
    Output(component_id='additional_cols', component_property='children'),
    [Input(component_id='postgresql_supplier_table', component_property='selected_columns'),
     Input(component_id='postgresql_supplier_table', component_property='selected_rows')]
)
def show_selected_file_and_true_col(selected_columns, rows):
    if rows and selected_columns:
        return f"The selected files are: {df_supplier.loc[rows][['NomFichier']].values.tolist()} " \
               f"and Selected columns are: {df_supplier.loc[rows][selected_columns].values.tolist()}"

    else:
        return ""


@app.callback(Output(component_id='output_schema_table', component_property='data'),
              Output(component_id='output_schema_table', component_property='columns'),
              [
                  Input(component_id='postgresql_supplier_table', component_property='selected_columns'),
                  Input(component_id='postgresql_supplier_table', component_property='selected_rows'),
                  Input(component_id='table-dropdown', component_property='data')
              ])
def populate_output_schema(selected_columns, rows, data):
    new_output = [final_data_dict.copy()]
    if rows and selected_columns:
        col_values = df_supplier.loc[rows][selected_columns].values.tolist()
        list_infos = df_supplier.loc[rows][['Marque', 'Founisseur', "DateEffective", "DateReception"]].values.tolist()

        for info in list_infos:
            new_output[0]['_'.join(info)] = ''

        for tmp in col_values:
            for col_tmp in tmp:
                new_output[0][col_tmp] = ''
        # Add data from additional columns
        if data:
            for elmt in data:
                if elmt["nouvelle_colonne"]:
                    new_output[0][elmt["nouvelle_colonne"]] = ''

        return new_output, [{"name": i, "id": i} for i in new_output[0].keys()]

    if data:
        for elmt in data:
            if elmt["nouvelle_colonne"]:
                new_output[0][elmt["nouvelle_colonne"]] = ''
    return new_output, [{"name": i, "id": i} for i in new_output[0].keys()]


@app.callback(
    Output(component_id="table-dropdown", component_property="selected_cells"),
    Output(component_id="table-dropdown", component_property="active_cell"),
    Input(component_id="clear", component_property="n_clicks"),
)
def clear(n_clicks):
    return [], None


@app.callback(
    Output('table-dropdown', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('table-dropdown', 'data'),
    State('table-dropdown', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows



@app.callback(
    Output('output_cart_filter', 'children'),

    Input('postgresql_cart_table', 'filter_query'))
def update_table(filter):
    output_string = ""
    if filter:
        print(filter)
        filtering_expressions = filter.split(' && ')
        print(filtering_expressions)
        for filter_part in filtering_expressions:
            filter_col_name, operator, filter_value = split_filter_part(filter_part)
            print(f"col_name: {filter_col_name}, operator: {operator}, filter_value: {filter_value}")
            output_string += f"col_name: {filter_col_name}, operator: {operator}, filter_value: {filter_value} \n"
        return output_string
    else:
        return output_string


app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(html.P('Données paniers')),
    dbc.Row(dt_cart, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.P('Données fournisseurs')),
    dbc.Row(postgresql_supplier_table, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    html.P("Ajouter des colonnes pour l'analyse"),
    dbc.Row(html.Button("clear selection", id="clear")),
    dbc.Row(html.Br(), class_name=".mb-4"),
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
        row_deletable=True,
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
    dbc.Row(html.Button('Add Row', id='editing-rows-button', n_clicks=0)),

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
    # dbc.Row(dcc.Markdown(id="output_schema"), style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown("Schéma attendu de l'analyse"), ),
    dbc.Row(output_schema_table, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown("Test affichage des filtres sur données panier: "), ),
    dbc.Row(dcc.Markdown(id="output_cart_filter", style={"white-space": "pre", },), ),
    dbc.Row(html.Br(), class_name=".mb-4"),

])

if __name__ == '__main__':
    print("Running second run_server")
    app.run_server(host='0.0.0.0', debug=True, port=8050)
