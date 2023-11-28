import os
import sys
from collections import OrderedDict

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table
from dash import dcc, html, Patch, ctx, ALL
from dash.dependencies import Input, Output, State
from sqlalchemy import create_engine
from datetime import datetime
import boto3
from io import StringIO

def get_s3_client():
    s3_endpoint = os.environ['AWS_S3_ENDPOINT']
    s3_region = os.environ['AWS_REGION_NAME']
    return boto3.client("s3", endpoint_url=s3_endpoint, region_name=s3_region)

def list_objects(s3_client, bucket_name: str, prefix: str):
    s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    if 'Contents' not in s3_result:
        # print(s3_result)
        return []

    file_list = [key['Key'].replace(prefix, "") for key in s3_result['Contents']]
    while s3_result['IsTruncated']:
        continuation_key = s3_result['NextContinuationToken']
        s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/",
                                              ContinuationToken=continuation_key)
        file_list.extend(
            key['Key'].replace(prefix, "") for key in s3_result['Contents']
        )
    return file_list

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

s3_bucket = os.environ['SUPPLIER_S3_BUCKET']
s3_dir = os.environ['SUPPLIER_S3_PROCESSED_DIR']

# Connect to the postgresql database
postgresql_string_connecion = f'postgresql://{postgresql_user}:{postgresql_pwd}@{postgresql_host}:{postgresql_port}/{postgresql_db}?sslmode=require'
pg_engine = create_engine(postgresql_string_connecion)

s3_client = get_s3_client()

# postgresql table name
# TODO: change to env var: $POSTGRESQL_ANALYSE_TABLE
pg_analyse_table = "test_analyse_panier_fournisseur"

# Define the postgresql_supplier_table component
df_supplier = pd.read_sql(f'SELECT * FROM test_supplier', pg_engine)
cols_metadata = ["NomFichier", "Marque", "Fournisseur",
                 "DateEffective", "DateReception"]

generic_columns_name = [
    'Reference brute', 'Reference DTP', 'Information1', 'Information2',
    'Information3', 'Information4', 'Information5', 'Information6',
    'Information7', 'Information8', 'Information9', 'Information10',
    'Coefficient1', 'Coefficient2', 'Coefficient3', 'Coefficient4',
    'Coefficient5', 'Coefficient6', 'Coefficient7', 'Coefficient8',
    'Coefficient9', 'Coefficient10', 'Tarif1', 'Tarif2', 'Tarif3', 'Tarif4',
    'Tarif5', 'Tarif6', 'Tarif7', 'Tarif8', 'Tarif9', 'Tarif10',
    'NomFichier', 'Marque', 'Founisseur', 'DateEffective', 'DateReception'
]

postgresql_supplier_table = dash_table.DataTable(id="postgresql_supplier_table",
                                                 data=df_supplier[cols_metadata].to_dict('records'),
                                                 columns=[{"name": i, "id": i, "selectable": True,
                                                           'type': table_type(df_supplier[i])} for i in
                                                          cols_metadata],
                                                 cell_selectable=True,
                                                 style_cell={'textAlign': 'center'},
                                                 sort_action='native',
                                                 filter_action='native',
                                                 page_size=20,
                                                 row_selectable='multi',
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
    Output(component_id="output_div", component_property="children"),

    [Input(component_id='postgresql_supplier_table', component_property='selected_rows'),
     Input(component_id='submit', component_property='n_clicks'),
     ],
)
def show_selected_file_name(rows, n_clicks):
    """

    """
    patched_children = Patch()
    if ctx.triggered_id == "submit":
        list_noms_fichiers = df_supplier.loc[rows][['NomFichier']].values.tolist()
        print(f"list_noms_fichiers: {list_noms_fichiers}")

        # files = list_objects(s3_client, s3_bucket, s3_dir)

        list_noms_cols = df_supplier.loc[
            rows, [c for c in df_supplier.columns if c not in cols_metadata]].values.tolist()
        
        temp_list_noms_cols = []

        for c in list_noms_cols:
            temp_c = []
            for col in c:
                if col.replace(" ", "") not in generic_columns_name:
                    temp_c.append(col)
            temp_list_noms_cols.append(temp_c)

        df = pd.DataFrame(list_noms_fichiers, columns=['selected_files'])
        # df["selected_columns"] = list_noms_cols
        df["selected_columns"] = temp_list_noms_cols
        for _, row in df.iterrows():
            # à voir si on peut check sur l'existence du champs pour ne pas le recréer à chaque fois

            print(f'{s3_dir}{row["selected_files"]}.csv')
            obj = s3_client.get_object(Bucket=s3_bucket, Key=f'{s3_dir}{row["selected_files"]}.csv')
            file_content = obj['Body'].read().decode('utf-8')
            df_file = pd.read_csv(StringIO(file_content), sep=",", encoding="utf-8")
            print(df_file[row["selected_columns"]].head(5))

            new_element = html.Div([
                html.Div(
                    row["selected_files"],
                    id={
                        'type': 'dynamic-output',
                        'index': row["selected_files"]
                    },
                ),
                dash_table.DataTable(
                    id={
                        'type': 'dynamic-table',
                        'index': n_clicks
                    },
                    data=df_file[row["selected_columns"]].head(5).to_dict('records'),
                    columns=[{"name": i, "id": i, "selectable": True} for i in row["selected_columns"]],
                    column_selectable="multi",
                    # cell_selectable=True,
                    style_cell={'textAlign': 'center'},
                    # sort_action='native',
                    # filter_action='native',
                    page_size=5,
                    style_table={'overflowX': 'scroll'}, 
                ),
                html.Br(),
            ])
            patched_children.append(new_element)

        return patched_children


@app.callback(Output(component_id='output_schema_table', component_property='data'),
              Output(component_id='output_schema_table', component_property='columns'),
              [Input(component_id='final_submit', component_property='n_clicks'),
               Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
               Input({'type': 'dynamic-output', 'index': ALL}, 'children'),
               State({'type': 'dynamic-dropdown', 'index': ALL}, 'id'),
               State({'type': 'dynamic-output', 'index': ALL}, 'id'),
               ])
def populate_output_schema(submit, value_dropdown, value_output, id_dropdown, id_output):
    new_output = [final_data_dict.copy()]
    if ctx.triggered_id == "final_submit" and value_dropdown and value_output:
        print(f"value_dropdown: {value_dropdown}")
        print(f"value_output: {value_output}")
        return new_output, [{"name": i, "id": i} for i in new_output[0].keys()]

    return new_output, [{"name": i, "id": i} for i in new_output[0].keys()]


@app.callback(
    Output('output_cart_filter', 'children'),

    Input('postgresql_cart_table', 'filter_query'))
def update_table(filter_expression):
    output_string = ""
    if filter_expression:
        print(filter_expression)
        filtering_expressions = filter_expression.split(' && ')
        print(filtering_expressions)
        for filter_part in filtering_expressions:
            filter_col_name, operator, filter_value = split_filter_part(filter_part)
            print(f"col_name: {filter_col_name}, operator: {operator}, filter_value: {filter_value}")
            output_string += f"col_name: {filter_col_name}, operator: {operator}, filter_value: {filter_value} \n"
        return output_string
    else:
        return output_string


@app.callback(
    Output('test', 'children'),
    Input('postgresql_cart_table', 'filter_query'),
    Input(component_id='final_submit', component_property='n_clicks'),
    Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'dynamic-output', 'index': ALL}, 'children'),
)
def submit_to_job(filter_expression, n_click, value_dropdown, value_output):
    list_filter_query = []
    list_file_column = []
    dict_result = {}
    if filter_expression:
        print(filter_expression)
        filtering_expressions = filter_expression.split(' && ')
        print(filtering_expressions)
        for filter_part in filtering_expressions:
            filter_col_name, operator, filter_value = split_filter_part(filter_part)
            print(f"col_name: {filter_col_name}, operator: {operator}, filter_value: {filter_value}")
            list_filter_query.append(
                {"col_name": filter_col_name, "operator": operator, "filter_value": filter_value})
            # structure de la table
            # date_analyse | filtres_panier (text) | fichiers_fournisseurs (text) | colonnes_fournisseurs (text) | already_done (bool) | nom_fichier_final (text)
            #[{'nom_fichier': 'fichier1', 'colonnes': ['col1', 'col2', 'col3']}, 
            #{'nom_fichier': 'fichier2', 'colonnes': ['col1', 'col2', 'col3']}]
        dict_result["filtres_panier"] = list_filter_query
    if ctx.triggered_id == "final_submit" and value_dropdown and value_output:
        dict_result["date_analyse"] = datetime.now()
        dict_result["fichiers_fournisseurs"] = value_output

        for i in range(len(value_output)):
            print(f"'nom_fichier': {value_output[i]}, 'colonnes': {value_dropdown[i]}")
            list_file_column.append({f"'nom_fichier': {value_output[i]}, 'colonnes': {value_dropdown[i]}"})
        # envoie sur pg dans une table:
        dict_result["colonnes_fournisseurs"] = list_file_column
        dict_result["already_done"] = False
        dict_result["nom_fichier_final"] = None
        df = pd.DataFrame([dict_result])
        with pg_engine.connect() as conn_pg:
            with conn_pg.begin():
                df.to_sql(pg_analyse_table, pg_engine, if_exists="append", index=False)
                # call API pour lancer le job
        return "Insertion dans la table pg_analyse_table réussie, l'analyse va bientôt démarrer."

    return ""


app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(html.P('Données paniers')),
    dbc.Row(dt_cart, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown("Les filtres sur les données panier: "), ),
    dbc.Row(dcc.Markdown(id="output_cart_filter", style={"white-space": "pre", }, ), ),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.P('Veuillez choisir un ou plusieurs fichiers fournisseurs: ')),
    dbc.Row(postgresql_supplier_table, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dbc.Button("Submit", id='submit', color="primary", className="mr-1", n_clicks=0)),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.P('Veuillez choisir le/les colonnes des fichiers fournisseurs: ')),
    dbc.Row(html.Br(), class_name=".mb-4"),
    html.Div(id="output_div", children=[]),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dbc.Button("Submit", id='final_submit', color="primary", className="mr-1", n_clicks=0)),
    dbc.Row(dcc.Markdown(id="test"), ),
    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(dcc.Markdown("Schéma attendu de l'analyse"), ),
    dbc.Row(output_schema_table, style={'margin-left': '2%', 'margin-right': '2%', }),
    dbc.Row(html.Br(), class_name=".mb-4"),

    dbc.Row(html.Br(), class_name=".mb-4"),
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

    dbc.Row(html.Br(), class_name=".mb-4"),
    dbc.Row(html.Br(), class_name=".mb-4"),

])

if __name__ == '__main__':
    print("Running second run_server")
    app.run_server(host='0.0.0.0', debug=True, port=8050)
