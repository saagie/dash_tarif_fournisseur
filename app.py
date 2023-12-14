import os

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table
from dash import dcc, html, Patch, ctx, ALL
from dash.dependencies import Input, Output
from datetime import datetime
from io import StringIO
from saagieapi import SaagieApi
import json
import utils

# Create the dash app
app = dash.Dash(__name__,
                url_base_pathname=os.environ["SAAGIE_BASE_PATH"] + "/",
                external_stylesheets=[dbc.icons.BOOTSTRAP],
                suppress_callback_exceptions=True)

# Environment variables
postgresql_user = os.environ["POSTGRESQL_WRITER_USER"]
postgresql_pwd = os.environ["POSTGRESQL_WRITER_PWD"]
postgresql_db = os.environ["POSTGRESQL_DATABASE"]

s3_bucket = os.environ['SUPPLIER_S3_BUCKET']
s3_dir = os.environ['SUPPLIER_S3_PROCESSED_DIR']
analyse_job_id = os.environ["ANALYSE_JOB_ID"]

# postgresql table name
pg_analyse_table = os.environ['POSTGRESQL_ANALYSE_TABLE']
supplier_table_name = os.environ['SUPPLIER_TABLE_NAME']
cart_table_name = os.environ['POSTGRESQL_CART_TABLE']

# Local variables
cols_metadata = ["Marque", "Fournisseur",
                 "DateEffective", "DateReception", "NomFichier"]

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

# Connect to the postgresql database
pg_engine = utils.get_postgresql_client(postgresql_user=postgresql_user,
                                        postgresql_pwd=postgresql_pwd,
                                        database=postgresql_db)

# Connect to the s3 bucket
s3_client = utils.get_s3_client()

# Connect to the saagie api
saagie_client = SaagieApi(url_saagie=os.environ["SAAGIE_URL"],
                          id_platform=os.environ["SAAGIE_PLATFORM_ID"],
                          user=os.environ["TECHNICAL_SAAGIE_LOGIN"],
                          password=os.environ["TECHNICAL_SAAGIE_PWD"],
                          realm=os.environ["SAAGIE_REALM"])


# Define the postgresql_supplier_table component
df_supplier = pd.read_sql(f'SELECT * FROM {supplier_table_name}', pg_engine)

postgresql_supplier_table = dash_table.DataTable(id="postgresql_supplier_table",
                                                 data=df_supplier[cols_metadata].to_dict('records'),
                                                 columns=[{"name": i, "id": i, "selectable": True,
                                                           'type': utils.table_type(df_supplier[i])} for i in
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
df_cart = pd.read_sql(f'SELECT * FROM  {cart_table_name} LIMIT 100', pg_engine)
df_cart["Quantité de réception"] = df_cart["Quantité de réception"].astype(int)
dt_cart = dash_table.DataTable(id="postgresql_cart_table",
                               data=df_cart.to_dict('records'),
                               columns=[{"name": i, "id": i, 'type': utils.table_type(df_cart[i])} for i in
                                        df_cart.columns],
                               cell_selectable=True,
                               style_cell={'textAlign': 'center'},
                               sort_action='native',
                               filter_action='native',
                               editable=True,
                               page_size=20,
                               style_table={'overflowX': 'scroll'}, )


def generate_frontpage():
    return html.Div(
        id="header",
        children=[
            html.Img(id="logo", src="assets/logo.png", style={'display': 'inline-block',
                                                              'height': '15%',
                                                              'width': '15%'}),
            html.Div(
                id="header-text",
                children=[
                    html.H2("Analyse des fichiers fournisseurs")
                ],
                style={'display': 'inline-block', 'marginLeft': '5%', 'verticalAlign': 'top'}
            )
        ],
    )


@app.callback(Output(component_id='postgresql_cart_table', component_property='data'),
              Output(component_id='postgresql_supplier_table', component_property='data'),
              [Input(component_id='refresh', component_property='n_clicks')])
def populate_table(refresh):
    if ctx.triggered_id == "refresh":
        print("refresh")
        df_cart_data = pd.read_sql(f'SELECT * FROM {cart_table_name} LIMIT 100', pg_engine)
        df_cart_data["Quantité de réception"] = df_cart_data["Quantité de réception"].astype(int)
        df_supplier_data = pd.read_sql(f"""SELECT *
                                            FROM {supplier_table_name}""",
                                       pg_engine)
        return df_cart_data.to_dict('records'), df_supplier_data[cols_metadata].to_dict('records')
    return df_cart.to_dict('records'), df_supplier[cols_metadata].to_dict('records')


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
        list_other_info = df_supplier.loc[rows][
            ['Marque', 'Fournisseur', 'DateEffective', 'DateReception']].values.tolist()
        print(f"list_noms_fichiers: {list_noms_fichiers}")

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
        df["selected_info"] = list(map('-'.join, list_other_info))
        df["selected_columns"] = temp_list_noms_cols
        for _, row in df.iterrows():
            # à voir si on peut check sur l'existence du champs pour ne pas le recréer à chaque fois
            print(f'Read {s3_dir}{row["selected_files"]}.csv')
            obj = s3_client.get_object(Bucket=s3_bucket, Key=f'{s3_dir}{row["selected_files"]}.csv')
            file_content = obj['Body'].read().decode('utf-8')
            df_file = pd.read_csv(StringIO(file_content), sep=";", encoding="utf-8", engine="python")

            new_element = html.Div([
                html.Div(
                    row["selected_files"],
                    id={
                        'type': 'dynamic-output',
                        'index': row["selected_files"]
                    },
                    style=dict(display='none'),
                ),
                html.Div(
                    row["selected_info"],
                    id={
                        'type': 'dynamic-output',
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
                    style_cell={'textAlign': 'center'},
                    page_size=5,
                    style_table={'overflowX': 'scroll'},
                ),
                html.Br(),
            ])
            patched_children.append(new_element)

        return patched_children


@app.callback(
    Output(component_id='finishing-output', component_property='children'),
    Input('postgresql_cart_table', 'derived_filter_query_structure'),
    Input(component_id='final_submit', component_property='n_clicks'),
    Input({'type': 'dynamic-table', 'index': ALL}, 'selected_columns'),
    Input({'type': 'dynamic-output', 'index': ALL}, 'children'),
    Input(component_id='prefix_file', component_property='value')
)
def submit_to_job(derived_filter, n_click, selected_columns, value_output, prefix_file, ):
    list_file_column = []
    dict_result = {}
    if derived_filter:
        dict_result["filtres_panier"] = derived_filter

    if ctx.triggered_id == "final_submit" and selected_columns and value_output:
        dict_result["date_analyse"] = datetime.now()
        dict_result["fichiers_fournisseurs"] = value_output

        for i in range(len(value_output)):
            print(f"'nom_fichier': {value_output[i]}, 'colonnes': {selected_columns[i]}")
            list_file_column.append({'nom_fichier': f"{value_output[i]}",
                                     'colonnes': selected_columns[i]})
        # envoie sur pg dans une table:
        dict_result["colonnes_fournisseurs"] = list_file_column
        dict_result["already_done"] = False
        prefix_file = prefix_file if prefix_file else "analyse"
        dict_result[
            "nom_fichier_final"] = f"{prefix_file}_{dict_result['date_analyse'].strftime('%Y-%m-%dT%H:%M:%S')}.csv"
        df = pd.DataFrame([dict_result])
        # Change type to string
        df["fichiers_fournisseurs"] = df["fichiers_fournisseurs"].astype(str)
        df["filtres_panier"] = df["filtres_panier"].astype(str)
        df["colonnes_fournisseurs"] = df["colonnes_fournisseurs"].astype(str)

        with pg_engine.connect() as conn_pg:
            with conn_pg.begin():
                df.to_sql(pg_analyse_table, pg_engine, if_exists="append", index=False)
                # call API pour lancer le job
                print("API call")
                saagie_client.jobs.run(job_id=analyse_job_id)

                return "Insertion dans la table pg_analyse_table réussie." \
                       f"L'analyse sera disponible sous le nom de: {dict_result['nom_fichier_final']}"

    return ""


@app.callback(
    Output(component_id='postgresql_cart_table', component_property='filter_query'),
    Input(component_id='filter-query-input', component_property='value')
)
def write_query(query):
    if query is None:
        return ''
    return query


@app.callback(
    Output('datatable-query-structure', 'children'),
    Input('postgresql_cart_table', 'derived_filter_query_structure')
)
def display_query(query):
    if query is None:
        return ''
    df_tmp = pd.read_sql(f'SELECT * FROM  {cart_table_name}', pg_engine)
    query_encoded = json.dumps(query, indent=4)
    print(f"query_encoded: {query_encoded}")
    (pd_query_string, df_cart_filtered) = utils.construct_filter(eval(query_encoded), df_tmp, df_tmp.columns)
    print(f"Filtering query: {pd_query_string}")
    if pd_query_string != '':
        df_cart_filtered = df_cart_filtered.query(pd_query_string)

    return html.Details([
        html.Summary(f'Les filtres sur les données panier (nombre de lignes filtrés: {len(df_cart_filtered.index)}):'),
        html.Div(dcc.Markdown('''```json
{}
```'''.format(query_encoded, indent=4)))
    ])


app.layout = html.Div(
    [
        html.Div(id="frontpage", className="page", children=generate_frontpage()),
        dcc.ConfirmDialog(
            id='confirm-danger',
            message='Danger danger! Are you sure you want to continue?',
        ),
        html.Div(
            className="section",
            children=[
                html.Div(className="section-title",
                         children="Données paniers"),
                dbc.Row(html.Br(), class_name=".mb-4"),
                dbc.Alert(
                    [
                        html.I(className="bi bi-info-circle-fill me-2"),
                        "Vous pourriez utiliser soit une requête de filtrage pour les filtres complexes, "
                        "soit utiliser l'option des filtres directement dans la table.",
                    ],
                    color="info",
                    className="d-flex align-items-center",
                ),
                dbc.Label("Saisir la requête de filtrage:", style={'float': 'left'}),
                html.Abbr("\u2753", title="""La requête de filtrage est sous format: {nom de la colonne} opérateur valeur.
                La liste des opérateurs est: >=, <=, <, >, !=, =,  contains.
                Exemple: {Année} = 2020 and {Année} = 2023
                /!\ Attention: Les parenthèses ne sont pas encore accpetés. """),
                dbc.Input(id='filter-query-input', type="text", placeholder="Enter filter query",
                          style={'width': '96%', 'margin-left': '2%', 'margin-right': '2%', }),

                html.Hr(),
                dbc.Row(dt_cart, style={'margin-left': '2%', 'margin-right': '2%', }),
                html.Hr(),

            ]
        ),
        html.Div(className="section", id='datatable-query-structure', style={'whitespace': 'pre'}),

        html.Div(
            className="section",
            children=[
                html.Div(className="section-title",
                         children="Veuillez choisir un ou plusieurs fichiers fournisseurs: "),
                html.Div(
                    className="page",
                    children=[
                        html.Div(id="supplier-table", children=postgresql_supplier_table)
                    ],
                ),
            ],
        ),
        html.Div(
            className="section",
            children=[
                dbc.Button("Submit", id='submit', color="primary", className="mr-1", n_clicks=0)
            ]

        ),
        html.Div(
            className="section",
            children=[
                html.Div(className="section-title",
                         children="Veuillez choisir le/les colonnes des fichiers fournisseurs: "),
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(className="page", id="output_div", children=[])
                ),
            ]

        ),
        html.Div(
            className="section",
            children=[
                dbc.Label("Saisir le prefix du fichier d'analyse, par défaut 'analyse':"),
                dbc.Input(id="prefix_file", type="text",
                          placeholder="Enter the prefix of analyse file that you want to use"),
                dbc.Row(html.Br(), class_name=".mb-4"),

                dbc.Row(dbc.Button("Submit", id='final_submit', color="primary", className="mr-1", n_clicks=0)),
                dbc.Row(dcc.Markdown(id="finishing-output"), ),
                dbc.Row(html.Br(), class_name=".mb-4"),

                dbc.Row(dbc.Button("Refresh", id='refresh', color="primary", className="mr-1", n_clicks=0)),
            ]
        ),

        dbc.Row(html.Br(), class_name=".mb-4"),

    ])

if __name__ == '__main__':
    print("Running second run_server")
    app.run_server(host='0.0.0.0', debug=True, port=8050)
