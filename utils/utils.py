import os
import sys

import boto3
import pandas as pd
from sqlalchemy import create_engine


def get_s3_client():
    """
    Get a S3 client
    :return: boto3 client, S3 client
    """
    s3_endpoint = os.environ['AWS_S3_ENDPOINT']
    s3_region = os.environ['AWS_REGION_NAME']
    return boto3.client("s3", endpoint_url=s3_endpoint, region_name=s3_region)


def list_objects(s3_client, bucket_name: str, prefix: str):
    """
    List all objects in a S3 bucket
    :param s3_client: boto3 client, S3 client
    :param bucket_name: str, S3 bucket name
    :param prefix: str, prefix of the files to list.
    :return: List, list of files
    """
    s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    if 'Contents' not in s3_result:
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


def transform_dict_to_list(input_dict):
    output_list = []
    for file, columns in input_dict.items():
        if columns:
            output_list.extend([f"{file}.{col}" for col in columns])
    return output_list


def table_type(df_column):
    # Note - this only works with Pandas >= 1.0.0

    if sys.version_info < (3, 0):  # Pandas 1.0.0 does not support Python 2
        return 'any'

    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
          isinstance(df_column.dtype, pd.BooleanDtype) or
          isinstance(df_column.dtype, pd.CategoricalDtype) or
          isinstance(df_column.dtype, pd.PeriodDtype) or
          df_column.dtype == 'object'):
        return 'text'
    elif (isinstance(df_column.dtype, pd.SparseDtype) or
          isinstance(df_column.dtype, pd.IntervalDtype) or
          isinstance(df_column.dtype, pd.Int8Dtype) or
          isinstance(df_column.dtype, pd.Int16Dtype) or
          isinstance(df_column.dtype, pd.Int32Dtype) or
          isinstance(df_column.dtype, pd.Int64Dtype) or
          df_column.dtype == 'int64' or
          df_column.dtype == 'int32' or
          df_column.dtype == 'int16' or
          df_column.dtype == 'int8'):
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


def split_filter_part(operators, filter_part):
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


def get_postgresql_client(**kwargs):
    """
    Connect to postgresql, return sql alchemy engine
    Accepts database, postgresql_user and postgresql_pwd as kwargs
    Returns:
        engine: sql alchemy engine
    """
    postgresql_ip = os.environ['POSTGRESQL_IP']
    postgresql_port = os.environ['POSTGRESQL_PORT']
    postgresql_user = kwargs.get('postgresql_user', os.environ['POSTGRESQL_WRITER_USER'])
    postgresql_pwd = kwargs.get('postgresql_pwd', os.environ['POSTGRESQL_WRITER_PWD'])
    database = kwargs.get('database', os.environ['POSTGRESQL_DATABASE'])
    engine = create_engine(
        f'postgresql://{postgresql_user}:{postgresql_pwd}@{postgresql_ip}:{postgresql_port}/{database}?sslmode=require')

    return engine


def to_string(filter_query, list_cols_name):
    operator_type = filter_query.get('type')
    operator_subtype = filter_query.get('subType')

    if operator_type == 'relational-operator':
        if operator_subtype == '=':
            return '=='
        else:
            return operator_subtype
    elif operator_type == 'logical-operator':
        if operator_subtype == '&&':
            return '&'
        else:
            return '|'
    elif operator_type == 'expression' and operator_subtype == 'value' and type(filter_query.get('value')) == str:
        return '"{}"'.format(filter_query.get('value'))
    else:
        value = filter_query.get('value')
        if value in list_cols_name:
            return f"`{value}`"
        else:
            return value


def construct_filter(derived_query_structure, df, list_cols_name, complex_operator=None):
    # there is no query; return an empty filter string and the
    # original dataframe
    if derived_query_structure is None:
        return '', df

    # the operator typed in by the user; can be both word-based or
    # symbol-based
    operator_type = derived_query_structure.get('type')

    # the symbol-based representation of the operator
    operator_subtype = derived_query_structure.get('subType')

    # the LHS and RHS of the query, which are both queries themselves
    left = derived_query_structure.get('left', None)
    right = derived_query_structure.get('right', None)

    # the base case
    if left is None and right is None:
        return to_string(derived_query_structure, list_cols_name), df

    # recursively apply the filter on the LHS of the query to the
    # dataframe to generate a new dataframe
    (left_query, left_df) = construct_filter(left, df, list_cols_name)

    # apply the filter on the RHS of the query to this new dataframe
    (right_query, right_df) = construct_filter(right, left_df, list_cols_name)

    # 'datestartswith' and 'contains' can't be used within a pandas
    # filter string, so we have to do this filtering ourselves
    if complex_operator is not None:
        right_query = right.get('value')
        # perform the filtering to generate a new dataframe
        if complex_operator == 'datestartswith':
            return '', right_df[right_df[left_query.replace("`", "")].astype(str).str.startswith(right_query)]
        elif complex_operator == 'contains':
            return '', right_df[right_df[left_query.replace("`", "")].astype(str).str.contains(right_query, case=False)]

    if operator_type == 'relational-operator' and operator_subtype in ['contains', 'datestartswith']:
        return construct_filter(derived_query_structure, df, list_cols_name, complex_operator=operator_subtype)

    # construct the query string; return it and the filtered dataframe
    return ('{} {} {}'.format(
        left_query,
        to_string(derived_query_structure, list_cols_name) if left_query != '' and right_query != '' else '',
        right_query
    ).strip(), right_df)
