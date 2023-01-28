# -*- coding: utf-8 -*-
"""Fetches and clean data from csv files and pandas Dataframes.
This module contains functions to fetch data from csv files in the format needed
by the tools in the project. Also contains methods to clean values when needed.
"""

import csv
import re
from typing import Literal, Union

import geopandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fetch_csv_data(
    csv_path: str,
    features: list[str],
    id_feature: str,
    id_feature_label: str,
    clean_cols_names: bool = False,
    scale_data: bool = False
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    r"""Fetches data from a csv file and renames columns as required for the
  rest of the code.

  Args:
    csv_path: A string representing the name of the csv file to fetch data
      from.
    features: A list of strings representing the key of the columns to
      fetch.
    id_feature: A string representing the key of the columns to fetch as id.
    id_feature_label: A string representing the key of the columns to fetch
      as name for the ids (e.g. name of districts).
    clean_cols_names: If True, specials chars will be remove from each
      column name.
    scale_data: indicates if data must be rescaled using the min and max values
      of each feature.

  Returns:
    A tuple containing a pandas DataFrame and a dict. The pandas DataFrame
    contains the fetched data; each column name in the features list is
    renamed as "'f' + index", in witch the index is the one in the features
    list.

    The dict maps the new name of the column to the one given in the
    features list.
  """
    if not clean_cols_names:
        df = pd.read_csv(
            csv_path,
            na_values=['#DIV/0!'],
            usecols=features +
            [id_feature, id_feature_label])[features +
                                            [id_feature, id_feature_label]]
    else:
        df = pd.read_csv(csv_path, na_values=['#DIV/0!'])
        cols: list[str] = df.columns.to_list()
        df.rename(columns={col: clean_colnames(col)
                           for col in cols},
                  inplace=True)
        df = df.loc[:, features + [id_feature, id_feature_label]]

    df_dict = dict(('f' + str(i + 1), features[i])
                   for i in range(0,
                                  len(df.columns) - 2))
    df_dict['id'] = id_feature
    df_dict['id_name'] = id_feature_label
    df_dict_inv = {v: k for k, v in df_dict.items()}

    df.columns = pd.Series(
        ['f' + str(i + 1)
         for i in range(0,
                        len(df.columns) - 2)] + ['id', 'id_name'])

    if scale_data:
        scaler: MinMaxScaler = MinMaxScaler()
        for i in range(len(features)):
            array = df['f' + str(i + 1)].to_numpy().reshape(-1, 1)
            if array.shape[0] > 0:
                df['f' + str(i + 1)] = scaler.fit_transform(array)

    return df, df_dict, df_dict_inv


def fetch_geo_data(csv_path: str) -> geopandas.GeoDataFrame:
    r"""Fetches geodata from a csv file and creates a GeoPandas GeoDataFrame.

  Args:
    csv_path: A string representing the name of the csv file to fetch data
      from.

  Returns:
    A GeoPandas GeoDataFrame containing the data obtained from the file passed
    as argument.
  """
    geo_df = pd.read_csv(csv_path)
    geo_df = geo_df.set_index('code_district', drop=False)

    region_df = geopandas.GeoDataFrame(data=geo_df.filter(
        items=[c for c in geo_df.columns if c != 'geometry']),
                                       geometry=geopandas.GeoSeries.from_wkt(
                                           geo_df['geometry']),
                                       crs='epsg:3005')  # type:ignore

    return region_df


def filter_quartiles(
        dataframe: pd.DataFrame,
        other_than: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Aplly box-plot filter to remove outliers from the dataframe. The filter is
  applied to all columns except the ones especified.

  Args:
    dataframe: A pandas DataFrame with the data to be filtered.
    other_than: A list of strings containg the keys of the columns that
      should not be filtered. It is not necessary to include the ids related
      ones ('id', 'id_name').

  Returns:
    A tuple containing two pandas Dataframes; the first one containing the
    filtered dataframe and the second containing the removed elements.
  """

    other_than += ['id', 'id_name']

    removed: pd.DataFrame = pd.DataFrame(columns=list(dataframe.columns) +
                                         ['removed_by'])

    cols: list[str] = dataframe.columns.to_list()
    for field in cols:
        if field not in other_than:
            q1 = dataframe[field].quantile(0.25)
            q3 = dataframe[field].quantile(0.75)
            iqr = q3 - q1
            mask = ((dataframe[field] >= q1 - 1.5 * iqr) &
                    (dataframe[field] <= q3 + 1.5 * iqr))

            r = dataframe.loc[mask is False]
            r['removed_by'] = field
            removed = pd.concat([removed, r], join='inner')  # type: ignore

            dataframe = dataframe.loc[mask]
    dataframe.reset_index(inplace=True)
    dataframe.drop(columns=['index'], inplace=True)
    return dataframe, removed


def fetch_weights(weights_path: str,
                  clean_cols_names: bool = False) -> dict[str, int]:
    r"""Creates a features weights dict based on data provided by a csv file.

  Args:
    weights_path: A string representing the name of the csv file to fetch
      data from.
    clean_cols_names: If True, specials chars will be remove from each
      column name.

  Returns:
    A dict containing the name of each feature associated with its weight.
  """

    weights_dict = {}

    with open(weights_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        if not clean_cols_names:
            weights_dict = {str(rows[0]): int(rows[1]) for rows in reader}
        else:
            weights_dict = {
                clean_colnames(str(rows[0])): int(rows[1])
                for rows in reader
            }
    return weights_dict


def fetch_features(
        features_path: str,
        join_classes: bool = False,
        join_begin_offset: int = 0,
        join_end_offset: int = 0,
        join_number_lists: int = 1) -> tuple[list[list[str]], pd.DataFrame]:
    r"""Creates features input list based on data provided by a csv file.

  Args:
    features_path: A string representing the name of the csv file to fetch
      data from.
    join_classes: if true, indicates that the lists are divided in 'classes'
      and that same 'class' must be handled together.
    join_begin_offset: has effect only if join_classes=True. Indicates which
      row the code should start the joining. Default: 0, i.e., the first
      row.
    join_end_offset: has effect only if join_classes=True. Indicates which
      row the code should end the joining. Default: 0, i.e., the first row.
    join_number_lists: A integer that has effect only if join_classes=True.
      Indicates the ammount of individuals that exist in each class.
  Returns:
    A tuple containing a list of the variables to be used as inputs, and a
    pandas DataFrame with the lists and their names.
  """

    df = pd.read_csv(features_path, usecols=['name_scenario', 'variables'])
    clean_lists_names(df)

    if join_classes:
        join_lists_by_classes(df, join_begin_offset, join_end_offset,
                              join_number_lists)

    features_list = []
    for i in range(len(df)):
        features = df['variables'].iloc[i][:]
        if features is not None:
            features = np.unique(features.split(', ')).tolist()
            df['variables'].iloc[i] = features
            features_list += [features]

        # features_list += [list(df['variables'].iloc[i][:].split(', '))]

    return features_list, df


def filter_features_weights(features: list[str],
                            weights_dict: dict[str, int]) -> list[str]:
    r"""Filter a list of features by the ones that have specific weights defined.

  Args:
    features: A list of strings representing the features to be filtered.
    weights_dict: a dict of strings representing the name of the features
      and ints representing the weights associated.

  Returns:
    A list of strings containing the filtered features, that have specific
    weights defined.
  """

    features_list_weights = []

    for bf in features:
        if bf in weights_dict:
            features_list_weights += [bf]
    return features_list_weights


def clean_colnames(column_name: str) -> str:
    r"""Remove special chars from a string.

  Note:
    Chars that are removed includes:
      _, (space), [, ], /, <, >, ., %, =, (, ), ,(comma), '(single quote),
      \, +, \n

  Args:
    column_name: a string to remove special chars from.

  Returns:
    A string with the special chars removed.
  """
    for c in [
            '_', ' ', '[', ']', '/', '<', '>', '.', '%', '=', '(', ')', ',',
            '\'', '\\', '+', '\n'
    ]:
        if c in column_name:
            column_name = column_name.replace(c, '')
    return column_name


def clean_lists_names(df: pd.DataFrame) -> None:
    r"""Remove special chars from names of lists and variables.

  Note:
    All chars that are not alphanumeric or '_' are removed.

  Args:
    df: a pandas dataframe with, at least, two columns: 'name_scenario',
      containg the names of lists of features, and 'variables', containg
      lists of features.
  """
    df['name_scenario'] = df['name_scenario'].str.replace(r'[^A-Za-z0-9\_]+',
                                                          '',
                                                          regex=True)
    df['variables'] = df['variables'].str.replace(r'[\'\[\]]+', '', regex=True)


def join_lists_by_classes(df: pd.DataFrame, begin_offset: int, end_offset: int,
                          number_lists: int):
    r"""Join lists of features.

  Args:
    df: a pandas dataframe with, at least, two columns: 'name_scenario',
      containg the names of lists of features, and 'variables', containg
      lists of features.
    begin_offset: has effect only if join_classes=True. Indicates which row
      the code should start the joining. Default: 0, i.e., the first row.
    end_offset: has effect only if join_classes=True. Indicates which row
      the code should end the joining. Default: 0, i.e., the first row.
    number_lists: A integer that has effect only if join_classes=True.
      Indicates the ammount of individuals that exist in each class.
  Returns:
    A pandas dataframe containg the aggregated lists of features.
  """
    for i in range(begin_offset, len(df) - end_offset, number_lists):
        new_row = ''
        for j in range(number_lists):
            if j == 0:
                new_row += df.iloc[begin_offset + j]['variables']
            else:
                new_row += ', ' + df.iloc[begin_offset + j]['variables']
        string = str(df.at[i + number_lists - 1, 'name_scenario'])
        df.drop(list(range(i, i + number_lists)), axis=0, inplace=True)
        df.at[i + number_lists - 1, 'variables'] = new_row

        regex_search = re.search(r'[a-z][\d].*_', string)
        if regex_search is not None:
            df.at[i + number_lists - 1,
                  'name_scenario'] = 'df_' + regex_search.group().removesuffix(
                      '_')
        else:
            df.at[i + number_lists - 1, 'name_scenario'] = 'df_' + string

    return df


def join_data(df: pd.DataFrame,
              csv_path: str,
              on: tuple[str, str],
              add_cols: Union[list[str], Literal['all']],
              labels: Union[dict[str, str], None] = None,
              clean_cols_names: bool = False):
    r"""Join external data into existing dataframe using pandas joins.

  Args:
    df: a pandas DataFrame containg the existing data.
    csv_path: a string representing the path to a csv file containg the
      additional data.
    on: a tuple containing two strings indicating which feature to join on; the
      first one refers to the dataframe; the second, to the csv file. If the
      second string is 'index', use the index of the tuples.
    add_cols: either a list of strings, indicating which features of the file
      should be used, or 'all' indicating that all features of the file should
      be joined.
    labels: a dict used to rename features after the joining process.
    clean_cols_names: If True, specials chars will be remove from each
      column name joined from the file.
  """
    if add_cols == 'all':
        df_all = pd.read_csv(csv_path, na_values=['#DIV/0!'])
    else:
        df_all = pd.read_csv(csv_path, na_values=['#DIV/0!'], usecols=add_cols)

    if clean_cols_names:
        cols: list[str] = df_all.columns.to_list()
        df_all.rename(columns={col: clean_colnames(col)
                               for col in cols},
                      inplace=True)

    df_copy = df.copy()

    if on[1] == 'index':
        df_all[on[0]] = [(i + 1) for i in range(len(df_all))]
    else:
        df_all.rename(columns={on[1]: on[0]}, inplace=True)

    if labels is not None:
        df_all.rename(columns=labels, inplace=True)

    df_copy = pd.merge(df_copy,
                       df_all,
                       how='left',
                       on=on[0],
                       suffixes=('', '_remove'),
                       copy=False)
    cols: list[str] = df_copy.columns.to_list()
    df_copy.drop([i for i in cols if '_remove' in i], axis=1, inplace=True)

    return df_copy
