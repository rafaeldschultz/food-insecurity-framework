# -*- coding: utf-8 -*-
"""Build Plots and maps to help visualization of the data.

This module contains methods to build plots and maps to help visualization
of the data.
"""

import math
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas
from typing import Union
import deap.benchmarks.tools as bt

import libpysal
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from yellowbrick.features import RadViz
from yellowbrick.features import rank2d
from yellowbrick.features import pca_decomposition
from yellowbrick.features import joint_plot

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from splot.esda import plot_local_autocorrelation
from esda import Moran_Local

import user_inputs as user_inputs


def scatter_plots(
    dataframe: pd.DataFrame,
    comb_features: list[tuple[str, ...]],
    color_key: str,
    labels: dict[str, str],
    color_scale: str,
    min_color_dict: dict[str, float],
    max_color_dict: dict[str, float],
    path: Path,
    only_3d_plots: bool = False,
) -> None:
  r"""Builds graphs for all especified combinations of features in the 
  dataframe.

  Args:
    dataframe: A pandas dataframe containing the data to be ploted.
    comb_features: A list of strings containing combinations of features to
      be ploted.
    color_key: A string containing the key to be used to color the plots.
    labels: A dict of strings containing the labels for the features in the
      dataframe.
    color_scale: A string containing the name of a valid plotly color scale
      (https://plotly.com/python/builtin-colorscales/).
    min_color_dict: A dict of [str, float] containg the minimum value of the
      specified feature.
    max_color_dict: A dict of [str, float] containg the maximum value of the
      specified feature.
    only_3d_plots: a boolean value indicating if only 3d graphs must be
      plotted.
    path: The path to save the exported graphs.
  """

  gridElements = sum(map(lambda x: len(x) == 2, comb_features))
  maxRows = math.ceil(gridElements / 2)
  maxCols = 2

  fig3d = 0
  fig = make_subplots(rows=maxRows,
                      cols=maxCols,
                      subplot_titles=tuple([i + 1 for i in range(gridElements)
                                           ]))

  names = {}
  row = col = cont = 1

  for axis in comb_features:
    if (len(axis) == 2 and not only_3d_plots):

      fig.add_trace(go.Scatter(
          x=dataframe[axis[0]],
          y=dataframe[axis[1]],
          mode='markers',
          marker=dict(color=dataframe[color_key],
                      coloraxis="coloraxis",
                      cmin=min_color_dict[color_key],
                      cmax=max_color_dict[color_key]),
          customdata=list(dataframe[['id_name']].to_numpy()),
      ),
                    row=row,
                    col=col)

      fig.update_xaxes(title_text=labels[axis[0]],
                       row=row,
                       col=col,
                       showgrid=True,
                       gridwidth=1,
                       gridcolor='#EFEEF3',
                       showline=True,
                       linewidth=2,
                       linecolor='#EFEEF3')
      fig.update_yaxes(title_text=labels[axis[1]],
                       row=row,
                       col=col,
                       showgrid=True,
                       gridwidth=1,
                       gridcolor='#EFEEF3',
                       showline=True,
                       linewidth=2,
                       linecolor='#EFEEF3')
      fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br><br>" +
                        labels[axis[0]] + ": %{x}<br>" + labels[axis[1]] +
                        ":  %{y}<br>",
                        row=row,
                        col=col)

      names[str(cont)] = labels[axis[0]] + ' x ' + labels[axis[1]]
      cont += 1
    elif (len(axis) > 2):
      fig3d = px.scatter_3d(
          dataframe,
          x=dataframe[axis[0]],
          y=dataframe[axis[1]],
          z=dataframe[axis[2]],
          color=dataframe[color_key],
          title=labels[axis[0]] + " x " + labels[axis[1]] + " x " +
          labels[axis[2]],
          width=800,
          height=800,
          labels=dict([(i, labels[i]) for i in axis] +
                      [(color_key, labels[color_key])]),
          color_continuous_scale=color_scale,
          template="plotly_white",
          custom_data=dataframe[['id_name']],
          range_color=[min_color_dict[color_key], max_color_dict[color_key]])
      fig3d.update_traces(marker_size=4)
      fig3d.update_traces(hovertemplate="<b>%{customdata[0]}</b><br><br>" +
                          labels[axis[0]] + ": %{x}<br>" + labels[axis[1]] +
                          ":  %{y}<br>" + labels[axis[2]] + ":  %{z}<br>")

    col += 1
    if (col > maxCols):
      col = 1
      row += 1

  if not only_3d_plots:
    fig.update_layout(height=maxRows * 600,
                      width=1600,
                      coloraxis=dict(colorscale=color_scale,
                                     colorbar_title=labels[color_key]),
                      showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.for_each_annotation(lambda a: a.update(text=names[a.text]))
    # fig.show()

    fig.write_html(
        path.joinpath('scatter_2d_{}_color_{}.html'.format(
            labels[color_key], color_key)))
    fig.write_image(path.joinpath('scatter_2d_{}_color_{}.png'.format(
        labels[color_key], color_key)),
                    scale=user_inputs.fig_scale)

  if (fig3d != 0):
    fig3d.show()

    fig3d.write_html(
        path.joinpath('scatter_3d_{}.html'.format(labels[color_key])))
    fig3d.write_image(path.joinpath('scatter_3d_{}.png'.format(
        labels[color_key])),
                      scale=user_inputs.fig_scale)


def nsga2_stats_plots(fronts: list, path: Path) -> None:
  r"""Plots stats that can be used to evaluate the quality of the NSGA-2 solutions.

  Args:
    fronts: A list containing the fronts obtained by the NSGA-2 algorithm.
    path: The path to save the exported graphs.
  """

  fig = make_subplots(
      rows=1,
      cols=3,
      subplot_titles=('NSGA-2 Hyp. non-dom fronts - With outliers',
                      'NSGA-2 % hyp. non-dom fronts - With outliers',
                      'NSGA-2 n solutions dominated by front - With outliers'))

  reference = np.max([
      np.max([ind.fitness.values for ind in front], axis=0) for front in fronts
  ],
                     axis=0) + 1
  hypervols = [bt.hypervolume(front, reference) for front in fronts]

  fig.add_trace(go.Scatter(
      y=hypervols,
      mode='lines',
  ), row=1, col=1)

  fig.update_xaxes(title_text='Front',
                   row=1,
                   col=1,
                   showgrid=True,
                   gridwidth=1,
                   gridcolor='#EFEEF3',
                   showline=True,
                   linewidth=2,
                   linecolor='#EFEEF3')
  fig.update_yaxes(title_text='Hypervolume',
                   row=1,
                   col=1,
                   showgrid=True,
                   gridwidth=1,
                   gridcolor='#EFEEF3',
                   showline=True,
                   linewidth=2,
                   linecolor='#EFEEF3')

  hypervols_np = np.array(hypervols)
  hypervols_ = hypervols_np / hypervols_np.max() * 100

  fig.add_trace(go.Scatter(
      y=hypervols_,
      mode='lines',
  ), row=1, col=2)

  fig.update_xaxes(title_text='Front',
                   row=1,
                   col=2,
                   showgrid=True,
                   gridwidth=1,
                   gridcolor='#EFEEF3',
                   showline=True,
                   linewidth=2,
                   linecolor='#EFEEF3')
  fig.update_yaxes(title_text='%',
                   row=1,
                   col=2,
                   showgrid=True,
                   gridwidth=1,
                   gridcolor='#EFEEF3',
                   showline=True,
                   linewidth=2,
                   linecolor='#EFEEF3')

  tot, num_dom, perc = 0, [], []
  for f in fronts:
    tot += len(f)
  subtot = tot
  for f in fronts:
    r = subtot - len(f)
    num_dom += [r]
    perc += [r / tot * 100]
    subtot -= len(f)

  fig.add_trace(go.Scatter(
      y=num_dom,
      mode='lines',
  ), row=1, col=3)

  fig.update_xaxes(title_text='Front',
                   row=1,
                   col=3,
                   showgrid=True,
                   gridwidth=1,
                   gridcolor='#EFEEF3',
                   showline=True,
                   linewidth=2,
                   linecolor='#EFEEF3')
  fig.update_yaxes(title_text='#',
                   row=1,
                   col=3,
                   showgrid=True,
                   gridwidth=1,
                   gridcolor='#EFEEF3',
                   showline=True,
                   linewidth=2,
                   linecolor='#EFEEF3')

  fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')

  # fig.show()

  fig.write_html(path.joinpath('nsga_stats.html'))
  fig.write_image(path.joinpath('nsga_stats.png'), scale=user_inputs.fig_scale)


def pareto_fronts_plot(
    dataframe: pd.DataFrame,
    fronts: list[int],
    labels: dict[str, str],
    axis: list[str],
    title: str,
    color_pallete: list[str],
    path: Path,
) -> None:
  r"""Plots 3 selected pareto's fronts.

  Args:
    dataframe: A pandas dataframe containing the data to be ploted.
    fronts: A list of integers containing the fronts to be ploted.
    labels: A dict of strings used to retrieve the real names of the
      features representing the columns of the dataframe. Must have the same
      lenght as "fronts".
    axis: A list of strings containing the axis to be ploted.
    title: A string containing the title of the plot.
    color_pallete: A list of strings representing hex colors.Must have the
      same lenght as "fronts" and "labels" lists.
    path: The path to save the exported graphs.
  """

  categories = ['Front ' + str(i) for i in fronts]

  x = []
  y = []
  z = []
  custom_data = []

  for front in fronts:
    dfs = dataframe.query('front == ' + str(front))
    x += [dfs[list(labels.keys())[0]]]
    y += [dfs[list(labels.keys())[1]]]
    z += [dfs[list(labels.keys())[2]]]
    custom_data += [dfs['id_name']]

  fig = go.Figure(data=[
      go.Mesh3d(x=x[i],
                y=y[i],
                z=z[i],
                color=color_pallete[i],
                opacity=0.50,
                name=categories[i],
                customdata=custom_data[i]) for i in range(len(x))
  ])

  fig.update_traces(hovertemplate='<b>%{customdata}</b><br><br>' + axis[0] +
                    ': %{x}<br>' + axis[1] + ':    %{y}<br>' + axis[2] +
                    ':    %{z}',
                    selector=dict(type='mesh3d'))

  fig.update_layout(title=title,
                    scene=dict(
                        xaxis_title=axis[0],
                        yaxis_title=axis[1],
                        zaxis_title=axis[2],
                    ))

  fig.write_html(
      path.joinpath('pareto_fronts_{}x{}x{}.html'.format(
          axis[0], axis[1], axis[2])))
  fig.write_image(path.joinpath('pareto_fronts_{}x{}x{}.png'.format(
      axis[0], axis[1], axis[2])),
                  scale=user_inputs.fig_scale)


def find_graph_fronts(dataframe: pd.DataFrame,
                      min_points: int = 3) -> list[int]:
  r"""Calculates 3 fronts (bottom, mid, upper) that have at least a certain 
  amount of points to be ploted.

  Args:
    dataframe: A pandas dataframe containing the data to be ploted.
    min_points: An integer that represents the minimum amount of points a
      front has to have to be considered valid.

  Returns:
    A list of integers containing 3 front numbers to be ploted: the lowest,
    the highest and the middlest possible.
  """

  fronts_min = dataframe['front'].min()
  fronts_max = dataframe['front'].max()

  fronts_list = [fronts_min, fronts_max // 2, fronts_max]
  fronts_count_list = [
      dataframe[dataframe['front'] == str(i)].count()[0] for i in fronts_list
  ]

  while fronts_count_list[0] < min_points:
    fronts_list[0] += 1
    if fronts_list[0] >= fronts_max:
      fronts_list[0] = -1
      break
    fronts_count_list[0] = dataframe[dataframe['front'] == str(
        fronts_list[0])].count()[0]

  while fronts_count_list[2] < min_points:
    fronts_list[2] -= 1
    if fronts_list[2] <= fronts_min:
      fronts_list[2] = -1
      break
    fronts_count_list[2] = dataframe[dataframe['front'] == str(
        fronts_list[2])].count()[0]

  delta = 1
  while fronts_count_list[1] < min_points:
    if delta % 2 == 0:
      fronts_list[1] += delta
    else:
      fronts_list[1] -= delta
    delta += 1

    if (fronts_list[1] >= fronts_max or fronts_list[1] <= fronts_min):
      fronts_list[1] = -1
      break
    fronts_count_list[1] = dataframe[dataframe['front'] == str(
        fronts_list[1])].count()[0]

  fronts_list = [i for i in fronts_list if i != -1]

  return fronts_list


def maps_plots(
    dataframe: pd.DataFrame,
    features: list[str],
    labels: dict[str, str],
    color_scale: str,
    weights: list[int],
    geo_df: geopandas.GeoDataFrame,
    min_color_dict: dict[str, float],
    max_color_dict: dict[str, float],
    path: Path,
) -> None:
  r"""Builds maps based on geographic and other given data.

  Args:
    dataframe: A pandas dataframe containing the data to be ploted.
    features: A list of strings contaning the features to be used to plot the
      maps.
    labels: A dict of strings used to retrieve the real names of the features
      representing the columns of the dataframe.
    color_scale: A string containing the name of a valid plotly color scale
      (https://plotly.com/python/builtin-colorscales/).
    weights: A list of integers containing the weights of each key.
    geo_df: A valid geopandas dataframe containing the geographic data of the
      map to be ploted.
    path: The path to save the exported graphs.
  """
  region_json = json.loads(geo_df.to_json())

  for i in range(len(features)):
    fig = px.choropleth(
        dataframe,
        geojson=region_json,
        color_continuous_scale=color_scale if weights[i] > 0 else color_scale +
        '_r',
        locations=dataframe['id'],
        projection='mercator',
        hover_name=dataframe['id_name'],
        labels=labels,
        color=features[i],
        title=labels[features[i]],
        width=1200,
        range_color=[min_color_dict[features[i]], max_color_dict[features[i]]])
    fig.update_geos(fitbounds='locations', visible=False)

    fig.write_html(path.joinpath('map_{}.html'.format(labels[features[i]])))
    fig.write_image(path.joinpath('map_{}.png'.format(labels[features[i]])),
                    scale=user_inputs.fig_scale)


def outliers_map_plot(
    dataframe: pd.DataFrame,
    outliers_df: pd.DataFrame,
    filtered_by: str,
    labels: dict[str, str],
    geo_df: geopandas.GeoDataFrame,
    hover_data: list[str],
    path: Path,
) -> None:
  r"""Builds a map that shows the geographic position of the outliers.

  Args:
    dataframe: A pandas dataframe containing the filtered data.
    outliers_df: A pandas dataframe containing the outliers' data.
    filtered_by: A string indicating the name of the field containing the
      'filtered_by' indicator.
    labels: A dict of strings used to retrieve the real names of the features
      representing the columns of the dataframe.
    geo_df: A valid pandas dataframe containing the geographic data of the map
      to be ploted.
    hover_data: A list of strings containing the name of fields to show on
      hover.
    path: The path to save the exported graphs.
  """
  region_json = json.loads(geo_df.to_json())

  df = dataframe.copy()
  df[filtered_by] = 'not an outlier'
  df = pd.concat([df, outliers_df], join='inner')
  df[filtered_by] = df[filtered_by].replace(labels)

  fig = px.choropleth(df,
                      geojson=region_json,
                      locations=df['id'],
                      projection='mercator',
                      hover_name=df['id_name'],
                      labels=labels,
                      color=filtered_by,
                      color_discrete_map={'not an outlier': 'lightgray'},
                      title='Outliers',
                      hover_data=hover_data,
                      width=1200,
                      basemap_visible=True)
  fig.update_geos(fitbounds='locations', visible=False)

  fig.write_html(path.joinpath('map_outliers.html'))
  fig.write_image(path.joinpath('map_outliers.png'),
                  scale=user_inputs.fig_scale)


def get_matplotlib_colorscale(
    plotly_colorscale: str,
    reverse: bool = False) -> mcolors.LinearSegmentedColormap:
  r"""Generates a matplotlib colorscale based on a plotly colorscale, providing
  compatibility between the two libraries.

  Args:
    plotly_colorscale: A string containing the name of the plotly colorscale.
    reverse: A boolean value indicating if the colorscale must be reversed.

  Returns:
    A matplotlib colormap (mcolors.LinearSegmentedColormap) based on the plotly
    colors of the colorscale given as argument.
  """
  plotly_colorscale = plotly_colorscale + '_r' if reverse else plotly_colorscale
  cmap = [
      mcolors.to_rgba(i[1]) for i in colors.get_colorscale(plotly_colorscale)
  ]
  matplotlib_colorscale = mcolors.LinearSegmentedColormap.from_list('', cmap)
  return matplotlib_colorscale


def radviz_plot(df_data: pd.DataFrame,
                df_color: pd.Series,
                classes: list[str],
                labels: dict[str, str],
                cmap: mcolors.LinearSegmentedColormap,
                path: Path,
                outliers_detection: bool = False,
                title: Union[str, None] = None,
                color_feature_label: Union[str, None] = None) -> None:
  r"""Plot a RadViz (Radial Visualization) graph.

  Args:
    df_data: A pandas dataframe with the data to be plotted.
    df_color: A pandas series with the data to be used to color the samples.
    classes: A list of strings containg the classes of the data.
    labels: A dictionary containg the labels of the features plotted.
    cmap: A matplotlib colormap to be used to color the map.
    path: The path to save the exported graphs.
    outliers_detection: A boolean value indicating if outliers were removed from
      the data. Default: False.
    title: A string indicating the title to be added to the map. If not
      provided, a standard title is generated. Default: None.
    color_feature_label: A string containg the color feature label, to be used
      in file export naming.
  """

  x = df_data.rename(columns=labels)
  y = df_color

  cols: list[str] = x.columns.to_list()

  outpath = ''
  if outliers_detection:
    title = 'Radviz \n Without Outliers \n' + ' x '.join(
        cols) if title is None else title
    if color_feature_label is None:
      outpath = path.joinpath('RadViz_without_outliers.png')
    else:
      title += '\nColored by {}\n'.format(color_feature_label)
      outpath = path.joinpath(
          'RadViz_without_outliers_{}.png'.format(color_feature_label))
  else:
    title = 'Radviz \n With Outliers \n' + ' x '.join(
        cols) if title is None else title
    if color_feature_label is None:
      outpath = path.joinpath('RadViz_with_outliers.png')
    else:
      title += '\nColored by {}\n'.format(color_feature_label)
      outpath = path.joinpath(
          'RadViz_without_outliers_{}.png'.format(color_feature_label))

  visualizer = RadViz(classes=classes,
                      colormap=cmap,
                      size=(1920, 920),
                      title=title)

  visualizer.fit(x, y)
  visualizer.transform(x)

  visualizer.show(outpath=outpath)
  plt.close('all')


def rank2d_pearson_plot(
    df_data: pd.DataFrame,
    labels: dict[str, str],
    cmap: mcolors.LinearSegmentedColormap,
    path: Path,
    outliers_detection: bool = False,
    df_data_original: Union[pd.DataFrame, None] = None,
) -> None:
  r"""Plot a Rank2D graph based on the Pearson's coefficient.

  Args:
    df_data: A pandas dataframe with the data to be plotted.
    labels: A dictionary containg the labels of the features plotted.
    cmap: A matplotlib colormap to be used to color the map.
    path: The path to save the exported graphs.
    outliers_detection: A boolean value indicating if outliers were removed
      from the data. Default: False.
    df_data_original: A pandas dataframe containg the original data in case
      outliers had been removed. It only takes effect if
      'outliers_detection = True'.
  """
  x = df_data.rename(columns=labels)

  if outliers_detection and df_data_original is not None:
    x_original = df_data_original.rename(columns=labels)
    _, axes = plt.subplots(ncols=2, figsize=(30, 10))
    rank2d(
        x,
        colormap=cmap,    # type:ignore
        title='Pearson Ranking\nWithout Outliers',
        ax=axes[0],
        show=False)
    rank2d(
        x_original,
        colormap=cmap,    # type:ignore
        title='Pearson Ranking\nWith Outliers',
        ax=axes[1],
        show=False)

    plt.savefig(path.joinpath('rank2d.png'))
  else:
    rank2d(
        x,
        colormap=cmap,    # type:ignore
        title='Pearson Ranking\nWith Outliers',
        show=False)

    plt.savefig(path.joinpath('rank2d.png'))

  plt.close('all')


def rename_classes_as_integer(df: pd.DataFrame) -> pd.DataFrame:
  r"""Convert the string names of classes in a DataFrame to integers values.

  Args:
    df: A pandas dataframe with the data containg the classes to be renamed.

  Returns:
    A pandas dataframe containg the classes renamed as integers values.
  """

  y = df.copy()
  y = y.astype(int)

  unique_y = pd.unique(y.squeeze())
  unique_y.sort()

  k = 0
  replace_dict = {}
  for v in unique_y:
    replace_dict[v] = k
    k += 1
  y.replace(replace_dict, inplace=True)
  return y


def pca_plot(
    df_data: pd.DataFrame,
    color_feature: str,
    classes: list[str],
    labels: dict[str, str],
    cmap: mcolors.LinearSegmentedColormap,
    path: Path,
    outliers_detection: bool = False,
    df_data_original: Union[pd.DataFrame, None] = None,
    classes_original: Union[list[str], None] = None,
    dim: int = 2,
    color_as_axis: bool = False,
) -> None:
  r"""Plot a PCA graph.

  Args:
    df_data: A pandas dataframe with the data to be plotted.
    color_feature: A string indicating the featured tha should be used to
      color the graph.
    classes: A list of strings indicating the classes of the samples.
    labels: A dictionary containg the labels of the features plotted.
    cmap: A matplotlib colormap to be used to color the map.
    path: The path to save the exported graphs.
    outliers_detection: A boolean value indicating if outliers were removed
      from the data. Default: False.
    df_data_original: A pandas dataframe containg the original data in case
      outliers had been removed. It only takes effect if
      'outliers_detection = True'.
    classes_original: a list of strings containg the original data classes
      (before the removal of outliers) of the samples.
    dim: A integer indicating the dimensions of the PCA plot.
      Must be 2 or 3.
    color_as_axis: A boolean value indicating if the feature used as color
      must also be plotted.
  """

  x = df_data.copy()
  y = rename_classes_as_integer(x[color_feature].to_frame())
  x[color_feature] = y
  if not color_as_axis:
    x.drop(columns=color_feature, inplace=True)
  x = x.rename(columns=labels)

  if outliers_detection and df_data_original is not None:
    x_original = df_data_original.copy()
    y_original = rename_classes_as_integer(x_original[color_feature].to_frame())
    x_original[color_feature] = y_original
    if not color_as_axis:
      x_original.drop(columns=color_feature, inplace=True)
    x_original = x_original.rename(columns=labels)

    if dim == 3:
      _, axes = plt.subplots(ncols=2,
                             figsize=(30, 10),
                             subplot_kw=dict(projection='3d'))
      outpath = path.joinpath(
          'pca_3d_without_outliers_{}.png'.format(color_feature))
    else:
      _, axes = plt.subplots(ncols=2, figsize=(30, 10))
      outpath = path.joinpath(
          'pca_2d_without_outliers_{}.png'.format(color_feature))

    pca_decomposition(x,
                      y,
                      scale=True,
                      classes=classes,
                      colormap=cmap,
                      proj_features=True,
                      ax=axes[0],
                      show=False,
                      projection=dim)
    pca_decomposition(x_original,
                      y_original,
                      scale=True,
                      classes=classes_original,
                      colormap=cmap,
                      proj_features=True,
                      ax=axes[1],
                      show=False,
                      projection=dim)
    plt.savefig(outpath)
  else:
    pca_decomposition(x,
                      y,
                      scale=False,
                      classes=classes,
                      colormap=cmap,
                      proj_features=True,
                      projection=dim,
                      show=False)

    if dim == 3:
      outpath = path.joinpath(
          'pca_3d_with_outliers_{}.png'.format(color_feature))
    else:
      outpath = path.joinpath(
          'pca_2d_with_outliers_{}.png'.format(color_feature))

    plt.savefig(outpath)
  plt.close('all')


def joint_plots(df_data: pd.DataFrame,
                labels: dict[str, str],
                path: Path,
                outliers_detection: bool = False,
                df_data_original: Union[pd.DataFrame, None] = None) -> None:
  r"""Plot a joint graph.

  Args:
    df_data: A pandas dataframe with the data to be plotted.
    labels: A dictionary containg the labels of the features plotted.
    path: The path to save the exported graphs.
    outliers_detection: A boolean value indicating if outliers were removed
      from the data. Default: False.
    df_data_original: A pandas dataframe containg the original data in case
      outliers had been removed. It only takes effect if
      'outliers_detection = True'.
  """

  x = df_data.rename(columns=labels)
  cols: list[str] = x.columns.to_list()
  combinations = list(itertools.combinations(cols, 2))

  if outliers_detection and df_data_original is not None:
    x_original = df_data_original.rename(columns=labels)

    for comb in combinations:
      fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
      joint_plot(x, None, columns=comb, ax=axes[0], show=False)
      joint_plot(x_original, None, columns=comb, ax=axes[1], show=False)

      fig.tight_layout()
      fig.subplots_adjust(top=0.90)
      plt.suptitle('x'.join(comb))

      axes[0].set_title("With Outliers", y=1.0, pad=95)
      axes[1].set_title("Without Outliers", y=1.0, pad=95)

      outpath = path.joinpath('joint_plots_{}'.format('x'.join(comb)))
      plt.savefig(outpath)
      plt.close('all')
  else:
    for comb in combinations:
      visualizer = joint_plot(x,
                              None,
                              columns=comb,
                              show=False,
                              size=(1840, 920))

      visualizer.fig.tight_layout()
      visualizer.fig.subplots_adjust(top=0.92)
      plt.suptitle('x'.join(comb))

      outpath = path.joinpath('joint_plots_{}'.format('x'.join(comb)))
      plt.savefig(outpath)
      plt.close('all')


def lisa_plot(df: pd.DataFrame, geo_df: geopandas.GeoDataFrame,
              features: list[str], color_feature: str, path: Path):
  """Plot a lisa autocorrelation plot.

  Args:
    df: A pandas DataFrame containg the data to be analysed.
    geo_df: A geopandas GeoDataFrame containg spatial data to be plotted.
    features: A list of strings containg the features being analysed.
    path: The path to save the exported graph.
  """
  df_spatial = geopandas.GeoDataFrame(
      df.merge(
          geo_df.rename(columns={'code_district': 'id'})[['id', 'geometry']],
          on='id'))

  w_rook = libpysal.weights.KNN.from_dataframe(df_spatial, ids='id')
  lisa = Moran_Local(df_spatial[color_feature], w_rook)
  fig, _ = plot_local_autocorrelation(lisa,
                                      df_spatial,
                                      color_feature,
                                      figsize=(15, 10))
  fig.suptitle(" x ".join(features) +
               "\n Colored by {} \n\n".format(color_feature),
               fontsize=16)

  plt.savefig(path.joinpath(
      'lisa_autocorrelation_{}.png'.format(color_feature)))
  plt.close()


def dot_plot(df: pd.DataFrame, x_vars: list[str], y_vars: list[str], path: Path,
             file_name: str):
  g = sns.PairGrid(df,
                   x_vars=x_vars,
                   y_vars=y_vars,
                   height=0.5 * len(df),
                   aspect=0.40)    #type:ignore

  g.map(sns.stripplot,
        size=10,
        orient='h',
        jitter=False,
        palette='flare_r',
        linewidth=1,
        edgecolor='w')
  g.set(xlabel="Frequency", ylabel="")

  titles = x_vars

  for ax, title in zip(g.axes.flat, titles):    # type: ignore
    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

  sns.despine(left=True, bottom=True)

  outpath = path.joinpath(file_name)
  g.fig.savefig(outpath, bbox_inches='tight')
  plt.close('all')
