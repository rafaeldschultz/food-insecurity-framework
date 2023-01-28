# -*- coding: utf-8 -*-
"""Executes MOEAs (MultiObjective Evolutionary Algorithms)
This module contains functions to execute MOEAs in order to obtain the
Pareto's Fronts.
"""

import array
import copy
from typing import Any, Generator, Union

import pandas as pd
from deap import algorithms, base, creator, tools

import user_inputs as inputs


def evaluate_indivs(individual: Any,
                    df: pd.DataFrame) -> list[Union[float, str]]:
    r"""Evaluate individuals from a population.

  Args:
    individual: the individual to be evaluated.
    df: a pandas DataFrame containing the data to be used to evaluate.

  Returns:
    A list of scores of the individual for each feature.
  """

    fvec = list(df.loc[individual[0]])
    return fvec


def allintallele(bound_low: int,
                 bound_up: int) -> Generator[list[int], None, None]:
    r"""Index individuals dinamically.

  Args:
    bound_low: the index of the first individual.
    bound_up: the index of the last individual.

  Returns:
    The index of the individual.
  """

    i = bound_low
    while True:
        yield [i % bound_up]
        i += 1


def run_ea(toolbox: base.Toolbox,
           stats: Union[tools.Statistics, None] = None,
           verbose: bool = False) -> tuple[Any, tools.Logbook]:
    r"""Run the deap's eaMuPlusLambda algorithm in the population.

  Args:
    toolbox: A toolbox containing the evolutionary operators.
    stats: An object containg statistic functions.
    verbose: A boolean value indicating weather or not to log the statistics.

  Returns:
    The final population obtained using the deap's eaMuPlusLambda algorithm.
	"""

    pop = toolbox.population(n=toolbox.pop_size)  # type:ignore
    pop = toolbox.select(pop, len(pop))  # type:ignore

    # (mu + lambda) -> versao de algoritmos evolutivos em que os filhos e
    # pais sao adicionados a populacao. Retorna a populacao final.

    return algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=toolbox.pop_size,  # type:ignore
        lambda_=toolbox.pop_size,  # type:ignore
        cxpb=1 - toolbox.mut_prob,  # type:ignore
        mutpb=toolbox.mut_prob,  # type:ignore
        stats=stats,
        ngen=toolbox.max_gen,  # type:ignore
        verbose=verbose)


def initialize_moea(
    dataframe: pd.DataFrame,
    n_dim: int,
    max_gen: int,
    bound_up: int,
    bound_low: int,
    weights: list[int],
    allint: Generator[list[int], None, None],
    func_id: int,
) -> tuple[base.Toolbox, tools.Statistics, list]:
    r"""Initialize the toolbox needed to run the moea.

  Args:
    dataframe: A pandas dataframe containing the data of the individuals.
    n_dim: An integer indicating the number of dimensions in the problem.
    max_gen: An integer indicating the maximum number of generations that the
      moea should produce.
    bound_up: An integer indicating the index of the first individual.
    bound_low: An integer indicating the index of the last individual.
    weights: A list containing the weights that should be used to each of
      the considered variables. Positive values indicates 'maximization'
      and negative values indicates 'minimization'.
    allint: A generator of integers to generate the index of individuals.
    func_id: A id for generating multiple fitness' and individuals functions in
      deap.

  Returns:
    A tuple containing a toolbox that has the necessary functions to execute
    the moea along with a box that has the statistics functions and an initial
    population.
	"""

    toolbox = base.Toolbox()

    fitness_name = 'Fitness_' + str(func_id)
    individual_name = 'Individual_' + str(func_id)

    c = getattr(creator, fitness_name, None)
    if c is not None:
        delattr(creator, fitness_name)
    c = getattr(creator, individual_name, None)
    if c is not None:
        delattr(creator, individual_name)

    creator.create(fitness_name, base.Fitness, weights=weights)

    creator.create(individual_name,
                   array.array,
                   typecode='f',
                   fitness=getattr(creator, fitness_name))

    toolbox.register('evaluate', lambda ind: evaluate_indivs(ind, dataframe))

    toolbox.register('attr_int', lambda: next(allint))
    toolbox.register('individual', tools.initIterate,
                     getattr(creator, individual_name),
                     toolbox.attr_int)  # type:ignore

    toolbox.register('population', tools.initRepeat, list,
                     toolbox.individual)  # type:ignore
    toolbox.pop_size = dataframe.shape[0]  # type:ignore
    pop = toolbox.population(n=toolbox.pop_size)  # type:ignore

    toolbox.register('select', tools.selNSGA2)
    toolbox.register('mate',
                     tools.cxSimulatedBinaryBounded,
                     low=bound_low,
                     up=bound_up,
                     eta=20.0)
    toolbox.register('mutate',
                     tools.mutPolynomialBounded,
                     low=bound_low,
                     up=bound_up,
                     eta=20.0,
                     indpb=1.0 / n_dim)

    toolbox.max_gen = max_gen  # type:ignore
    toolbox.mut_prob = 0.2  # type:ignore

    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)

    return toolbox, stats, pop


def build_fronts(n_dim: int, dataframe: pd.DataFrame, res: list,
                 toolbox: base.Toolbox) -> tuple[list, pd.DataFrame]:
    r"""Classifies every individual in its respective Pareto's Front.

  Args:
    n_dim: An integer indicating the number of dimensions in the problem.
    dataframe: A pandas dataframe containing the data of the individuals.
    res: A list containing the population used to build the Pareto's Fronts.
    toolbox: A deap Toolbox containing functions needed to execute the moea.

  Returns:
    A tuple containing the fronts of the individuals and a pandas dataframe
    containing all the data from the original dataframe of individuals
    passed as argument plus the calculated nsga2 rank and the Pareto's
    Front for each individual.
  """

    # generating visualizations
    fronts = tools.sortNondominated(res, k=len(res))

    evaluations = [toolbox.evaluate(ind) for ind in res]  # type:ignore
    df = pd.DataFrame(evaluations,
                      columns=['f' + str(i + 1) for i in range(n_dim)])

    nsga2_sel = tools.selNSGA2(res, len(res))
    ev_all_points = [toolbox.evaluate(ind) for ind in nsga2_sel]  # type:ignore

    df_all = pd.DataFrame(list(ev_all_points), columns=df.columns)

    peeloff = [g[0] for g in nsga2_sel]
    df_all['id'] = [dataframe.iloc[int(i)]['id'] for i in peeloff]
    df_all['id'] = df_all['id'].astype(int)

    df_all['nsga2_rank'] = list(df_all.index + 1)
    df_all['nsga2_rank'] = df_all['nsga2_rank'].astype(int)

    fronts = tools.sortNondominated(res, k=len(res))

    df_all['front'] = [e for e, f in enumerate(fronts) for i in f]

    df_all = df_all.merge(dataframe[['id', 'id_name']], how='outer', on='id')

    return fronts, df_all


def run_moea(df: pd.DataFrame, n_dim: int, weights: list[int],
             id_analysis: int) -> tuple[pd.DataFrame, list[int]]:
    r"""Executes the MOEA for classification using Pareto's Fronts.

  Args:
    df: a pandas DataFrame containg the data to be used in the MOEA.
    n_dim: a integer indicating the number of dimensions of the problem.
    weights: a list of size 'n_dim' containg the weights to be used in the
      problem.
    id_analysis: a integer indicating the identification of the analisys. Can
      be sequential and prevents overwrite in DEAP functions.

  Returns:
    A pandas dataframe containg the original data classified in fronts and a
    list of integers representing each df's tuple front.
  """
    bound_low, bound_up = 0, df.shape[0] - 1
    allint = allintallele(bound_low, bound_up + 1)

    toolbox, stats, pop = initialize_moea(
        df[['f' + str(i + 1) for i in range(n_dim)]],
        n_dim,
        inputs.max_gen,
        bound_up,
        bound_low,
        weights,
        allint=allint,
        func_id=id_analysis)

    res, _ = run_ea(toolbox, stats=stats)

    fronts, df_all = build_fronts(n_dim, df, res, toolbox)

    del toolbox, allint, pop, stats

    return (df_all, fronts)
