# -*- coding: utf-8 -*-

import os

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pulp

pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, pkg_path)
from VarianceConstraints import VarianceConstraints  # noqa: E402

_test_link = r'https://web.stanford.edu/~hastie/CASI_files/DATA/'


def test_data(n_obs=None, n_vars=None):

    try:
        data = pd.read_csv('leukemia_big.csv')
    except FileNotFoundError:
        print("from remote")
        data = pd.read_csv(_test_link+'leukemia_big.csv')

    x = data.values
    y = np.matrix(['ALL' in c for c in data.columns])*1

    if n_obs is not None:
        x = x[:, :n_obs]
        y = y[:, :n_obs]

    n_obs = x.shape[1]

    if n_vars is not None:
        x = x[:n_vars, :]

    x = x - np.repeat(np.atleast_2d(np.mean(x, 1)).T, n_obs, axis=1)
    y = y - np.mean(y)

    x = x / np.repeat(np.atleast_2d(np.std(x, 1, ddof=1)).T, n_obs, axis=1)
    y = y / np.std(y, ddof=1)
    data = np.concatenate((x, y))
    covar = np.cov(data)
    return covar


def timed_solve(problem):
    start_time = time.time()
    result = problem.solve()
    solve_time = time.time()-start_time
    return result, solve_time


def best_subset(covar, n_select, max_time=300, constraint_cutoff=10**-8,
                print_iter=False):
    arb_large = 1/n_select

    nx = covar.shape[0]
    X = [pulp.LpVariable("x_"+str(i)) for i in range(nx)]

    problem = pulp.LpProblem("best_subset")
    variance_variable = pulp.LpVariable("VarianceEstimate")

    problem += pulp.LpConstraint(X[-1], rhs=-1, name="def_of_y")
    problem += pulp.LpConstraint(variance_variable, sense=-1, rhs=1,
                                 name="variance_cap")

    cnz = pulp.LpVariable("count_non_zero")
    problem.setObjective(variance_variable)
    problem += pulp.LpConstraint(cnz, sense=-1, rhs=n_select,
                                 name="count_non_zero_cap")

    def_cnz = pulp.LpConstraint(cnz, sense=1, name="def_count_non_zero")
    problem += def_cnz

    for ix in range(nx-1):
        six = str(ix)
        xnz = pulp.LpVariable("X_non_zero_" + six,
                              cat="Integer", lowBound=0, upBound=1)
        def_cnz.addterm(xnz, -1)

    def define_non_zero(problem, X, arb_large):
        pvd = problem.variablesDict()

        for ix in range(len(X)-1):
            six = str(ix)

            if "xp_" + six in problem.constraints:
                problem.constraints.pop("xp_"+six)
                problem.constraints.pop("xn_"+six)

            xnz = pvd["X_non_zero_"+six]
            problem += pulp.LpConstraint(xnz*arb_large - X[ix],
                                         sense=1, name="xp_"+six)

            problem += pulp.LpConstraint(-xnz*arb_large - X[ix],
                                         sense=-1, name="xn_"+six)

    define_non_zero(problem, X, arb_large)
    ve = VarianceConstraints(covar, variance_variable, X)

    y_alone = np.zeros((nx))
    y_alone[-1] = 1

    ve.build_constraints_at_value(cutoff=constraint_cutoff/2,
                                  x_vals=y_alone,
                                  problem=problem)

    for c in ve.constraints:
        problem += c

    start_time = time.time()
    iteration = 0
    summary = {}
    while (time.time()-start_time) < max_time:
        iteration_summary = {}
        summary[iteration] = iteration_summary

        iteration_summary['constraints'] = len(problem.constraints)
        iteration_summary['arb_large'] = arb_large
        result, iteration_summary['solve_time'] = timed_solve(problem)
        iteration_summary['cumulative_time'] = time.time()-start_time
        if result == -1:
            print("Infeasible")
            break

        max_abs_x = max(abs(ve.x[:-1]))
        iteration_summary['estimate'] = problem.objective.value()
        iteration_summary['solution_true_variance'] = ve.variance_true

        ve.build_constraints_at_value(cutoff=constraint_cutoff,
                                      problem=problem)

        iteration_summary['x_star_variance'], x_star = \
            ve.build_constraints_at_subset_regression(
                                            cutoff=constraint_cutoff,
                                            problem=problem)

        max_abs_xstar = max(abs(x_star[:-1]))

        arb_large = (max_abs_xstar + max_abs_x) / 2 * 1.1
        define_non_zero(problem, X, arb_large)

        iteration_summary['max_abs_xstar'] = max_abs_xstar
        iteration_summary['max_abs_x'] = max_abs_x

        if print_iter:
            print(iteration_summary)
        iteration += 1
    summary = pd.DataFrame(summary).T
    return ve, problem, summary


if __name__ == "__main__":
    problems = [
            {'n_vars': 10, 'n_select': 2, 'max_time': 10},
            {'n_vars': 30, 'n_select': 5, 'max_time': 60},
            {'n_vars': 50, 'n_select': 10, 'max_time': 180}]

    summaries = []

    plotfields = [
                ['estimate', 'r', 'estimate'],
                ['solution_true_variance', 'k', 'solution true'],
                ['x_star_variance', 'g', 'x star']]

    for params in problems:
        covar = test_data(n_vars=params['n_vars'])
        ve, problem, summary = best_subset(covar, n_select=params['n_select'],
                                           max_time=params['max_time'],
                                           print_iter=True)

        fig, ax = plt.subplots()
        ax.set_title(str(params))
        for series in plotfields:

            summary.iloc[1:, :].plot.scatter(
                    x='cumulative_time',
                    y=series[0], color=series[1],
                    label=series[2], ax=ax)

        ax.legend()
        ax.set_ylabel('Variance')
        fig.show()

        summaries.append(summary)
