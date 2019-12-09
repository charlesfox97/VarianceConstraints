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
    x = x
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

    nx = covar.shape[0]
    arb_large = 1 / nx
    X = [pulp.LpVariable("x_"+str(i)) for i in range(nx)]

    problem = pulp.LpProblem("best_subset")
    variance_variable = pulp.LpVariable("VarianceEstimate")

    problem += pulp.LpConstraint(X[-1], rhs=-1, name="def_of_y")
    problem += pulp.LpConstraint(variance_variable, sense=-1, rhs=1,
                                 name="variance_cap")

    problem.setObjective(variance_variable)

    cnz = pulp.LpVariable("count_non_zero")
    problem += pulp.LpConstraint(cnz, sense=-1, rhs=n_select,
                                 name="count_non_zero_cap")
    def_cnz = pulp.LpConstraint(cnz, sense=1, name="def_count_non_zero")

    ta = pulp.LpVariable("total_abs_x")
    def_ta = pulp.LpConstraint(ta, sense=0, name="def_total_abs_x")
    cap_total = n_select*arb_large
    problem += pulp.LpConstraint(ta, sense=-1, rhs=n_select*arb_large,
                                 name="total_abs_x_cap")

    for ix in range(nx-1):
        six = str(ix)

        xnz = pulp.LpVariable("non_zero_" + six,
                              cat="Integer", lowBound=0, upBound=1)

        pos = pulp.LpVariable("pos_part_"+six, lowBound=0)
        neg = pulp.LpVariable("neg_part_"+six, lowBound=0)
        abs_x = pulp.LpVariable("abs_" + six)

        problem += pulp.LpConstraint(pos+neg-abs_x)
        problem += pulp.LpConstraint(X[ix] - pos + neg)

        def_cnz.addterm(xnz, -1)
        def_ta.addterm(abs_x, -1)

    problem += def_cnz
    problem += def_ta

    def define_non_zero(problem, X, arb_large, pvd):
        for ix in range(len(X)-1):
            six = str(ix)
            xnz = pvd["non_zero_" + six]
            pos = pvd["pos_part_" + six]
            neg = pvd["neg_part_"+six]

            xpdn = "xp_"+six
            if xpdn in problem.constraints:
                problem.constraints.pop(xpdn)
                problem.constraints.pop("xn_"+six)

            problem += pulp.LpConstraint(xnz*arb_large - pos,
                                         sense=1, name=xpdn)

            problem += pulp.LpConstraint(xnz*arb_large - neg,
                                         sense=1, name="xn_"+six)

    pvd = problem.variablesDict()
    define_non_zero(problem, X, arb_large, pvd)
    ve = VarianceConstraints(covar, variance_variable, X)

    # TODO: find a systematic way to determine pre-solve cuts
    # TODO: find a systematic way to tune constraint_cutoff
    y_alone = np.zeros((nx))
    y_alone[-1] = 1
    for factor in [2, 1, 0.5, 0.25, 0.125]:

        ve.build_constraints_at_value(cutoff=None,
                                      x_vals=y_alone*factor,
                                      problem=problem,
                                      symmetric=True)

    for c in ve.constraints:
        problem += c

    start_time = time.time()
    iteration = 1
    summary = {}
    best = {'var': 1, 'x': y_alone, 'iteration': 0}

    while (time.time()-start_time) < max_time:
        iteration_summary = {}
        summary[iteration] = iteration_summary

        iteration_summary['constraints'] = len(problem.constraints)
        iteration_summary['arb_large'] = arb_large
        result, iteration_summary['solve_time'] = timed_solve(problem)
        iteration_summary['cumulative_time'] = time.time()-start_time
        if result == -1:
            print("Infeasible")
            print(problem.constraints['variance_cap'])
            print(problem.constraints['total_abs_x_cap'])
            print(problem.constraints["xp_0"])
            break
        if not(result == 1):
            print("Unexpected result")
            print(result)
            raise

        max_abs_x = float(max(abs(ve.x[:-1])))
        iteration_summary['estimate'] = problem.objective.value()
        iteration_summary['solution_true_variance'] = ve.variance_true

        ve.build_constraints_at_value(cutoff=constraint_cutoff,
                                      problem=problem,
                                      symmetric=True)

        iteration_summary['x_star_variance'], x_star = \
            ve.build_constraints_at_subset_regression(
                                            cutoff=constraint_cutoff,
                                            problem=problem)

        max_abs_xstar = float(max(abs(x_star[:-1])))
        if iteration_summary['x_star_variance'] < best['var']:
            best['var'] = iteration_summary['x_star_variance']
            best['x'] = x_star
            best['iteration'] = iteration
            print("new_best")
        elif iteration_summary['estimate'] >= best['var']:
            print("estimate was not better than best")

            # TODO: find a better tuning strategy for arb_large
            arb_large *= 2
            cap_total *= 2

            problem.constraints['total_abs_x_cap'].changeRHS(cap_total)

            define_non_zero(problem, X, arb_large, pvd)
        else:
            print("Not improved, but estimate was lower than best found")

        iteration_summary['max_abs_xstar'] = max_abs_xstar
        iteration_summary['max_abs_x'] = max_abs_x

        if print_iter:
            print(iteration)
            for k in iteration_summary:
                print("\t"+k+": " + str(np.round(iteration_summary[k], 4)))

        iteration += 1
    summary = pd.DataFrame(summary).T
    return ve, problem, summary


if __name__ == "__main__":
    problems = [{'n_vars': 50, 'n_select': 15, 'max_time': 30}]

    summaries = []
    #  Todo: I'm not sure how to tune this parameter
    constraint_cutoff = 10**-10
    plotfields = [
                ['estimate', 'r', 'estimate'],
                ['solution_true_variance', 'k', 'solution true'],
                ['x_star_variance', 'g', 'x star']]

    for params in problems:
        covar = test_data(n_vars=params['n_vars'])
        ve, problem, summary = best_subset(covar,
                                           constraint_cutoff=constraint_cutoff,
                                           n_select=params['n_select'],
                                           max_time=params['max_time'],
                                           print_iter=True)

        fig, ax = plt.subplots()
        ax.set_title(str(params))
        for series in plotfields:

            summary.plot.scatter(
                    x='cumulative_time',
                    y=series[0], color=series[1],
                    label=series[2], ax=ax)

        ax.legend()
        ax.set_ylabel('Variance')

        summaries.append(summary)
