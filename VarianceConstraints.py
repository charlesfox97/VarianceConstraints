# -*- coding: utf-8 -*-
"""
This module creates linear estimates of variance as PuLP constriants.
"""

import numpy as np
import pandas as pd
import pulp

import time

_test_link = r'https://web.stanford.edu/~hastie/CASI_files/DATA/'


class VarianceConstraints:

    """
    VarianceConstraints creates linear estimates of variance as PuLP
    constraints.

    The variance estimate is a sum of estimates of principal component
    contributions to variance plus a remainder term for excluded components.

    """

    def __init__(self, covar, variance_variable, X):
        """
        Inputs:

            covar: a covariance matrix
            variance_variable: a pulp variable representing variance
            X: a list of pulp variables corresponding to the covariance matrix

        Properties:
            pca_variance_explained: sorted eigenvalues of covar
            pca_vectors: eigenvectors of covar

           components: a list of Component objects
            nc: number of components
            rc: range of components

            X: the inputted list of variables
            x: the current values of X
            nx: length of X
            rx: range(nx)

            total_constraint: variance estimate = sum of component estimates

            remainder_constraints: if, when building the constraints around a
            solution point, some of the components do not have a significant
            contribution to error, they will be lumped together into a single
            constraint.

        """

        self.covar = covar
        self.variance_variable = variance_variable
        self.X = X

        # covariance -> principal components
        self.pca_variance_explained, self.pca_vectors = decompose(covar)

        # number of variables
        self.nx = covar.shape[0]
        self.rx = range(self.nx)

        # number of components
        self.nc = self.pca_variance_explained.shape[0]
        self.rc = range(self.nc)

        # build component objects
        components = []
        for ic in self.rc:
            components.append(Component(
                    self.pca_variance_explained[ic][0],
                    self.pca_vectors[:, ic], X, ic))

        self.components = components

        # variance = sum of component contributions
        self.total_constraint = pulp.LpConstraint(
            variance_variable - sum([c.contrib_variable for c in components]),
            name="VarianceSumOfComponents")

        self.remainder_constraints = []
        self.remainder_covars = []

    def calc_estimate_at_point(self, x_vals=None):
        """Estimates variance using the current set of constraints.

        Parameters
        ----------
        x_vals : np.ndarray, optional
            The set of x values to test.  Defaults to values of self.X,
            the X variables.

        Returns
        -------
        float
            The estimated variance.
        """

        # Get current solution values if needed.
        if x_vals is None:
            x_vals = self.x

        # Create a new, empty problem
        problem = pulp.LpProblem("test")
        problem.objective = self.variance_variable

        # The only constraints of the new problem are the ones used for
        # estimating variance.
        for c in self.constraints:
            problem += c

        # Constraint all X variables to be equal to their corresponding
        # value in x_vals
        for ix in self.rx:
            problem += pulp.LpConstraint(self.X[ix], rhs=x_vals[ix])

        # Because X[i]==x[i] for all i, only the contribution variables may
        # be optimized.
        problem.solve()

        # The objective is the estimate of variance.
        return problem.objective.value()

    def build_constraints_at_value(self, cutoff=None, problem=None,
                                   x_vals=None):
        """Creates new constraints for variance estimation.

        Components are sorted by contribution to error.  If the total
        contribution to error of two or more components is less than cutoff,
        one constraint is generated to estimate the variance contribution of
        the group of components.  One new constraint is generated for each
        component not belonging to the group of small error contributors.


        Parameters
        ----------
        cutoff : float, optional
            Determines the maximum total error contributed by components that
            are consolidated into a single constraint.  An arbitrarily large
            value for cutoff will cause this function to generate only a single
            constraint.  A zero cutoff will cause the function to create
            one constraint per component with a non-zero contribution to error.

        Returns
        -------
        list(pulp.LpConstraint)
            A set of constraints that together force the variance estimate
            at the current solution to equal the true variance.
        """

        def remainder_covar(ix_components, vectors, variance_explained):
            """
            Calculate covariance matrix of low error components.

            Parameters
            ----------
            ix_components : np.ndarray(int)
                Which components have a low contribution to error.  If all
                components are listed in ix_components, the original
                covariance matrix is returned.

            Returns
            -------
            np.ndarray
                The remainder covariance matrix
            """

            v = vectors[:, ix_components]
            e = np.diag(variance_explained[ix_components, 0])
            return np.matmul(v, np.matmul(e, v.T))

        if (x_vals is not None) and (cutoff is not None):
            self.calc_estimate_at_point(x_vals)

        if x_vals is None:
            x_vals = self.x

        new_c = []
        # If no cutoff is given, create one constraint per component without
        # error testing.
        if cutoff is None:
            components_to_update = self.rc
        else:
            # Sort the errors, identify which components get a new constraint.
            e = self.errors
            error_sort = np.argsort(e)
            cum_e = np.cumsum(e[error_sort])
            components_to_update = error_sort[cum_e >= cutoff]

        # Create a new constraint for each component that is not in the set
        # fitting below the error cutoff.
        for icomponent in components_to_update:
            component = self.components[icomponent]
            c = component.add_estimate(component.calc_load(x_vals))
            new_c.append(c)
            if problem is not None:
                problem += c

        if cutoff is None:
            return new_c

        # Identify the set of components whose total error fits under the
        # cutoff.
        as_single = error_sort[cum_e < cutoff]

        # Create a single constraint to estimate the total variance of this
        # set of components.
        if len(as_single) > 0:
            sg = sum([self.components[ic].contrib_true for ic in as_single])
            expr = sum([self.components[ic].contrib_variable
                       for ic in as_single])

            c = pulp.LpConstraint(expr, sense=1, rhs=-sg)

            remainder = remainder_covar(as_single,
                                        self.pca_vectors,
                                        self.pca_variance_explained)

            for ix in self.rx:
                for ix2 in self.rx:
                    c.addterm(self.X[ix], -2*self.x[ix2]*remainder[ix, ix2])

            problem += c
            self.remainder_constraints.append(c)
            new_c.append(c)
            self.remainder_covars.append(remainder)

        return new_c

    @property
    def variance_estimated(self):
        return sum([self.components[ic].contrib_estimated for ic in self.rc])

    @property
    def variance_true(self):
        x = np.atleast_2d(self.x)
        return float(np.matmul(x, np.matmul(self.covar, x.T)))

    @property
    def errors(self):
        err = np.zeros(self.nc)
        for ic in self.rc:
            err[ic] = self.components[ic].error
        return err

    @property
    def error(self):
        return sum(self.errors)

    @property
    def x(self):
        _x = np.zeros(self.nx)
        for ix in self.rx:
            _x[ix] = self.X[ix].value()
        return _x

    @property
    def constraints(self):
        _constraints = [self.total_constraint]

        for ic in self.rc:
            lc = self.components[ic].constraints
            _constraints += lc
        _constraints += self.remainder_constraints
        return _constraints


class Component:
    """The Component class is used to estimate the contribution to variance
    from a single PCA component.

    ...

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self,
                 variance_explained: float,
                 vector: np.ndarray,
                 X: list,
                 number: int):
        """
        Parameters:
        -----------
            variance_explained : float
                the component's eigenvalue
            vector: the component's eigenvector
            X: a list of LpVariables corresponding to the vector
            number: an identifying number

        Properties:


        """
        sn = str(number)
        self.contrib_variable = pulp.LpVariable("component_contrib_"+sn)

        self.variance_explained = variance_explained
        self.vector = vector

        self.X = X
        self.nx = len(X)
        self.rx = range(self.nx)

        self.loads = {}
        self.add_estimate(0)

    def add_estimate(self, load):
        c = pulp.LpConstraint(
                self.contrib_variable -
                2*self.variance_explained*load*(
                    sum([self.vector[ix]*self.X[ix] for ix in self.rx])),
                rhs=-load**2*self.variance_explained, sense=1)

        self.loads[load] = c
        return c

    def calc_load(self, x):
        xs = np.reshape(x, (1, -1))
        return np.dot(xs, self.vector)[0]

    @property
    def x(self):
        _x = np.zeros(self.nx)
        for ix in self.rx:
            _x[ix] = self.X[ix].value()
        return _x

    @property
    def contrib_estimated(self):
        return self.contrib_variable.value()

    @property
    def contrib_true(self):
        return self.variance_explained*self.load**2

    @property
    def error(self):
        return self.contrib_true - self.contrib_estimated

    @property
    def load(self):
        return self.calc_load(self.x)

    @property
    def constraints(self):

        return self.loads.values()


def decompose(covar, min_to_keep=10**-20):
    e_vals, e_vectors = np.linalg.eigh(covar)
    idx = e_vals.argsort()[::-1]
    e_vals = e_vals[idx]
    e_vectors = e_vectors[:, idx]
    keep = e_vals > min_to_keep
    e_vals = e_vals[keep]
    e_vectors = e_vectors[:, keep]

    return np.atleast_2d(e_vals).T, e_vectors


def ols_from_cov(covar):

    return np.matmul(covar[-1, :-1], np.linalg.inv(covar[:-1, :-1]))


def test(n_vars=10, n_select=3, n_obs=100, max_error=0.05):
    arb_large = 1/n_select

    def test_data():
        try:
            data = pd.read_csv('leukemia_big.csv')
        except FileNotFoundError:
            data = pd.read_csv(_test_link+'leukemia_big.csv')
        x = data.values
        y = np.matrix(['ALL' in c for c in data.columns])*1
        return x, y

    x, y = test_data()
    n_obs = min(n_obs,x.shape[1])
    x = x[:, :n_obs]
    y = y[:, :n_obs]
    x = x - np.repeat(np.atleast_2d(np.mean(x, 1)).T, n_obs, axis=1)
    y = y - np.mean(y)

    x = x / np.repeat(np.atleast_2d(np.std(x, 1, ddof=1)).T, n_obs, axis=1)
    y = y / np.std(y, ddof=1)

    data = np.concatenate((x[:n_vars, :], y))
    covar = np.cov(data)

    nx = covar.shape[0]
    X = [pulp.LpVariable("x_"+str(i)) for i in range(nx)]

    problem = pulp.LpProblem("best_subset")
    variance_variable = pulp.LpVariable("VarianceEstimate")
    problem.setObjective(variance_variable)
    problem += pulp.LpConstraint(X[-1],rhs=-1,name="def_of_y")
    problem += pulp.LpConstraint(variance_variable,sense=-1,rhs=1,name="variance_cap")

    cnz = pulp.LpVariable("count_non_zero")
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

    ve.build_constraints_at_value(cutoff=max_error/2,
                                  x_vals=y_alone,
                                  problem=problem)

    if False:
        ve.add_constraints_at_value(y_alone)
        ve.add_constraints_at_value(-y_alone)

    for c in ve.constraints:
        problem += c

    start_time = time.time()

    problem.solve()
    print(time.time()-start_time)
    bgst = np.max(abs(ve.x[:-1]))
    while (ve.error > max_error) or (bgst >= arb_large*0.5):

        problem.constraints["variance_cap"].changeRHS(ve.variance_true)
        nz = abs(ve.x) > 0

        nz = np.where(nz)[0]
        x_base = ve.x
        xstar = ols_from_cov(ve.covar[nz, :][:, nz])
        x_star = np.zeros(ve.nx)
        x_star[nz[:-1]] = xstar
        x_star[-1] = -1

        ve.calc_estimate_at_point(x_star)
        ve.build_constraints_at_value(cutoff=max_error/2, problem=problem)
        print(time.time()-start_time)
        new_estimate = ve.calc_estimate_at_point(x_base)

        if (new_estimate+max_error) <= ve.variance_true:
            ve.build_constraints_at_value(cutoff=max_error/2, problem=problem)

        if bgst >= arb_large*0.5:
            arb_large = max(bgst*(2.01**0.5), arb_large*1.1)
            define_non_zero(problem, X, arb_large)
            print("expanded arb large to " + str(arb_large))

        problem.solve()
        bgst = np.max(abs(ve.x[:-1]))
        print("\ntime: " + str(np.round(time.time()-start_time, 1)))
        print("\tErr: " + str(np.round(ve.error, 4)))
        print("\tTrue Variance: " + str(np.round(ve.variance_true, 4)))
    print("\tConstraints: " + str(len(problem.constraints)))

    return ve, problem, data


if __name__ == "__main__":
    ve, problem, data = test(n_vars=15, n_select=5, max_error=0.01)












