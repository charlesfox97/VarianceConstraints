# -*- coding: utf-8 -*-
"""
This module creates linear estimates of variance as PuLP constriants.
"""

import numpy as np
import pulp


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
        self.X = np.atleast_2d(np.array(X)).T

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

        def get_value(X):
            return X.value()
        self.fvx = np.vectorize(get_value)

    def calc_variance_true(self, x_vals):
        x = np.atleast_2d(x_vals)
        return float(np.matmul(x, np.matmul(self.covar, x.T)))

    def solve_to_estimate(self, x_vals=None):
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
            problem += pulp.LpConstraint(self.X[ix][0], rhs=x_vals[ix])

        # Because X[i]==x[i] for all i, only the contribution variables may
        # be optimized.
        problem.solve()

        # The objective is the estimate of variance.
        return problem.objective.value()

    def calc_estimate(self, x_vals=None):
        x_vals = self.x if x_vals is None else x_vals
        estimates = [self.components[ic].calc_contrib_estimated(x_vals)
                     for ic in self.rc]
        return sum(estimates)

    def calc_errors(self, x=None):
        x = self.x if x is None else x
        errors = [self.components[ic].calc_error(x)
                  for ic in self.rc]
        return np.array(errors)

    def build_constraints_at_value(self, cutoff=None, problem=None,
                                   x_vals=None, symmetric=False):
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
        arb_small = 10**-18

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

        x = self.x if x_vals is None else x_vals

        new_c = []
        # If no cutoff is given, create one constraint per component without
        # error testing.
        if cutoff is None:
            components_to_update = self.rc
        else:
            # Sort the errors, identify which components get a new constraint.
            e = self.calc_errors(x)

            error_sort = np.argsort(e)

            cum_e = np.cumsum(e[error_sort])
            components_to_update = error_sort[cum_e >= cutoff]

        # Create a new constraint for each component that is not in the set
        # fitting below the error cutoff.
        for icomponent in components_to_update:
            component = self.components[icomponent]
            ld = component.calc_load(x)
            if ld**2 * component.variance_explained > arb_small:
                c = component.add_estimate(ld)
                new_c.append(c)
                if symmetric:
                    c = component.add_estimate(-ld)
                    new_c.append(c)
            else:
                if len(component.load_points) == 0:
                    c = component.add_estimate(0)
                    new_c.append(c)

        def cat_all(problem,new_c):
            if problem is not None:
                for c in new_c:
                    problem.addConstraint(c)

        cat_all(problem,new_c)
        if cutoff is None:
            return new_c

        # Identify the set of components whose total error fits under the
        # cutoff.
        as_single = error_sort[(cum_e < cutoff) & (e[error_sort] > arb_small)]

        # Create a single constraint to estimate the total variance of this
        # set of components.
        if len(as_single) > 0:
            sg = sum([self.components[ic].contrib_true for ic in as_single])
            expr = sum([self.components[ic].contrib_variable
                       for ic in as_single])

            remainder = remainder_covar(as_single,
                                        self.pca_vectors,
                                        self.pca_variance_explained)


            expr += -2*np.matmul(x.T,np.matmul(remainder,self.X))
            c = pulp.LpConstraint(expr, sense=1, rhs=-sg)

            problem += c
            self.remainder_constraints.append(c)
            new_c.append(c)
            self.remainder_covars.append(remainder)

        return new_c

    def build_constraints_at_subset_regression(self, cutoff=None,
                                               problem=None):

        # Identify which variables are non-zero
        nz = np.where(abs(self.x) > 0)[0]

        # Use ordinary least squares with the non-zero variables to find the
        # optimal weighting for this set.
        xstar = ols_from_cov(self.covar[nz, :][:, nz])
        x_star = np.zeros(self.nx)
        x_star[nz[:-1]] = xstar
        x_star[-1] = -1

        # Build constraints around the optimal weighting.
        self.build_constraints_at_value(cutoff=cutoff, problem=problem,
                                        x_vals=x_star,
                                        symmetric=True)

        return self.calc_variance_true(x_star), x_star

    @property
    def variance_estimated(self):
        return sum([self.components[ic].contrib_estimated for ic in self.rc])

    @property
    def variance_true(self):
        x = self.x
        return float(np.matmul(x.T, np.matmul(self.covar, x)))

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
        return self.fvx(self.X)

    @property
    def constraints(self):
        _constraints = [self.total_constraint]

        for ic in self.rc:
            lc = self.components[ic].constraints
            _constraints += lc
        _constraints += self.remainder_constraints
        return _constraints

    def contributions(self, x=None):
        contributions = np.zeros(self.nc)
        for c in self.rc:
            contributions[c] = self.components[c].calc_contrib(x)
        return contributions



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
                 X,
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
        self.vector = np.atleast_2d(vector)

        self.X = np.atleast_2d(X).T
        self.vX = np.dot(self.vector,self.X)
        self.nx = len(X)
        self.rx = range(self.nx)

        self.loads = {}
        self.load_points = np.array([])

        def get_value(X):
            return X.value()
        self.fvx = np.vectorize(get_value)

    def add_estimate(self, load):
        c = pulp.LpConstraint(
                self.contrib_variable -
                2*self.variance_explained*load*self.vX,
                rhs=-load**2*self.variance_explained, sense=1)
        # sum([self.vector[ix]*self.X[ix] for ix in self.rx])),
        self.loads[load] = c
        self.load_points = np.append(self.load_points,load)
        return c

    def calc_load(self, x):
        return float(np.matmul(self.vector, x))

    def calc_contrib_estimated(self, x=None):
        x = self.x if x is None else x
        load = self.calc_load(x)
        ld_pts = self.load_points
        load_sqd_est = ld_pts**2 + 2*(load - ld_pts)*ld_pts
        load_sqd_est = np.max(load_sqd_est)
        return load_sqd_est*self.variance_explained

    def calc_contrib(self, x=None):
        x = self.x if x is None else x
        ld = self.calc_load(x=x)
        return ld**2 * self.variance_explained

    def calc_error(self, x=None):
        x = self.x if x is None else x
        return self.calc_contrib(x) - self.calc_contrib_estimated(x)

    @property
    def x(self):
        return self.fvx(self.X)

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
    """Calculate the regression coefficients"""
    # the last column of covar is Y
    return np.matmul(covar[-1, :-1], np.linalg.inv(covar[:-1, :-1]))


if __name__ == "__main__":
    pass
