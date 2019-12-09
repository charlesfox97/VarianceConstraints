# VarianceConstraints
This class builds PuLP constraints for variance minimization problems.

Rather than apply cuts on the covariance matrix directly, the cuts are applied on the principal components.  

Suppose we want to minimize the sum of a vector X:

    Variance = X * Cov(X) * X.T 
    = X * (Eigenvectors * Eigenvalues * Eigenvectors.T) * X.T 
    = sum[Eigenvalue_i * (X*Eigenvector_i)^2]
    = sum[Component_Contribution_i]
    
    Component_Contribution_i = Eigenvalue_i * (X*Eigenvector_i)^2

Near a reference vector X_0:
       
      cc_0 = Eigenvalue_i * (X_0*Eigenvector_i)^2
      Component_Contribution_i >= cc_0 + 2 * Eigenvalue_i * (X_0*Eigenvector_i)*(X*Eigenvector_i-X_0*Eigenvector_i)
    
 Note that there are many potential X_0 that have an equal value for X_0 * Eigenvector_i.  The above estimate for the component's contribution to variance does not depend on X_0.  It only depends on X_0 * Eigenvector_i.  This allows this approximation to be useful near many different X_0.  It is an approximation near a load value.  Substituting:
 
    L_0 = X_0*Eigenvector_i
    Component_Contribution_i >= Eigenvalue_i*L^2 + 2*Eigenvalue_i*L*(X*Eigenvector_i - L)
    Component_Contribution_i >= Eigenvalue_i*(2*L*X*Eigenvector_i - L^2)
    
This allows each component's contribution to be estimated independently.  We may choose to build more constraints (more L values) for estimating the first component than for the smallest.  By always building the constraint for -L when the constraint for L is created, the minimum for the estimate of the component will occur when X*Eigenvector_i = 0, which is consistent with the true contribution of the component.
