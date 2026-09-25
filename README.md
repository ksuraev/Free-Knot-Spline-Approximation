# Free-knot Chebyshev approximation by continuous polynomial splines

Approximating a continuous function by a polynomial or piecewise polynomial in the Chebyshev sense means minimising the maximum absolute approximation error over an interval. For polynomial splines, the points joining neighbouring polynomial pieces are called knots.

This project focuses on continuous free-knot polynomial splines of highest defect. In this setting, both the spline coefficients and knot locations are optimisation variables, giving a nonconvex and nondifferentiable problem.

Our approach reformulates the problem as a bilevel optimisation problem. For fixed knots, the lower-level problem computes the best Chebyshev spline approximation using modifications of existing algorithms. The upper-level problem then optimises the knot locations using a descent method, with quasidifferential calculus used to determine descent directions.
