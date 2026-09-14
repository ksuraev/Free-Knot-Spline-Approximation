import Spline
import numpy as np


def test_polynomial():
    c = np.array([1., -1., 2., -2.])
    p = Spline.Polynomial(c, offset=0.5)
    assert p(1.5) == 0 and p(-0.5) == 6 and p(2.5) == -9
    return p


def create_Spline_from_polynomial():
    c = [
        Spline.Polynomial(np.array([1., -1., 2.])),
        Spline.Polynomial(np.array([1., 1., -2.])),
        ]
    knots = np.array([-1., 0., 1.])
    S = Spline.Spline(knots, c)
    return S, c


def create_Spline_from_arrays():
    c = [
        np.array([1., -1., 2.]),
        np.array([1., 1., -2.]),
        ]
    knots = np.array([-1., 0., 1.])
    S = Spline.Spline(knots, c)
    return S


def test_Spline_from_polynomials():
    S, c = create_Spline_from_polynomial()
    assert (S(-1) == c[0](-1)) and (
            S(-0.5) == c[0](-0.5)) and (
            S(0) == c[1](0)) and (
            S(0.5) == c[1](0.5)) and (
            S(1) == c[1](1))


def test_Concatenate_Splines():
    S, c = create_Spline_from_polynomial()
    S1 = Spline.Spline(S.knots[:-1], [c[0]])
    S2 = Spline.Spline(S.knots[1:], [c[1]])
    S3 = S1.concatenate(S2)
    assert (S(-1) == S3(-1)) and (
            S(-0.5) == S3(-0.5)) and (
            S(0) == S3(0)) and (
            S(0.5) == S3(0.5)) and (
            S(1) == S3(1))


def test_Spline_from_arrays():
    S1 = create_Spline_from_arrays()
    S2, _ = create_Spline_from_polynomial()
    assert S1(-1) == S2(-1)
    assert S1(-0.5) == S2(-0.5)
    assert S1(0) == S2(0)
    assert S1(0.5) == S2(0.5)
    assert S1(1) == S2(1)


def test_SUSpline_from_Spline():
    S1, _ = create_Spline_from_polynomial()
    S2 = S1.to_SUSpline()
    assert S1(-1.0) == S2(-1.0)
    assert S1(-0.5) == S2(-0.5)
    assert S1(-0.25) == S2(-0.25)
    assert S1(0.) == S2(0.)
    assert S1(0.5) == S2(0.5)
    assert S1(0.75) == S2(0.75)
    assert S1(1.) == S2(1.)
    return S2

def test_SSpline_from_Spline():
    S1, _ = create_Spline_from_polynomial()
    S2 = S1.to_SSpline()
    assert S1(-1.0) == S2(-1.0)
    assert S1(-0.5) == S2(-0.5)
    assert S1(-0.25) == S2(-0.25)
    assert S1(0.) == S2(0.)
    assert S1(0.5) == S2(0.5)
    assert S1(0.75) == S2(0.75)
    assert S1(1.) == S2(1.)
    return S2

def test_Spline_rebase():
    S1, _ = create_Spline_from_polynomial()
    S2 = S1.rebase(-1)
    assert S1(-1.0) == S2(-1.0)
    assert S1(-0.5) == S2(-0.5)
    assert S1(-0.25) == S2(-0.25)
    assert S1(0.) == S2(0.)
    assert S1(0.5) == S2(0.5)
    assert S1(0.75) == S2(0.75)
    assert S1(1.) == S2(1.)
    return S2


if __name__ == "__main__":
    p = test_polynomial()
    test_Spline_from_polynomials()
    test_Spline_from_arrays()
    test_Concatenate_Splines()
    S2 = test_SUSpline_from_Spline()
    S3 = test_SSpline_from_Spline()
    S4 = test_Spline_rebase()
