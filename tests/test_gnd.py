import numpy as np

from pyhrebsd.gnd import (dislocation_matrix, fcc_perfect_dislocations,
                         partial_nye_from_maps,
                         silicon_perfect_dislocations, solve_gnd_l1)


def test_silicon_basis_has_expected_burgers_lengths_and_rank():
    types = silicon_perfect_dislocations()
    assert sum(t.kind == "edge" for t in types) == 12
    assert sum(t.kind == "screw" for t in types) == 6
    for kind in types:
        np.testing.assert_allclose(np.linalg.norm(kind.burgers_crystal_m),
                                   0.5431e-9 / np.sqrt(2))
        np.testing.assert_allclose(np.linalg.norm(kind.line_crystal), 1)
        if kind.kind == "edge":
            np.testing.assert_allclose(np.dot(kind.burgers_crystal_m,
                                               kind.line_crystal), 0, atol=1e-24)
    assert np.linalg.matrix_rank(dislocation_matrix(types, np.eye(3))) == 6


def test_nickel_fcc_basis_uses_h5_lattice_parameter():
    types = fcc_perfect_dislocations(3.57e-10)
    assert len(types) == 18
    np.testing.assert_allclose(np.linalg.norm(types[0].burgers_crystal_m),
                               3.57e-10 / np.sqrt(2))


def test_partial_nye_from_linear_rotation_field():
    y, x = np.mgrid[:3, :4]
    w = np.stack((0.002 * y, 0.003 * x, 0.004 * x + 0.005 * y), axis=-1) * 1000
    observed = partial_nye_from_maps(w, 2e-6, 4e-6)
    expected = np.array([-1500, -2000, -500, -1250, 0, 0], dtype=float)
    np.testing.assert_allclose(observed[:-1, :-1], np.broadcast_to(expected, (2, 3, 6)))
    assert np.isnan(observed[-1]).all()
    assert np.isnan(observed[:, -1]).all()


def test_gradient_span_keeps_linear_curvature_and_expands_border():
    y, x = np.mgrid[:8, :9]
    w = np.stack((0.002 * y, 0.003 * x, 0.004 * x + 0.005 * y), axis=-1) * 1000
    near = partial_nye_from_maps(w, 2e-6, 4e-6)
    wide = partial_nye_from_maps(w, 2e-6, 4e-6, gradient_span=3)
    np.testing.assert_allclose(wide[:-3, :-3], near[:-3, :-3], atol=1e-9)
    assert np.isnan(wide[-3:]).all()
    assert np.isnan(wide[:, -3:]).all()


def test_partial_nye_includes_measured_strain_gradient():
    rotation = np.zeros((3, 4, 3))
    strain = np.zeros((3, 4, 3, 3))
    strain[:, :, 0, 1] = np.arange(4)[None, :] * 0.001
    strain[:, :, 1, 0] = strain[:, :, 0, 1]
    observed = partial_nye_from_maps(rotation, 2e-6, 4e-6, strain)
    np.testing.assert_allclose(observed[:-1, :-1, 1], 500)
    np.testing.assert_allclose(observed[:-1, :-1, [0, 2, 3, 4, 5]], 0)


def test_l1_solution_reconstructs_six_observables():
    types = silicon_perfect_dislocations()
    matrix = dislocation_matrix(types, np.eye(3))
    truth = np.zeros(len(types))
    truth[0], truth[7], truth[14] = 2e12, -1e12, 3e12
    observed = matrix @ truth
    solved, residual = solve_gnd_l1(observed, matrix, types)
    np.testing.assert_allclose(matrix @ solved, observed, atol=1e-4)
    np.testing.assert_allclose(residual, 0, atol=1e-4)
