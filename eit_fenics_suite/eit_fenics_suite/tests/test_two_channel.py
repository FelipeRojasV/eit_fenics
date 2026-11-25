import numpy as np

from eit_lib import TwoChannelEIT


def test_roundtrip_recovers_sigma_epsilon():
    two_ch = TwoChannelEIT(nx=16, ny=16, electrode_width=0.4)

    sigma_true = 0.25
    epsilon_true = 5000.0 * 8.854e-12
    omega = 2.0 * np.pi * 50e3

    V_meas, _ = two_ch.forward_solve(sigma_true, epsilon_true, omega)
    sigma_rec, eps_rec, gamma_rec = two_ch.inverse_solve(V_meas, omega)

    assert np.isclose(sigma_rec, sigma_true, rtol=1e-10)
    assert np.isclose(eps_rec, epsilon_true, rtol=1e-10)
    assert np.isclose(gamma_rec.real, sigma_true, rtol=1e-10)