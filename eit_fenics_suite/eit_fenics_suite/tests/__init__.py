"""
exercises/ex5_two_channel_eit.py
Ejercicio 5: Example 1.5 — Two-channel EIT (Ammari, pág. 11).
Reconstrucción de la conductividad compleja gamma = sigma + i * omega * epsilon
en un medio con propiedades eléctricas constantes y corriente sinusoidal.

Requiere FEniCS Legacy. Sin fallback analítico.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eit_lib import TwoChannelEIT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_exercise_5():
    print("=" * 65)
    print("EJERCICIO 5: EXAMPLE 1.5 — TWO-CHANNEL EIT (CONDUCTIVIDAD COMPLEJA)")
    print("=" * 65)

    sigma_true = 0.25         # Conductividad [S/m]
    eps_0 = 8.854e-12         # Permitividad del vacío [F/m]
    eps_r = 5000.0            # Permitividad relativa biológica típica
    epsilon_true = eps_r * eps_0

    freq_hz = 50e3            # Frecuencia: 50 kHz
    omega = 2.0 * np.pi * freq_hz
    gamma_true = complex(sigma_true, omega * epsilon_true)

    print(f"\n1. Propiedades Eléctricas Verdaderas del Medio:")
    print(f"   - Frecuencia f:               {freq_hz / 1e3:.1f} kHz")
    print(f"   - Frecuencia angular omega:   {omega:.2e} rad/s")
    print(f"   - Conductividad sigma:        {sigma_true:.4f} S/m")
    print(f"   - Permitividad epsilon:       {epsilon_true:.4e} F/m")
    print(f"   - Reactancia omega*epsilon:   {omega * epsilon_true:.4f} S/m")
    print(f"   - gamma = sigma + i*omega*eps:{gamma_true.real:.4f} + {gamma_true.imag:.4f}j S/m")

    two_ch = TwoChannelEIT(nx=32, ny=32, electrode_width=0.4, current_intensity=1.0)
    V_meas, _ = two_ch.forward_solve(sigma_true, epsilon_true, omega)
    sigma_recon, eps_recon, gamma_recon = two_ch.inverse_solve(V_meas, omega)
    C_Omega = two_ch.energy_u0

    print(f"\n2. Respuesta Eléctrica y Medición en Electrodo E1:")
    print(f"   - Factor geométrico C_Omega:  {C_Omega:.4f}")
    print(f"   - Voltaje complejo V_meas:    {V_meas.real:.4f} + {V_meas.imag:.4f}j V")
    print(f"   - Amplitud |V|:               {abs(V_meas):.4f} V")
    phase_deg = np.angle(V_meas, deg=True)
    print(f"   - Desfase angular phi:        {phase_deg:.2f}°")

    print(f"\n3. Reconstrucción Inversa:")
    print(f"   - gamma reconstruida:         {gamma_recon.real:.4f} + {gamma_recon.imag:.4f}j S/m")
    print(f"   - sigma reconstruida:         {sigma_recon:.4f} S/m (Error: {abs(sigma_recon - sigma_true):.2e})")
    print(f"   - epsilon reconstruida:       {eps_recon:.4e} F/m (Error: {abs(eps_recon - epsilon_true):.2e})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    t = np.linspace(0, 2.0 / freq_hz, 400)
    I_t = np.cos(omega * t)
    V_amp = abs(V_meas)
    phi_rad = np.angle(V_meas)
    V_t = V_amp * np.cos(omega * t + phi_rad)

    t_us = t * 1e6
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    p1, = ax1.plot(t_us, I_t, "b-", lw=2, label=r"Corriente inyectada $I(t)$")
    p2, = ax1_twin.plot(t_us, V_t, "r--", lw=2, label=r"Voltaje medido $V(t)$")
    ax1.set_xlabel(r"Tiempo [$\mu$s]")
    ax1.set_ylabel(r"Corriente $I(t)$ [A]", color="b")
    ax1_twin.set_ylabel(r"Voltaje $V(t)$ [V]", color="r")
    ax1.set_title(f"Señales sinusoidales temporales (Desfase $\\phi = {phase_deg:.1f}^\\circ$)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    lines = [p1, p2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    freqs = np.logspace(2, 6, 50)
    omegas = 2.0 * np.pi * freqs
    sigmas_rec = []
    omegas_eps_rec = []
    for w_i in omegas:
        g_i = complex(sigma_true, w_i * epsilon_true)
        v_i = C_Omega / g_i
        g_rec = C_Omega / v_i
        sigmas_rec.append(g_rec.real)
        omegas_eps_rec.append(g_rec.imag)

    axes[1].semilogx(freqs, sigmas_rec, "b-", lw=2,
                     label=r"$\sigma$ reconstruida (Conductividad)")
    axes[1].semilogx(freqs, omegas_eps_rec, "g--", lw=2,
                     label=r"$\omega\varepsilon$ reconstruida (Reactancia)")
    axes[1].set_xlabel("Frecuencia [Hz]")
    axes[1].set_ylabel("Magnitud admisible [S/m]")
    axes[1].set_title(r"Espectro de Conductividad Compleja $\gamma(\omega) = \sigma + i\omega\varepsilon$")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex5_two_channel_complex.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Guardado] Imagen guardada en: {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_5()