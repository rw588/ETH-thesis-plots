import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
d = 1.0               # grating period
wavelength = 0.5e-6   # wavelength in meters
f = 0.3               # open fraction
M = 10                # number of Fourier orders: -M..M

# Fourier coefficients for binary grating
m_vals = np.arange(-M, M+1)
a_m = np.zeros_like(m_vals, dtype=complex)
for idx, m in enumerate(m_vals):
    if m == 0:
        a_m[idx] = f
    else:
        a_m[idx] = np.sin(np.pi * m * f) / (np.pi * m)

# Talbot length
L_T = 2 * d**2 / wavelength

# -------------------------
# Grid
# -------------------------
x_min, x_max = -5*d, 5*d
num_x = 500
num_z = 300
z_max = 2 * L_T

x_vals = np.linspace(x_min, x_max, num_x)
z_vals = np.linspace(0, z_max, num_z)

# -------------------------
# Allocate intensity arrays
# -------------------------
I_coh = np.zeros((num_z, num_x))
I_incoh = np.zeros((num_z, num_x))

# -------------------------
# Evaluate
# -------------------------
for i, z in enumerate(z_vals):
    phase_L = np.exp(1j * np.pi * wavelength * L_T / d**2 * (m_vals[:, None]**2 - m_vals[None, :]**2))
    for j, x in enumerate(x_vals):
        exp_x = np.exp(1j * 2*np.pi/d * (m_vals[:, None] - m_vals[None, :]) * x)
        # coherent sum
        I_coh[i,j] = np.abs(np.sum(a_m[:, None] * a_m[None, :] * exp_x * phase_L))**2
    # incoherent sum (diagonal only)
    I_incoh[i,:] = np.abs(np.sum(np.abs(a_m)**2))**2

# -------------------------
# Plot results
# -------------------------
fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True)

im0 = axs[0].pcolormesh(x_vals/d, z_vals/L_T, I_coh, shading='auto', cmap='viridis')
axs[0].set_ylabel('z / L_T')
axs[0].set_title(f'Coherent source intensity |S(x,L+z)|^2, f={f}')
fig.colorbar(im0, ax=axs[0], label='Intensity')

im1 = axs[1].pcolormesh(x_vals/d, z_vals/L_T, I_incoh, shading='auto', cmap='viridis')
axs[1].set_xlabel('x / d')
axs[1].set_ylabel('z / L_T')
axs[1].set_title(f'Incoherent source intensity |sum_n a_n|^2, f={f}')
fig.colorbar(im1, ax=axs[1], label='Intensity')

plt.tight_layout()
plt.show()