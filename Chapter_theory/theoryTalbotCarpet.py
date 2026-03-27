import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
d = 1.0             # grating period
wavelength = 1.6e-9  # wavelength in meters
f = 0.5             # open fraction of binary grating
M = 10              # number of Fourier orders to include: -M..M

# Fourier coefficients for binary grating
m_vals = np.arange(-M, M+1)
a_m = np.zeros_like(m_vals, dtype=complex)
for idx, m in enumerate(m_vals):
    if m == 0:
        a_m[idx] = f
    else:
        a_m[idx] = np.sin(np.pi * m * f) / (np.pi * m)

# -------------------------
# Grid
# -------------------------
x_min, x_max = -5*d, 5*d
num_x = 500
num_z = 300
L_T = 2 * d**2 / wavelength
z_max = 2 * L_T  # two Talbot lengths

x_vals = np.linspace(x_min, x_max, num_x)
z_vals = np.linspace(0, z_max, num_z)

# -------------------------
# Evaluate sum
# -------------------------
S = np.zeros((num_z, num_x), dtype=complex)

for i, z in enumerate(z_vals):
    for j, x in enumerate(x_vals):
        total = 0
        for m_idx, m in enumerate(m_vals):
            for n_idx, n in enumerate(m_vals):
                total += (a_m[m_idx] * a_m[n_idx] *
                          np.exp(1j * 2 * np.pi / d * (m - n) * x) *
                          np.exp(1j * np.pi * wavelength * z / d**2 * (m**2 - n**2)))
        S[i, j] = total

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8,6))
plt.pcolormesh(x_vals/d, z_vals/L_T, np.abs(S)**2, shading='auto', cmap='viridis')
plt.xlabel('x / d')
plt.ylabel('z / L_T')
plt.title(f'Binary grating intensity |S(x,z)|^2, f={f}')
plt.colorbar(label='Intensity')
plt.show()