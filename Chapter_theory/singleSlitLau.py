import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
d = 1.0       # slit spacing
L = 100.0     # propagation distance (final plane)
m_values = np.arange(-10, 11)
n_values = m_values.copy()

z_min, z_max = 0, L*1.2
x_min, x_max = -15, 15
Nz, Nx = 1200, 800

z = np.linspace(z_min, z_max, Nz)
x = np.linspace(x_min, x_max, Nx)
Z, X = np.meshgrid(z, x)

# -----------------------------
# Hyperbola centre function
# -----------------------------
def hyperbola_center(x_m, x_n, z, L):
    """Far-field approximation for hyperbola centre"""
    Delta = (n**2 - m**2) * d**2 / (2*L)
    x_c = (x_m + x_n)/2 + Delta * z / (x_n - x_m)
    return x_c

# -----------------------------
# 1️⃣ Hyperbola lines plot
# -----------------------------
plt.figure(figsize=(6, 8))

for m in m_values:
    for n in n_values:
        if m == n:
            continue
        x_m = m * d
        x_n = n * d
        x_c = hyperbola_center(x_m, x_n, z, L)

        plt.plot(z, x_c, 'k', linewidth=0.8)
        plt.plot(z, -x_c, 'k', linewidth=0.8)

        # Foci
        plt.scatter([0, 0], [x_m, x_n], color='red', s=25, zorder=5)

plt.xlim(z_min, z_max)
plt.ylim(x_min, x_max)
plt.xlabel("z")
plt.ylabel("x")
plt.title("Matched Path Length Hyperbolae (Lines) with Foci")
plt.tight_layout()
plt.show()

# -----------------------------
# 2️⃣ Intensity map
# -----------------------------
intensity = np.zeros_like(Z)

for m in m_values:
    for n in n_values:
        if m == n:
            continue
        x_m = m * d
        x_n = n * d
        x_c = hyperbola_center(x_m, x_n, z, L)

        sigma_z = 0.01 * z
        sigma_z[sigma_z == 0] = 0.01

        intensity += np.exp(-(X - x_c)**2 / (2 * sigma_z**2))
        intensity += np.exp(-(X + x_c)**2 / (2 * sigma_z**2))

plt.figure(figsize=(6, 8))
plt.imshow(
    intensity,
    extent=[z_min, z_max, x_min, x_max],
    origin='lower',
    aspect='auto'
)
plt.xlabel("z")
plt.ylabel("x")
plt.title("Gaussian Intensity Map Along Hyperbola Centres")
plt.colorbar(label="Intensity")
plt.tight_layout()
plt.show()

# -----------------------------
# 3️⃣ Intensity slice at z = L
# -----------------------------
# find index closest to z = L
idx_L = np.argmin(np.abs(z - L))
I_slice = intensity[:, idx_L]

# compute contrast
I_max = np.max(I_slice)
I_min = np.min(I_slice)
contrast = (I_max - I_min) / (I_max + I_min)

plt.figure(figsize=(8, 4))
plt.plot(x, I_slice, 'b')
plt.xlabel("x")
plt.ylabel("Intensity")
plt.title(f"Intensity at z = L, Contrast C = {contrast:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()