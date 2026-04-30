import numpy as np
import matplotlib.pyplot as plt

# --- Styling (thesis-quality) ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
})

# --- Data ---
phi = np.linspace(0, 2*np.pi, 1000)

C_values = [0.001, 0.2, 0.3, 0.4, 0.995]
colors = ["C0", "C1", "C2", "C3", "C4"]

eps = 0.02  # regularisation to avoid divergence

# --- Plot ---
fig, ax_lam = plt.subplots(figsize=(7, 4))
ax_fi = ax_lam.twinx()

lam_values = []
fisher_values = []
for C, col in zip(C_values, colors):
    lam = 1 + C * np.cos(phi)
    fisher = (C**2 * np.sin(phi)**2) / (lam + eps)

    ax_lam.plot(phi, lam, color=col, label=rf"$C={C}$")
    ax_fi.plot(phi, fisher, color=col, linestyle="--")
    lam_values.append(lam)
    fisher_values.append(fisher)

# Locus of Fisher information maxima: cos(phi_max) = (-1 + sqrt(1 - C^2/2)) / C
C_dense = np.linspace(0.001, 0.999, 1000)
cos_phi_max = (-(1 + eps) + np.sqrt(np.clip((1 + eps)**2 - C_dense**2, 0, None))) / C_dense
cos_phi_max = np.clip(cos_phi_max, -1, 1)
phi_max = np.arccos(cos_phi_max)
lam_max = 1 + C_dense * cos_phi_max
fisher_max = (C_dense**2 * (1 - cos_phi_max**2)) / (lam_max + eps)
ax_fi.plot(phi_max, fisher_max, color="black", linewidth=1.5, zorder=5,
           label=r"maxima locus")

# --- Formatting ---
ax_lam.set_xlabel(r"$\phi$")
ax_lam.set_ylabel(r"$\lambda(\phi)/\lambda_0$")
ax_lam.set_xlim(0, 2*np.pi)
ax_lam.set_ylim(0, 1.1 * np.max(lam_values))
ax_lam.set_title(
    r"Mean photon number $\lambda(\phi)$ and Fisher information $I(\phi)$"
    "\n"
    r"for a displaced thermal state as a function of phase $\phi$",
    fontsize=12, pad=8
)
ax_lam.grid(True, which="both", linestyle=":", linewidth=0.7)

ax_fi.set_ylabel(r"$I(\phi)$")
ax_fi.set_ylim(0, 1.1 * np.max(fisher_values))

# Single legend: solid = lambda, dashed = I(phi)
ax_lam.legend(loc="upper left", bbox_to_anchor=(1.08, 1), frameon=False, title=r"$\lambda$ (solid), $I$ (dashed)")

# Tight layout
plt.tight_layout()

# Save as vector PDF (best for thesis)
plt.savefig("fisher_information_plot.pdf")

plt.show()