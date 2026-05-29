import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Thesis style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.linestyle":    ":",
    "grid.alpha":        0.6,
    "errorbar.capsize":  4,
    "figure.dpi":        150,
})

# ── Data ──────────────────────────────────────────────────────────────────────
T      = [80, 124, 164, 240, 350]  # mK
T_err  = [3,  2,   2,   2,   2]   # mK (symmetric)

v_emission = [1960, 1900, 1820, 1940, 1960]  # m/s
t0      = [1.2,  1.0,  1.1,  1.0,  2.2]  # µs
mu      = [8.3,  6.9,  5.1,  6.2,  3.2]  # %
slope   = [2.5,  2.5,  2.3,  2.0,  0.9]  # ns⁻¹

# Asymmetric error bars: [lower, upper] for each data point
v_emission_err = [[220, 180, 190, 480, 840],
               [220, 180, 190, 480, 840]]
t0_err      = [[0.5, 0.4, 0.4, 1.1, 2.5],
               [0.5, 0.4, 0.4, 1.1, 2.5]]
mu_err      = [[0.9, 0.8, 0.8, 1.4, 0.9],
               [0.9, 0.8, 0.8, 1.4, 0.9]]
slope_err   = [[0.3, 0.5, 0.5, 0.4, 0.2],
               [0.3, 0.5, 0.5, 0.4, 0.2]]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
fig.suptitle("Superthermal muonium from He-II parameters variation with temperature", fontsize=12)

datasets = [
    (axes[0, 0], v_emission, v_emission_err,
     "Longitudinal propagation velocity",         r"$v_\mathrm{sound}$  (m\,s$^{-1}$)", "C0"),

    (axes[0, 1], t0,      t0_err,
     "Diffusion time",     r"$t_0$  ($\mu$s)",                    "C1"),

    (axes[1, 0], mu,      mu_err,
     "Conversion efficiency",      r"$\mu$  (\%)",                        "C2"),

    (axes[1, 1], slope,   slope_err,
     "Signal rise slope (layer 4)", r"slope  (ns$^{-1}$)",            "C3"),
]

for ax, y, yerr, title, ylabel, colour in datasets:
    ax.errorbar(T, y, xerr=T_err, yerr=yerr,
                marker='+', markersize=5,
                color=colour, linewidth=1.4,
                elinewidth=1.1, capthick=1.1)
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True)

for ax in axes[1]:
    ax.set_xlabel(r"Temperature  (mK)")

# x-axis ticks at each data point for clarity
for ax in axes.flat:
    ax.set_xticks(T)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("tempHeII.pdf", bbox_inches="tight")
plt.show()
