"""
Classical simulation: contrast and transmission vs. grating closed fraction.

Reads classicalContrast.csv from the same directory.

Figure 1 (dual y-axis): Contrast C (left) and transmitted counts N (right).
Figure 2: Figure of merit C/sqrt(N).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "text.usetex":         False,
    "mathtext.fontset":    "cm",
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      11,
    "legend.fontsize":     9.5,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "axes.linewidth":      0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":          150,
})

# ── load and clean ─────────────────────────────────────────────────────────────
df = pd.read_csv(OUT / "classicalContrast.csv", header=0)
df["Closed fraction"] = pd.to_numeric(df["Closed fraction"], errors="coerce")
df = (df.dropna(subset=["Closed fraction"])
        .sort_values("Closed fraction")
        .reset_index(drop=True))

x   = df["Closed fraction"].values
C   = pd.to_numeric(df["Contrast"],     errors="coerce").values
N   = pd.to_numeric(df.iloc[:, 11],     errors="coerce").values   # raw counts
σ_C = pd.to_numeric(df["error on Con"], errors="coerce").values

# Poisson uncertainty on N (conservative; fit errors in p3 column are smaller)
σ_N = np.where(np.isfinite(N), np.sqrt(N), np.nan)

# ── figure of merit C/√N and propagated error ─────────────────────────────────
valid = np.isfinite(N) & (N > 0) & np.isfinite(C) & (C > 0)
fom   = np.full_like(C, np.nan)
σ_fom = np.full_like(C, np.nan)
fom[valid]   = C[valid] / np.sqrt(N[valid])
σ_fom[valid] = np.sqrt(
    (σ_C[valid] / np.sqrt(N[valid]))**2 +
    (C[valid] * σ_N[valid] / (2.0 * N[valid]**1.5))**2
)

# ── Figure 1: contrast + N on dual y-axes ─────────────────────────────────────
fig1, ax_C = plt.subplots(figsize=(7, 4.5))
ax_N = ax_C.twinx()

cC, cN = "C0", "C3"

σ_C_safe = np.where(np.isfinite(σ_C), σ_C, 0.0)
ax_C.fill_between(x, C - σ_C_safe, C + σ_C_safe, color=cC, alpha=0.20)
ax_C.plot(x, C, "+-", color=cC, ms=8, lw=1.3, label="Contrast with third grating displacement")

mask_N = np.isfinite(N)
xN, Nk, σNk = x[mask_N], N[mask_N] * 1e-3, σ_N[mask_N] * 1e-3
ax_N.fill_between(xN, Nk - σNk, Nk + σNk, color=cN, alpha=0.20)
ax_N.plot(xN, Nk, "+--", color=cN, ms=8, lw=1.0, label=r"Fraction of successful particles $(\times 10^3)$")

ax_C.axvline(0.5, color="0.65", lw=0.8, ls=":", zorder=0)

ax_C.set_xlabel("Open fraction")
ax_C.set_ylabel(r"Contrast $\mathcal{C}$",                    color=cC)
ax_N.set_ylabel(r"Transmitted particles $N\;(\times 10^3)$",  color=cN)
ax_C.tick_params(axis="y", which="both", colors=cC)
ax_N.tick_params(axis="y", which="both", colors=cN)
ax_C.spines["left"].set_color(cC)
ax_N.spines["right"].set_color(cN)
ax_C.set_ylim(bottom=0)
ax_N.set_ylim(bottom=0)

ax_C.xaxis.set_minor_locator(AutoMinorLocator())
ax_C.yaxis.set_minor_locator(AutoMinorLocator())
ax_N.yaxis.set_minor_locator(AutoMinorLocator())

lines1, labs1 = ax_C.get_legend_handles_labels()
lines2, labs2 = ax_N.get_legend_handles_labels()
ax_C.legend(lines1 + lines2, labs1 + labs2, loc="upper center")

ax_C.set_title("Classical simulation: contrast and transmission vs. duty cycle")

fig1.tight_layout()
fig1.savefig(OUT / "classicalContrast_CN.pdf", bbox_inches="tight")
fig1.savefig(OUT / "classicalContrast_CN.png", dpi=200, bbox_inches="tight")
print("Figure 1 saved.")

# ── Figure 2: figure of merit C√η and related curves ─────────────────────────
fig2, ax_fom = plt.subplots(figsize=(7, 4.5))
ax_fom2 = ax_fom.twinx()

fom_mask = np.isfinite(fom)
xf, fk, σfk = x[fom_mask], fom[fom_mask] * 1e3, σ_fom[fom_mask] * 1e3
ax_fom.fill_between(xf, fk - σfk, fk + σfk, color="C2", alpha=0.20)
ax_fom.plot(xf, fk, "+-", color="C2", ms=8, lw=1.2,
            label=r"$\mathcal{C}\sqrt{\eta}$, $\eta$ = open fraction (probability of single particle traversal success)")

valid_C = np.isfinite(C)
ax_fom2.plot(x[valid_C], C[valid_C] * x[valid_C],       "--", color="C0", lw=1.4,
             label=r"$\mathcal{C}\cdot\eta$")
ax_fom2.plot(x[valid_C], C[valid_C] * x[valid_C]**1.5,  "--", color="C3", lw=1.4,
             label=r"$\mathcal{C}\cdot\eta^{3/2}$")
ax_fom2.set_ylim(bottom=0)
ax_fom2.set_ylabel(r"$\mathcal{C}\cdot\eta,\;\mathcal{C}\cdot\eta^{3/2}$")
ax_fom2.yaxis.set_minor_locator(AutoMinorLocator())

ax_fom.axvline(0.5, color="0.65", lw=0.8, ls=":", zorder=0)
ax_fom.set_ylim(bottom=0)

ax_fom.set_xlabel("Open fraction $\\eta$")
ax_fom.set_ylabel(r"$\mathcal{C}\sqrt{\eta}\;\;(\times 10^{-3})$")
ax_fom.set_title(r"$\mathcal{C}\sqrt{\eta}$: Information about $g_3$")

lines1, labs1 = ax_fom.get_legend_handles_labels()
lines2, labs2 = ax_fom2.get_legend_handles_labels()
ax_fom.legend(lines1 + lines2, labs1 + labs2, loc="upper center", fontsize=8)

ax_fom.xaxis.set_minor_locator(AutoMinorLocator())
ax_fom.yaxis.set_minor_locator(AutoMinorLocator())

fig2.tight_layout()
fig2.savefig(OUT / "classicalContrast_FOM.pdf", bbox_inches="tight")
fig2.savefig(OUT / "classicalContrast_FOM.png", dpi=200, bbox_inches="tight")
print("Figure 2 saved.")

plt.show()
