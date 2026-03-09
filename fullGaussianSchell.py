"""
Talbot Carpet – Gaussian-Schell Model (GSM) + Coherent Comparison
==================================================================
All axes are dimensionless:
    ξ = x / d                 (transverse position in grating periods)
    ζ = z / z_T               (post-grating propagation in Talbot lengths)
    z_T = 2 d² / λ            (standard Talbot length)

GSM beam propagation  (z = 0  →  z = L = L_norm · z_T):
─────────────────────────────────────────────────────────
    p_d   = 1/(4 s_I0²) + 1/s_c0²          (dimensionless spreading parameter)
    s_IL² = s_I0²(1 – L_norm² η²) + L_norm² p_d / π²
    s_cL² = s_c0² · s_IL² / s_I0²          (coherence / intensity ratio preserved)
    ρ     = π² s_IL² / (L_norm p_d)         (dimensionless R(L) / z_T)

where  η = inv_r0 = z_T / R(0)  (0 → collimated, > 0 → diverging).

Intensity pattern  (GSM, post-grating, ζ = z'/z_T):
────────────────────────────────────────────────────
    I(ξ,ζ) = e^{–ξ²/(2s_IL²)} · Re[Σ_{m,n} aₘ aₙ · T1·T2·T3·T4·T5]

    T1 = exp{ i 2π [(m–n) + (m+n) ζ/ρ] ξ }         fringe + curvature tilt
    T2 = exp{ i 2π ζ (m²–n²) (1 + ζ/ρ) }            Talbot + curvature phase
    T3 = exp{ –(m–n) ζ ξ / s_IL² }                  envelope tilt
    T4 = exp{ –(m²+n²) ζ² / s_IL² }                 order attenuation
    T5 = exp{ –2(m+n)² ζ² / s_cL² }                 coherence filter

Coherent reference (plane wave, infinite beam and coherence):
─────────────────────────────────────────────────────────────
    I_coh(ξ,ζ) = |Σ_m aₘ e^{i2πmξ} e^{i2πm²ζ}|²

Binary grating coefficients:
    aₘ = f · sinc(m f)   [np.sinc convention: sinc(x) = sin(πx)/(πx)]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import PowerNorm

matplotlib.rcParams.update({
    "figure.facecolor":  "#08080e",
    "axes.facecolor":    "#08080e",
    "axes.edgecolor":    "#1e1e2e",
    "axes.labelcolor":   "#888",
    "xtick.color":       "#555",
    "ytick.color":       "#555",
    "text.color":        "#c8c8d8",
    "font.family":       "monospace",
    "font.size":         8,
})

# ── Helpers ──────────────────────────────────────────────────────────────────

def grating_coeffs(M: int, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (m_vals, a_m) for a binary grating with open fraction f."""
    m = np.arange(-M, M + 1, dtype=float)
    a = np.where(m == 0, f, f * np.sinc(m * f))   # np.sinc(x) = sin(πx)/(πx)
    return m, a


# ── GSM propagation ──────────────────────────────────────────────────────────

def gsm_propagate(s_I0: float, s_c0: float, L_norm: float,
                  inv_r0: float = 0.0) -> tuple[float, float, float]:
    """
    Propagate GSM beam parameters from z=0 to z = L_norm · z_T.

    Parameters (all in units of d and z_T)
    ----------------------------------------
    s_I0   : σ_I(0)/d   – intensity half-width at input
    s_c0   : σ_c(0)/d   – coherence half-width at input
    L_norm : L/z_T       – propagation distance to grating
    inv_r0 : z_T/R(0)   – inverse radius of curvature; 0 = collimated

    Returns
    -------
    s_IL : σ_I(L)/d
    s_cL : σ_c(L)/d
    rho  : R(L)/z_T      (inf if effectively flat)
    """
    p_d    = 1.0 / (4 * s_I0**2) + 1.0 / s_c0**2   # dimensionless
    spread = L_norm**2 * p_d / np.pi**2              # extra variance from diffraction

    # σ_I²(L)/d²  – ABCD result including initial curvature
    s_IL_sq = s_I0**2 * (1.0 - (L_norm * inv_r0)**2) + spread
    s_IL_sq = max(s_IL_sq, 1e-10)

    # σ_c²(L)/d²  – ratio σ_c/σ_I is preserved for collimated input;
    # approximately true for gently curved inputs (valid when s_I0 ≫ L·inv_r0)
    s_cL_sq = s_c0**2 * s_IL_sq / s_I0**2
    s_cL_sq = max(s_cL_sq, 1e-10)

    # R(L)/z_T   (exact for collimated; approximate for finite R_0)
    if L_norm < 1e-14 or spread < 1e-20:
        rho = np.inf
    else:
        rho = np.pi**2 * s_IL_sq / (L_norm * p_d)

    return np.sqrt(s_IL_sq), np.sqrt(s_cL_sq), rho


# ── Carpet computation ───────────────────────────────────────────────────────

def compute_coherent_carpet(M: int, f: float,
                             Nx: int, Nz: int,
                             xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """
    Coherent Talbot carpet (plane wave, infinite beam).
    I = |Σ_m aₘ exp(i2πmξ) exp(i2πm²ζ)|²
    ζ = z/z_T,  z_T = 2d²/λ  →  self-image at ζ = 1.
    """
    m_vals, a_vals = grating_coeffs(M, f)
    z2 = zeta[:, None]   # (Nz, 1)
    x2 = xi[None, :]     # (1, Nx)
    field = np.zeros((Nz, Nx), dtype=complex)
    for m, a in zip(m_vals, a_vals):
        field += a * np.exp(1j * 2 * np.pi * m * x2) \
                   * np.exp(1j * 2 * np.pi * m**2 * z2)
    return np.abs(field)**2


def compute_gsm_carpet(M: int, f: float,
                        s_IL: float, s_cL: float, rho: float,
                        Nx: int, Nz: int,
                        xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """
    GSM post-grating intensity carpet – double-sum formula.

    In dimensionless units (ξ=x/d, ζ=z/z_T, ρ=R(L)/z_T, s=σ/d):

    I(ξ,ζ) = e^{−ξ²/(2s_IL²)} · Re[Σ_{m,n} aₘ aₙ · T1·T2·T3·T4·T5]

    T1 = exp{i2π[(m−n)+(m+n)ζ/ρ]ξ}          fringe + curvature tilt
    T2 = exp{i2πζ(m²−n²)(1+ζ/ρ)}             Talbot + curvature phase
    T3 = exp{−(m−n)ζξ/s_IL²}                 envelope tilt
    T4 = exp{−(m²+n²)ζ²/s_IL²}               order attenuation
    T5 = exp{−2(m+n)²ζ²/s_cL²}               coherence filter

    Loop over (m,n) pairs; broadcasting over ξ and ζ.
    """
    m_vals, a_vals = grating_coeffs(M, f)
    rho_eff = rho if (np.isfinite(rho) and abs(rho) > 1e-6) else 1e18

    z2 = zeta[:, None]   # (Nz, 1)
    x2 = xi[None, :]     # (1, Nx)

    double_sum = np.zeros((Nz, Nx), dtype=complex)

    for m, am in zip(m_vals, a_vals):
        for n, an in zip(m_vals, a_vals):
            amn = am * an

            # ζ-only factors: shape (Nz, 1)  → broadcast over ξ automatically
            T2 = np.exp(1j * 2*np.pi * z2 * (m**2 - n**2) * (1.0 + z2/rho_eff))
            T4 = np.exp(-(m**2 + n**2) * z2**2 / s_IL**2)
            T5 = np.exp(-2.0 * (m + n)**2 * z2**2 / s_cL**2)

            # (ζ,ξ)-dependent factor: T1·T3 = exp(φ · ξ),  φ shape (Nz,1)
            phi = (1j * 2*np.pi * ((m - n) + (m + n) * z2 / rho_eff)
                   - (m - n) * z2 / s_IL**2)       # (Nz, 1)
            T1T3 = np.exp(phi * x2)                 # (Nz, Nx)

            double_sum += amn * T2 * T4 * T5 * T1T3

    envelope = np.exp(-x2**2 / (2.0 * s_IL**2))   # (1, Nx)
    I = envelope * np.real(double_sum)
    return np.maximum(I, 0.0)


# ── Plotting helpers ─────────────────────────────────────────────────────────

DARK   = "#08080e"
MID    = "#0f0f1a"
BORDER = "#1e1e2e"
AMBER  = "#e08030"
TEAL   = "#50b0a0"
LILAC  = "#9988cc"
WHITE  = "#e8e8f0"
DIM    = "#444455"


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.set_xlabel(xlabel, color="#888", fontsize=8, labelpad=3)
    ax.set_ylabel(ylabel, color="#888", fontsize=8, labelpad=3)
    ax.set_title(title, color="#aaa", fontsize=9, pad=5)
    ax.tick_params(colors="#555", labelsize=7)


def _draw_talbot_lines(ax, rho=None):
    """Annotate Talbot self-image planes."""
    marks = [
        (0.5, "½·z_T  shift",  "#607060"),
        (1.0, "z_T  self-img", AMBER),
        (1.5, "3/2·z_T",       "#607060"),
        (2.0, "2z_T  self-img",AMBER),
    ]
    for z_m, lbl, col in marks:
        ax.axhline(z_m, color=col, lw=0.7, ls="--", alpha=0.5)


def _draw_carpet(ax, xi, zeta, I, cmap, hline=None, title="", norm_gamma=0.5):
    ax.cla()
    _style_ax(ax, "ξ = x/d", "ζ = z/z_T", title)
    peak = I.max() if I.max() > 0 else 1.0
    im = ax.imshow(
        I / peak,
        origin="lower",
        extent=[xi[0], xi[-1], zeta[0], zeta[-1]],
        aspect="auto",
        cmap=cmap,
        norm=PowerNorm(gamma=norm_gamma, vmin=0, vmax=1),
        interpolation="bilinear",
    )
    _draw_talbot_lines(ax)
    if hline is not None:
        ax.axhline(hline, color=WHITE, lw=0.9, ls="-", alpha=0.8)
    return im


# ── Main app ─────────────────────────────────────────────────────────────────

NX, NZ  = 350, 450      # carpet resolution (increase for publication quality)
ZETA_MAX = 2.0
ZETA_MARKS = {0.5: "½z_T (shift)", 1.0: "z_T (self-image)", 2.0: "2z_T"}


def main():
    # ── Default parameters ──
    def_f      = 0.50
    def_M      = 8
    def_s_I0   = 5.0     # σ_I(0)/d
    def_s_c0   = 2.0     # σ_c(0)/d
    def_L_norm = 1.0     # L/z_T
    def_inv_r0 = 0.0     # z_T/R(0)  (0 = collimated)
    def_zeta_s = 1.0     # slice z/z_T

    # ── Figure layout ──
    fig = plt.figure(figsize=(14, 8), facecolor=DARK)
    fig.canvas.manager.set_window_title("GSM Talbot Carpet")

    gs_top = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.06, right=0.98, top=0.93, bottom=0.38,
        wspace=0.32,
    )
    gs_bot = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.98, top=0.34, bottom=0.05,
        wspace=0.35,
    )

    ax_coh   = fig.add_subplot(gs_top[0, 0])
    ax_gsm   = fig.add_subplot(gs_top[0, 1])
    ax_diff  = fig.add_subplot(gs_top[0, 2])
    ax_slice = fig.add_subplot(gs_bot[0, 0])
    ax_info  = fig.add_subplot(gs_bot[0, 1])
    ax_info.axis("off")

    # ── Sliders ──
    slider_defs = [
        # (label, left, bottom, vmin, vmax, valinit, step, color)
        ("f  (open fraction)",   0.06, 0.305, 0.05, 0.95, def_f,      0.01, AMBER),
        ("M  (order limit ±M)",  0.06, 0.270, 1,    25,   def_M,      1,    LILAC),
        ("σ_I(0)/d",             0.06, 0.235, 0.3,  20.0, def_s_I0,   0.1,  TEAL),
        ("σ_c(0)/d",             0.06, 0.200, 0.1,  20.0, def_s_c0,   0.1,  "#80b050"),
        ("L/z_T",                0.06, 0.165, 0.05, 8.0,  def_L_norm, 0.05, "#c060a0"),
        ("z_T/R(0)  [inv curv]", 0.06, 0.130, -0.5, 0.5,  def_inv_r0, 0.01, "#c09040"),
        ("slice  ζ = z/z_T",     0.55, 0.170, 0.0,  ZETA_MAX, def_zeta_s, 0.01, WHITE),
    ]

    sliders = {}
    for lbl, l, b, vmin, vmax, vinit, step, col in slider_defs:
        ax_sl = fig.add_axes([l, b, 0.42 if l < 0.5 else 0.38, 0.018],
                              facecolor=MID)
        for sp in ax_sl.spines.values():
            sp.set_edgecolor(BORDER)
        sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vinit, valstep=step,
                    color=col,
                    handle_style={"facecolor": col, "edgecolor": WHITE, "size": 7})
        sl.label.set_color("#999"); sl.label.set_fontsize(8)
        sl.valtext.set_color("#e0d090"); sl.valtext.set_fontsize(8)
        sliders[lbl] = sl

    # ── Reset button ──
    ax_btn = fig.add_axes([0.90, 0.055, 0.07, 0.03], facecolor=MID)
    btn = Button(ax_btn, "Reset", color=MID, hovercolor=BORDER)
    btn.label.set_color("#888"); btn.label.set_fontsize(8)

    # ── Header ──
    fig.text(0.5, 0.972,
             "I(ξ,ζ) = e^{−ξ²/2s_IL²} · Re[Σ_{m,n} aₘaₙ T1·T2·T3·T4·T5]   "
             "aₘ=f·sinc(mf)   z_T=2d²/λ   ρ=R(L)/z_T",
             ha="center", va="top", color="#444", fontsize=8.5, fontfamily="monospace")

    state = {}

    # ── Compute and draw ──
    def redraw():
        f      = float(sliders["f  (open fraction)"].val)
        M      = int(sliders["M  (order limit ±M)"].val)
        s_I0   = float(sliders["σ_I(0)/d"].val)
        s_c0   = float(sliders["σ_c(0)/d"].val)
        L_norm = float(sliders["L/z_T"].val)
        inv_r0 = float(sliders["z_T/R(0)  [inv curv]"].val)
        zeta_s = float(sliders["slice  ζ = z/z_T"].val)

        # GSM propagation
        s_IL, s_cL, rho = gsm_propagate(s_I0, s_c0, L_norm, inv_r0)

        # x range: ±4 σ_I(L), but at least 3 grating periods
        xi_lim = max(4.0 * s_IL, 3.5)
        xi     = np.linspace(-xi_lim, xi_lim, NX)
        zeta   = np.linspace(0, ZETA_MAX, NZ)

        # Coherent carpet
        I_coh = compute_coherent_carpet(M, f, NX, NZ, xi, zeta)

        # GSM carpet
        I_gsm = compute_gsm_carpet(M, f, s_IL, s_cL, rho, NX, NZ, xi, zeta)

        # Difference
        peak_c = I_coh.max() or 1.0
        peak_g = I_gsm.max() or 1.0
        I_diff = I_coh / peak_c - I_gsm / peak_g

        state.update(dict(xi=xi, zeta=zeta, I_coh=I_coh, I_gsm=I_gsm,
                          I_diff=I_diff, s_IL=s_IL, s_cL=s_cL, rho=rho,
                          f=f, M=M, s_I0=s_I0, s_c0=s_c0, L_norm=L_norm, inv_r0=inv_r0))

        # Carpet plots
        _draw_carpet(ax_coh, xi, zeta, I_coh, "viridis", zeta_s,
                     "Coherent (plane wave, ∞ beam)")
        _draw_carpet(ax_gsm, xi, zeta, I_gsm, "inferno", zeta_s,
                     "GSM  (partially coherent)")

        # Difference
        ax_diff.cla()
        _style_ax(ax_diff, "ξ = x/d", "ζ = z/z_T", "Coherent − GSM  (norm. each)")
        vmax_d = max(abs(I_diff).max(), 0.01)
        ax_diff.imshow(I_diff, origin="lower",
                       extent=[xi[0], xi[-1], zeta[0], zeta[-1]],
                       aspect="auto", cmap="RdBu_r",
                       vmin=-vmax_d, vmax=vmax_d,
                       interpolation="bilinear")
        _draw_talbot_lines(ax_diff)
        ax_diff.axhline(zeta_s, color=WHITE, lw=0.9, ls="-", alpha=0.8)

        # Slice
        _draw_slice(ax_slice, xi, zeta, I_coh, I_gsm, zeta_s)

        # Info panel
        _draw_info(ax_info, f, M, s_I0, s_c0, L_norm, inv_r0, s_IL, s_cL, rho)

        fig.canvas.draw_idle()

    def _draw_slice(ax, xi, zeta, I_coh, I_gsm, zeta_s):
        ax.cla()
        _style_ax(ax, "ξ = x/d", "I (normalised)", f"Slice at ζ = {zeta_s:.3f}")

        iz = np.argmin(np.abs(zeta - zeta_s))

        def norm(v):
            pk = v.max()
            return v / pk if pk > 0 else v

        sl_coh = norm(I_coh[iz, :])
        sl_gsm = norm(I_gsm[iz, :])

        ax.fill_between(xi, sl_coh, alpha=0.12, color=TEAL)
        ax.plot(xi, sl_coh, color=TEAL,  lw=1.2, label="coherent")
        ax.fill_between(xi, sl_gsm, alpha=0.12, color=AMBER)
        ax.plot(xi, sl_gsm, color=AMBER, lw=1.2, label="GSM")
        ax.axhline(0, color=BORDER, lw=0.5)
        ax.set_xlim(xi[0], xi[-1])
        ax.set_ylim(-0.05, 1.18)
        ax.legend(fontsize=7, facecolor=MID, edgecolor=BORDER, labelcolor="#ccc",
                  loc="upper right")

        # mark Talbot planes
        if zeta_s in ZETA_MARKS:
            ax.set_title(f"Slice at ζ={zeta_s:.3f}  ({ZETA_MARKS[zeta_s]})",
                         color="#aaa", fontsize=9, pad=5)

    def _draw_info(ax, f, M, s_I0, s_c0, L_norm, inv_r0, s_IL, s_cL, rho):
        ax.cla()
        ax.set_facecolor(MID)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        _, a_vals = grating_coeffs(M, f)
        m_vals = np.arange(-M, M+1)

        rho_str = f"{rho:.2f}" if np.isfinite(rho) and abs(rho) < 1e10 else "∞"
        R0_str  = f"{1/inv_r0:.1f}" if abs(inv_r0) > 1e-6 else "∞"

        entries = [
            ("── Input (z=0) ──", "", "#666"),
            ("σ_I(0)/d",   f"{s_I0:.3f}",  TEAL),
            ("σ_c(0)/d",   f"{s_c0:.3f}",  "#80b050"),
            ("R(0)/z_T",   R0_str,          "#c09040"),
            ("L/z_T",      f"{L_norm:.3f}", "#c060a0"),
            ("── At grating (z=L) ──", "", "#666"),
            ("σ_I(L)/d",   f"{s_IL:.4f}",  TEAL),
            ("σ_c(L)/d",   f"{s_cL:.4f}",  "#80b050"),
            ("R(L)/z_T  ρ",rho_str,         AMBER),
            ("── Grating ──", "", "#666"),
            ("f  (open frac.)", f"{f:.3f}",     AMBER),
            ("orders  ±M",      f"±{M}",         LILAC),
            ("a₀ = f",         f"{f:.4f}",       "#9988cc"),
            ("a₁ = f·sinc(f)", f"{a_vals[M+1]:.4f}", "#9988cc"),
            ("Σ|aₘ|² = f",     f"{np.sum(a_vals**2):.4f}", "#70a080"),
        ]

        for i, (lbl, val, col) in enumerate(entries):
            y = 0.97 - i * 0.064
            ax.text(0.04, y, lbl, transform=ax.transAxes, color="#666" if not val else "#555",
                    fontsize=7.5, va="top", fontfamily="monospace")
            ax.text(0.96, y, val, transform=ax.transAxes, color=col,
                    fontsize=8, va="top", ha="right", fontfamily="monospace")

        # mini bar chart of |aₘ|
        ax2 = ax.inset_axes([0.03, 0.02, 0.94, 0.13])
        ax2.set_facecolor(DARK)
        colors = [AMBER if mm == 0 else LILAC for mm in m_vals]
        ax2.bar(m_vals, np.abs(a_vals), color=colors, width=0.75)
        ax2.set_xlim(-M - 0.8, M + 0.8)
        ax2.tick_params(colors="#444", labelsize=6)
        ax2.set_ylabel("|aₘ|", color="#444", fontsize=6, labelpad=1)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#1a1a2a")

    # ── Wire sliders ──
    def on_change(_):
        redraw()

    for sl in sliders.values():
        sl.on_changed(on_change)

    def on_reset(_):
        for sl in sliders.values():
            sl.reset()

    btn.on_clicked(on_reset)

    # ── Click carpet to update slice ──
    def on_click(event):
        for ax in (ax_coh, ax_gsm, ax_diff):
            if event.inaxes is ax and event.ydata is not None:
                new_z = float(np.clip(event.ydata, 0, ZETA_MAX))
                sliders["slice  ζ = z/z_T"].set_val(
                    round(new_z / 0.01) * 0.01)
                break

    fig.canvas.mpl_connect("button_press_event", on_click)

    redraw()

    # ── Colorbar ──
    if "I_coh" in state:
        cb_ax = fig.add_axes([0.985, 0.38, 0.008, 0.55])
        sm = plt.cm.ScalarMappable(cmap="inferno",
                                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=1))
        cb = fig.colorbar(sm, cax=cb_ax)
        cb.set_label("I / I_max", color="#555", fontsize=7, labelpad=4)
        cb.ax.yaxis.set_tick_params(colors="#444", labelsize=6)
        cb.outline.set_edgecolor(BORDER)

    plt.show()


if __name__ == "__main__":
    main()