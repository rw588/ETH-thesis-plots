"""
Talbot Carpet – GSM beam  (v3)
==============================
Speed fixes vs v2
-----------------
  v2 bug: GSM kernel looped over Nz z-slices → one GPU kernel launch per row.
          CuPy overhead dominated; no benefit over numpy.

  v3 fix: loop over K² = (2M+1)² (mn-pairs) instead.
          Each iteration accumulates a full (Nz, Nx) array.
          Arrays stay resident on the GPU (or in numpy) for the whole carpet.
          CuPy now gives genuine ~5-15x speedup for M >= 6.

  Numba removed – the numpy/CuPy kernel is already well-vectorised and
  Numba doesn't help the remaining Python overhead.

  L is now a set of three radio buttons (0.5 / 1 / 770 z_T) rather than
  a continuous slider, matching the typical experimental regimes.

COORDINATES
-----------
  xi   = x / d           (grating periods)
  zeta = z / z_T         (Talbot lengths,  z_T = 2d²/λ)
  z/L  = zeta / L_norm   (displayed y-axis)
  rho  = R(L) / z_T      (dimensionless radius at grating)

FORMULA
-------
Coherent:
  I = |Σ_m a_m exp(i2π m ξ) exp(i2π m² ζ)|²

GSM:
  I = exp(−ξ²/2s_IL²) Re[ Σ_{m,n} a_m a_n T1·T2·T3·T4·T5 ]

  T1 = exp{i2π[(m−n)+(m+n)ζ/ρ]ξ}        fringe + curvature tilt
  T2 = exp{i2πζ(m²−n²)(1+ζ/ρ)}           Talbot + curvature phase
  T3 = exp{−(m−n)ζξ/s_IL²}               envelope tilt
  T4 = exp{−(m²+n²)ζ²/s_IL²}             order attenuation
  T5 = exp{−2(m+n)²ζ²/s_cL²}             coherence filter

  T1·T3 = exp{φ(ζ)·ξ}  where  φ = i2π[(m−n)+(m+n)ζ/ρ] − (m−n)ζ/s_IL²

GSM propagation (z=0 → z=L):
  p_d    = 1/(4s_I0²) + 1/s_c0²
  s_IL²  = s_I0²(1−(Lη)²) + L²p_d/π²
  s_cL²  = s_c0² · s_IL²/s_I0²
  ρ      = π²s_IL²/(L·p_d)
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import PowerNorm

# ── Backend detection ─────────────────────────────────────────────────────────

try:
    import cupy as cp
    # quick smoke-test
    _t = cp.array([1.0, 2.0])
    del _t
    xp = cp
    BACKEND = "CuPy (GPU)"
except Exception:
    xp = np
    BACKEND = "NumPy (CPU)"

print(f"[talbot v3] backend: {BACKEND}")

# ── Style ─────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "figure.facecolor": "#08080e",
    "axes.facecolor":   "#08080e",
    "axes.edgecolor":   "#1e1e2e",
    "axes.labelcolor":  "#888",
    "xtick.color":      "#555",
    "ytick.color":      "#555",
    "text.color":       "#c8c8d8",
    "font.family":      "monospace",
    "font.size":        8,
})

DARK, MID, BORDER = "#08080e", "#0f0f1a", "#1e1e2e"
AMBER, TEAL, LILAC, WHITE = "#e08030", "#50b0a0", "#9988cc", "#e8e8f0"
L_CHOICES = [0.5, 1.0, 770.0]   # fixed L / z_T options


# ═══════════════════════════════════════════════════════════════════════════════
#  Physics
# ═══════════════════════════════════════════════════════════════════════════════

def grating_coeffs(M: int, f: float):
    """a_m = f·sinc(m·f),  m = −M…M.  Uses np.sinc(x)=sin(πx)/(πx)."""
    m = np.arange(-M, M + 1, dtype=np.float64)
    a = np.where(m == 0, f, f * np.sinc(m * f))
    return m, a


def gsm_propagate(s_I0: float, s_c0: float, L_norm: float,
                  inv_r0: float = 0.0):
    """Return (s_IL, s_cL, rho) after propagation through distance L_norm·z_T."""
    p_d     = 1.0 / (4.0 * s_I0**2) + 1.0 / max(s_c0**2, 1e-20)
    s_IL_sq = max(
        s_I0**2 * (1.0 - (L_norm * inv_r0)**2) + L_norm**2 * p_d / np.pi**2,
        1e-12,
    )
    s_cL_sq = max(s_c0**2 * s_IL_sq / s_I0**2, 1e-12)
    rho     = np.pi**2 * s_IL_sq / (L_norm * p_d) if L_norm > 1e-12 else np.inf
    return np.sqrt(s_IL_sq), np.sqrt(s_cL_sq), rho


# ── Coherent kernel ───────────────────────────────────────────────────────────

def compute_coherent(M: int, f: float, xi: np.ndarray, zeta: np.ndarray):
    """
    Fully vectorised: single loop over 2M+1 orders.
    field shape: (Nz, Nx) accumulated via broadcasting (Nz,1)×(1,Nx).
    """
    t0 = time.perf_counter()
    m_vals, a_vals = grating_coeffs(M, f)

    # move to accelerator if available
    z2 = xp.asarray(zeta[:, None])
    x2 = xp.asarray(xi[None, :])
    field = xp.zeros((len(zeta), len(xi)), dtype=xp.complex128)

    for m, a in zip(m_vals, a_vals):
        field += a * xp.exp(1j * 2.0 * np.pi * (m * x2 + m * m * z2))

    I = xp.abs(field)**2
    if xp is not np:
        I = xp.asnumpy(I)
    print(f"  coherent: {time.perf_counter()-t0:.2f}s")
    return I


# ── GSM kernel  (K²-outer-loop, GPU-friendly) ─────────────────────────────────

def compute_gsm(M: int, f: float,
                s_IL: float, s_cL: float, rho: float,
                xi: np.ndarray, zeta: np.ndarray):
    """
    Loop over K² = (2M+1)² (m,n) pairs.
    Each iteration adds a (Nz, Nx) contribution:

        dI[z, x] += Re{ B_k(zeta) · exp(phi_k(zeta) · xi) }

    where
        B_k(ζ)   = a_m a_n · T2(ζ) · T4(ζ) · T5(ζ)    shape (Nz,)
        phi_k(ζ) = i2π[(p + q·ζ/ρ)] − p·ζ/s_IL²        shape (Nz,)
        exp(phi · xi)                                     shape (Nz, Nx)  ← outer

    The full (Nz, Nx) array stays resident on GPU/CPU throughout.
    GPU benefit: K² launches of (Nz × Nx) kernels vs v2's Nz launches of (K² × Nx).
    For Nz = 500, M = 8: K² = 289 iterations  vs  500 in v2.
    For large Nz (e.g. 500 rows, 770 z_T range): K² stays constant, Nz grows → big win.
    """
    t0 = time.perf_counter()
    m_vals, a_vals = grating_coeffs(M, f)
    rho_eff = rho if (np.isfinite(rho) and abs(rho) > 1e-8) else 1e18

    # Build flattened (m,n) pair arrays
    mg, ng  = np.meshgrid(m_vals, m_vals, indexing='ij')
    amn_f   = (a_vals[:, None] * a_vals[None, :]).ravel()
    p_f     = (mg - ng).ravel()    # m−n
    q_f     = (mg + ng).ravel()    # m+n
    msq_f   = (mg**2).ravel()
    nsq_f   = (ng**2).ravel()

    # Move grids to accelerator
    z  = xp.asarray(zeta)          # (Nz,)
    xi_d = xp.asarray(xi)          # (Nx,)
    env  = xp.exp(-xi_d**2 / (2.0 * s_IL**2))   # (Nx,)

    I_out = xp.zeros((len(zeta), len(xi)), dtype=xp.float64)

    for k in range(len(amn_f)):
        amn_k = float(amn_f[k])
        if abs(amn_k) < 1e-14:
            continue

        p  = float(p_f[k])
        q  = float(q_f[k])
        ms = float(msq_f[k])
        ns = float(nsq_f[k])

        # ζ-dependent scalar factors → shape (Nz,)
        T2 = xp.exp(1j * 2.0 * np.pi * z * (ms - ns) * (1.0 + z / rho_eff))
        T4 = xp.exp(-(ms + ns) * z**2 / s_IL**2)
        T5 = xp.exp(-2.0 * q**2 * z**2 / s_cL**2)
        B  = amn_k * T2 * T4 * T5   # (Nz,)

        # ζ-dependent complex phase coefficient for ξ → shape (Nz,)
        phi = (1j * 2.0 * np.pi * (p + q * z / rho_eff)
               - p * z / s_IL**2)   # (Nz,)

        # outer product: exp(phi[:,None] * xi[None,:]) → (Nz, Nx)
        # then weighted sum into I_out
        I_out += xp.real(B[:, None] * xp.exp(phi[:, None] * xi_d[None, :]))

    # apply transverse envelope
    I_out *= env[None, :]

    if xp is not np:
        I_out = xp.asnumpy(I_out)

    I_out = np.maximum(I_out, 0.0)
    print(f"  GSM:      {time.perf_counter()-t0:.2f}s  (K²={len(amn_f)}, Nz={len(zeta)}, Nx={len(xi)})")
    return I_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.set_xlabel(xlabel, color="#888", fontsize=8, labelpad=3)
    ax.set_ylabel(ylabel, color="#888", fontsize=8, labelpad=3)
    ax.set_title(title, color="#aaa", fontsize=9, pad=5)
    ax.tick_params(colors="#555", labelsize=7)


def _talbot_lines(ax, L_norm, z_min_zT, z_max_zT):
    """Amber dashes at full Talbot planes, dim dots at half-planes (z/L coords)."""
    z_T_in_L = 1.0 / L_norm
    k_lo = max(0, int(np.floor(z_min_zT)) - 1)
    k_hi = int(np.ceil(z_max_zT)) + 2
    y_lo, y_hi = z_min_zT / L_norm, z_max_zT / L_norm
    for k in range(k_lo, k_hi):
        for half, col, ls, lw in [(0, AMBER, "--", 0.6), (1, "#607060", ":", 0.5)]:
            yv = (k + 0.5 * half) * z_T_in_L
            if y_lo <= yv <= y_hi:
                ax.axhline(yv, color=col, lw=lw, ls=ls, alpha=0.5)


def _draw_carpet(ax, xi, zeta_L, I, cmap, hline_L, title):
    ax.cla()
    _style_ax(ax, "ξ = x/d", "z / L", title)
    peak = I.max() if I.max() > 0 else 1.0
    im = ax.imshow(
        I / peak, origin="lower",
        extent=[xi[0], xi[-1], zeta_L[0], zeta_L[-1]],
        aspect="auto", cmap=cmap,
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=1),
        interpolation="bilinear",
    )
    if hline_L is not None:
        ax.axhline(hline_L, color=WHITE, lw=0.9, alpha=0.85)
    return im


def _draw_slice(ax, xi, I_coh, I_gsm, z_over_L, z_over_zT):
    ax.cla()
    _style_ax(ax, "ξ = x/d", "I (norm.)",
              f"Slice   z/L = {z_over_L:.4g}   z/z_T = {z_over_zT:.4g}")

    def _n(v):
        pk = v.max()
        return v / pk if pk > 0 else v

    ax.fill_between(xi, _n(I_coh), alpha=0.12, color=TEAL)
    ax.plot(xi, _n(I_coh), color=TEAL,  lw=1.2, label="coherent")
    ax.fill_between(xi, _n(I_gsm), alpha=0.12, color=AMBER)
    ax.plot(xi, _n(I_gsm), color=AMBER, lw=1.2, label="GSM")
    ax.set_xlim(xi[0], xi[-1])
    ax.set_ylim(-0.05, 1.18)
    ax.legend(fontsize=7, facecolor=MID, edgecolor=BORDER, labelcolor="#ccc")


def _draw_info(ax, f, M, s_I0, s_c0, L_norm, inv_r0,
               s_IL, s_cL, rho, z_min_zT, z_max_zT):
    ax.cla()
    ax.set_facecolor(MID)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _, a_vals = grating_coeffs(M, f)
    rho_str = f"{rho:.3g}" if np.isfinite(rho) and abs(rho) < 1e9 else "∞"
    R0_str  = f"{1/inv_r0:.3g}" if abs(inv_r0) > 1e-6 else "∞"

    rows = [
        ("─── source → grating ───", "",                 "#444"),
        ("σ_I(0)/d",               f"{s_I0:.3g}",        TEAL),
        ("σ_c(0)/d",               f"{s_c0:.3g}",        "#80b050"),
        ("R(0)/z_T",               R0_str,               "#c09040"),
        ("L/z_T",                  f"{L_norm:.3g}",      "#c060a0"),
        ("─── at grating (z=L) ───", "",                 "#444"),
        ("σ_I(L)/d",               f"{s_IL:.4g}",        TEAL),
        ("σ_c(L)/d",               f"{s_cL:.4g}",        "#80b050"),
        ("ρ = R(L)/z_T",           rho_str,              AMBER),
        ("z_T/L",                  f"{1/L_norm:.4g}",    "#888"),
        ("─── window ───",         "",                   "#444"),
        ("z_min/z_T",              f"{z_min_zT:.3g}",    "#6688aa"),
        ("z_max/z_T",              f"{z_max_zT:.3g}",    "#8866aa"),
        ("─── grating ───",        "",                   "#444"),
        ("f (open fraction)",      f"{f:.3f}",           AMBER),
        ("orders ±M",              f"±{M}",              LILAC),
        ("Σ|aₘ|² (= f)",          f"{np.sum(a_vals**2):.4f}", "#70a080"),
    ]

    for i, (lbl, val, col) in enumerate(rows):
        y = 0.97 - i * 0.056
        if y < 0.13:
            break
        ax.text(0.03, y, lbl, transform=ax.transAxes,
                color="#555" if val else "#444",
                fontsize=7, va="top", fontfamily="monospace")
        ax.text(0.97, y, val, transform=ax.transAxes, color=col,
                fontsize=7.5, va="top", ha="right", fontfamily="monospace")

    ax2 = ax.inset_axes([0.03, 0.01, 0.94, 0.10])
    ax2.set_facecolor(DARK)
    m_v  = np.arange(-M, M + 1)
    cols = [AMBER if m == 0 else LILAC for m in m_v]
    ax2.bar(m_v, np.abs(a_vals), color=cols, width=0.75)
    ax2.set_xlim(-M - 0.8, M + 0.8)
    ax2.tick_params(colors="#444", labelsize=5)
    ax2.set_ylabel("|aₘ|", color="#444", fontsize=5, labelpad=1)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#1a1a2a")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

NX, NZ = 380, 500    # carpet pixel resolution

def main():

    # ── state ──
    state = {
        "L_norm":  1.0,
        "f":       0.50,
        "M":       8,
        "s_I0":    5.0,
        "s_c0":    2.0,
        "inv_r0":  0.0,
        "z_min":   0.0,
        "z_max":   2.0,
        "z_sl":    1.0,
        # computed arrays
        "xi":      None, "zeta": None, "zeta_L": None,
        "I_coh":   None, "I_gsm": None,
    }

    # ── figure ──
    fig = plt.figure(figsize=(15, 8.8), facecolor=DARK)
    try:
        fig.canvas.manager.set_window_title("GSM Talbot Carpet v3")
    except Exception:
        pass

    gs_top = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.05, right=0.98,
                               top=0.92, bottom=0.42, wspace=0.30)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.05, right=0.98,
                               top=0.38, bottom=0.05, wspace=0.32)

    ax_coh  = fig.add_subplot(gs_top[0, 0])
    ax_gsm  = fig.add_subplot(gs_top[0, 1])
    ax_diff = fig.add_subplot(gs_top[0, 2])
    ax_sl   = fig.add_subplot(gs_bot[0, 0])
    ax_inf  = fig.add_subplot(gs_bot[0, 1])
    ax_inf.axis("off")

    # ── L radio buttons ──
    L_btn_axes = []
    L_btns     = []
    L_labels   = ["L = 0.5 z_T", "L = 1 z_T", "L = 770 z_T"]
    L_active   = [False, True, False]   # default: L=1

    def _btn_style(btn, active):
        col  = AMBER if active else MID
        ecol = AMBER if active else BORDER
        btn.ax.set_facecolor(col)
        btn.label.set_color(WHITE if active else "#666")
        for sp in btn.ax.spines.values():
            sp.set_edgecolor(ecol)

    for i, lbl in enumerate(L_labels):
        ax_b = fig.add_axes([0.55 + i * 0.13, 0.375, 0.11, 0.028], facecolor=MID)
        for sp in ax_b.spines.values():
            sp.set_edgecolor(BORDER)
        b = Button(ax_b, lbl, color=MID, hovercolor="#1e1e2e")
        b.label.set_fontsize(8)
        L_btn_axes.append(ax_b)
        L_btns.append(b)
        _btn_style(b, L_active[i])

    # ── Continuous sliders ──
    SL_DEFS = [
        ("f",     "f  (open fraction)",  0.05, 0.375, 0.05, 0.95,  0.50, 0.01, AMBER),
        ("M",     "M  (order limit ±M)", 0.05, 0.340, 1,    25,    8,    1,    LILAC),
        ("sI0",   "σ_I(0) / d",          0.05, 0.305, 0.3,  30.0,  5.0,  0.1,  TEAL),
        ("sc0",   "σ_c(0) / d",          0.05, 0.270, 0.1,  30.0,  2.0,  0.1,  "#80b050"),
        ("inv_r0","z_T / R(0)",          0.05, 0.235, -0.5, 0.5,   0.0,  0.005,"#c09040"),
        ("zmin",  "z_min / z_T  (crop)", 0.55, 0.305, 0.0,  800.0, 0.0,  0.5,  "#6688aa"),
        ("zmax",  "z_max / z_T  (crop)", 0.55, 0.270, 0.1,  800.0, 2.0,  0.5,  "#8866aa"),
        ("zsl",   "slice  z / z_T",      0.55, 0.235, 0.0,  800.0, 1.0,  0.05, WHITE),
    ]

    sld = {}
    for key, lbl, x, y, vmin, vmax, vi, step, col in SL_DEFS:
        w = 0.42 if x < 0.5 else 0.38
        ax_s = fig.add_axes([x, y, w, 0.018], facecolor=MID)
        for sp in ax_s.spines.values():
            sp.set_edgecolor(BORDER)
        sl = Slider(ax_s, lbl, vmin, vmax, valinit=vi, valstep=step, color=col,
                    handle_style={"facecolor": col, "edgecolor": WHITE, "size": 7})
        sl.label.set_color("#999");      sl.label.set_fontsize(8)
        sl.valtext.set_color("#e0d090"); sl.valtext.set_fontsize(8)
        sld[key] = sl

    # Reset button
    ax_rst = fig.add_axes([0.90, 0.068, 0.07, 0.028], facecolor=MID)
    btn_rst = Button(ax_rst, "Reset", color=MID, hovercolor=BORDER)
    btn_rst.label.set_color("#888"); btn_rst.label.set_fontsize(8)

    fig.text(
        0.5, 0.975,
        f"GSM Talbot carpet v3  |  y-axis: z/L  |  backend: {BACKEND}  |  click carpet → move slice",
        ha="center", va="top", color="#444", fontsize=8, fontfamily="monospace",
    )

    # ── Redraw ──
    def redraw():
        L_norm = state["L_norm"]
        f      = float(sld["f"].val)
        M      = int(sld["M"].val)
        s_I0   = float(sld["sI0"].val)
        s_c0   = float(sld["sc0"].val)
        inv_r0 = float(sld["inv_r0"].val)
        z_min  = float(sld["zmin"].val)
        z_max  = float(sld["zmax"].val)
        z_sl   = float(sld["zsl"].val)

        if z_min >= z_max:
            z_max = z_min + 0.1
        z_sl = float(np.clip(z_sl, z_min, z_max))

        s_IL, s_cL, rho = gsm_propagate(s_I0, s_c0, L_norm, inv_r0)

        xi_lim = max(5.0 * s_IL, 3.5)
        xi     = np.linspace(-xi_lim, xi_lim, NX)
        zeta   = np.linspace(z_min, z_max, NZ)   # z / z_T
        zeta_L = zeta / L_norm                    # z / L

        print(f"\n[redraw]  L={L_norm} z_T  window=[{z_min:.1f},{z_max:.1f}]z_T"
              f"  s_IL={s_IL:.3f}  s_cL={s_cL:.3f}  ρ={rho:.3g}  M={M}  f={f:.2f}")

        I_coh = compute_coherent(M, f, xi, zeta)
        I_gsm = compute_gsm(M, f, s_IL, s_cL, rho, xi, zeta)

        peak_c = I_coh.max() or 1.0
        peak_g = I_gsm.max() or 1.0
        I_diff = I_coh / peak_c - I_gsm / peak_g

        hline_L = z_sl / L_norm

        _draw_carpet(ax_coh,  xi, zeta_L, I_coh, "viridis", hline_L, "Coherent (plane wave)")
        _draw_carpet(ax_gsm,  xi, zeta_L, I_gsm, "inferno", hline_L, "GSM (partial coherence)")

        ax_diff.cla()
        _style_ax(ax_diff, "ξ = x/d", "z / L", "Coherent − GSM  (norm. each)")
        vd = max(abs(I_diff).max(), 0.01)
        ax_diff.imshow(I_diff, origin="lower",
                       extent=[xi[0], xi[-1], zeta_L[0], zeta_L[-1]],
                       aspect="auto", cmap="RdBu_r",
                       vmin=-vd, vmax=vd, interpolation="bilinear")
        ax_diff.axhline(hline_L, color=WHITE, lw=0.9, alpha=0.85)

        for ax in (ax_coh, ax_gsm, ax_diff):
            _talbot_lines(ax, L_norm, z_min, z_max)

        iz = int(np.argmin(np.abs(zeta - z_sl)))
        _draw_slice(ax_sl, xi, I_coh[iz], I_gsm[iz], zeta_L[iz], zeta[iz])

        _draw_info(ax_inf, f, M, s_I0, s_c0, L_norm, inv_r0,
                   s_IL, s_cL, rho, z_min, z_max)

        state.update(dict(xi=xi, zeta=zeta, zeta_L=zeta_L,
                          I_coh=I_coh, I_gsm=I_gsm))
        fig.canvas.draw_idle()

    # ── L button callbacks ──
    def make_L_cb(idx):
        def cb(event):
            for j, b in enumerate(L_btns):
                L_active[j] = (j == idx)
                _btn_style(b, L_active[j])
            state["L_norm"] = L_CHOICES[idx]
            # auto-adjust crop defaults for 770 case
            if L_CHOICES[idx] == 770.0:
                if sld["zmax"].val < 10.0:   # still at small default
                    sld["zmax"].set_val(2.0)
            redraw()
        return cb

    for i, b in enumerate(L_btns):
        b.on_clicked(make_L_cb(i))

    def on_change(_):
        redraw()

    for sl in sld.values():
        sl.on_changed(on_change)

    def on_reset(event):
        for sl in sld.values():
            sl.reset()
        # restore L=1 button
        for j, b in enumerate(L_btns):
            L_active[j] = (j == 1)
            _btn_style(b, L_active[j])
        state["L_norm"] = 1.0
        redraw()

    btn_rst.on_clicked(on_reset)

    # click carpet → move slice
    def on_click(event):
        for ax in (ax_coh, ax_gsm, ax_diff):
            if event.inaxes is ax and event.ydata is not None:
                if state["zeta"] is not None:
                    zT = event.ydata * state["L_norm"]
                    zT = float(np.clip(zT, sld["zmin"].val, sld["zmax"].val))
                    sld["zsl"].set_val(round(zT / 0.05) * 0.05)
                break

    fig.canvas.mpl_connect("button_press_event", on_click)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()