"""
Talbot Carpet – GSM beam  (v4)
==============================
Key fixes vs v3
---------------
  PHYSICS BUG: coherent panel used plane wave (ρ→∞) regardless of L, so
  self-images were always at ζ=1,2,... independent of source distance.
  Fix: coherent panel now uses a spherical wave (R=L, ρ=L_norm):

      I_coh(ξ,ζ) = |Σ_m a_m exp(i2πm·(1+ζ/ρ)·(ξ+mζ))|²

  Self-images satisfy ζ(1+ζ/ρ) = integer, not ζ = integer.
  For ρ→∞ (large L) this reduces to the standard plane-wave result.

  MARKER BUG: Talbot lines were drawn at fixed z/L = k/L_norm, ignoring ρ.
  Fix: solve ζ²/ρ + ζ = k/2  → ζ_k = ρ(-1+√(1+2k/ρ))/2

  READABILITY: all font sizes increased, axis labels explicit.

  L choices: 0.5, 1, 30, 770 z_T  (radio buttons)

COORDINATES
-----------
  λ          free-space wavelength
  d          grating period
  z_T = 2d²/λ   Talbot length
  L          source-to-grating distance     [user selects as multiple of z_T]
  z          post-grating propagation       [displayed axis: z/L]
  ξ = x/d   transverse position in grating periods
  ζ = z/z_T  post-grating propagation in Talbot lengths
  ρ = R(L)/z_T  dimensionless wavefront curvature at grating

  For coherent reference: ρ_coh = L/z_T = L_norm  (spherical wave, R=L)
  For GSM:               ρ_gsm = from gsm_propagate()

COHERENT SPHERICAL WAVE SELF-IMAGE CONDITION
--------------------------------------------
  ζ(1 + ζ/ρ) = n  →  ζ_n = ρ(−1 + √(1+4n/ρ))/2

  n=1: first full self-image
  n=½: half-period shifted copy
  n=2: second self-image  …

  Large ρ (L >> z_T): ζ_n → n  (standard plane wave Talbot)
  ρ=0.5:             ζ_1 = 0.5 z_T  (first self-image at half a Talbot length)
  ρ=1:               ζ_1 ≈ 0.618 z_T
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import PowerNorm
from concurrent.futures import ThreadPoolExecutor

# ── Backend ───────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    _t = cp.array([1.0, 2.0])
    _t2 = cp.exp(_t * 1j)          # needs NVRTC; fails gracefully if missing
    del _t, _t2
    xp = cp
    BACKEND = "CuPy (GPU)"
except Exception as _e:
    xp = np
    BACKEND = "NumPy (CPU)"
    print(f"  (CuPy → CPU fallback: {_e.__class__.__name__})")

print(f"[talbot v4] backend: {BACKEND}")

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor": "#08080e",
    "axes.facecolor":   "#08080e",
    "axes.edgecolor":   "#2a2a3e",
    "axes.labelcolor":  "#bbbbcc",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#ddddee",
    "font.family":      "sans-serif",
    "font.size":        10,
})

DARK   = "#08080e"
MID    = "#0f0f1a"
BORDER = "#2a2a3e"
AMBER  = "#e08030"
TEAL   = "#50c0b0"
LILAC  = "#aa88ee"
GREEN  = "#70cc70"
WHITE  = "#f0f0ff"
DIM    = "#555566"

L_CHOICES  = [0.5, 1.0, 30.0, 770.0]
L_LABELS   = ["L = ½ z_T", "L = 1 z_T", "L = 30 z_T", "L = 770 z_T"]

XI_CHOICES = [10.0, 20.0, 100.0]
XI_LABELS  = ["±10 d", "±20 d", "±100 d"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Physics
# ═══════════════════════════════════════════════════════════════════════════════

def grating_coeffs(M: int, f: float):
    """a_m = f·sinc(m·f),  m = −M…M.  np.sinc(x) = sin(πx)/(πx)."""
    m = np.arange(-M, M + 1, dtype=np.float64)
    a = np.where(m == 0, f, f * np.sinc(m * f))
    return m, a


def gsm_propagate(s_I0: float, s_c0: float, L_norm: float,
                  inv_r0: float = 0.0):
    """
    Propagate GSM beam from z=0 to z = L_norm·z_T.
    Returns (σ_I(L)/d, σ_c(L)/d, R(L)/z_T).
    """
    p_d     = 1.0 / (4.0 * s_I0**2) + 1.0 / max(s_c0**2, 1e-20)
    s_IL_sq = max(
        s_I0**2 * (1.0 - (L_norm * inv_r0)**2) + L_norm**2 * p_d / np.pi**2,
        1e-12,
    )
    s_cL_sq = max(s_c0**2 * s_IL_sq / s_I0**2, 1e-12)
    rho     = np.pi**2 * s_IL_sq / (L_norm * p_d) if L_norm > 1e-12 else np.inf
    return np.sqrt(s_IL_sq), np.sqrt(s_cL_sq), rho


def talbot_zeta(n_half: float, rho: float):
    """
    Solve ζ(1+ζ/ρ) = n_half  →  ζ²/ρ + ζ − n_half = 0.
    Returns ζ in z_T units, or None if unphysical.
    n_half = k/2 for k = 1,2,3,...
      k even → full self-image;  k odd → half-period shifted copy.
    """
    if rho > 1e8:
        return n_half
    disc = 1.0 + 4.0 * n_half / rho
    if disc < 0:
        return None
    return rho * (-1.0 + np.sqrt(disc)) / 2.0


def talbot_markers(rho: float, z_min_zT: float, z_max_zT: float):
    """
    Return list of (ζ_in_zT, is_full_image) within [z_min_zT, z_max_zT].
    """
    out = []
    # check enough multiples to cover the window
    k_max = max(20, int(4 * z_max_zT * rho / max(rho, 1)) + 10)
    for k in range(1, k_max + 1):
        z = talbot_zeta(k / 2.0, rho)
        if z is None:
            continue
        if z > z_max_zT * 1.001:
            break
        if z_min_zT <= z <= z_max_zT:
            out.append((z, k % 2 == 0))   # True = full image
    return out


# ── Kernels ───────────────────────────────────────────────────────────────────

def compute_coherent(M: int, f: float,
                     xi: np.ndarray, zeta: np.ndarray,
                     rho: float):
    """
    Coherent spherical-wave Talbot pattern (source at R=L, ρ=L/z_T).

    E[z,x] = Σ_m a_m · exp(i·2π·m²·u(z)·z) · exp(i·2π·m·u(z)·x)
    where u(z) = 1 + z/ρ.  I = |E|².

    Fully vectorised — no Python loop over m.
    """
    t0 = time.perf_counter()
    rho_eff = rho if (np.isfinite(rho) and abs(rho) > 1e-8) else 1e18
    m_vals, a_vals = grating_coeffs(M, f)   # (K,)

    z  = xp.asarray(zeta)   # (Nz,)
    xi_d = xp.asarray(xi)   # (Nx,)
    m_xp = xp.asarray(m_vals)   # (K,)
    a_xp = xp.asarray(a_vals)   # (K,)

    u = 1.0 + z / rho_eff                                            # (Nz,)
    # w[m, z] = a_m · exp(i·2π·m²·u(z)·z)
    w = a_xp[:, None] * xp.exp(
        1j * 2.0 * np.pi * m_xp[:, None]**2 * u[None, :] * z[None, :]
    )                                                                 # (K, Nz)
    # k_eff[m, z] = 2π·m·u(z)
    k_eff = 2.0 * np.pi * m_xp[:, None] * u[None, :]                # (K, Nz) real

    # E[z, x] = Σ_m w[m,z] · exp(i·k_eff[m,z]·x)
    # exp_term[m, z, x]: for K=17, Nz=500, Nx=380 → ~51 MB — acceptable
    exp_term = xp.exp(1j * k_eff[:, :, None] * xi_d[None, None, :]) # (K, Nz, Nx)
    E = (w[:, :, None] * exp_term).sum(axis=0)                       # (Nz, Nx)

    I = xp.abs(E)**2
    if xp is not np:
        I = xp.asnumpy(I)
    print(f"  coherent (spherical ρ={rho_eff:.2g}): {time.perf_counter()-t0:.2f}s")
    return I


def compute_gsm(M: int, f: float,
                s_IL: float, s_cL: float, rho: float,
                xi: np.ndarray, zeta: np.ndarray):
    """
    GSM post-grating intensity — fully vectorised, no Python loop over (m,n).

    Strategy:
      1. Precompute B[k,z] and phi[k,z] for all K² pairs simultaneously.
      2. I[z,x] = Re[Σ_k B[k,z] · exp(phi[k,z]·x)] · env[x]
         evaluated in x-chunks to cap memory at ~80 MB per chunk.
    """
    t0 = time.perf_counter()
    rho_eff = rho if (np.isfinite(rho) and abs(rho) > 1e-8) else 1e18
    m_vals, a_vals = grating_coeffs(M, f)

    # ── All (m,n) pair quantities — precomputed once ─────────────────────────
    mg, ng   = np.meshgrid(m_vals, m_vals, indexing='ij')   # (K,K)
    amn_f    = (a_vals[:, None] * a_vals[None, :]).ravel()  # (K²,)
    p_f      = (mg - ng).ravel()          # m − n
    q_f      = (mg + ng).ravel()          # m + n
    pq_f     = (p_f * q_f)               # m² − n²
    ms_pns_f = (mg**2 + ng**2).ravel()   # m² + n²

    # Drop negligible pairs
    mask     = np.abs(amn_f) > 1e-14
    amn_f    = amn_f[mask];  p_f = p_f[mask];  q_f = q_f[mask]
    pq_f     = pq_f[mask];   ms_pns_f = ms_pns_f[mask]
    KK       = len(amn_f)

    # ── To device ────────────────────────────────────────────────────────────
    z        = xp.asarray(zeta)           # (Nz,)
    xi_d     = xp.asarray(xi)            # (Nx,)
    amn_xp   = xp.asarray(amn_f)         # (KK,)
    p_xp     = xp.asarray(p_f)
    q_xp     = xp.asarray(q_f)
    pq_xp    = xp.asarray(pq_f)
    ms_ns_xp = xp.asarray(ms_pns_f)

    z2 = z[None, :]   # (1, Nz) for broadcasting against (KK, 1)

    # ── B[k, z] = amn · T2 · T4 · T5 ────────────────────────────────────────
    T2    = xp.exp(1j * 2.0*np.pi * pq_xp[:, None] * z2 * (1.0 + z2 / rho_eff))
    T4    = xp.exp(-ms_ns_xp[:, None] * z2**2 / s_IL**2)
    T5    = xp.exp(-2.0 * q_xp[:, None]**2 * z2**2 / s_cL**2)
    B_all = amn_xp[:, None] * T2 * T4 * T5                          # (KK, Nz)

    # ── phi[k, z] = i2π(p + q·z/ρ) − p·z/s_IL² ─────────────────────────────
    phi_all = (1j*2.0*np.pi * (p_xp[:, None] + q_xp[:, None] * z2 / rho_eff)
               - p_xp[:, None] * z2 / s_IL**2)                      # (KK, Nz)

    # ── Envelope ─────────────────────────────────────────────────────────────
    env   = xp.exp(-xi_d**2 / (2.0 * s_IL**2))                      # (Nx,)

    # ── Sum over k, chunked over x to limit memory ───────────────────────────
    # Chunk size 32: (KK=289, Nz=500, 32) × 16 B ≈ 74 MB peak
    Nx    = len(xi)
    I_out = xp.zeros((len(zeta), Nx))
    CHUNK = 32
    for x0 in range(0, Nx, CHUNK):
        xi_c     = xi_d[x0:x0+CHUNK]                                 # (c,)
        exp_term = xp.exp(phi_all[:, :, None] * xi_c[None, None, :]) # (KK, Nz, c)
        I_out[:, x0:x0+CHUNK] = xp.real(
            (B_all[:, :, None] * exp_term).sum(axis=0)               # (Nz, c)
        )

    I_out *= env[None, :]
    if xp is not np:
        I_out = xp.asnumpy(I_out)
    print(f"  GSM      (ρ={rho_eff:.3g}): {time.perf_counter()-t0:.2f}s"
          f"  KK={KK}  Nz={len(zeta)}")
    return np.maximum(I_out, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

FS_TITLE  = 12
FS_LABEL  = 11
FS_TICK   = 9
FS_ANNOT  = 10
FS_SLIDER = 9
FS_INFO   = 9


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=4)
    ax.set_title(title,   fontsize=FS_TITLE, pad=6)
    ax.tick_params(labelsize=FS_TICK, colors="#888899")



def _draw_carpet(ax, xi, zeta_L, I, cmap, hline_L, title):
    ax.cla()
    _style_ax(ax,
              xlabel="ξ = x / d",
              ylabel="z / L",
              title=title)
    peak = I.max() if I.max() > 0 else 1.0
    ax.imshow(
        I / peak, origin="lower",
        extent=[xi[0], xi[-1], zeta_L[0], zeta_L[-1]],
        aspect="auto", cmap=cmap,
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=1),
        interpolation="bilinear",
    )
    if hline_L is not None:
        ax.axhline(hline_L, color=WHITE, lw=1.2, alpha=0.9, zorder=3)


def _draw_diff(ax, xi, zeta_L, I_diff, hline_L):
    ax.cla()
    _style_ax(ax,
              xlabel="ξ = x / d",
              ylabel="z / L",
              title="Coherent − GSM  (norm. each)")
    vd = max(abs(I_diff).max(), 0.01)
    ax.imshow(I_diff, origin="lower",
              extent=[xi[0], xi[-1], zeta_L[0], zeta_L[-1]],
              aspect="auto", cmap="RdBu_r",
              vmin=-vd, vmax=vd, interpolation="bilinear")
    ax.axhline(hline_L, color=WHITE, lw=1.2, alpha=0.9, zorder=3)


def _draw_slice(ax, xi, I_coh, I_gsm, z_over_L, z_over_zT):
    ax.cla()
    _style_ax(ax,
              xlabel="ξ = x / d",
              ylabel="I  (normalised)",
              title=f"Intensity slice   z = {z_over_zT:.3g} z_T  =  {z_over_L:.3g} L")

    def _n(v):
        pk = v.max()
        return v / pk if pk > 0 else v

    ax.fill_between(xi, _n(I_coh), alpha=0.13, color=TEAL)
    ax.plot(xi, _n(I_coh), color=TEAL,  lw=1.5, label="Coherent (spherical)")
    ax.fill_between(xi, _n(I_gsm), alpha=0.13, color=AMBER)
    ax.plot(xi, _n(I_gsm), color=AMBER, lw=1.5, label="GSM (partial coherence)")
    ax.set_xlim(xi[0], xi[-1])
    ax.set_ylim(-0.05, 1.20)
    ax.legend(fontsize=FS_ANNOT, facecolor=MID, edgecolor=BORDER,
              labelcolor=WHITE, loc="upper right")
    ax.axhline(0, color=BORDER, lw=0.5)


def _draw_info(ax, f, M, s_I0, s_c0, L_norm, inv_r0,
               s_IL, s_cL, rho_gsm, z_min_zT, z_max_zT):
    ax.cla()
    ax.set_facecolor(MID)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _, a_vals = grating_coeffs(M, f)

    def _fmt(v, n=4):
        if not np.isfinite(v) or abs(v) > 9999:
            return "∞"
        return f"{v:.{n}g}"

    R0_str  = _fmt(1.0/inv_r0) if abs(inv_r0) > 1e-6 else "∞"

    rows = [
        ("── Source & propagation ─────", "",                    "#555"),
        ("z_T = 2d²/λ",                "reference scale",       DIM),
        ("L / z_T",                    f"{L_norm:.4g}",         "#cc88ee"),
        ("σ_I(z=0) / d",               f"{s_I0:.3g}",           TEAL),
        ("σ_c(z=0) / d",               f"{s_c0:.3g}",           GREEN),
        ("R(z=0) / z_T",               R0_str,                  "#c09040"),
        ("── At grating (z = L) ──────", "",                    "#555"),
        ("σ_I(L) / d",                 f"{s_IL:.4g}",           TEAL),
        ("σ_c(L) / d",                 f"{s_cL:.4g}",           GREEN),
        ("ρ_GSM = R_GSM(L) / z_T",     _fmt(rho_gsm),           AMBER),
        ("ρ_coh = ρ_GSM  (matched)",    _fmt(rho_gsm),           "#aaaacc"),
        ("── Carpet window ───────────", "",                    "#555"),
        ("z_min",                      f"{z_min_zT:.3g} z_T",   "#6688bb"),
        ("z_max",                      f"{z_max_zT:.3g} z_T",   "#8866bb"),
        ("── Grating ─────────────────", "",                    "#555"),
        ("f  (open fraction)",         f"{f:.3f}",              AMBER),
        ("Orders ±M",                  f"±{M}",                 LILAC),
        ("Σ|aₘ|²  (should = f)",      f"{np.sum(a_vals**2):.4f}", GREEN),
    ]

    y0 = 0.975
    dy = 0.053
    for i, (lbl, val, col) in enumerate(rows):
        y = y0 - i * dy
        if y < 0.12:
            break
        is_header = val == ""
        ax.text(0.03, y, lbl, transform=ax.transAxes,
                color="#666677" if is_header else "#9999aa",
                fontsize=FS_INFO - (0 if is_header else 0),
                va="top", style="italic" if is_header else "normal")
        if val:
            ax.text(0.97, y, val, transform=ax.transAxes, color=col,
                    fontsize=FS_INFO + 0.5, va="top", ha="right", fontweight="bold")

    # |a_m| bar chart
    ax2 = ax.inset_axes([0.03, 0.01, 0.94, 0.10])
    ax2.set_facecolor(DARK)
    m_v  = np.arange(-M, M + 1)
    cols = [AMBER if mm == 0 else LILAC for mm in m_v]
    ax2.bar(m_v, np.abs(a_vals), color=cols, width=0.75, zorder=2)
    ax2.set_xlim(-M - 0.8, M + 0.8)
    ax2.tick_params(colors="#555566", labelsize=6)
    ax2.set_ylabel("|aₘ|", color="#666677", fontsize=7, labelpad=2)
    ax2.set_xlabel("order m", color="#666677", fontsize=7, labelpad=1)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#1a1a2a")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

NX, NZ = 380, 500


def main():
    state = {
        "L_norm":   1.0,
        "xi_half":  10.0,
        "x_center": 0.0,
        "xi":       None, "zeta": None, "zeta_L": None,
        "I_coh":    None, "I_gsm": None,
    }

    # ── Figure ──
    fig = plt.figure(figsize=(16, 9.2), facecolor=DARK)
    try:
        fig.canvas.manager.set_window_title("GSM Talbot Carpet v4")
    except Exception:
        pass

    # ── Plot layout: top 65% plots, bottom 30% controls ──────────────────────
    gs_top = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.99,
                               top=0.96, bottom=0.60,
                               wspace=0.26)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.04, right=0.99,
                               top=0.55, bottom=0.33,
                               wspace=0.26)

    ax_coh  = fig.add_subplot(gs_top[0, 0])
    ax_gsm  = fig.add_subplot(gs_top[0, 1])
    ax_diff = fig.add_subplot(gs_top[0, 2])
    ax_sl   = fig.add_subplot(gs_bot[0, 0])
    ax_inf  = fig.add_subplot(gs_bot[0, 1])
    ax_inf.axis("off")

    # Thin separator line between plots and controls
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.305, 0.305],
                              transform=fig.transFigure,
                              color=BORDER, lw=0.8))

    # Header
    fig.text(0.5, 0.988,
             f"GSM Talbot Carpet  |  z_T = 2d²/λ  |  backend: {BACKEND}",
             ha="center", va="top", color=DIM,
             fontsize=FS_SLIDER + 0.5)

    # ── Button style helper ────────────────────────────────────────────────────
    def _btn_style(b, active):
        col = AMBER if active else MID
        b.ax.set_facecolor(col)
        b.label.set_color(DARK if active else "#888899")
        b.label.set_fontweight("bold" if active else "normal")
        for sp in b.ax.spines.values():
            sp.set_edgecolor(AMBER if active else BORDER)

    def _make_btn_ax(x, y, w, h):
        ax_b = fig.add_axes([x, y, w, h], facecolor=MID)
        for sp in ax_b.spines.values():
            sp.set_edgecolor(BORDER)
        return ax_b

    # ── LEFT CONTROL COLUMN:  Grating & Source  (x: 0.02 → 0.49) ─────────────
    # Sliders start at x=0.22 so labels (right-aligned, extending left) have room
    SL_LEFT = [
        #  key      label                   vmin    vmax    vi    step  colour
        ("f",      "open fraction  f",      0.05,   0.95,   0.50, 0.01, AMBER),
        ("M",      "Fourier orders  M",      1,      25,     8,    1,    LILAC),
        ("sI0",    "σ_I(z=0) / d",          0.3,    30.0,   0.5,  0.1,  TEAL),
        ("sc0",    "σ_c(z=0) / d",          0.1,    30.0,   0.5,  0.1,  GREEN),
        ("inv_r0", "curvature  z_T/R(z=0)", -0.5,   0.5,    0.0,  0.005,"#c09040"),
    ]
    SX, SW = 0.22, 0.27   # slider left edge and width
    SH = 0.019             # slider height
    SY0, SDY = 0.263, 0.036  # top slider y and row step

    fig.text(0.02, SY0 + SH + 0.014, "─── Grating & Source ───",
             color="#777788", fontsize=FS_SLIDER, va="bottom")

    sld = {}
    for i, (key, lbl, vmin, vmax, vi, step, col) in enumerate(SL_LEFT):
        y = SY0 - i * SDY
        ax_s = fig.add_axes([SX, y, SW, SH], facecolor=MID)
        for sp in ax_s.spines.values():
            sp.set_edgecolor(BORDER)
        sl = Slider(ax_s, lbl, vmin, vmax, valinit=vi, valstep=step, color=col,
                    handle_style={"facecolor": col, "edgecolor": WHITE, "size": 8})
        sl.label.set_color("#ccccdd"); sl.label.set_fontsize(FS_SLIDER)
        sl.valtext.set_color("#f0d080"); sl.valtext.set_fontsize(FS_SLIDER + 0.5)
        sld[key] = sl

    # ── RIGHT CONTROL COLUMN:  View  (x: 0.54 → 0.99) ────────────────────────
    RX0 = 0.54   # left edge of right column

    fig.text(RX0, SY0 + SH + 0.014, "─── View ───",
             color="#777788", fontsize=FS_SLIDER, va="bottom")

    # L buttons  (4 buttons)
    L_btns   = []
    L_active = [False, True, False, False]
    bw, bh = 0.099, 0.030
    by_L   = SY0 - 0 * SDY + (SH - bh) / 2
    for i, lbl in enumerate(L_LABELS):
        b = Button(_make_btn_ax(RX0 + i*(bw+0.005), by_L, bw, bh),
                   lbl, color=MID, hovercolor="#1e1e30")
        b.label.set_fontsize(FS_SLIDER)
        L_btns.append(b)
        _btn_style(b, L_active[i])
    fig.text(RX0, by_L + bh + 0.005, "L  (source → grating)",
             color="#aaaacc", fontsize=FS_SLIDER - 0.5, va="bottom")

    # xi buttons  (2 buttons)
    xi_btns   = []
    xi_active = [True, False, False]
    bw_xi, bh_xi = 0.090, 0.026
    by_xi = SY0 - 1 * SDY + (SH - bh_xi) / 2
    for i, lbl in enumerate(XI_LABELS):
        b = Button(_make_btn_ax(RX0 + i*(bw_xi+0.006), by_xi, bw_xi, bh_xi),
                   lbl, color=MID, hovercolor="#1e1e30")
        b.label.set_fontsize(FS_SLIDER)
        xi_btns.append(b)
        _btn_style(b, xi_active[i])
    fig.text(RX0, by_xi + bh_xi + 0.005, "x-axis range",
             color="#aaaacc", fontsize=FS_SLIDER - 0.5, va="bottom")

    # z sliders — label column 0.54-0.74, slider 0.74-0.98
    ZSX, ZSW = 0.74, 0.24
    SL_RIGHT = [
        ("zmin",     "z_min / z_T",   0.0,    800.0,  0.0,  0.5,  "#6688bb"),
        ("zmax",     "z_max / z_T",   0.1,    800.0,  1.5,  0.5,  "#8866bb"),
        ("zsl",      "slice  z/z_T",  0.0,    800.0,  1.0,  0.05, WHITE),
        ("x_center", "x-centre / d", -2000.0, 2000.0, 0.0,  10.0, "#cc8888"),
    ]
    for i, (key, lbl, vmin, vmax, vi, step, col) in enumerate(SL_RIGHT):
        y = SY0 - (i + 2) * SDY
        ax_s = fig.add_axes([ZSX, y, ZSW, SH], facecolor=MID)
        for sp in ax_s.spines.values():
            sp.set_edgecolor(BORDER)
        sl = Slider(ax_s, lbl, vmin, vmax, valinit=vi, valstep=step, color=col,
                    handle_style={"facecolor": col, "edgecolor": WHITE, "size": 8})
        sl.label.set_color("#ccccdd"); sl.label.set_fontsize(FS_SLIDER)
        sl.valtext.set_color("#f0d080"); sl.valtext.set_fontsize(FS_SLIDER + 0.5)
        sld[key] = sl

    # Reset button — bottom right
    btn_rst = Button(_make_btn_ax(0.915, 0.315, 0.072, 0.026),
                     "Reset", color=MID, hovercolor=BORDER)
    btn_rst.label.set_color("#aaaacc"); btn_rst.label.set_fontsize(FS_SLIDER)

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

        # GSM propagation
        s_IL, s_cL, rho_gsm = gsm_propagate(s_I0, s_c0, L_norm, inv_r0)
        rho_coh = rho_gsm  # coherent panel uses same wavefront curvature as GSM

        # Grids
        xi_lim    = state["xi_half"]
        x_center  = float(sld["x_center"].val)
        state["x_center"] = x_center
        xi        = np.linspace(x_center - xi_lim, x_center + xi_lim, NX)
        zeta   = np.linspace(z_min, z_max, NZ)    # z / z_T
        zeta_L = zeta / L_norm                     # z / L

        print(f"\n[redraw]  L={L_norm} z_T  "
              f"ρ_coh={rho_coh:.3g}  ρ_gsm={rho_gsm:.3g}  "
              f"s_IL={s_IL:.3f}  s_cL={s_cL:.3f}  "
              f"window=[{z_min:.2g},{z_max:.2g}]z_T  M={M}  f={f:.2f}")

        # Cache key — everything that affects the 2-D arrays (not z_sl)
        phys_key = (int(M), round(f, 4), round(s_I0, 4), round(s_c0, 4),
                    round(inv_r0, 5), round(L_norm, 4),
                    round(xi_lim, 2), round(x_center, 1),
                    round(z_min, 3), round(z_max, 3))

        if phys_key == state.get("_phys_key") and state["I_coh"] is not None:
            # Only the slice position changed — skip expensive recompute
            xi    = state["xi"]
            zeta  = state["zeta"]
            zeta_L = state["zeta_L"]
            I_coh = state["I_coh"]
            I_gsm = state["I_gsm"]
            iz = int(np.argmin(np.abs(zeta - z_sl)))
            _draw_slice(ax_sl, xi, I_coh[iz], I_gsm[iz],
                        zeta_L[iz], zeta[iz])
            fig.canvas.draw_idle()
            return

        # Run coherent and GSM in parallel (numpy releases GIL for vectorised ops)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_coh = pool.submit(compute_coherent, M, f, xi, zeta, rho_coh)
            fut_gsm = pool.submit(compute_gsm, M, f, s_IL, s_cL, rho_gsm, xi, zeta)
            I_coh = fut_coh.result()
            I_gsm = fut_gsm.result()

        state["_phys_key"] = phys_key

        peak_c = I_coh.max() or 1.0
        peak_g = I_gsm.max() or 1.0
        I_diff = I_coh / peak_c - I_gsm / peak_g

        hline_L = z_sl / L_norm

        _draw_carpet(ax_coh, xi, zeta_L, I_coh, "viridis", hline_L,
                     "Coherent  (spherical wave, R = L)")
        _draw_carpet(ax_gsm, xi, zeta_L, I_gsm, "inferno", hline_L,
                     "GSM  (partially coherent)")
        _draw_diff(ax_diff, xi, zeta_L, I_diff, hline_L)

        iz = int(np.argmin(np.abs(zeta - z_sl)))
        _draw_slice(ax_sl, xi, I_coh[iz], I_gsm[iz],
                    zeta_L[iz], zeta[iz])

        _draw_info(ax_inf, f, M, s_I0, s_c0, L_norm, inv_r0,
                   s_IL, s_cL, rho_gsm, z_min, z_max)

        state.update(dict(xi=xi, zeta=zeta, zeta_L=zeta_L,
                          I_coh=I_coh, I_gsm=I_gsm))
        fig.canvas.draw_idle()

    # ── L buttons ──
    def make_L_cb(idx):
        def cb(_):
            for j, b in enumerate(L_btns):
                L_active[j] = (j == idx)
                _btn_style(b, L_active[j])
            L = L_CHOICES[idx]
            state["L_norm"] = L
            # auto-set z window: 0 → L·1.25, slice at L
            sld["zmin"].set_val(0.0)
            sld["zmax"].set_val(round(L * 1.5 / 0.5) * 0.5)
            sld["zsl"].set_val(min(L, sld["zmax"].val))
            redraw()
        return cb

    for i, b in enumerate(L_btns):
        b.on_clicked(make_L_cb(i))

    def make_xi_cb(idx):
        def cb(_):
            for j, b in enumerate(xi_btns):
                xi_active[j] = (j == idx)
                _btn_style(b, xi_active[j])
            state["xi_half"] = XI_CHOICES[idx]
            redraw()
        return cb

    for i, b in enumerate(xi_btns):
        b.on_clicked(make_xi_cb(i))

    def on_change(_):
        redraw()

    for sl in sld.values():
        sl.on_changed(on_change)

    def on_reset(_):
        for sl in sld.values():
            sl.reset()
        for j, b in enumerate(L_btns):
            L_active[j] = (j == 1)
            _btn_style(b, L_active[j])
        for j, b in enumerate(xi_btns):
            xi_active[j] = (j == 0)
            _btn_style(b, xi_active[j])
        state["L_norm"]   = 1.0
        state["xi_half"]  = 10.0
        state["x_center"] = 0.0
        redraw()

    btn_rst.on_clicked(on_reset)

    # Click carpet → move slice
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