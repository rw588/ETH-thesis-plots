import numpy as np
import matplotlib.pyplot as plt

# ── Parameters (edit these) ──────────────────────────────────────────────────
f       = 0.5   # open fraction of binary grating
M_upper = 2    # include Fourier orders  m = -M_upper … +M_upper

# Physical units — only the ratio d²/λ matters; axes are normalised below
d          = 1.0
wavelength = 1.0

# ── Derived quantities ───────────────────────────────────────────────────────
L_T = 2 * d**2 / wavelength   # Talbot length

# Fourier coefficients  a_n = f · sinc(π f n)
# numpy sinc is the normalised sinc: sinc(x) = sin(π x)/(π x), sinc(0) = 1
# so  f * sinc(f·m)  gives  sin(π f m)/(π m)  for m≠0  and  f  for m=0
m_vals = np.arange(-M_upper, M_upper + 1)
a_m    = f * np.sinc(f * m_vals)   # real, shape (2M+1,)

# ── Grid ─────────────────────────────────────────────────────────────────────
num_x = 800
num_z = 600
x_vals = np.linspace(-10 * d,  10 * d,  num_x)
z_vals = np.linspace(  0,       4 * L_T, num_z)

# ── Vectorised evaluation ─────────────────────────────────────────────────────
# Γ(x, z) = Σ_{m,n} a_m a_n exp(i·2π(m-n)x/d) exp(i·π·λ·z·(m²-n²)/d²)
#          = |E(x,z)|²
# where  E(x,z) = Σ_m a_m exp(i·2π·m·x/d) exp(i·π·λ·z·m²/d²)

# phase_x[m, x_idx] — shape (2M+1, num_x)
phase_x = np.exp(1j * 2*np.pi / d * m_vals[:, None] * x_vals[None, :])

# phase_z[z_idx, m] — shape (num_z, 2M+1)
phase_z = np.exp(1j * np.pi * wavelength / d**2 * z_vals[:, None] * m_vals[None, :]**2)

# E[z_idx, x_idx] = Σ_m (a_m · phase_z[z,m]) · phase_x[m,x]   — one matmul
E     = (a_m[None, :] * phase_z) @ phase_x   # (num_z, num_x), complex
Gamma = (E * E.conj()).real                   # |E|², guaranteed real

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.pcolormesh(
    x_vals / d,
    z_vals / L_T,
    Gamma,
    shading='auto',
    cmap='inferno',
)
ax.set_xlabel(r'$x\,/\,d$', fontsize=13)
ax.set_ylabel(r'$z\,/\,L_T$', fontsize=13)
ax.set_title(
    r'$\Gamma(x,z)=\sum_{m,n}a_m a_n\,'
    r'e^{i2\pi(m-n)x/d}\,e^{i\pi\lambda z(m^2-n^2)/d^2}$'
    '\n'
    rf'Binary grating,  $f={f}$,  $M_\mathrm{{upper}}={M_upper}$',
    fontsize=11,
)
fig.colorbar(im, ax=ax, label=r'$\Gamma(x,z)$')
plt.tight_layout()
plt.show()
