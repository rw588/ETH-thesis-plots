"""
Accelerometer data plotter
──────────────────────────
Reads FLAC/WAV vibration recordings from accelerationData/ and produces:

  Figure 1 — spectrogram grid  (time × frequency × amplitude)
             one panel per measurement, frequency axis 1–2000 Hz
  Figure 2 — overlaid PSD comparison (Welch method)
             all measurements on the same axes for direct comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import soundfile as sf
from scipy.signal import spectrogram, welch
from pathlib import Path

# ── file definitions ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "accelerationData"

FILES = [
    ("5KandCoolingBelowUnfixed.flac", "5 K + cooling below\n(unfixed)"),
    ("5KandCoolingBelowFixed.wav",    "5 K + cooling below\n(fixed)"),
    ("coldDecoupled.flac",            "Cold decoupled"),
    ("PSIcryoCold.flac",              "PSI cryo cold"),
    ("PSIcryoColdFreeBelow.flac",     "PSI cryo cold\n(free below)"),
]

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":         False,
    "mathtext.fontset":    "cm",
    "font.family":         "serif",
    "font.size":           10,
    "axes.labelsize":      10,
    "axes.titlesize":      9.5,
    "legend.fontsize":     8.5,
    "xtick.labelsize":     8.5,
    "ytick.labelsize":     8.5,
    "axes.linewidth":      0.7,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":          150,
})

# ── spectrogram parameters ────────────────────────────────────────────────────
NPERSEG   = 8192               # FFT window — ~186 ms at 44.1 kHz → Δf ≈ 5.4 Hz
NOVERLAP  = NPERSEG * 3 // 4   # 75 % overlap
F_MIN     = 60.0               # Hz — display lower bound (cuts 50 Hz mains)
F_MAX     = 2000.0             # Hz — display upper bound
# colour limits are auto-computed from data after loading (see below)

COLORS = plt.cm.tab10(np.linspace(0, 1, len(FILES)))

# ── helpers ───────────────────────────────────────────────────────────────────
def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load file; average channels if stereo."""
    data, sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float64), sr


def db(x):
    """Convert power to dB with a hard noise floor."""
    return 10 * np.log10(np.maximum(x, 1e-30))


# ── load all files ────────────────────────────────────────────────────────────
recordings = []
for fname, label in FILES:
    sig, sr = load_mono(DATA_DIR / fname)
    recordings.append({"sig": sig, "sr": sr, "label": label})
    print(f"Loaded  {fname:40s}  {len(sig)/sr:5.1f} s   sr={sr} Hz")

# ── auto colour limits from global data distribution ─────────────────────────
_all_db = []
for rec in recordings:
    f0, _, Sxx = spectrogram(rec["sig"], fs=rec["sr"], nperseg=NPERSEG,
                              noverlap=NOVERLAP, window='hann', scaling='density')
    mask = (f0 >= F_MIN) & (f0 <= F_MAX)
    _all_db.append(db(Sxx[mask, :]).ravel())
_all_db = np.concatenate(_all_db)
VMIN_DB = float(np.percentile(_all_db, 2))    # noise floor
VMAX_DB = float(np.percentile(_all_db, 99.5)) # peak signal
print(f"Colour limits: {VMIN_DB:.1f} dB  →  {VMAX_DB:.1f} dB")

# ── Figure 1: spectrogram grid ────────────────────────────────────────────────
n = len(recordings)
fig1, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n),
                          gridspec_kw=dict(hspace=0.45))

im_ref = None
for ax, rec in zip(axes, recordings):
    sig, sr = rec["sig"], rec["sr"]

    f, t, Sxx = spectrogram(sig, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP,
                             window='hann', scaling='density')

    mask    = (f >= F_MIN) & (f <= F_MAX)
    Sxx_db  = db(Sxx[mask, :])

    im = ax.pcolormesh(t, f[mask], Sxx_db,
                       shading='gouraud',
                       cmap='inferno',
                       norm=Normalize(vmin=VMIN_DB, vmax=VMAX_DB),
                       rasterized=True)
    im_ref = im

    ax.set_yscale('log')
    ax.set_ylim(F_MIN, F_MAX)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks([60, 100, 200, 500, 1000, 2000])
    ax.set_xlim(0, t[-1])
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(rec["label"].replace('\n', '  '), loc='left', pad=3)

    rms = np.sqrt(np.mean(sig ** 2))
    ax.text(0.995, 0.96,
            f'dur = {len(sig)/sr:.1f} s     RMS = {rms*1e3:.3f} m·g',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
            color='white',
            bbox=dict(fc='black', alpha=0.45, pad=2, ec='none'))

axes[-1].set_xlabel('Time (s)')

fig1.colorbar(im_ref, ax=axes.tolist(), shrink=0.55, pad=0.02,
              label='PSD (dB re 1 a.u.²/Hz)')
fig1.suptitle('Vibration spectrograms — time × frequency', y=1.001,
              fontsize=11, fontweight='bold')

fig1.savefig(DATA_DIR.parent / "acc_spectrograms.pdf", bbox_inches='tight')
fig1.savefig(DATA_DIR.parent / "acc_spectrograms.png", dpi=180, bbox_inches='tight')
print("Saved acc_spectrograms.pdf / .png")

# ── Figure 2: overlaid PSD comparison ────────────────────────────────────────
fig2, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5),
                                       gridspec_kw=dict(wspace=0.35))

for color, rec in zip(COLORS, recordings):
    sig, sr = rec["sig"], rec["sr"]
    label   = rec["label"].replace('\n', ' — ')

    f_w, Pxx = welch(sig, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP,
                     window='hann', scaling='density')

    mask = (f_w >= F_MIN) & (f_w <= F_MAX)
    f_pl = f_w[mask]
    P_db = db(Pxx[mask])

    ax_lin.plot(f_pl, P_db, color=color, lw=1.3, label=label)
    ax_log.semilogx(f_pl, P_db, color=color, lw=1.3, label=label)

for ax, suffix in [(ax_lin, 'linear freq. axis'),
                   (ax_log, 'log freq. axis')]:
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re 1 a.u.²/Hz)')
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(VMIN_DB - 5, None)
    ax.grid(which='major', color='0.88', lw=0.6)
    ax.grid(which='minor', color='0.93', lw=0.4)
    ax.set_title(f'Welch PSD — {suffix}')

ax_log.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax_log.set_xticks([60, 100, 200, 500, 1000, 2000])
ax_lin.legend(fontsize=8, loc='upper right', framealpha=0.9)

fig2.suptitle('Vibration PSD comparison — all measurements', fontsize=11,
              fontweight='bold')

fig2.savefig(DATA_DIR.parent / "acc_psd_comparison.pdf", bbox_inches='tight')
fig2.savefig(DATA_DIR.parent / "acc_psd_comparison.png", dpi=180, bbox_inches='tight')
print("Saved acc_psd_comparison.pdf / .png")

plt.show()
