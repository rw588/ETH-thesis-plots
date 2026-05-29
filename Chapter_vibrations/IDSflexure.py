"""
IDS fibre length measurements from attocube .aws CSV export.
Plots Pos0, Pos1, Pos2 vs time with initial values subtracted.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

CSV_PATH = "/Users/robertwaddy/Downloads/flexure23_topR3L1front2_z50.0V_r6.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, sep=";", skiprows=4, header=0)
df.columns = df.columns.str.strip()

time = df["Time"].astype(float)
pos  = {
    "Pos0": df["Pos0"].astype(float),
    "Pos1": df["Pos1"].astype(float),
    "Pos2": df["Pos2"].astype(float),
}
pos_rel = {k: v - v.iloc[0] for k, v in pos.items()}

# Difference signal Pos2 − Pos0
diff_sig = pos_rel["Pos2"] - pos_rel["Pos0"]

# ── Sample rate ───────────────────────────────────────────────────────────────
dt = float(time.iloc[1] - time.iloc[0])
fs = 1.0 / dt
n  = len(time)

# ── Noise metrics ─────────────────────────────────────────────────────────────
def noise_metrics(arr):
    return {"pp": arr.max() - arr.min(), "rms": arr.std()}

metrics = {k: noise_metrics(v.values) for k, v in pos_rel.items()}
metrics["Pos2−Pos0"] = noise_metrics(diff_sig.values)

print(f"\n{'Channel':<12}  {'Peak-to-peak [pm]':>18}  {'RMS [pm]':>10}")
print("-" * 46)
for name, m in metrics.items():
    print(f"{name:<12}  {m['pp']:>18.1f}  {m['rms']:>10.2f}")
print()

# ── FFT ───────────────────────────────────────────────────────────────────────
freq = np.fft.rfftfreq(n, d=dt)
def fft_amplitude(arr):
    return np.abs(np.fft.rfft(arr)) * 2 / n

fft_amp = {k: fft_amplitude(v.values) for k, v in pos_rel.items()}
fft_amp["Pos2−Pos0"] = fft_amplitude(diff_sig.values)

# ── Amplitude spectral density (Welch) ────────────────────────────────────────
nperseg = min(n, max(256, int(fs)))
def compute_asd(arr):
    f_w, psd = welch(arr, fs=fs, nperseg=nperseg, window="hann", scaling="density")
    return f_w, np.sqrt(psd)

asd = {k: compute_asd(v.values) for k, v in pos_rel.items()}
asd["Pos2−Pos0"] = compute_asd(diff_sig.values)

# ── Windowed noise filter  H(f) = 2 sin(π f τ), clamped at H=2 above f_c ────
tau  = 10e-6
f_c  = 1.0 / (2 * tau)
f_w  = asd["Pos0"][0]
H    = np.where(f_w <= f_c, 2 * np.abs(np.sin(np.pi * f_w * tau)), 2.0)
df_w = f_w[1] - f_w[0]

def windowed_noise(arr):
    _, psd = welch(arr, fs=fs, nperseg=nperseg, window="hann", scaling="density")
    return np.sqrt(psd) * H, float(np.sqrt(np.sum(psd * H**2 * df_w)))

windowed_asd = {}
A_window     = {}
for k, v in pos_rel.items():
    windowed_asd[k], A_window[k] = windowed_noise(v.values)
windowed_asd["Pos2−Pos0"], A_window["Pos2−Pos0"] = windowed_noise(diff_sig.values)

print(f"{'Channel':<12}  {'A_window [pm]':>14}  {'A_window [nm]':>14}")
print("-" * 46)
for name, a in A_window.items():
    print(f"{name:<12}  {a:>14.1f}  {a/1000:>14.4f}")
print()

# ── Plot ──────────────────────────────────────────────────────────────────────
colours = {
    "Pos0":     "#1f77b4",
    "Pos1":     "#ff7f0e",
    "Pos2":     "#2ca02c",
    "Pos2−Pos0": "#e377c2",
}

fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        "Time domain",
        "FFT amplitude spectrum",
        "Amplitude spectral density  (Welch)",
        rf"Windowed ASD  ×H(f)=2sin(πfτ), τ={tau*1e6:.0f} µs,  clamped at f>{f_c/1e3:.0f} kHz",
    ),
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}]] * 3 + [[{"secondary_y": True}]],
)

# ── Individual channel traces (indices 0–11) ──────────────────────────────────
for name in ("Pos0", "Pos1", "Pos2"):
    y = pos_rel[name]
    m = metrics[name]
    label = f"{name}  p-p {m['pp']:.0f} pm  RMS {m['rms']:.1f} pm"
    fig.add_trace(go.Scattergl(x=time, y=y, mode="lines", name=label,
                               line=dict(color=colours[name], width=1),
                               legendgroup=name),
                  row=1, col=1)

for name in ("Pos0", "Pos1", "Pos2"):
    fig.add_trace(go.Scattergl(x=freq, y=fft_amp[name], mode="lines", name=name,
                               line=dict(color=colours[name], width=1),
                               legendgroup=name, showlegend=False),
                  row=2, col=1)

for name in ("Pos0", "Pos1", "Pos2"):
    f_w_ch, a = asd[name]
    fig.add_trace(go.Scattergl(x=f_w_ch, y=a, mode="lines", name=name,
                               line=dict(color=colours[name], width=1),
                               legendgroup=name, showlegend=False),
                  row=3, col=1)

for name in ("Pos0", "Pos1", "Pos2"):
    fig.add_trace(go.Scattergl(x=f_w, y=windowed_asd[name], mode="lines",
                               name=f"{name}  A={A_window[name]:.0f} pm",
                               line=dict(color=colours[name], width=1),
                               legendgroup=name, showlegend=False),
                  row=4, col=1, secondary_y=False)

# H(f) transfer function — index 12
fig.add_trace(go.Scattergl(x=f_w, y=H, mode="lines", name="H(f)",
                           line=dict(color="white", width=1.2, dash="dot"),
                           legendgroup="H"),
              row=4, col=1, secondary_y=True)

# ── Difference traces (indices 13–16), initially hidden ───────────────────────
dname = "Pos2−Pos0"
dc    = colours[dname]
dm    = metrics[dname]
fig.add_trace(go.Scattergl(x=time, y=diff_sig, mode="lines",
                           name=f"{dname}  p-p {dm['pp']:.0f} pm  RMS {dm['rms']:.1f} pm",
                           line=dict(color=dc, width=1),
                           legendgroup=dname, visible=False),
              row=1, col=1)

fig.add_trace(go.Scattergl(x=freq, y=fft_amp[dname], mode="lines", name=dname,
                           line=dict(color=dc, width=1),
                           legendgroup=dname, showlegend=False, visible=False),
              row=2, col=1)

f_w_d, a_d = asd[dname]
fig.add_trace(go.Scattergl(x=f_w_d, y=a_d, mode="lines", name=dname,
                           line=dict(color=dc, width=1),
                           legendgroup=dname, showlegend=False, visible=False),
              row=3, col=1)

fig.add_trace(go.Scattergl(x=f_w, y=windowed_asd[dname], mode="lines",
                           name=f"{dname}  A={A_window[dname]:.0f} pm",
                           line=dict(color=dc, width=1),
                           legendgroup=dname, showlegend=False, visible=False),
              row=4, col=1, secondary_y=False)

# ── Layout ────────────────────────────────────────────────────────────────────
# Trace indices: Pos0/1/2 × 4 rows = 0–11, H(f) = 12, diff × 4 rows = 13–16
vis_all  = [True]  * 13 + [False] * 4
vis_diff = [False] * 12 + [True]  + [True]  * 4   # keep H(f), show diff

fig.update_layout(
    title=dict(text=CSV_PATH.split("/")[-1], font=dict(size=13)),
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    height=1200,
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.0, y=1.04, xanchor="left", yanchor="bottom",
        showactive=True,
        buttons=[
            dict(label="All channels",
                 method="restyle",
                 args=[{"visible": vis_all}]),
            dict(label="Pos2 − Pos0",
                 method="restyle",
                 args=[{"visible": vis_diff}]),
        ],
    )],
)
fig.update_xaxes(title_text="Time  [s]",            row=1, col=1, showgrid=True, gridcolor="#333")
fig.update_yaxes(title_text="ΔLength  [pm]",        row=1, col=1, showgrid=True, gridcolor="#333")
fig.update_xaxes(title_text="Frequency  [Hz]",      row=2, col=1, showgrid=True, gridcolor="#333")
fig.update_yaxes(title_text="Amplitude  [pm]",      row=2, col=1, showgrid=True, gridcolor="#333", type="log")
fig.update_xaxes(title_text="Frequency  [Hz]",      row=3, col=1, showgrid=True, gridcolor="#333")
fig.update_yaxes(title_text="ASD  [pm/√Hz]",        row=3, col=1, showgrid=True, gridcolor="#333", type="log")
fig.update_xaxes(title_text="Frequency  [Hz]",      row=4, col=1, showgrid=True, gridcolor="#333")
fig.update_yaxes(title_text="ASD×H(f)  [pm/√Hz]",  row=4, col=1, showgrid=True, gridcolor="#333",
                 type="log", secondary_y=False)
fig.update_yaxes(title_text="|H(f)|",               row=4, col=1, secondary_y=True,
                 showgrid=False, range=[0, 2.2])

# annotation: A_window summary (all channels + diff)
annotation_lines = [f"<b>RMS motion in {tau*1e6:.0f} µs window</b>"]
for name, a in A_window.items():
    c = colours[name].lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    annotation_lines.append(
        f"<span style='color:rgb({r},{g},{b})'>{name}: {a:.0f} pm  ({a/1000:.3f} nm)</span>"
    )
fig.add_annotation(
    text="<br>".join(annotation_lines),
    xref="paper", yref="paper",
    x=0.01, y=0.01, xanchor="left", yanchor="bottom",
    showarrow=False,
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#555", borderwidth=1,
    font=dict(size=12),
)

print("=" * 46)
print(f"  RMS motion in {tau*1e6:.0f} µs window:")
for name, a in A_window.items():
    print(f"  {name:<12}:  {a:.0f} pm  =  {a/1000:.3f} nm")
print("=" * 46)

fig.show()

# ── Save individual panels as PDFs ────────────────────────────────────────────
import os
out_dir  = os.path.dirname(CSV_PATH)
basename = os.path.splitext(os.path.basename(CSV_PATH))[0]

def save_pdf(panel_fig, suffix):
    path = os.path.join(out_dir, f"{basename}_{suffix}.pdf")
    panel_fig.write_image(path, format="pdf", width=1000, height=400)
    print(f"  saved {path}")

print("\nSaving PDFs...")

# 1 — time domain
f1 = go.Figure()
for name in ("Pos0", "Pos1", "Pos2"):
    y = pos_rel[name]; m = metrics[name]
    f1.add_trace(go.Scattergl(x=time, y=y, mode="lines",
                              name=f"{name}  p-p {m['pp']:.0f} pm  RMS {m['rms']:.1f} pm",
                              line=dict(color=colours[name], width=1)))
f1.add_trace(go.Scattergl(x=time, y=diff_sig, mode="lines",
                           name=f"Pos2−Pos0  p-p {metrics['Pos2−Pos0']['pp']:.0f} pm  "
                                f"RMS {metrics['Pos2−Pos0']['rms']:.1f} pm",
                           line=dict(color=colours["Pos2−Pos0"], width=1)))
f1.update_layout(template="plotly_dark", hovermode="x unified",
                 xaxis_title="Time  [s]", yaxis_title="ΔLength  [pm]",
                 title="Time domain")
save_pdf(f1, "1_time")

# 2 — FFT
f2 = go.Figure()
for name in ("Pos0", "Pos1", "Pos2", "Pos2−Pos0"):
    f2.add_trace(go.Scattergl(x=freq, y=fft_amp[name], mode="lines",
                              name=name, line=dict(color=colours[name], width=1)))
f2.update_layout(template="plotly_dark", hovermode="x unified",
                 xaxis_title="Frequency  [Hz]", yaxis_title="Amplitude  [pm]",
                 yaxis_type="log", title="FFT amplitude spectrum")
save_pdf(f2, "2_fft")

# 3 — ASD
f3 = go.Figure()
for name in ("Pos0", "Pos1", "Pos2", "Pos2−Pos0"):
    fw, a = asd[name]
    f3.add_trace(go.Scattergl(x=fw, y=a, mode="lines",
                              name=name, line=dict(color=colours[name], width=1)))
f3.update_layout(template="plotly_dark", hovermode="x unified",
                 xaxis_title="Frequency  [Hz]", yaxis_title="ASD  [pm/√Hz]",
                 yaxis_type="log", title="Amplitude spectral density (Welch)")
save_pdf(f3, "3_asd")

# 4 — windowed ASD
f4 = make_subplots(specs=[[{"secondary_y": True}]])
for name in ("Pos0", "Pos1", "Pos2", "Pos2−Pos0"):
    f4.add_trace(go.Scattergl(x=f_w, y=windowed_asd[name], mode="lines",
                              name=f"{name}  A={A_window[name]:.0f} pm",
                              line=dict(color=colours[name], width=1)),
                 secondary_y=False)
f4.add_trace(go.Scattergl(x=f_w, y=H, mode="lines", name="H(f)",
                           line=dict(color="white", width=1.2, dash="dot")),
             secondary_y=True)
f4.update_layout(template="plotly_dark", hovermode="x unified",
                 title=rf"Windowed ASD  ×H(f)=2sin(πfτ), τ={tau*1e6:.0f} µs")
f4.update_xaxes(title_text="Frequency  [Hz]")
f4.update_yaxes(title_text="ASD×H(f)  [pm/√Hz]", type="log", secondary_y=False)
f4.update_yaxes(title_text="|H(f)|", secondary_y=True, showgrid=False, range=[0, 2.2])
save_pdf(f4, "4_windowed_asd")
