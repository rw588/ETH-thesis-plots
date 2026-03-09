import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------
# basic parameters
# ----------------------------

d = 1.0
lam = 1.0
k = 2*np.pi/lam

TL = 2*d**2/lam

mmax = 30
x = np.linspace(-2*d,2*d,1200)

I_L = 1.0
R = 1e6*d   # effectively flat wavefront

# ----------------------------
# grating Fourier coefficients
# ----------------------------

def a_m(m,f):
    if m == 0:
        return f
    return np.sin(np.pi*m*f)/(np.pi*m)

# ----------------------------
# GSM mutual intensity
# ----------------------------

def W0(x1,x2,sigma_I,sigma_c):

    envelope = np.exp(-(x1**2+x2**2)/(4*sigma_I**2))
    coherence = np.exp(-(x1-x2)**2/(2*sigma_c**2))
    curvature = np.exp(1j*k/(2*R)*(x2**2-x1**2))

    return I_L * envelope * coherence * curvature

# ----------------------------
# intensity calculation
# ----------------------------

def compute_intensity(f,sigma_I_d,sigma_c_d,L_TL):

    sigma_I = sigma_I_d*d
    sigma_c = sigma_c_d*d
    L = L_TL*TL

    mvals = np.arange(-mmax,mmax+1)
    nvals = np.arange(-mmax,mmax+1)

    I = np.zeros_like(x,dtype=complex)

    for m in mvals:

        xm = m*d
        am = a_m(m,f)

        for n in nvals:

            xn = n*d
            an = a_m(n,f)

            phase_x = np.exp(1j*2*np.pi*(m-n)*x/d)
            phase_z = np.exp(1j*np.pi*lam*L/d**2*(m**2-n**2))

            W = W0(xm,xn,sigma_I,sigma_c)

            I += am*an*phase_x*phase_z*W

    I = np.real(I)

    Imax = np.max(I)
    Imin = np.min(I)

    contrast = (Imax-Imin)/(Imax+Imin)

    return I, contrast

# ----------------------------
# initial parameters
# ----------------------------

f0 = 0.4
sigmaI0 = 20
sigmaC0 = 5
L0 = 0.5

I0, C0 = compute_intensity(f0,sigmaI0,sigmaC0,L0)

# ----------------------------
# plotting
# ----------------------------

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35)

line, = ax.plot(x/d,I0)

title = ax.set_title(f"Contrast = {C0:.3f}")

ax.set_xlabel("x / d")
ax.set_ylabel("Intensity")

# ----------------------------
# sliders
# ----------------------------

ax_f = plt.axes([0.1,0.25,0.8,0.03])
ax_sigmaI = plt.axes([0.1,0.20,0.8,0.03])
ax_sigmaC = plt.axes([0.1,0.15,0.8,0.03])
ax_L = plt.axes([0.1,0.10,0.8,0.03])

s_f = Slider(ax_f,"f",0.05,0.95,valinit=f0)
s_sigmaI = Slider(ax_sigmaI,"σ_I / d",0,10,valinit=sigmaI0)
s_sigmaC = Slider(ax_sigmaC,"σ_c / d",0,10,valinit=sigmaC0)
s_L = Slider(ax_L,"L / T_L",0,770,valinit=L0)

# ----------------------------
# update function
# ----------------------------

def update(val):

    f = s_f.val
    sigmaI = s_sigmaI.val
    sigmaC = s_sigmaC.val
    L = s_L.val

    I, C = compute_intensity(f,sigmaI,sigmaC,L)

    line.set_ydata(I)
    title.set_text(f"Contrast = {C:.3f}")

    fig.canvas.draw_idle()

s_f.on_changed(update)
s_sigmaI.on_changed(update)
s_sigmaC.on_changed(update)
s_L.on_changed(update)

plt.show()