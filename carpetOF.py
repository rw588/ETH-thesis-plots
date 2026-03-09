import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------
# constants
# ----------------------------

d = 1.0
lam = 1.0
k = 2*np.pi/lam

TL = d**2/lam

mmax = 25

x = np.linspace(-2*d,2*d,400)
z_vals = np.linspace(0,2*TL,200)

R = 1e6*d
I_L = 1

# ----------------------------
# Fourier coefficients
# ----------------------------

def a_m(m,f):
    if m == 0:
        return f
    return np.sin(np.pi*m*f)/(np.pi*m)

# ----------------------------
# GSM mutual intensity
# ----------------------------

def W0(x1,x2,sigmaI,sigmaC):

    envelope = np.exp(-(x1**2+x2**2)/(4*sigmaI**2))
    coherence = np.exp(-(x1-x2)**2/(2*sigmaC**2))
    curvature = np.exp(1j*k/(2*R)*(x2**2-x1**2))

    return I_L*envelope*coherence*curvature

# ----------------------------
# compute Talbot carpet
# ----------------------------

def compute_carpet(f,sigmaI_d,sigmaC_d):

    sigmaI = sigmaI_d*d
    sigmaC = sigmaC_d*d

    mvals = np.arange(-mmax,mmax+1)
    nvals = np.arange(-mmax,mmax+1)

    carpet = np.zeros((len(z_vals),len(x)))

    for zi,z in enumerate(z_vals):

        I = np.zeros_like(x,dtype=complex)

        for m in mvals:

            xm = m*d
            am = a_m(m,f)

            for n in nvals:

                xn = n*d
                an = a_m(n,f)

                phase_x = np.exp(1j*2*np.pi*(m-n)*x/d)
                phase_z = np.exp(1j*np.pi*lam*z/d**2*(m**2-n**2))

                W = W0(xm,xn,sigmaI,sigmaC)

                I += am*an*phase_x*phase_z*W

        carpet[zi,:] = np.real(I)

    return carpet

# ----------------------------
# initial parameters
# ----------------------------

f0 = 0.5
sigmaI0 = 0.5
sigmaC0 = 0.5

carpet = compute_carpet(f0,sigmaI0,sigmaC0)

# ----------------------------
# plotting
# ----------------------------

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.25)

im = ax.imshow(
    carpet,
    extent=[x[0]/d,x[-1]/d,z_vals[0]/TL,z_vals[-1]/TL],
    aspect='auto',
    origin='lower'
)

ax.set_xlabel("x / d")
ax.set_ylabel("z / T_L")
ax.set_title("Talbot carpet")

# ----------------------------
# sliders
# ----------------------------

ax_f = plt.axes([0.1,0.15,0.8,0.03])
ax_sigmaI = plt.axes([0.1,0.10,0.8,0.03])
ax_sigmaC = plt.axes([0.1,0.05,0.8,0.03])

s_f = Slider(ax_f,"f",0.05,0.95,valinit=f0)
s_sigmaI = Slider(ax_sigmaI,"σ_I / d",0.1,100,valinit=sigmaI0)
s_sigmaC = Slider(ax_sigmaC,"σ_c / d",0.1,50,valinit=sigmaC0)

# ----------------------------
# update
# ----------------------------

def update(val):

    f = s_f.val
    sigmaI = s_sigmaI.val
    sigmaC = s_sigmaC.val

    carpet = compute_carpet(f,sigmaI,sigmaC)

    im.set_data(carpet)
    fig.canvas.draw_idle()

s_f.on_changed(update)
s_sigmaI.on_changed(update)
s_sigmaC.on_changed(update)

plt.show()