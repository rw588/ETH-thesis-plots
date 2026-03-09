#!/opt/homebrew/anaconda3/envs/eth/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

def gaussian_schell_model(x, z, sigma_I, sigma_c, wavelength=1.6e-9):
    """
    Calculate Gaussian Schell model beam intensity
    """
    # Beam parameters
    w0 = sigma_I
    z_R = np.pi * w0**2 / wavelength
    
    # Beam width at distance z
    wz = w0 * np.sqrt(1 + (z/z_R)**2)
    
    # On-axis intensity (follows 1/(1+(z/z_R)^2) law)
    I0_z = 1.0 / (1 + (z/z_R)**2)
    
    # Intensity profile
    intensity = I0_z * np.exp(-2 * x**2 / wz**2)
    
    # Coherence factor
    mu = np.exp(-x**2 / (2 * sigma_c**2))
    
    return intensity * mu

# Set up parameters
sigma_I = 50e-9  # 50 nm
sigma_c = 1.6 * 2175/70 * 1e-9  # ≈ 49.71 nm
wavelength = 1.6e-9  # 1.6 nm (EUV/X-ray region)

# Calculate Rayleigh range
z_R = np.pi * sigma_I**2 / wavelength
print(f"Rayleigh range: {z_R*1e6:.3f} μm = {z_R*1e9:.1f} nm")

# Set propagation distance
z_max = 50 * z_R  # 50 Rayleigh ranges
print(f"Propagation distance: {z_max*1e6:.3f} μm ({z_max/z_R:.0f} × Rayleigh range)")

# Create coordinate grids
x_range = 5 * sigma_I * np.sqrt(1 + (z_max/z_R)**2)
x = np.linspace(-x_range, x_range, 500)
z = np.linspace(0, z_max, 200)

# Create meshgrid
X, Z = np.meshgrid(x, z)

# Calculate beam intensity
I = np.zeros_like(X)
for i in range(len(z)):
    I[i, :] = gaussian_schell_model(x, z[i], sigma_I, sigma_c, wavelength)

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Plot 1: Absolute intensity (linear scale)
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.pcolormesh(Z*1e6, X*1e9, I, shading='auto', cmap=cm.viridis)
ax1.set_xlabel('Propagation distance z (μm)')
ax1.set_ylabel('Transverse position x (nm)')
ax1.set_title(f'Absolute Intensity\n$\sigma_I$={sigma_I*1e9:.1f} nm, $\sigma_c$={sigma_c*1e9:.2f} nm, $\lambda$=1.6 nm')
plt.colorbar(im1, ax=ax1, label='Intensity (a.u.)')
ax1.axvline(x=z_R*1e6, color='r', linestyle='--', alpha=0.5, label=f'Rayleigh range ({z_R*1e6:.3f} μm)')
ax1.legend()

# Plot 2: Absolute intensity (log scale)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.pcolormesh(Z*1e6, X*1e9, np.log10(I + 1e-10), shading='auto', cmap=cm.viridis)
ax2.set_xlabel('Propagation distance z (μm)')
ax2.set_ylabel('Transverse position x (nm)')
ax2.set_title('Log10 Intensity')
plt.colorbar(im2, ax=ax2, label='log10(Intensity)')
ax2.axvline(x=z_R*1e6, color='r', linestyle='--', alpha=0.5, label=f'Rayleigh range ({z_R*1e6:.3f} μm)')
ax2.legend()

# Plot 3: Normalized intensity
ax3 = fig.add_subplot(gs[1, 0])
I_normalized = I / np.max(I, axis=1, keepdims=True)
im3 = ax3.pcolormesh(Z*1e6, X*1e9, I_normalized, shading='auto', cmap=cm.viridis)
ax3.set_xlabel('Propagation distance z (μm)')
ax3.set_ylabel('Transverse position x (nm)')
ax3.set_title('Normalized Intensity (shows beam spreading)')
plt.colorbar(im3, ax=ax3, label='Normalized Intensity')
ax3.axvline(x=z_R*1e6, color='r', linestyle='--', alpha=0.5, label=f'Rayleigh range ({z_R*1e6:.3f} μm)')
ax3.legend()

# Plot 4: Beam profiles at different z positions
ax4 = fig.add_subplot(gs[1, 1])
z_positions = [0, 0.5*z_R, 1*z_R, 2*z_R, 5*z_R, 10*z_R, 20*z_R, 50*z_R]
# Filter to those <= z_max
z_positions = [z for z in z_positions if z <= z_max]
colors = plt.cm.viridis(np.linspace(0, 1, len(z_positions)))

for i, z_pos in enumerate(z_positions):
    intensity = gaussian_schell_model(x, z_pos, sigma_I, sigma_c, wavelength)
    beam_width = sigma_I * np.sqrt(1 + (z_pos/z_R)**2)
    ax4.plot(x*1e9, intensity, color=colors[i], 
             label=f'z={z_pos/z_R:.1f}$z_R$ ({z_pos*1e6:.2f} μm)', 
             linewidth=2)
    # Mark the 1/e² width
    ax4.axvline(x=beam_width*1e9, color=colors[i], linestyle=':', alpha=0.5)
    ax4.axvline(x=-beam_width*1e9, color=colors[i], linestyle=':', alpha=0.5)

ax4.set_xlabel('Transverse position x (nm)')
ax4.set_ylabel('Intensity (a.u.)')
ax4.set_title('Beam Profiles at Different z (dotted = 1/e² width)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Add text box with key parameters
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = f'$\sigma_I$ = {sigma_I*1e9:.1f} nm\n$\sigma_c$ = {sigma_c*1e9:.2f} nm\n$\lambda$ = 1.6 nm\n$z_R$ = {z_R*1e6:.3f} μm'
ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Print parameters
print(f"\nBeam Parameters:")
print(f"----------------------------------------")
print(f"Intensity width (σ_I): {sigma_I*1e9:.2f} nm")
print(f"Coherence width (σ_c): {sigma_c*1e9:.4f} nm")
print(f"Coherence ratio (σ_c/σ_I): {sigma_c/sigma_I:.4f}")
print(f"Wavelength: 1.6 nm")
print(f"\nPropagation Information:")
print(f"Rayleigh range: {z_R*1e6:.4f} μm")
print(f"Beam width at z=0: {sigma_I*1e9:.2f} nm")
print(f"Beam width at z={z_max/z_R:.0f}z_R: {sigma_I*np.sqrt(1+(z_max/z_R)**2)*1e9:.2f} nm")
print(f"Peak intensity reduction factor: {(1+(z_max/z_R)**2):.2e}x")