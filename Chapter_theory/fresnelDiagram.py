import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import matplotlib.patches as mpatches

# Set up the figure with horizontal layout
fig, axes = plt.subplots(1, 6, figsize=(16, 6), 
                         gridspec_kw={'width_ratios': [1.2, 1.2, 0.8, 1.2, 2, 1.2]})
plt.subplots_adjust(wspace=0.3)

# x coordinate (transverse) for profiles
x = np.linspace(-3, 3, 1000)

# 1. Gaussian at z=0 - vertical profile
ax1 = axes[0]
gaussian0 = np.exp(-x**2 / 1.2)
ax1.plot(gaussian0, x, 'b-', linewidth=2)
ax1.fill_betweenx(x, 0, gaussian0, alpha=0.3, color='blue')
ax1.set_title('z = 0', fontsize=12)
ax1.set_xlabel('Intensity')
ax1.set_ylabel('x (transverse)')
ax1.set_xlim(0, 1.2)
ax1.set_ylim(-3, 3)
ax1.grid(True, alpha=0.3)
ax1.text(0.6, 2.5, 'Gaussian\nbeam', ha='center', fontsize=10)

# 2. Expanded Gaussian at z=L-
ax2 = axes[1]
gaussianL = 1.2 * np.exp(-x**2 / 2.5)
ax2.plot(gaussianL, x, 'c-', linewidth=2)
ax2.fill_betweenx(x, 0, gaussianL, alpha=0.3, color='cyan')
ax2.set_title('z = L⁻', fontsize=12)
ax2.set_xlabel('Intensity')
ax2.set_xlim(0, 1.2)
ax2.set_ylim(-3, 3)
ax2.grid(True, alpha=0.3)
ax2.text(0.6, 2.5, 'Expanded\nGaussian', ha='center', fontsize=10)

# 3. Grating at z=L
ax3 = axes[2]
# Draw grating as vertical bars
for i, xi in enumerate(np.linspace(-2.5, 2.5, 9)):
    if i % 2 == 0:  # alternate bars
        rect = Rectangle((0, xi-0.25), 0.5, 0.5, 
                        facecolor='gray', edgecolor='black', alpha=0.8)
        ax3.add_patch(rect)
ax3.set_xlim(0, 1)
ax3.set_ylim(-3, 3)
ax3.set_title('z = L', fontsize=12)
ax3.set_xlabel('Grating')
ax3.set_xticks([])
ax3.grid(False)
ax3.text(0.5, 2.5, 'Grating', ha='center', fontsize=10)

# 4. Modulated field at z=L+
ax4 = axes[3]
# Create square wave modulation
square_wave = np.zeros_like(x)
for i, xi in enumerate(x):
    if int((xi + 3) * 4) % 2 == 0:
        square_wave[i] = 1
    else:
        square_wave[i] = 0.15
modulated = 1.2 * np.exp(-x**2 / 2.5) * square_wave
ax4.plot(modulated, x, 'r-', linewidth=2)
ax4.fill_betweenx(x, 0, modulated, alpha=0.3, color='red', where=(modulated>0))
ax4.set_title('z = L⁺', fontsize=12)
ax4.set_xlabel('Field amp.')
ax4.set_xlim(0, 1.2)
ax4.set_ylim(-3, 3)
ax4.grid(True, alpha=0.3)
ax4.text(0.6, 2.5, 'Modulated\nfield', ha='center', fontsize=10)

# 5. Integral contributions (empty space for arrows)
ax5 = axes[4]
ax5.set_title('Fresnel Propagation', fontsize=12)
ax5.set_xlim(0, 1)
ax5.set_ylim(-3, 3)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.grid(False)
for spine in ax5.spines.values():
    spine.set_visible(False)

# Draw Huygens wavelets and arrows
source_points = [-2.0, -1.0, 0, 1.0, 2.0]
for i, x0 in enumerate(source_points):
    # Calculate amplitude at source (L+)
    idx = np.argmin(np.abs(x - x0))
    amp_src = modulated[idx]
    
    # Source position (x, y) in data coordinates
    src_x = 0.2 + amp_src * 0.3  # Convert to axis coordinates
    src_y = x0
    
    # Draw wavelets (semi-circles)
    for r in [0.1, 0.2, 0.3]:
        circle = plt.Circle((src_x, src_y), r, fill=False, 
                           color='orange', alpha=0.3, linestyle='--')
        ax5.add_patch(circle)
    
    # Draw arrow to observation plane
    ax5.annotate('', xy=(0.8, src_y), xytext=(src_x, src_y),
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle='->', color='orange', 
                              linewidth=2, alpha=0.7))

ax5.text(0.5, 2.2, r'integral', 
         ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# 6. Diffraction pattern at z=2L
ax6 = axes[5]
# Simulate diffraction pattern
diffraction = 0.8 * (np.sinc(x*1.5))**2 + 0.2 * np.exp(-x**2/3)
ax6.plot(diffraction, x, 'purple', linewidth=2)
ax6.fill_betweenx(x, 0, diffraction, alpha=0.3, color='purple')
ax6.set_title('z = 2L', fontsize=12)
ax6.set_xlabel('Intensity')
ax6.set_xlim(0, 1.0)
ax6.set_ylim(-3, 3)
ax6.grid(True, alpha=0.3)
ax6.text(0.5, 2.5, 'Diffraction\npattern', ha='center', fontsize=10)

# Add optical axis line
for ax in axes:
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Add z-axis at the bottom
fig.text(0.5, 0.02, 'Propagation distance z →', ha='center', fontsize=12)

# Add title
fig.suptitle('Gaussian Beam Propagation Through a Grating: Fresnel Diffraction Integral', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('grating_propagation_vertical.png', dpi=300, bbox_inches='tight')
plt.show()