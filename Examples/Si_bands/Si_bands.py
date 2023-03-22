import numpy as np
from CASTEPbands import Spectral
import matplotlib.pyplot as plt
import matplotlib

# Set up Matplotlib for publication quality figures
matplotlib.rc('text', usetex = True)
plt.style.use("classic")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Fig, ax and properties
fig,ax=plt.subplots()
fontsize=20

# Initialise the bands data
Si_bands  = Spectral.Spectral('Si',zero_fermi=True)

# Plot simple band structure
Si_bands.plot_bs(ax)

plt.tight_layout()
plt.savefig("Si_bands.png")

