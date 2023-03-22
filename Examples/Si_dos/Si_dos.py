import numpy as np
import CASTEP_bands as cb
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
Si_bands  = cb.Spectral('Si',zero_fermi=True)

# Plot simple band structure
Si_bands.plot_dos(ax)

plt.tight_layout()
plt.savefig("Si_dos.png")

