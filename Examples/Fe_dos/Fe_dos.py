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
Fe_bands  = cb.Spectral('Fe',zero_fermi=True)

# Plot simple dos with projected mulliken
Fe_bands.plot_dos(ax,
                  pdos=True,
                  Elim=[-7,20],
                  dE=1e-1,
                  pdos_colors = ['red','green','blue','gold']) # We have 4 angular momenta, s,p,d,f so specify 4 colours

plt.tight_layout()
plt.savefig("Fe_dos.png")

