import numpy as np
import CASTEP_bands as cb
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

# Set up Matplotlib for publication quality figures
matplotlib.rc('text', usetex = True)
plt.style.use("classic")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# A more complicated version with bands and dos side by side using gridspec

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(1, 2,wspace=0.15, width_ratios = [3,1.5])

bs_ax = fig.add_subplot(gs[0])
dos_ax = fig.add_subplot(gs[1])


fontsize=20

# Initialise the bands data
Fe_bands  = cb.Spectral('Fe_bands',zero_fermi=True)
# Initialise the dos dat
Fe_dos  = cb.Spectral('Fe_dos',zero_fermi=True)

# Band structure axis, add some customisations

Fe_bands.plot_bs(bs_ax,
                 mono=True,
                 mono_color='blue')

# Dos axis, turned on its side
Fe_dos.plot_dos(dos_ax,
                swap_axes=True,
                color='blue',
                labely = False)

plt.tight_layout()
plt.savefig("Fe_bands_dos.png")

