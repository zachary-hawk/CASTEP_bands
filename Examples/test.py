import numpy as np
import bands
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('text', usetex = True)
plt.style.use("classic")

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fontsize=20

dos = bands.Spectral('TiNbS',zero_fermi=True)

fig,ax=plt.subplots()
dos.plot_dos(ax,
             fontsize=fontsize,
             shade=True,
             spin_polarised=True,
             swap_axes=True,
             broadening='adaptive',
             pdos=True,
             dE=0.5e-1,
             pdos_species=[0,1,2],
             pdos_orbitals=[2],
             Elim=[-6,6],
             show_total=False,
             spin_share_axis=False)


plt.tight_layout()


plt.show()

