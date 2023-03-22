import warnings
import numpy as np
import sys
import ase
import ase.io as io
import ase.dft.bz as bz
import os
import matplotlib.pyplot as plt
from itertools import cycle
from ase.data import atomic_numbers
import matplotlib
import time
import cycler


class Spectral:
    '''Class containing bands information from a CASTEP .bands file'''
    def __init__(self,
                 seed,
                 zero_fermi=True,
                 convert_to_eV = True,
                 flip_spins=False):
        ''' Initalise the class, it will require the CASTEP seed to read the file ''' 
        self.start_time=time.time()
        self.pdos_has_read = False
        # Functions
        def _check_sym(vec):
            fracs=np.array([0.5,0.0,0.25,0.75,0.33333333,0.66666667])
            frac=[]
            for i in vec:
                #frac.append(i.as_integer_ratio()[0])                                                                                                                         
                #frac.append(i.as_integer_ratio()[1])                                                                                                                         
                buff=[]
                for j in fracs:
                    buff.append(np.isclose(i,j))
                frac.append(any(buff))
        
        
        
            if all(frac):
                #print(vec)                                                                                                                                                   
                return True
            else:
                return False
    
        if convert_to_eV:
            eV=27.2114
            self.eV=eV
        else :
            eV=1
            self.eV=eV
        self.convert_to_eV=convert_to_eV
        self.seed=seed
        self.zero_fermi=zero_fermi 
        # First we try to open the file 
        
        # Open the bands file
        try:
            bands_file=seed+".bands"
            bands=open(bands_file,'r')
        except:
            raise Exception("No .bands file")
    
        lines=bands.readlines()
    
        no_spins=int(lines[1].split()[-1])
        no_kpoints=int(lines[0].split()[-1])
        fermi_energy=float(lines[4].split()[-1])
        
        if no_spins==1:
            fermi_energy=float(lines[4].split()[-1])
            no_electrons =float(lines[2].split()[-1])
            no_eigen  = int(lines[3].split()[-1])
            max_eig = no_eigen
            no_eigen_2=None
            spin_polarised=False
        if no_spins==2:
            spin_polarised=True
            no_eigen  = int(lines[3].split()[-2])
            no_eigen_2=int(lines[3].split()[-1])
            max_eig=np.max([no_eigen,no_eigen_2])
            n_up=float(lines[2].split()[-2])
            n_down=float(lines[2].split()[-1])
        # Set all of the bands information
        self.spin_polarised=spin_polarised
        self.Ef=fermi_energy*eV
        self.n_kpoints=no_kpoints
        if spin_polarised:
            self.nup=n_up
            self.ndown=n_down
            self.electrons=n_up+n_down
        else:
            self.nup=None
            self.ndown=None
            self.electrons=no_electrons
        self.eig_up=no_eigen
        self.eig_down=no_eigen_2
        self.n_kpoints=no_kpoints
    
    
        # bands, kpt, spin
        band_structure=np.zeros((max_eig,no_kpoints,no_spins))
    
        kpt_weights=np.zeros(no_kpoints)

        kpoint_array=np.empty(shape=(no_kpoints)) # the array holding the number of the kpoint                                                                            
        kpoint_list=[] # array of the kpoint vectors            
    
        if no_spins==1:
            kpoint_string=lines[9::no_eigen+2]
        else:
            kpoint_string=lines[9::no_eigen+3+no_eigen_2]
        for i in range(len(kpoint_string)):
            kpt_weights[i]=float(kpoint_string[i].split()[-1])
    
        for i in range(len(kpoint_string)):
            kpoint_array[i]=int(kpoint_string[i].split()[1])
            
            #Empty list for vectors                                                                                                                                       
            vec=[]
            vec.append(float(kpoint_string[i].split()[2]))
            vec.append(float(kpoint_string[i].split()[3]))
            vec.append(float(kpoint_string[i].split()[4]))
            kpoint_list.append(vec)
        # fill up the arrays

        if not zero_fermi:
            fermi_energy=0
        else:
            self.Ef=0
            
        
        for k in range(0,no_kpoints):
            if no_spins==1:
                ind=9+k*no_eigen+2*(k+1)
                band_structure[:,k,0]=eV*np.array([float(i)-fermi_energy for i in lines[ind:ind+no_eigen]])
    
            if no_spins==2:
                ind=9+k*(no_eigen+no_eigen_2+1)+2*(k+1)
                band_structure[:,k,0]=eV*np.array([float(i)-fermi_energy for i in lines[ind:ind+no_eigen]])
                band_structure[:,k,1]=eV*np.array([float(i)-fermi_energy for i in lines[ind+no_eigen+1:ind+no_eigen+1+no_eigen_2]])
        sort_array=kpoint_array.argsort()
        kpoint_array=kpoint_array[sort_array]
        kpoint_list=np.array(kpoint_list)[sort_array]
        self.kpt_sort=sort_array
        for nb in range(max_eig):
            for ns in range(no_spins):
                band_structure[nb,:,ns]=band_structure[nb,:,ns][sort_array]


        if no_spins ==2 and flip_spins:
            band_structure[:,:,[0,1]] = band_structure[:,:,[1,0]]
        
        self.kpoints=kpoint_array
        self.kpoint_list=kpoint_list
        self.kpt_weights=kpt_weights[sort_array]
        self.BandStructure=band_structure
        self.nbands=max_eig
        self.nspins=no_spins
        #do the high symmetry points

        k_ticks=[]
        for i,vec in enumerate(kpoint_list):
            if _check_sym(vec):
                k_ticks.append(kpoint_array[i])
    
        tol=1e-5

    
        kpoint_grad=[]
        for i in range(1,len(kpoint_list)):
            diff=kpoint_list[i]-kpoint_list[i-1]
            kpoint_grad.append(diff)
    
        kpoint_2grad=[]
        high_sym=[0]
        for i in range(1,len(kpoint_grad)):
            diff=kpoint_grad[i]-kpoint_grad[i-1]
            kpoint_2grad.append(diff)
            #print(diff)                                                                                                                                                  
    
            if any(np.abs(diff)>tol):
    
                # print(diff)                                                                                                                                             
                high_sym.append(i)
        high_sym.append(len(kpoint_list)-1)
        high_sym=np.array(high_sym)+1
        self.high_sym=high_sym


        # Set up the special points
        warnings.filterwarnings("ignore")
        #try:
        #sys.stdout = open(os.devnull, 'w')

        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            cell=io.read(seed+".cell")
            bv_latt=cell.cell.get_bravais_lattice()
            special_points=bv_latt.get_special_points()
            sys.stdout = old_stdout

        #sys.stdout = sys.__stdout__
        

        
        atoms=np.unique(cell.get_chemical_symbols())[::-1]
        mass=[]
        for i in atoms:
            mass.append(atomic_numbers[i])
        atom_sort=np.argsort(mass)
        mass=np.array(mass)[atom_sort]
        atoms=np.array(atoms)[atom_sort]
        self.atoms=atoms
        self.mass=mass
        

        #except:
        #sys.stdout = sys.__stdout__
        #    warnings.warn("No .cell file found for generating high symmetry labels")

        

        ticks=[""]*len(high_sym)
        found=False
        for k_count,k in enumerate(kpoint_list[high_sym-1]):
            found=False

            for i in special_points:

                if abs(special_points[i][0]-k[0])<tol and abs(special_points[i][1]-k[1])<tol and abs(special_points[i][2]-k[2])<tol:
                    if i=="G":
                        ticks[k_count]="$\Gamma$"
                    else:
                        ticks[k_count]=i
                    found=True

        self.high_sym_labels=ticks
        self.dk=np.sum((np.sum(kpoint_list,axis=0)/no_kpoints)**2)
        # We have all the info now we can break it up
        #warnings.filterwarnings('always')

        
    def _pdos_read(self,
                  species_only= False,
                  popn_select = [None,None]):
        ''' Internal function for reading the pdos_bin file. This contains all of the projected DOS from the Mulliken '''
        from scipy.io import FortranFile as FF
    
        f=FF(self.seed+'.pdos_bin', 'r','>u4')
        self.pdos_has_read = True
        version=f.read_reals('>f8')
        header=f.read_record('a80')[0]
        num_kpoints=f.read_ints('>u4')[0]
        num_spins=f.read_ints('>u4')[0]
        num_popn_orb=f.read_ints('>u4')[0]
        max_eigenvalues=f.read_ints('>u4')[0]
    
        orbital_species=f.read_ints('>u4')
        num_species=len(np.unique(orbital_species))
        orbital_ion=f.read_ints('>u4')
        orbital_l=f.read_ints('>u4')


        self.orbital_species=orbital_species
        self.num_species=num_species
        self.orbital_ion=orbital_ion
        self.orbital_l=orbital_l

        kpoints=np.zeros((num_kpoints,3))
        pdos_weights=np.zeros((num_popn_orb,max_eigenvalues,num_kpoints,num_spins))
    
        pdos_orb_spec=np.zeros((num_species,4,max_eigenvalues,num_kpoints,num_spins))
    
        for nk in range(0,num_kpoints):
            record=f.read_record('>i4','>3f8')
            kpt_index,kpoints[nk,:]=record
            for ns in range(0,num_spins):
                spin_index=f.read_ints('>u4')[0]
                num_eigenvalues=f.read_ints('>u4')[0]
    
                for nb in range(0,num_eigenvalues):
                    pdos_weights[0:num_popn_orb,nb,nk,ns]=f.read_reals('>f8')
                    norm=np.sum((pdos_weights[0:num_popn_orb,nb,nk,ns]))
                    pdos_weights[0:num_popn_orb,nb,nk,ns]=pdos_weights[0:num_popn_orb,nb,nk,ns]/norm



        # sort it based on kpoint ordering
        for i in range( len(pdos_weights[:,0,0,0])):
            for nb in range(num_eigenvalues):
                for ns in range(num_spins):
                    pdos_weights[i,nb,:,ns]=pdos_weights[i,nb,:,ns][self.kpt_sort]


        # Return the raw weights
        self.raw_pdos = pdos_weights
        
        # reshape so we can work out which bands are which                                                                                                                
        for i in range(len(orbital_species)):
            l_ind=orbital_l[i]
            spec_ind=orbital_species[i]-1
       
            pdos_orb_spec[spec_ind,l_ind,:,:,:]=pdos_orb_spec[spec_ind,l_ind,:,:,:] + pdos_weights[i,:,:,:]

        # Go through each kpoint, band and spin to find the species and orbital with highest occupancy. Then we can set it to 1 to find the mode.                         
        for nk in range(num_kpoints):
            for nb in range(max_eigenvalues):
                for ns in range(num_spins):
                    max_spec,max_l=np.where(pdos_orb_spec[:,:,nb,nk,ns]==np.max(pdos_orb_spec[:,:,nb,nk,ns]))
       
                    pdos_orb_spec[:,:,nb,nk,ns]=0
                    pdos_orb_spec[max_spec[0],max_l[0],nb,nk,ns]=1
       
        pdos_bands=np.sum(pdos_orb_spec,axis=3)
        
        band_char=np.zeros((2,max_eigenvalues,num_spins))

        
        for nb in range(0,max_eigenvalues):
            for ns in range(0,num_spins):
                max_spec,max_l=np.where(pdos_bands[:,:,nb,ns]==np.max(pdos_bands[:,:,nb,ns]))
    
                band_char[0,nb,ns]=max_spec[0]+1
                band_char[1,nb,ns]=max_l[0]
    

        # Now filter based on user input                                                                                                                                  
        popn_bands=np.zeros((max_eigenvalues,num_spins),dtype=bool)
        if popn_select[0] is not None:
            for nb in range(max_eigenvalues):
                for ns in range(num_spins):

                    if band_char[0,nb,ns]==popn_select[0] and band_char[1,nb,ns]==popn_select[1]:
                        popn_bands[nb,ns]=1
            self.popn_bands=popn_bands
            return


        if species_only:
            num_species=len(np.unique(orbital_species))
            pdos_weights_sum=np.zeros((num_species,max_eigenvalues,num_kpoints,num_spins))
        
            for i in range(0,num_species):
                loc=np.where(orbital_species==i+1)[0]
                pdos_weights_sum[i,:,:,:]=np.sum(pdos_weights[loc,:,:,:],axis=0)
        
        
        else:
            num_orbitals=4
            pdos_weights_sum=np.zeros((num_orbitals,max_eigenvalues,num_kpoints,num_spins))
            pdos_colours=np.zeros((3,max_eigenvalues,num_kpoints,num_spins))
            
            r=np.array([1,0,0])
            g=np.array([0,1,0])
            b=np.array([0,0,1])
            k=np.array([0,0,0])
            
        
        
            for i in range(0,num_orbitals):
                loc=np.where(orbital_l==i)[0]
                if len(loc)>0:
                    pdos_weights_sum[i,:,:,:]=np.sum(pdos_weights[loc,:,:,:],axis=0)
        
        pdos_weights_sum=np.where(pdos_weights_sum>1,1,pdos_weights_sum)
        pdos_weights_sum=np.where(pdos_weights_sum<0,0,pdos_weights_sum)
        self.pdos=np.round(pdos_weights_sum,7)




    def _gradient_read(self):
        ''' Internal function for reading the gradient file .dome_bin. This is used in the calculation of the adaptive broadening. If using  cite Jonathan R. Yates, Xinjie Wang, David Vanderbilt, and Ivo Souza
        Phys. Rev. B 75, 195121 ''' 
        from scipy.io import FortranFile as FF
        try:        
            f=FF(self.seed+'.dome_bin', 'r','>u4')
        except:
            raise Exception('Unable to read .dome_bin file, change broadening="gaussian" or "lorentzian".')
        version=f.read_reals('>f8')
        header=f.read_record('a80')[0]

        bands_grad=np.zeros((3,self.nbands,self.n_kpoints,self.nspins))
    
    
        for nk in range(0,self.n_kpoints):
            for ns in range(0,self.nspins):
                #for nb in range(0,self.nbands):
                bands_grad[:,:,nk,ns]=f.read_reals('>f8').reshape(3,self.nbands)

        # Convert the gradients to eV
        bands_grad=bands_grad*self.eV*0.52917720859
        grad_bands_2=np.sqrt(np.sum((bands_grad**2),axis=0))

        for nb in range(self.nbands):
            for ns in range(self.nspins):
                grad_bands_2[nb,:,ns]=grad_bands_2[nb,:,ns][self.kpt_sort]

        adaptive_weights=grad_bands_2*self.dk

        adaptive_weights[adaptive_weights<1e-2]=1e-2

        
        self.adaptive_weights=adaptive_weights

    def _split_pdos(self,species,ion=1):
        '''Internal function for splitting the pdos into various components'''
        
        self.castep_parse()
        self.pdos_read()
        #except:
        #    raise Exception("No .castep file.")
        #Do the masking
        mask = np.where(self.orbital_species==species)[0]
        orbital_species=self.orbital_species[mask]
        orbital_l=self.orbital_l[mask]
        orbital_ion=self.orbital_ion[mask]
        orbital_n = []
        if ion is not None:
            mask2=np.where(orbital_ion==ion)[0]
            orbital_species=orbital_species[mask2]
            orbital_l=orbital_l[mask2]
            orbital_ion=orbital_ion[mask2]

        
        sn=self.low_n[species-1]
        pn=self.low_n[species-1]
        dn=self.low_n[species-1]
        fn=self.low_n[species-1]

        si=0
        pi=0
        di=0
        fi=0
        
        
        s=['s']
        p=['p$_{x}$','p$_{y}$','p$_{z}$']
        d=['d$_{z^2}$','d$_{zy}$','d$_{zx}$','d$_{x^2-y^2}$','d$_{xy}$']
        f=['f$_{x^3}$','f$_{y^3}$','f$_{z^3}$','f$_{xyz}$','f$_{z(x^2-y^2)}$','f$_{y(z^2-x^2)}$','f$_{x(y^2-z^2)}$']
        labels = ['' for i in range(len(orbital_l))]

        for i in range(len(orbital_l)):
            if i>0:
                if orbital_ion[i]!=orbital_ion[i-1]:
                    sn=self.low_n[species-1]
                    pn=self.low_n[species-1]
                    dn=self.low_n[species-1]
                    fn=self.low_n[species-1]
                    
                    si=0
                    pi=0
                    di=0
                    fi=0
                    
            if orbital_l[i] == 0 :
                # s
                if sn<=dn and di>3:
                    sn=dn+1
                labels[i]=str(sn)+s[si]
                orbital_n.append(sn)
                sn+=1
            elif  orbital_l[i] == 1:
                if pi > 2:
                    pi=0
                    pn+=1
                if pn<=dn and di>3:
                    pn=dn+1
                labels[i]=str(pn)+p[pi]
                orbital_n.append(pn)
                pi+=1
            elif  orbital_l[i] == 2:
                if di > 4:
                    di=0
                    dn+=1
                labels[i]=str(dn)+d[di]
                orbital_n.append(dn)
                di+=1
            elif  orbital_l[i] == 1:
                if fi > 6:
                    fi=0
                    fn+=1
                labels[i]=str(fn)+f[fi]
                orbital_n.append(fn)
                fi+=1

        labels=labels
        
        return labels


    def plot_bs(self,
                ax,
                mono=False,
                mono_color='k',
                spin_polarised=False,
                spin_up_color='red',
                spin_down_color='blue',
                spin_up_color_hi='black',
                spin_down_color_hi='black',             
                pdos=False,
                fontsize=20,
                cmap='tab20c',
                show_fermi=True,
                fermi_line_style="--",
                fermi_line_color='0.5',
                fermi_linewidth=1,
                linestyle="-",
                linewidth=1.2,
                sym_lines=True,
                spin_index=None,
                Elim=None,
                axes_only=False,
                pdos_species=False,
                pdos_popn_select=[None,None],
                band_ids=None):
        ''' Function for plotting a Band structure, provide an ax object'''
        import matplotlib
        #cycle_color = plt.get_cmap(cmap).colors
        #plt.rcParams['axes.prop_cycle'] = cycler(color=cycle_color)

        # Set dedaults for spins
        if self.spin_polarised and spin_index is None:
            spin_index = [0,1]
        elif not self.spin_polarised and spin_index is None:
            spin_index = [0]
        if spin_index is not None:
            if not isinstance(spin_index,list) :
                spin_index=[spin_index]
                spin_polarised=True
        # spin colors
        if spin_polarised:
            spin_colors= [spin_up_color,spin_down_color]
            spin_colors_select = [spin_up_color_hi,spin_down_color_hi]

        # Set up the band ids
        band_ids_mask = np.ones((self.nbands,self.nspins),dtype=bool)
        if band_ids is not None:
            band_ids=np.array(band_ids)
            if band_ids.ndim==2:
                # We have different spins for the different bands
                for nb in range(self.nbands):
                    for ns in range(self.nspins):

                        if nb not in band_ids[:,ns]:
                            band_ids_mask[nb,ns] = False
            elif band_ids.ndim==1:
                # We have only one spin
                for nb in range(self.nbands):
                    for ns in spin_index:
                        if nb not in band_ids[:]:
                            band_ids_mask[nb,ns] = False
                
                            
        # Set the boring stuff

        if self.convert_to_eV  and self.zero_fermi :
            ax.set_ylabel(r"E-E$_{\mathrm{F}}$ (eV)",fontsize=fontsize)
        elif not self.convert_to_eV and self.zero_fermi:
            ax.set_ylabel(r"E-E$_{\mathrm{F}}$ (Ha)",fontsize=fontsize)
        elif not self.convert_to_eV and  not self.zero_fermi:
            ax.set_ylabel(r"E (Ha)",fontsize=fontsize)
        elif self.convert_to_eV and not self.zero_fermi:
            ax.set_ylabel(r"E (eV)",fontsize=fontsize)

        ax.set_xlim(1,len(self.kpoints))
        ax.tick_params(axis='both', direction='in',which='major', labelsize=fontsize*0.8,length=12,width=1.2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize*0.8,length=6, right=True, top=False, bottom=False,left=True,width=1.2)
        ax.set_xticks(self.high_sym)
        ax.set_xticklabels(self.high_sym_labels)
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        



        
        # energy lims
        if Elim is not None:
            ax.set_ylim(Elim[0],Elim[1])

        # Add in all the lines
        if show_fermi:
            ax.axhline(self.Ef,linestyle=fermi_line_style,c=fermi_line_color,linewidth=fermi_linewidth)

        if sym_lines:
            for i in self.high_sym:
                ax.axvline(i,color='k',linewidth=1)


        # We have set up the axes, not we terminate if the user only wants axes
        if axes_only:
            return

        
        # Do the standard plotting, no pdos here
        if not pdos:
        # Here we plot all of the bands. We can provide a mechanism latter for plotting invididual ones
            for nb in range(self.nbands):
                for ns in spin_index:

                    if not band_ids_mask[nb,ns]:
                        continue
                            # Mono
                    if mono:
                        ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth,color=mono_color)

                    elif spin_polarised:
                        ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth,color=spin_colors[ns])
                    else:
                        ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth)
            
        #now pdos is a thing
        else:
            #calculate the pdos if needed                                                                                                                                     
            self._pdos_read(pdos_species,pdos_popn_select)

            # first do the plotting with the popn_select
            if pdos_popn_select[0] is not None:
                for nb in range(self.nbands):
                    for ns in spin_index:
                        if not band_ids_mask[nb,ns]:
                            continue

                        # Mono
                        if mono:
                            if self.popn_bands[nb,ns]:
                                ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth,color=mono_color_select)
                            else:
                                ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth,color=mono_color)

                        elif spin_polarised:
                            if self.popn_bands[nb,ns]:
                                ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth,color=spin_colors_select[ns])
                            else:
                                ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linestyle=linestyle,linewidth=linewidth,color=spin_colors[ns])
                        else:
                            raise Exception("Highlighting by population analysis unavailable for non-mono plots.")
            #Now we do the horrid part of plotting the colors                
            else:
                from matplotlib import colors
                from matplotlib.colors import ListedColormap
                from matplotlib.lines import Line2D
                import matplotlib.collections as mcoll
                import matplotlib.path as mpath
                # Define the colours we'll use for the plotting
                n_colors=cycle(['blue','red','green','black','purple','orange','yellow','cyan'])
                
                def make_segments(x, y):
                    """                                                                                                                                                   
                    Create list of line segments from x and y coordinates, in the correct format                                                                          
                    for LineCollection: an array of the form numlines x (points per line) x 2 (x                                                                          
                    and y) array                                                                                                                                          
                    """
    
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
    
    
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
                    return segments
    
                def colorline(
                        x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
                        linewidth=3, alpha=1.0):
                    """                                                                                                                                                   
                    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb                                                          
                    http://matplotlib.org/examples/pylab_examples/multicolored_line.html                                                                                  
                    Plot a colored line with coordinates x and y                                                                                                          
                    Optionally specify colors in the array z                                                                                                              
                    Optionally specify a colormap, a norm function and a line width                                                                                       
                    """
    
                    # Default colors equally spaced on [0,1]:                                                                                                             
                    if z is None:
                        z = np.linspace(0.0, 1.0, len(x))
                    z = np.asarray(z)
                    segments = make_segments(x, y)
                    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                              linewidth=linewidth, alpha=alpha)
                    ax.add_collection(lc)
                    return lc
    
    
                if pdos_species:
                    n_cat=len(self.atoms )
                else:
                    n_cat=4
    
    
                basis=[]
                for i in range(n_cat):
                    basis.append(np.array(colors.to_rgba(next(n_colors))))
    
    
    
                for nb in range(self.nbands):
                    for ns in spin_index:

                        if not band_ids_mask[nb,ns]:
                            continue

                        # calculate the colour                                                                                                                                
                        cmap_array=np.zeros((len(self.kpoints),4))
                        for i in range(n_cat):
                            
                            cmap_array[:,0]+=self.pdos[i,nb,:,ns]*basis[i][0]#/n_cat                                                                                        
                            cmap_array[:,1]+=self.pdos[i,nb,:,ns]*basis[i][1]#/n_cat                                                                                        
                            cmap_array[:,2]+=self.pdos[i,nb,:,ns]*basis[i][2]#/n_cat                                                                                        
                            cmap_array[:,3]+=self.pdos[i,nb,:,ns]*basis[i][3]#/n_cat                                                                                        
                            
                            #cmap_array[:,0:3]=cmap_array[:,0:3]/n_cat                                                                                                        
                            cmap_array=np.where(cmap_array>1,1,cmap_array)
                            cmap = ListedColormap(cmap_array)
    
                        z = np.linspace(0, 1, len(self.kpoints))
                        colorline(self.kpoints, self.BandStructure[nb,:,ns], z, cmap=cmap, linewidth=3)
                        ax.plot(self.kpoints,self.BandStructure[nb,:,ns],linewidth=linewidth,alpha=0)
                            
                custom_lines = []
                labels=[]
                for i in range(n_cat):
                    custom_lines.append(Line2D([0], [0], color=basis[i], lw=3))
                    if pdos_species:
                        labels.append(self.atoms[i])
                    else:
                        labels=["s","p","d","f"]
    
    
                ax.legend(custom_lines,labels,fontsize=fontsize)
        return
    def pdos_filter(self,species,l,ion=None):
        ''' Function for filtering the pdos for a particular species, ion and angular momentum'''
        ls=np.where(self.orbital_l ==l)[0]
        ss=np.where(self.orbital_species == species+1)[0]


        cross = np.intersect1d(ls,ss)

        if ion is not None:
            ions=np.where(self.orbital_ion==ion)[0]
            cross=np.intersect1d(cross,ions)
        return self.raw_pdos[cross,:,:,:]

    def plot_dos(self,
                 ax,
                 spin_polarised=None,
                 color='black',
                 spin_up_color='red',
                 spin_down_color='blue',
                 spin_share_axis = False,
                 dE=None,
                 fontsize=20,
                 cmap='tab20c',
                 show_fermi=True,
                 fermi_line_style="--",
                 fermi_line_color='0.5',
                 fermi_linewidth=1,
                 zero_line=True,
                 zero_linestyle = "-",
                 zero_linewidth=1,
                 zero_line_color="black",
                 linestyle="-",
                 linewidth=1.2,
                 spin_index=None,
                 Elim=None,
                 glim=None,
                 swap_axes=False,
                 axes_only=False,
                 labelx=True,
                 labely=True,
                 pdos=False,
                 pdos_colors=None,
                 show_total=False,
                 pdos_species=None,
                 pdos_orbitals=None,
                 shade=False,
                 alpha=0.4,
                 temperature = None,
                 broadening="adaptive",
                 width=0.05,
                 loc='upper right'):
        ''' Function for calculating and plotting a DOS '''
        
        def _fermi_dirac(T,E):

            if T==None:
                return 1.0

            elif T==0:
                fd=np.ones(len(E))
                fd[E>self.Ef] = 0
                return fd
            else:
                K=8.617333e-5
                beta=1/(K*T)
                
                return 1/(np.exp(beta*(E-self.Ef))+1)
        
        def _gaussian(Ek,E,width):
            dist=Ek-E
            mask=np.where(np.abs(dist)<5/self.eV)
            result=np.zeros(np.shape(dist))
            factor= 1/(width*np.sqrt(2*np.pi))
            exponent= np.exp(-0.5*np.square(dist[mask]/(width)))
            result[mask] = factor*exponent
            return result

        def _adaptve(Ek,E,width):
            dist=Ek-E
            mask=np.where(np.abs(dist)<5/self.eV)
            result=np.zeros(np.shape(dist))
            factor= 1/(width*np.sqrt(2*np.pi))
            exponent= np.exp(-0.5*np.square(dist[mask]/(width[mask])))
            result[mask] = factor*exponent
            return result

        def _lorentzian(Ek,E,width=width):
            return 1/(np.pi*width)*(width**2/((Ek-E)**2+width**2))

        def _adaptive(Ek,E,width):
            return 1/(width*np.sqrt(2*np.pi))*np.exp(-0.5*((Ek-E)/(width))**2)

        self.start_time=time.time()
        # Set dedaults for spins
        if spin_polarised is None:
            spin_polarised=self.spin_polarised
        if self.spin_polarised and spin_index is None:
            spin_index = [0,1]
        elif not self.spin_polarised and spin_index is None:
            spin_index = [0]
        if spin_index is not None:
            if not isinstance(spin_index,list) :
                spin_index=[spin_index]
                spin_polarised=True
        # spin colors
        if spin_polarised:
            spin_colors= [spin_up_color,spin_down_color]


        # Orbital defs
        orbs=['$s$','$p$','$d$','$f$']
        # Set energy spacing
        if dE is None:
            dE=self.dk

        # Set the boring stuff
        if swap_axes:

            if self.convert_to_eV  and self.zero_fermi :
                ax.set_ylabel(r"E-E$_{\mathrm{F}}$ (eV)",fontsize=fontsize)
            elif not self.convert_to_eV and self.zero_fermi:
                ax.set_ylabel(r"E-E$_{\mathrm{F}}$ (Ha)",fontsize=fontsize)
            elif not self.convert_to_eV and  not self.zero_fermi:
                ax.set_ylabel(r"E (Ha)",fontsize=fontsize)
            elif self.convert_to_eV and not self.zero_fermi:
                ax.set_ylabel(r"E (eV)",fontsize=fontsize)
            
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8,length=12,width=1.2)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize*0.8,length=6, right=True, top=False, bottom=False,left=True,width=1.2)
            
            ax.set_xlabel(r"$\mathit{g}(\mathit{E}$) (states/eV)",fontsize=fontsize)
        else:        
            if self.convert_to_eV and self.zero_fermi:
                ax.set_xlabel(r"E-E$_{\mathrm{F}}$ (eV)",fontsize=fontsize)
            elif not self.convert_to_eV and self.zero_fermi:
                ax.set_xlabel(r"E-E$_{\mathrm{F}}$ (Ha)",fontsize=fontsize)
            elif not self.convert_to_eV and  not self.zero_fermi:
                ax.set_xlabel(r"E (Ha)",fontsize=fontsize)
            elif self.convert_to_eV and not self.zero_fermi:
                ax.set_xlabel(r"E (eV)",fontsize=fontsize)
            
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8,length=12,width=1.2)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize*0.8,length=6, right=True, top=False, bottom=True,left=True,width=1.2)
            
            ax.set_ylabel(r"$\mathit{g}(\mathit{E}$) (states/eV)",fontsize=fontsize)


        ax.minorticks_on()

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

        if not labely:
            ax.set_ylabel('')
        if not labelx:
            ax.set_xlabel('')
        
        # energy lims
        if Elim is not None:
            if swap_axes:
                ax.set_ylim(Elim[0],Elim[1])
            else:
                ax.set_xlim(Elim[0],Elim[1])
        if glim is not None:
            if swap_axes:
                ax.set_xlim(glim[0],glim[1])
            else:
                ax.set_ylim(glim[0],glim[1])
        # Add in all the lines
        if show_fermi:
            if swap_axes:
                ax.axhline(self.Ef,linestyle=fermi_line_style,c=fermi_line_color)
            else:
                ax.axvline(self.Ef,linestyle=fermi_line_style,c=fermi_line_color)
        if zero_line:
            if swap_axes:
                ax.axvline(0,linestyle=zero_linestyle,linewidth=zero_linewidth,color=zero_line_color)
            else:
                ax.axhline(0,linestyle=zero_linestyle,linewidth=zero_linewidth,color=zero_line_color)
        # We have set up the axes, not we terminate if the user only wants axes



        if axes_only:
            return

        # Set up the calculation limits
        E = np.arange(np.min(self.BandStructure),np.max(self.BandStructure),dE)
        #E = np.arange(-2,2,dE)
        
        dos=np.zeros((len(E),self.nspins))
        spin_dir=np.ones((len(E),self.nspins))
        all_dos=np.zeros((self.nbands,self.n_kpoints,self.nspins,len(E)))

        # Initialise the class object to prevent recaculation
        recalculate_dos = True
        recalculate_pdos= True
        try:
            if self.all_dos.shape==all_dos.shape:
                recalculate_dos = False
                all_dos=self.all_dos
        except:
            recalculate_dos = True
            
        
        
        if broadening=='adaptive':
            print("Please cite 'Jonathan R. Yates, Xinjie Wang, David Vanderbilt, and Ivo Souza Phys. Rev. B 75, 195121' in all publications including these DOS.")
            self._gradient_read()
            
        # Set up the pdos stuff
        if pdos:

            if not self.pdos_has_read:                
                self._pdos_read()

                
            if pdos_species is None:
                pdos_species=np.arange(0,len(self.atoms))
            if pdos_orbitals is None:
                pdos_orbitals=np.array([0,1,2,3])                                       
            pdos_dos=np.zeros((len(self.atoms),4,len(E),self.nspins))
            try:
                if self.pdos_dos.shape==pdos_dos.shape:
                    recalculate_pdos = False
                    pdos_dos=self.pdos_dos
            except:
                recalculate_pdos = True                

        if len(spin_index)==2 and spin_polarised:
          if spin_share_axis:
              spin_dir[:,1]=1
          else:
              spin_dir[:,1]=-1

        # We now decide if we are plotting a pdos or not

        if not pdos:
            '''
            for s in spin_index:
    
                if broadening=='adaptive':
                    # we are going to brute force it for now
                    for nb in range(self.nbands):
                        for nk in range(self.n_kpoints):
                            dos[:,s]+=_adaptive(self.BandStructure[nb,nk,s],E,self.adaptive_weights[nb,nk,s])*self.kpt_weights[nk]
                elif broadening=='gaussian':
                    # we are going to brute force it for now
                    for nb in range(self.nbands):
                        for nk in range(self.n_kpoints):
                            dos[:,s]+=_gaussian(self.BandStructure[nb,nk,s],E,width)*self.kpt_weights[nk]
                elif broadening=='lorentzian':
                    # we are going to brute force it for now
                    for nb in range(self.nbands):
                        for nk in range(self.n_kpoints):
                            dos[:,s]+=_lorentzian(self.BandStructure[nb,nk,s],E)*self.kpt_weights[nk]
                                
                    
                else:
                    raise Exception('Unknown broadening scheme')
            '''
            if recalculate_dos:
                if broadening=='gaussian':
                    # Lets change the shape of the bandstructure first
                    new_bs= np.repeat(self.BandStructure[:,:,:,np.newaxis],len(E),axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:,np.newaxis],self.nspins,axis=1)            
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:,:,np.newaxis],len(E),axis=2)                
    
                    for nb in range(self.nbands):            
                        for ns in range(self.nspins):
                            all_dos[nb,:,ns,:]=_gaussian(new_bs[nb,:,ns,:],E,width)*new_kpt_w[:,ns]
    
    
                            
                            
                elif broadening=='adaptive':
                    # Lets change the shape of the bandstructure first
                    new_bs= np.repeat(self.BandStructure[:,:,:,np.newaxis],len(E),axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:,np.newaxis],self.nspins,axis=1)            
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:,:,np.newaxis],len(E),axis=2)
                    new_weights = np.repeat(self.adaptive_weights[:,:,:,np.newaxis],len(E),axis=3)
    
                    for nb in range(self.nbands):            
                        for ns in range(self.nspins):
                            all_dos[nb,:,ns,:]=_adaptive(new_bs[nb,:,ns,:],E,new_weights[nb,:,ns,:])*new_kpt_w[:,ns]
                # Done, now store for next time
                self.all_dos = all_dos
                #print("Recaculated")
            
            # Multiply in the FD and spin flips

            dos = np.sum(np.sum(all_dos,axis=0),axis=0)
            dos= np.swapaxes(dos,0,1)
            
            dos=dos*spin_dir
            dos = dos * np.expand_dims(_fermi_dirac(temperature,E), axis=-1)

            # Sum over spins if not spin_polarised
            if not spin_polarised and self.spin_polarised:
                dos=np.sum(dos,axis=1)
            
            # Actually plot now
            if not swap_axes:
    
                if spin_polarised:
                    for s in spin_index:
                        ax.plot(E,dos[:,s],linestyle=linestyle,linewidth=linewidth,color=spin_colors[s])
                        if shade:
                            ax.fill_between(E,dos[:,s],color=spin_colors[s],alpha=alpha)
                            
        
                else:
                    ax.plot(E,dos,linestyle=linestyle,linewidth=linewidth,color=color)
                    if shade:
                        ax.fill_between(E,dos,color=color,alpha=alpha)
            else:
                if spin_polarised:
                    for s in spin_index:
                        ax.plot(dos[:,s],E,linestyle=linestyle,linewidth=linewidth,color=spin_colors[s])
                        if shade:
                            ax.fill_betweenx(E,dos[:,s],color=spin_colors[s],alpha=alpha)
                            
        
                else:
                    ax.plot(dos,E,linestyle=linestyle,linewidth=linewidth,color=color)
                    if shade:
                        ax.fill_betweenx(E,dos,color=color,alpha=alpha)
    
        else:
            '''
            for s in spin_index:
                for nb in range(self.nbands):
                    for nk in range(self.n_kpoints):            
                        if broadening=='adaptive':
                            # we are going to brute force it for now
                            temp_dos=_adaptive(self.BandStructure[nb,nk,s],E,self.adaptive_weights[nb,nk,s])*self.kpt_weights[nk]
                        elif broadening=='gaussian':
                            # we are going to brute force it for now
                            temp_dos=_gaussian(self.BandStructure[nb,nk,s],E,width)*self.kpt_weights[nk]
                        elif broadening=='lorentzian':
                            # we are going to brute force it for now                
                            tempdos=_lorentzian(self.BandStructure[nb,nk,s],E)*self.kpt_weights[nk]
                                        
                        
                        else:
                            raise Exception('Unknown broadening scheme')

                        # figure out the pdos factor
                        for ispec in pdos_species:
                            for iorb in pdos_orbitals:

                                pdos_factor = np.sum(self.pdos_filter(ispec,iorb)[:,nb,nk,s])
                                pdos_dos[ispec,iorb,:,s] = pdos_dos[ispec,iorb,:,s] + temp_dos*pdos_factor

            '''
            if recalculate_pdos:
                if broadening=='gaussian':
                    # Lets change the shape of the bandstructure first
                    new_bs= np.repeat(self.BandStructure[:,:,:,np.newaxis],len(E),axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:,np.newaxis],self.nspins,axis=1)            
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:,:,np.newaxis],len(E),axis=2)                
    
                    for nb in range(self.nbands):            
                        for ns in range(self.nspins):
                            all_dos[nb,:,ns,:]=_gaussian(new_bs[nb,:,ns,:],E,width)*new_kpt_w[:,ns]
                    
                    for ispec in pdos_species:
                        for iorb in pdos_orbitals:                    
                            pdos_factor = np.sum(self.pdos_filter(ispec,iorb),axis=0)
                            pdos_factor= np.repeat(pdos_factor[:,:,:,np.newaxis],len(E),axis=3)
                            # Multiply in the factor
                            temp_dos = all_dos *pdos_factor 
    
                            # Sum over the bands, kpoints
                            temp_dos= np.sum(temp_dos,axis=0)
                            temp_dos= np.sum(temp_dos,axis=0)
                            pdos_dos[ispec,iorb,:,:]= np.swapaxes(temp_dos,0,1)
                            
    
                elif broadening=='adaptive':
                    # Lets change the shape of the bandstructure first
                    new_bs= np.repeat(self.BandStructure[:,:,:,np.newaxis],len(E),axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:,np.newaxis],self.nspins,axis=1)            
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:,:,np.newaxis],len(E),axis=2)
                    new_weights = np.repeat(self.adaptive_weights[:,:,:,np.newaxis],len(E),axis=3)
    
                    for nb in range(self.nbands):            
                        for ns in range(self.nspins):
                            all_dos[nb,:,ns,:]=_adaptive(new_bs[nb,:,ns,:],E,new_weights[nb,:,ns,:])*new_kpt_w[:,ns]
    
    
                    for ispec in pdos_species:
                        for iorb in pdos_orbitals:                    
                            pdos_factor = np.sum(self.pdos_filter(ispec,iorb),axis=0)
                            #print(self.atoms[ispec],orbs[iorb],np.max(pdos_factor),np.min(pdos_factor))
                            pdos_factor= np.repeat(pdos_factor[:,:,:,np.newaxis],len(E),axis=3)
                            # Multiply in the factor
    
                            temp_dos = all_dos *pdos_factor 
    
                            # Sum over the bands, kpoints
                            temp_dos= np.sum(temp_dos,axis=0)
                            temp_dos= np.sum(temp_dos,axis=0)
                            pdos_dos[ispec,iorb,:,:]= np.swapaxes(temp_dos,0,1)
                # Done, store for subsequent runs
                self.pdos_dos=pdos_dos
                        
            # Multiply in the FD and spin flips

            pdos_dos=pdos_dos*spin_dir
            pdos_dos=pdos_dos * np.expand_dims(_fermi_dirac(temperature,E), axis=-1)

            if show_total:
                all_dos=np.swapaxes(np.sum(np.sum(all_dos,axis=0),axis=0),0,1)
                all_dos=all_dos*spin_dir
                all_dos=all_dos * np.expand_dims(_fermi_dirac(temperature,E), axis=-1)

                
            # Sum over spins if not spin_polarised
            if not spin_polarised and self.spin_polarised:
                pdos_dos=np.sum(pdos_dos,axis=3)


            
            # Set up the color stuff

            if pdos_colors is not None:
                try:
                    assert len(pdos_colors)==len(pdos_orbitals)*len(pdos_species)
                    color=pdos_colors
                    custom_cycler = (cycler.cycler(color=color))
                    ax.set_prop_cycle(custom_cycler)
                
                except:
                    warnings.warn("Warning: pdos_colors does not match number of colors")
                    n_lines=len(pdos_orbitals)*len(pdos_species)
                    color = plt.cm.bwr(np.linspace(0,1,n_lines))
                    for i in range(len(color)):
                        if np.all(np.round(color[i,0:3])==1):
                            color[i,0:3] = [0.5,0.5,0.5]
                    custom_cycler = (cycler.cycler(color=color))
                    ax.set_prop_cycle(custom_cycler)
               
            else:
                n_lines=len(pdos_orbitals)*len(pdos_species)
                color = plt.cm.bwr(np.linspace(0,1,n_lines))

                for i in range(len(color)):
                    if np.all(np.round(color[i,0:3])==1):
                        color[i,0:3] = [0.5,0.5,0.5]
                custom_cycler = (cycler.cycler(color=color))
                ax.set_prop_cycle(custom_cycler)
    

            # Actually plot now
            if not swap_axes:
                for s in spin_index:
                    if show_total:
                        ax.plot(E,all_dos[:,s],linestyle=linestyle,linewidth=linewidth,color='0.5')

                for ispec in pdos_species:
                    for iorb in pdos_orbitals:

                        color=next(ax._get_lines.prop_cycler)['color']

                        for s in spin_index:                                
                            ax.plot(E,pdos_dos[ispec,iorb,:,s],linestyle=linestyle,linewidth=linewidth,label=self.atoms[ispec]+"("+orbs[iorb]+')',color=color,zorder=len(pdos_species)-ispec)
                            if shade:
                                ax.fill_between(E,pdos_dos[ispec,iorb,:,s],alpha=alpha,color=color,zorder=len(pdos_species)-ispec)

            else:

                for s in spin_index:
                    if show_total:
                        ax.plot(all_dos[:,s],E,linestyle=linestyle,linewidth=linewidth,color='0.5')


                for ispec in pdos_species:
                    for iorb in pdos_orbitals:

                        color=next(ax._get_lines.prop_cycler)['color']

                        for s in spin_index:                                
                            ax.plot(pdos_dos[ispec,iorb,:,s],E,linestyle=linestyle,linewidth=linewidth,label=self.atoms[ispec]+"("+orbs[iorb]+')',color=color,zorder=len(pdos_species)-ispec)
                            if shade:
                                ax.fill_betweenx(E,pdos_dos[ispec,iorb,:,s],alpha=alpha,color=color,zorder=len(pdos_species)-ispec)
    


            # end pdos check 



            handles, labels = plt.gca().get_legend_handles_labels()
            
            ids = np.unique(labels, return_index=True)[1]
            labels=np.array(labels)[np.sort(ids)]

            handles = [handles[i] for i in np.sort(ids)]

            ax.legend(handles, labels, loc=loc,fontsize=fontsize*0.8,ncol=int(np.ceil((len(pdos_orbitals)*len(pdos_species))/4)),fancybox=True,frameon=False,handlelength=1.5,handletextpad=0.2)


        #print("Total time =",time.time()-self.start_time)
        # Autoscaling
        if swap_axes:
            if glim is None and Elim is not None: 
                self._autoscale(ax,'x')
        else:
            if Elim is not None and glim is None:
                self._autoscale(ax,'y')
        return




                
    def kpt_where(self,label):
        ''' Find the kpoint indices that correspond to a particular high symmetry label'''
        if label == "G":
            label = "$\\Gamma$"
        lab_loc = np.where(np.array(self.high_sym_labels) == label)[0]
        return self.high_sym[lab_loc]





    def _autoscale(self,ax=None, axis='y', margin=0.1):
        '''Autoscales the x or y axis of a given matplotlib ax object
        to fit the margins set by manually limits of the other axis,
        with margins in fraction of the width of the plot
    
        Defaults to current axes object if not specified.
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            ax = plt.gca()
        newlow, newhigh = np.inf, -np.inf
    
        for artist in ax.collections + ax.lines:
            x,y = self._get_xy(artist)
            if axis == 'y':
                setlim = ax.set_ylim
                lim = ax.get_xlim()
                fixed, dependent = x, y
            else:
                setlim = ax.set_xlim
                lim = ax.get_ylim()
                fixed, dependent = y, x
    
            low, high = self._calculate_new_limit(fixed, dependent, lim)
            newlow = low if low < newlow else newlow
            newhigh = high if high > newhigh else newhigh
    
        margin = margin*(newhigh - newlow)
    
        setlim(newlow-margin, newhigh+margin)
    
    def _calculate_new_limit(self,fixed, dependent, limit):
        '''Calculates the min/max of the dependent axis given 
        a fixed axis with limits
        '''
        if len(fixed) > 2:
            mask = (fixed>limit[0]) & (fixed < limit[1])
            window = dependent[mask]
            low, high = window.min(), window.max()
        else:
            low = dependent[0]
            high = dependent[-1]
            if low == 0.0 and high == 1.0:
                # This is a axhline in the autoscale direction
                low = np.inf
                high = -np.inf
        return low, high
    
    def _get_xy(self,artist):
        '''Gets the xy coordinates of a given artist
        '''
        if "Collection" in str(artist):
            x, y = artist.get_offsets().T
        elif "Line" in str(artist):
            x, y = artist.get_xdata(), artist.get_ydata()
        else:
            raise ValueError("This type of object isn't implemented yet")
        return x, y
