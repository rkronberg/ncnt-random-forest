
"""
Ad hoc data preprocessing for ML
of hydrogen adsorption on NCNTs

Based on CP2K output

author: Rasmus Kronberg
rasmus.kronberg@aalto.fi
"""

# Load necessary packages
import numpy as np
import subprocess
from ase.io import read, write
from ase.neighborlist import neighbor_list as NL
from tqdm import tqdm

def cart2pol(x, y):

    # Auxiliary function for transforming Cartesian 
    # coordinates to polar

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(r, theta)

def cylDist(xyz,atom1,atom2):

    # Auxiliary function to calculate curvilinear distance 
    # between two atoms on a cylinder (nanotube)
    
    x1 = xyz[atom1].position[0]
    x2 = xyz[atom2].position[0]
    y1 = xyz[atom1].position[1]
    y2 = xyz[atom2].position[1]
    z1 = xyz[atom1].position[2]
    z2 = xyz[atom2].position[2]

    r1, theta1 = cart2pol(x1,y1)
    r2, theta2 = cart2pol(x2,y2)

    r = np.mean([r1,r2])
    theta = theta1-theta2
    z = abs(z1-z2)

    #Check PBC
    if z > xyz.get_cell()[2,2]/2:
        z = xyz.get_cell()[2,2] - z

    return np.sqrt((r*theta)**2 + z**2)

def main():

    ####################################################################

    # Preparations

    print('Launching script for parsing features from CP2K output and .xyz coordinate files...')
    print('Implementation is very ad hoc, proceed with caution and check your results!\n')
    print('Processing...')

    Ead = []        # Adsorption energy
    cV = []         # Concentration of vacancies
    cN = []         # Concentration of nitrogen dopants
    cH = []         # Concentration of hydrogens
    Z = []          # Atomic number of adsorption site
    rmsd = []       # RMSD of atomic positions during geoopt
    rmaxsd = []     # Root maximum squared displacement of atomic positions
    dminNS = []     # Distance from adsorption site S to closest N
    daveNS = []     # Average distance from adsorption site to all N
    dminHS = []     # Distance from adsorption site to closest occupied site
    daveHS = []     # Averae distance from adsorption site to all occupied sites
    mult = []       # Nanotube multiplicity
    chir = []       # Nanotube chiral angle
    Egap = []       # HOMO-LUMO gap
    mu = []         # Hirshfeld spin moment on adsorption site
    q = []          # Residual Hirshfeld charge on adsorption site
    cnN = []        # Coordination number (CN) of closest N
    dcnN = []       # Change in CN_N during geoopt
    cnS = []        # CN of adsorption site
    dcnS = []       # Change in adsorption site CN
    aminS = []      # Smallest angle at the adsorption site
    amaxS = []      # Largest angle at the adsorption site
    aminN = []      # Smallest C-N-C angle
    amaxN = []      # Largest C-N-C angle
    adispN = []     # Angular displacement of S wrt. closest dopant
    adispH = []     # Angular displacement of S wrt. closest occupied site

    # Get H2 energy
    str1 = "grep 'ENERGY|' ../refs/H2/h2-geoopt.out | tail -1 | awk '{print $9}'"
    h2Ener = float(subprocess.check_output(str1,shell=True))

    # Dictionary of systems and number of hydrogen configurations
    dirs = {'1N_1H_14x0_graphitic': 119,
            '1N_1H_14x0_pyridinic': 102,
            '1N_1H_14x0_pyrrolic': 102,
            '1N_1H_14x0_pyrrolic_SW_A': 103,
            '1N_1H_14x0_pyrrolic_SW_B': 103,
            '2N_1H_14x0_graphitic': 103,
            '3N_1H_14x0_pyridinic': 102,
            '4N_1H_14x0_pyridinic': 101,
            '1N_1H_8x8_graphitic': 118,
            '1N_1H_8x8_pyridinic': 103,
            '1N_1H_8x8_pyrrolic': 103,
            '1N_1H_8x8_pyrrolic_SW_A': 104,
            '1N_1H_8x8_pyrrolic_SW_B': 104,
            '2N_1H_8x8_graphitic': 104,
            '3N_1H_8x8_pyridinic': 103,
            '4N_1H_8x8_pyridinic': 102,
            '1N_2H_14x0_graphitic': 102,
            '1N_2H_14x0_pyridinic': 101,
            '1N_2H_14x0_pyrrolic': 102,
            '1N_2H_14x0_pyrrolic_SW_A': 102,
            '1N_2H_14x0_pyrrolic_SW_B': 102,
            '2N_2H_14x0_graphitic': 102,
            '3N_2H_14x0_pyridinic': 102,
            '4N_2H_14x0_pyridinic': 101,
            '1N_2H_8x8_graphitic': 96,
            '1N_2H_8x8_pyridinic': 103,
            '1N_2H_8x8_pyrrolic': 103,
            '1N_2H_8x8_pyrrolic_SW_A': 103,
            '1N_2H_8x8_pyrrolic_SW_B': 96,
            '2N_2H_8x8_graphitic': 103,
            '3N_2H_8x8_pyridinic': 103,
            '4N_2H_8x8_pyridinic': 102,
            '1N_3H_14x0_graphitic': 101,
            '1N_3H_14x0_pyridinic': 100,
            '1N_3H_14x0_pyrrolic': 102,
            '1N_3H_14x0_pyrrolic_SW_A': 101,
            '1N_3H_14x0_pyrrolic_SW_B': 101,
            '2N_3H_14x0_graphitic': 101,
            '3N_3H_14x0_pyridinic': 102,
            '4N_3H_14x0_pyridinic': 101,
            '1N_3H_8x8_graphitic': 95,
            '1N_3H_8x8_pyridinic': 102,
            '1N_3H_8x8_pyrrolic': 103,
            '1N_3H_8x8_pyrrolic_SW_A': 102,
            '1N_3H_8x8_pyrrolic_SW_B': 95,
            '2N_3H_8x8_graphitic': 95,
            '3N_3H_8x8_pyridinic': 103,
            '4N_3H_8x8_pyridinic': 103,
            '1N_4H_14x0_graphitic': 100,
            '1N_4H_14x0_pyridinic': 102,
            '1N_4H_14x0_pyrrolic': 102,
            '1N_4H_14x0_pyrrolic_SW_A': 100,
            '1N_4H_14x0_pyrrolic_SW_B': 100,
            '2N_4H_14x0_graphitic': 100,
            '3N_4H_14x0_pyridinic': 101,
            '4N_4H_14x0_pyridinic': 100,
            '1N_4H_8x8_graphitic': 94,
            '1N_4H_8x8_pyridinic': 97,
            '1N_4H_8x8_pyrrolic': 103,
            '1N_4H_8x8_pyrrolic_SW_A': 94,
            '1N_4H_8x8_pyrrolic_SW_B': 94,
            '2N_4H_8x8_graphitic': 94,
            '3N_4H_8x8_pyridinic': 95,
            '4N_4H_8x8_pyridinic': 101,
    }

    ###########################################################################

    # Loop over the different nanotubes in dirs
    for d in tqdm(dirs):
        # Reference energy (one less hydrogen)
        str2 = "grep 'ENERGY|' ../refs/%s/ncnt-geoopt.out | tail -1 | awk '{print $9}'" % d
        refEner = float(subprocess.check_output(str2,shell=True))

        # Unoptimized reference
        ref = read('../refs/%s/ncnt.xyz' % d)
        cell = ref.get_cell()

        # Optimized reference
        refopt = read('../refs/%s/ncnt-geoopt-pos.xyz' % d)
        refopt.set_cell(cell)
        refopt.center(about=0)
        refopt.set_pbc([True,True,True])

        # Construct neighborlist for optimized reference system
        nlref = NL('ij',refopt,{('H', 'H'): 1.4, ('C', 'H'): 1.3, 
        ('N', 'H'): 1.3, ('C', 'C'): 1.85, ('C', 'N'): 1.85})

        # Loop over each adsorbed state
        for c in tqdm(range(dirs[d]+1),leave=False):
            str3 = "grep 'ENERGY|' ../adsorbed/%s/ncnt_%s-geoopt.out | tail -1 | awk '{print $9}'" % (d,c)
            convEner = float(subprocess.check_output(str3,shell=True))

            # Calculate adsorption energy, exclude outliers
            dE = (convEner - refEner - 0.5*h2Ener)*27.21138
            if abs(dE) > 2.5:
                continue

            xyz = read('../adsorbed/%s/ncnt_%s-geoopt-pos.xyz' % (d,c))
            xyz.set_cell(cell)

            # Important to get coordinate conversions correct!
            xyz.center(about=0)
            xyz.set_pbc([True,True,True])

            # Construct neighborlist for adsorbed state
            nl = NL('ij',xyz,{('H', 'H'): 1.4, ('C', 'H'): 1.3, 
                ('N', 'H'): 1.3, ('C', 'C'): 1.85, ('C', 'N'): 1.85})

            ###########################################################################

            # Get atom indices
            site = nl[1][-1]                                    # Adsorption site, ensure H index -1
            if xyz.symbols[site] == 'H':                        # Check for H2 formation
                continue
            nitro = np.where(xyz.symbols == 'N')[0]             # Nitrogen site(s)
            hydro = np.where(refopt.symbols == 'H')[0]          # Previous hydrogens
            siteNN = nlref[1][np.where(nlref[0] == site)]       # Ads. site nearest neighbors
            if 'H' in xyz.symbols[siteNN]:                      # Check if site already occupied
                continue

            dNS = []
            for N in nitro:
                dNS.append(cylDist(refopt,site,N))
            nearN = nitro[np.where(dNS == np.amin(dNS))][0]     # N closest to ads. site
            nearNNN = nlref[1][np.where(nlref[0]==nearN)]       # Nearest neighbors of closest N

            dHS = []
            occupied = []
            if len(hydro) != 0:
                for h in hydro:
                    occupied.append(nlref[1][np.where(nlref[0]==h)])  # Occupied sites  
                occupied = np.array(occupied).reshape(-1)
                for occ in occupied:
                    dHS.append(cylDist(refopt,site,occ))
                nearH = occupied[np.where(dHS == np.amin(dHS))][0]    # Occupied site closest to ads. site
            else:
                nearH = np.nan

            # Skip some extra sites
            if refopt[site].position[0] < 0:
                continue

            ###########################################################################

            # Log simple features

            # Adsorption energy
            Ead.append(dE)

            # Net atomic charge on adsorption site
            str4 = "grep ' %s       %s' ../refs/%s/ncnt-geoopt.out | tail -1 | awk '{print $8}'" % (site+1,
                xyz.symbols[site], d)
            nac = float(subprocess.check_output(str4,shell=True))
            q.append(nac)

            # Spin moment on adsorption site
            str5 = "grep ' %s       %s' ../refs/%s/ncnt-geoopt.out | tail -1 | awk '{print $7}'" % (site+1,
                xyz.symbols[site], d)
            spin = float(subprocess.check_output(str5,shell=True))
            mu.append(spin)

            # Band gap
            str6 = "grep 'LUMO gap' ../refs/%s/ncnt-geoopt.out | tail -2 | awk '{print $7}' | sort -n | head -1" % d
            gap = float(subprocess.check_output(str6,shell=True))
            Egap.append(gap)

            # Dopant/vacancy concentration, site atomic number, multiplicity
            cN.append(float(len(nitro))/float(len(ref)-len(hydro))*100)
            cV.append((224-len(ref)+len(hydro))/224.*100)
            cH.append(float(len(hydro)+1)/float(len(ref)-len(hydro))*100)
            Z.append(int(xyz.get_atomic_numbers()[site]))
            S = (len(nitro)*5+len(hydro))%2/2
            mult.append(int(2*S+1))

            # CNT type (zigzag or armchair)  
            if '14x0' in d:
                chir.append(np.arctan(np.sqrt(3)*0/(2*14+0)))
            elif '8x8' in d:
                chir.append(np.arctan(np.sqrt(3)*8/(2*8+8)))

            # Coordination numbers and relaxation induced changes in CN
            cnN.append(int(np.bincount(nlref[0])[nearN]))
            dcnN.append(int(np.bincount(nl[0])[nearN])-cnN[-1])
            cnS.append(int(np.bincount(nlref[0])[site]))
            dcnS.append(int(np.bincount(nl[0])[site])-cnS[-1])

            # Mean and max displacement of atoms during relaxation  
            rmsd.append(np.sqrt(np.mean((xyz.get_positions()[:-1]-refopt.get_positions())**2)))
            rmaxsd.append(np.sqrt(np.amax((xyz.get_positions()[:-1]-refopt.get_positions())**2)))

            # Minimum and average distance to dopants
            dminNS.append(np.amin(dNS))
            if len(nitro) > 1:
                daveNS.append(np.mean(dNS))
            else:
                daveNS.append(np.nan)

            # Minimum and average distance to occupied sites
            if np.isnan(nearH):
                dminHS.append(np.nan)
                daveHS.append(np.nan)
            elif len(dHS) < 2:
                dminHS.append(np.amin(dHS))
                daveHS.append(np.nan)
            else:
                dminHS.append(np.amin(dHS))
                daveHS.append(np.mean(dHS))

            # N-S angular displacement
            zaxis = (0,0,1)
            if site == nearN:
                adispN.append(np.nan)
            else:
                NSvec = refopt[site].position-refopt[nearN].position
                A = np.arccos(np.dot(NSvec,zaxis)/np.linalg.norm(NSvec))
                adispN.append(A)

            # H-S angular displacement
            if np.isnan(nearH):
                adispH.append(np.nan)
            else:
                HSvec = refopt[site].position-refopt[nearH].position
                A = np.arccos(np.dot(HSvec,zaxis)/np.linalg.norm(HSvec))
                adispH.append(A)

            # Adsorption and dopant NN angles
            angle=[]
            siteNNcycl = np.append(siteNN,siteNN[0])
            S = 1
            while S < len(siteNNcycl):
                angle.append(refopt.get_angle(siteNNcycl[S-1],site,siteNNcycl[S],mic=True))
                S+=1

            aminS.append(np.deg2rad(np.amin(angle)))
            if len(angle) == 2:
                amaxS.append(np.deg2rad(360-np.min(angle)))
            else:
                amaxS.append(np.deg2rad(np.amax(angle)))

            angle=[]
            NNNcycl = np.append(nearNNN,nearNNN[0])
            S = 1
            while S < len(NNNcycl):
                angle.append(refopt.get_angle(NNNcycl[S-1],nearN,NNNcycl[S],mic=True))
                S+=1

            aminN.append(np.deg2rad(np.amin(angle)))
            if len(angle) == 2:
                amaxN.append(np.deg2rad(360-np.min(angle)))
            else:
                amaxN.append(np.deg2rad(np.amax(angle)))

            ###########################################################################

    np.savetxt('masterdata.dat',
        np.c_[Ead,cV,cN,cH,Z,rmsd,rmaxsd,dminNS,daveNS,dminHS,daveHS,mult,chir,q,mu,Egap,cnN,dcnN,cnS,dcnS,aminS,amaxS,aminN,amaxN,adispN,adispH],
        header='Ead,cV,cN,cH,Z,rmsd,rmaxsd,dminNS,daveNS,dminHS,daveHS,mult,chir,q,mu,Egap,cnN,dcnN,cnS,dcnS,aminS,amaxS,aminN,amaxN,adispN,adispH',
        fmt='%8.3f,%8.3f,%8.3f,%8.3f,%4i,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%4i,%8.3f,%8.3f,%8.3f,%8.3f,%4i,%4i,%4i,%4i,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f,%8.3f',
        delimiter=',',comments='')

if __name__ == '__main__':
    main()
