
"""
Ad hoc data preprocessing for ML
of hydrogen adsorption on NCNTs

author: Rasmus Kronberg
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
	z = z1-z2

	#Check PBC
	if z > xyz.get_cell()[2,2]/2:
		z = xyz.get_cell()[2,2] - z

	return np.sqrt((r*theta)**2 + z**2)

def main():

	################################################################################

	# Preparations

	Ead = []			# Adsorption energy
	cV = []				# Concentration of vacancies
	cN = []				# Concentration of nitrogen dopants
	Zsite = []			# Atomic number of adsorption site
	rmsd = []			# RMSD of atomic positions during geoopt
	rmaxsd = []			# Root maximum squared displacement of at. pos.
	dmin = []			# Distance from adsorption site to closest N
	dave = []			# Average distance from adsorption site to all N
	mult = []			# Nanotube multiplicity
	n = []				# Nanotube n
	m = []				# Nanotube m
	CNN = []			# Coordination number (CN) of closest N
	dCNN = []			# Change in N CN during geoopt
	CNad = []			# CN of adsorption site
	dCNad = []			# Change in adsorption site CN
	aminad = []			# Smallest angle at the adsorption site
	amaxad = []			# Largest angle at the adsorption site
	aminN = []			# Smallest C-N-C angle
	amaxN = []			# Largest C-N-C angle
	angdisp = []		# Angular displacement of the adsorption site wrt. closest N

	# Get H2 energy
	str1 = "grep 'ENERGY|' ../refs/H2/h2-geoopt.out | tail -1 | awk '{print $9}'"
	h2Ener = float(subprocess.check_output(str1,shell=True))

	# Dictionary of systems and number of hydrogen configurations
	dirs = {'1N_1H_14x0_graphitic': 119, '1N_1H_14x0_pyridinic': 102, '1N_1H_14x0_pyrrolic_SW_A': 103, 
	'1N_1H_14x0_pyrrolic_SW_B': 103, '2N_1H_14x0_graphitic': 103, '3N_1H_14x0_pyridinic': 102, 
	'4N_1H_14x0_pyridinic': 101, '1N_1H_8x8_graphitic': 118, '1N_1H_8x8_pyridinic': 103,
	'1N_1H_8x8_pyrrolic_SW_A': 104, '2N_1H_8x8_graphitic': 104, '1N_1H_8x8_pyrrolic': 103,
	'1N_1H_8x8_pyrrolic_SW_B': 104,'3N_1H_8x8_pyridinic': 103,'4N_1H_8x8_pyridinic': 102,
	'1N_1H_14x0_pyrrolic': 102}

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
		refopt.center(about=0)				# Important to get coordinate conversions correct!
		refopt.set_pbc([True,True,True])

		# Construct neighborlist for optimized reference system
		nlref = NL('ij',refopt,{('H', 'H'): 1.85, ('C', 'H'): 1.3, 
		('N', 'H'): 1.3, ('C', 'C'): 1.85, ('C', 'N'): 1.85})

		# Loop over each adsorbed state
		for c in tqdm(np.linspace(0,dirs[d],dirs[d]+1,dtype=int)):
			str3 = "grep 'ENERGY|' ../adsorbed/%s/ncnt_%s-geoopt.out | tail -1 | awk '{print $9}'" % (d,c)
			convEner = float(subprocess.check_output(str3,shell=True))

			xyz = read('../adsorbed/%s/ncnt_%s-geoopt-pos.xyz' % (d,c))
			xyz.set_cell(cell)
			xyz.center(about=0)				# Important to get coordinate conversions correct!
			xyz.set_pbc([True,True,True])

			# Construct neighborlist for adsorbed state
			nl = NL('ij',xyz,{('H', 'H'): 1.85, ('C', 'H'): 1.3, 
				('N', 'H'): 1.3, ('C', 'C'): 1.85, ('C', 'N'): 1.85})

			################################################################################

			# Get atom indices
			site = nl[1][-1]								# Adsorption site, ensure H index -1
			nitro = np.where(xyz.symbols == 'N')[0]			# Nitrogen site(s)
			nitroNN = nlref[1][np.where(nlref[0]==nitro)]	# Nitrogen nearest neighbors
			siteNN = nlref[1][np.where(nlref[0]==site)]		# Ads. site nearest neighbors

			dNad = []
			for N in nitro:
				dNad.append(cylDist(refopt,site,N))

			nearN = nitro[np.where(dNad == np.amin(dNad))][0]		# N closest to ads. site
			nearNNN = nlref[1][np.where(nlref[0]==nearN)]			# Nearest neighbors of closest N
			################################################################################

			# Log features

			Ead.append((convEner - refEner - 0.5*h2Ener)*27.21138)
			cN.append(float(len(nitro))/float(len(ref))*100)
			cV.append((224-len(ref))/224.*100)
			Zsite.append(int(xyz.get_atomic_numbers()[site]))
			mult.append(int(2*(len(nitro)%2)/2.+1))

			if '14x0' in d:
				n.append(int(14))
				m.append(int(0))
			elif '8x8' in d:
				n.append(int(8))
				m.append(int(8))

			CNN.append(int(np.bincount(nlref[0])[nearN]))
			dCNN.append(int(np.bincount(nl[0])[nearN])-CNN[-1])
			CNad.append(int(np.bincount(nlref[0])[site]))
			dCNad.append(int(np.bincount(nl[0])[site])-CNad[-1])

			rmsd.append(np.sqrt(np.mean((xyz.get_positions()[:-1]-refopt.get_positions())**2)))
			rmaxsd.append(np.sqrt(np.amax((xyz.get_positions()[:-1]-refopt.get_positions())**2)))

			dmin.append(np.amin(dNad))
			dave.append(np.mean(dNad))

			yzAd = (refopt[site].position[1],refopt[site].position[2])
			zaxis = (0,1)
			cosA = np.dot(zaxis,yzAd)/(np.linalg.norm(yzAd)*np.linalg.norm(zaxis))
			angdisp.append(abs(cosA))

			angle=[]
			siteNNcycl = np.append(siteNN,siteNN[0])
			S = 1
			while S < len(siteNNcycl):
				angle.append(refopt.get_angle(siteNNcycl[S-1],site,siteNNcycl[S],mic=True))
				S+=1

			aminad.append(np.amin(angle))
			if len(angle) == 2:
				amaxad.append(360-np.min(angle))
			else:
				amaxad.append(np.amax(angle))

			angle=[]
			NNNcycl = np.append(nearNNN,nearNNN[0])
			S = 1
			while S < len(NNNcycl):
				angle.append(refopt.get_angle(NNNcycl[S-1],nearN,NNNcycl[S],mic=True))
				S+=1

			aminN.append(np.amin(angle))
			if len(angle) == 2:
				amaxN.append(360-np.min(angle))
			else:
				amaxN.append(np.amax(angle))

			################################################################################

	np.savetxt('masterdata.dat',
		np.c_[Ead,cV,cN,Zsite,rmsd,rmaxsd,dmin,dave,mult,n,m,CNN,dCNN,CNad,dCNad,aminad,amaxad,aminN,amaxN,angdisp],
		header='Ead,cV,cN,Zsite,rmsd,rmaxsd,dmin,dave,mult,n,m,CNN,dCNN,CNad,dCNad,aminad,amaxad,aminN,amaxN,angdisp',
		fmt='%10.3f,%10.3f,%10.3f,%i,%10.3f,%10.3f,%10.3f,%10.3f,%i,%i,%i,%i,%i,%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f',
		delimiter=',',comments='')

if __name__ == '__main__':
	main()