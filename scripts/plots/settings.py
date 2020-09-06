import matplotlib.pyplot as plt

def rcparams():

	# Change some default settings
	plt.rc('text', usetex=True)
	plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
	plt.rc('font', family='serif', size=24)
	plt.rc('axes', linewidth=2)
	plt.rc('lines', linewidth=2)
	plt.rc('xtick.major',width=2,size=7)
	plt.rc('xtick.minor',width=1,size=4)
	plt.rc('ytick.major',width=2,size=7)
	plt.rc('ytick.minor',width=1,size=4)

def flabels():

	# LaTeX formatting of feature symbols for plotting
	featureSym=[r'$x_\mathrm{V}$',
	            r'$x_\mathrm{N}$',
	            r'$x_\mathrm{H}$',
	            r'$Z$',
	            r'RMSD',
	            r'RmaxSD',
	            r'$\min\{d_\mathrm{NS}\}$',
	            r'$\langle d_\mathrm{NS}\rangle$',
	            r'$\min\{d_\mathrm{HS}\}$',
	            r'$\langle d_\mathrm{HS}\rangle$',
	            r'$M$',
	            r'$\chi$',
	            r'$q$',
	            r'$\mu$',
	            r'$E_g$',
	            r'$\mathrm{CN}_\mathrm{N}$',
	            r'$\Delta\mathrm{CN}_\mathrm{N}$',
	            r'$\mathrm{CN}_\mathrm{S}$',
	            r'$\Delta\mathrm{CN}_\mathrm{S}$',
	            r'$\min\{\varphi_\mathrm{S}\}$',
	            r'$\max\{\varphi_\mathrm{S}\}$',
	            r'$\min\{\varphi_\mathrm{N}\}$',
	            r'$\max\{\varphi_\mathrm{N}\}$',
	            r'$\alpha_\mathrm{N}$',
	            r'$\alpha_\mathrm{H}$']

	return featureSym

def funit():

	# LaTeX formatting of feature symbols for plotting
	featureSym=[r'at-\%',
	            r'at-\%',
	            r'at-\%',
	            r'',
	            r'\r{A}',
	            r'\r{A}',
	            r'\r{A}',
	            r'\r{A}',
	            r'\r{A}',
	            r'\r{A}',
	            r'',
	            r'rad',
	            r'e',
	            r'e',
	            r'eV',
	            r'',
	            r'',
	            r'',
	            r'',
	            r'rad',
	            r'rad',
	            r'rad',
	            r'rad',
	            r'rad',
	            r'rad']

	return featureSym
