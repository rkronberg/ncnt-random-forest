import numpy as np
import matplotlib.pyplot as plt
import os
from shap.plots import colors
from . import settings

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../data/gga"))

settings.rcparams()
featureSym = settings.flabels()
funit = settings.funit()

def plot(all):

    # Loop over CV splits and load data
    shap_values = []
    x = []
    shap_i=[]
    for k in np.arange(1,11):
        shap_tmp = np.loadtxt('%s/shap-split_%s.out' % (DATA_PATH,k), delimiter=',')
        shap_values.append(shap_tmp)
        x_tmp = np.loadtxt('%s/features-split_%s.out' % (DATA_PATH,k), delimiter=',')
        x.append(x_tmp)
        shap_i.append(np.mean(abs(shap_tmp),axis=0))

    # Sort by global feature importance
    shap_ave = np.mean(shap_i,axis=0)
    ind_shap = np.flipud(np.argsort(shap_ave))

    # Plot selected SHAP dependence plots without approximate interactions
    fig, ax = plt.subplots(2,2,figsize=(12,10))
    ax=ax.flatten()
    ind=[ind_shap[0],ind_shap[2],ind_shap[3],ind_shap[4]]
    for split in np.arange(0,10):
        for pos, i in enumerate(ind):
            ax[pos].plot(x[split][:,i],shap_values[split][:,i],'.',
                color=colors.blue_rgb,alpha=0.5,markersize=9,markeredgewidth=0)
            ax[pos].minorticks_on()
            ax[pos].tick_params(which='both',direction='in',top=True,right=True)
            ax[pos].set_xlabel("%s (%s)" % (featureSym[i],funit[i]))

    ax[0].set_ylabel(r'$\phi_j$ (eV)')
    ax[2].set_ylabel(r'$\phi_j$ (eV)')

    ax[0].set_xticks([-0.05,0.05,0.15,0.25])
    ax[0].set_yticks([-0.7,-0.5,-0.3,-0.1,0.1])
    ax[0].set_xlim(right=0.25)
    ax[1].set_xticks([0.0,0.3,0.6,0.9])
    ax[2].set_xticks([1.7,1.8,1.9,2.0,2.1])
    ax[2].set_yticks([-0.5,-0.3,-0.1,0.1])
    ax[2].set_xlim(right=2.15)
    ax[3].set_xticks([0,4,8,12])
    plt.subplots_adjust(hspace=0.25)

    plt.savefig('%s/shap_dependence.pdf' % DATA_PATH)

    if(all):
        fig, ax = plt.subplots(5,5,figsize=(25,25))
        ax=ax.flatten()
        for split in np.arange(0,10):
            for pos, i in enumerate(ind_shap):
                ax[pos].plot(x[split][:,i],shap_values[split][:,i],'.',
                    color=colors.blue_rgb,alpha=0.5,markersize=9,markeredgewidth=0)
                ax[pos].minorticks_on()
                ax[pos].tick_params(which='both',direction='in',top=True,right=True)
                ax[pos].set_xlabel("%s (%s)" % (featureSym[i],funit[i]))
        plt.subplots_adjust(hspace=0.5)
        
        plt.savefig('%s/shap_dependence_all.pdf' % DATA_PATH)