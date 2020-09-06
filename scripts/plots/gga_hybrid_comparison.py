import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import os
from shap.plots import colors
from . import settings
import pandas as pd

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../data"))

def plot():

    settings.rcparams()

    gga_data = pd.read_csv('%s/gga/masterdata.csv' % DATA_PATH)
    hybrid_data = pd.read_csv('%s/hybrid/masterdata.csv' % DATA_PATH)

    size = len(hybrid_data)

    Ead_gga = np.empty(size)
    Ead_hybrid = np.empty(size)
    Egap_gga = np.empty(size)
    Egap_hybrid = np.empty(size)
    q_gga = np.empty(size)
    q_hybrid = np.empty(size)
    mu_gga = np.empty(size)
    mu_hybrid = np.empty(size)

    k = 0
    for i, value_gga in enumerate(gga_data['rmaxsd'].values):
        for j, value_hybrid in enumerate(hybrid_data['rmaxsd'].values):
            if value_gga == value_hybrid:
                gga_conf = gga_data['conf'].values[i]
                gga_id = gga_data['id'].values[i]
                hybrid_id = hybrid_data['id'].values[j]
                Ead_gga[k] = gga_data['Ead'][i]
                Ead_hybrid[k] = hybrid_data['Ead'][j]
                Egap_gga[k] = gga_data['Egap'][i]
                Egap_hybrid[k] = hybrid_data['Egap'][j]
                q_gga[k] = gga_data['q'][i]
                q_hybrid[k] = hybrid_data['q'][j]
                mu_gga[k] = gga_data['mu'][i]
                mu_hybrid[k] = hybrid_data['mu'][j]
                k+=1

    Ead_ind = np.argsort(Ead_gga)
    Egap_ind = np.argsort(Egap_gga)
    q_ind = np.argsort(q_gga)
    mu_ind = np.argsort(mu_gga)

    fig,ax=plt.subplots(1,4,figsize=(30,6))
    ax[0].plot(Ead_hybrid[Ead_ind],'o',color=colors.blue_rgb,fillstyle='none',markeredgewidth=1.5,label=r'Hybrid (PBE0)')
    ax[0].plot(Ead_gga[Ead_ind],color=colors.red_rgb,label=r'GGA (PBE)')
    ax[0].set_ylabel(r'$\Delta E$ (eV)')
    ax[0].legend(frameon=False)
    ax[0].text(0.1,0.9,r'$\mathrm{RMSE=%.2f}$ eV' % np.sqrt(np.mean((Ead_hybrid-Ead_gga)**2))
        ,transform=ax[0].transAxes)
    
    ax[1].plot(Egap_hybrid[Egap_ind],'o',color=colors.blue_rgb,fillstyle='none',markeredgewidth=1.5)
    ax[1].plot(Egap_gga[Egap_ind],color=colors.red_rgb)
    ax[1].set_ylabel(r'$E_g$ (eV)')
    ax[1].set_yticks([0.4,0.8,1.2,1.6])
    ax[1].text(0.1,0.9,r'$\mathrm{RMSE=%.2f}$ eV' % np.sqrt(np.mean((Egap_hybrid-Egap_gga)**2))
        ,transform=ax[1].transAxes)
    
    ax[2].plot(q_hybrid[q_ind],'o',color=colors.blue_rgb,fillstyle='none',markeredgewidth=1.5)
    ax[2].plot(q_gga[q_ind],color=colors.red_rgb)
    ax[2].set_ylabel(r'$q$ (e)')
    ax[2].text(0.1,0.9,r'$\mathrm{RMSE=%.2f}$ e' % np.sqrt(np.mean((q_hybrid-q_gga)**2))
        ,transform=ax[2].transAxes)
    
    ax[3].plot(mu_hybrid[mu_ind],'o',color=colors.blue_rgb,fillstyle='none',markeredgewidth=1.5)
    ax[3].plot(mu_gga[mu_ind],color=colors.red_rgb)
    ax[3].set_ylabel(r'$\mu$ ($\mu_\mathrm{B}$)')
    ax[3].text(0.1,0.9,r'$\mathrm{RMSE=%.2f}$ $\mu_\mathrm{B}$' % np.sqrt(np.mean((mu_hybrid-mu_gga)**2))
        ,transform=ax[3].transAxes)

    for i in range(4):
        ax[i].minorticks_on()
        ax[i].tick_params(which='both',direction='in',top=True, right=True)
        ax[i].set_xlabel(r'Configuration')
        ax[i].set_xticks([0,50,100,150,200,250])

    plt.subplots_adjust(bottom=0.13, wspace=0.25)

    plt.savefig('%s/hybrid/gga_vs_hybrid.pdf' % DATA_PATH)