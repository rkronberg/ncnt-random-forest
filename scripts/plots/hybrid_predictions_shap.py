import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import os
from shap.plots import colors
from . import settings

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../data/hybrid"))

def plot():

    # Function for plotting predictions vs. DFT data and learning curve

    settings.rcparams()

    # Loop over CV splits and load data
    y_train = []
    y_pred_train = []
    y_test = [] 
    y_pred_test = []
    for k in np.arange(1,11):
        y_train.append(np.loadtxt('%s/y-train-split_%s.out' % (DATA_PATH,k), usecols=[0], delimiter=','))
        y_pred_train.append(np.loadtxt('%s/y-train-split_%s.out' % (DATA_PATH,k), usecols=[1], delimiter=','))
        y_test.append(np.loadtxt('%s/y-test-split_%s.out' % (DATA_PATH,k), usecols=[0], delimiter=','))
        y_pred_test.append(np.loadtxt('%s/y-test-split_%s.out' % (DATA_PATH,k), usecols=[1], delimiter=','))

    # Plot training and test set DFT data vs. predictions (one CV split as an example)
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    ax[0].plot([-3,3],[-3,3],'k--')
    ax[0].plot(y_train[4],y_pred_train[4],'o',color=colors.blue_rgb,fillstyle='none',markeredgewidth=1.5,
        label=r'Training set')
    ax[0].plot(y_test[4],y_pred_test[4],'x',color=colors.red_rgb,markeredgewidth=1.5,label=r'Test set')
    ax[0].set_xlim(np.amin(y_train[4])-0.1,np.amax(y_train[4])+0.1)
    ax[0].set_ylim(np.amin(y_train[4])-0.1,np.amax(y_train[4])+0.1)
    ax[0].set_xticks([-2,-1,0,1,2])
    ax[0].set_yticks([-2,-1,0,1,2])
    ax[0].minorticks_on()
    ax[0].tick_params(which='both',direction='in',right=True,top=True)
    ax[0].set_ylabel(r'$\Delta E_\mathrm{RF}$ (eV)')
    ax[0].set_xlabel(r'$\Delta E_\mathrm{PBE0}$ (eV)')
    ax[0].legend(frameon=False,handletextpad=0.1,loc='upper left')

    featureSym = settings.flabels()

    # Loop over splits and load data
    shap_values = []
    x = []
    shap_i = []
    shap_c = []
    for k in np.arange(1,11):
        shap_tmp = np.loadtxt('%s/shap-split_%s.out' % (DATA_PATH,k), delimiter=',')
        shap_values.append(shap_tmp)
        x_tmp = np.loadtxt('%s/features-split_%s.out' % (DATA_PATH,k), delimiter=',')
        x.append(x_tmp)
        shap_corrs = []
        for i, feature in enumerate(x_tmp[0]):
            shap_corrs.append(np.ma.corrcoef(np.ma.masked_invalid(x_tmp[:,i]),np.ma.masked_invalid(shap_tmp[:,i]))[1,0])
        shap_i.append(np.mean(abs(shap_tmp),axis=0))
        shap_c.append(shap_corrs)

    shap_ave = np.mean(shap_i,axis=0)
    corr_ave = np.mean(shap_c,axis=0)
    shap_std = np.std(shap_i,axis=0,ddof=1)
    ind_shap = np.argsort(shap_ave)

    # Plot global feature importances including the correlation
    for pos, i in enumerate(ind_shap[-10:]):
        ax[1].axhline(y=pos,color='k',alpha=0.5,lw=1,dashes=(1,5),zorder=-1)
        if corr_ave[i] >= 0:
            color = colors.red_rgb
            ax[1].barh(featureSym[i],shap_ave[i],xerr=shap_std[i],capsize=4,color=color)
        else:
            color = colors.blue_rgb
            ax[1].barh(featureSym[i],shap_ave[i],xerr=shap_std[i],capsize=4,color=color)
        if abs(shap_ave[i]) > 0.04:
            ax[1].text(shap_ave[i]-shap_std[i]-0.005,pos,r'$%.2f$' % corr_ave[i],verticalalignment='center', horizontalalignment='right',
                fontsize=16,color='white')
        else:
            ax[1].text(shap_ave[i]+shap_std[i]+0.005,pos,r'$%.2f$' % corr_ave[i],verticalalignment='center', horizontalalignment='left',
                fontsize=16,color=color)


    ax[1].minorticks_on()
    ax[1].tick_params(which='both',direction='in',top=True,left=False)
    ax[1].tick_params(axis='y',labelsize=16)
    ax[1].set_xlabel(r'$\langle|\phi_j|\rangle$ (eV)')
    #ax[1].set_xticks([0,0.02,0.04,0.06,0.08])
    ax[1].set_xlim(left=0)
    red_patch = mpa.Patch(color=colors.red_rgb,label=r'$r_{ij}\geq0$')
    blue_patch = mpa.Patch(color=colors.blue_rgb,label=r'$r_{ij}<0$')
    ax[1].legend(handles=[red_patch,blue_patch],frameon=False,loc='lower right')
    plt.subplots_adjust(bottom=0.13,wspace=0.25)

    plt.savefig('%s/predictions.pdf' % DATA_PATH)
    