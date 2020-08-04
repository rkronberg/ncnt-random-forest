import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import os
from shap.plots import colors
from . import settings

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../data"))
OUT_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../"))

def plot(lc):

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
    fig,ax=plt.subplots(figsize=(7,6))
    plt.plot([-3,3],[-3,3],'k--')
    plt.plot(y_train[-1],y_pred_train[-1],'o',color=colors.blue_rgb,fillstyle='none',markeredgewidth=1.5,
        label=r'Training set')
    plt.plot(y_test[-1],y_pred_test[-1],'x',color=colors.red_rgb,markeredgewidth=1.5,label=r'Test set')
    plt.xlim(np.amin(y_train[-1])-0.1,np.amax(y_train[-1])+0.1)
    plt.ylim(np.amin(y_train[-1])-0.1,np.amax(y_train[-1])+0.1)
    plt.xticks([-2,-1,0,1,2])
    plt.yticks([-2,-1,0,1,2])
    plt.minorticks_on()
    ax.tick_params(which='both',direction='in',right=True,top=True)
    plt.ylabel(r'$\Delta E_\mathrm{RF}$ (eV)')
    plt.xlabel(r'$\Delta E_\mathrm{DFT}$ (eV)')
    plt.subplots_adjust(left=0.17,bottom=0.13)
    plt.legend(frameon=False,handletextpad=0.1,loc='upper left')

    # Plot learning curve as inset
    if(lc):

        lcdata = np.loadtxt('%s/learning_curve.out' % DATA_PATH, delimiter=',')

        ax2 = fig.add_axes([0.61, 0.22, 0.26, 0.26])
        ax2.plot(lcdata[:,0],lcdata[:,1],'o-',color=colors.blue_rgb,fillstyle='none',lw=1.5,
            markeredgewidth=1.5)
        ax2.plot(lcdata[:,0],lcdata[:,3],'x-',color=colors.red_rgb,lw=1.5,markeredgewidth=1.5)
        ax2.fill_between(lcdata[:,0],lcdata[:,1] - lcdata[:,2],
            lcdata[:,1] + lcdata[:,2],alpha=0.2,color=colors.blue_rgb,lw=0)
        ax2.fill_between(lcdata[:,0],lcdata[:,3] - lcdata[:,4],
            lcdata[:,3] + lcdata[:,4],alpha=0.2,color=colors.red_rgb,lw=0)
        ax2.minorticks_on()
        ax2.tick_params(which='both',direction='in',top=True,right=True,
            labelsize=14)
        ax2.set_xlabel(r'Training set size',fontsize=14)
        ax2.set_ylabel(r'$R^2$',fontsize=14)
        ax2.set_xticks([1000,2000,3000,4000,5000,6000])
        ax2.set_xticklabels(['1k','2k','3k','4k','5k','6k'])
        ax2.xaxis.set_label_coords(0.5,-0.15)

    plt.savefig('%s/predictions.pdf' % OUT_PATH)
    