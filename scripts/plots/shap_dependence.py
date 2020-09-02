import numpy as np
import matplotlib.pyplot as plt
import os
from shap.plots import colors
from . import settings
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

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
    fig, ax = plt.subplots(1,4,figsize=(28,6))
    ax=ax.flatten()
    ind=[ind_shap[0],ind_shap[2],ind_shap[3],ind_shap[4]]

    xdata = [[] for i in range(4)]
    ydata = [[] for i in range(4)]

    for split in np.arange(0,10):
        for pos, i in enumerate(ind):
            for value in x[split][:,i]:
                xdata[pos].append(value)
            for value in shap_values[split][:,i]:
                ydata[pos].append(value)

    for pos, i in enumerate(ind):
        xtmp = np.array(xdata[pos])
        ytmp = np.array(ydata[pos])
        xy = np.vstack([xtmp,ytmp])
        z = gaussian_kde(xy).logpdf(xy)
        idx = z.argsort()
        xi, yi, zi = xtmp[idx], ytmp[idx], z[idx]
        ax[pos].scatter(xi, yi, c=zi, s=36, edgecolor='', cmap=colors.red_blue,alpha=0.5)
        #ax[pos].plot(xdata[pos],ydata[pos],'o',
        #    color=colors.blue_rgb,alpha=0.3,markeredgewidth=0)
        ax[pos].minorticks_on()
        ax[pos].tick_params(which='both',direction='in',top=True,right=True)
        ax[pos].set_xlabel("%s (%s)" % (featureSym[i],funit[i]))


    ax[0].set_ylabel(r'$\phi_j$ (eV)')

    ax[0].set_xticks([-0.05,0.05,0.15,0.25])
    ax[0].set_yticks([-0.7,-0.5,-0.3,-0.1,0.1])
    ax[0].set_xlim(right=0.25)
    ax[1].set_xticks([0.0,0.3,0.6,0.9])
    ax[2].set_xticks([1.8,1.9,2.0,2.1])
    ax[2].set_yticks([-0.5,-0.3,-0.1,0.1])
    ax[2].set_xlim(right=2.15)
    ax[3].set_xticks([0,4,8,12])

    ax[0].text(0, 1.06, r'\textbf{a)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes)
    ax[0].text(1, 1.06, r'$\langle|\phi_j|\rangle=%.2f$ eV' % shap_ave[ind[0]], horizontalalignment='right',verticalalignment='center', transform=ax[0].transAxes)

    ax[1].text(0, 1.06, r'\textbf{b)}', horizontalalignment='left',verticalalignment='center', transform=ax[1].transAxes)
    ax[1].text(1, 1.06, r'$\langle|\phi_j|\rangle=%.2f$ eV' % shap_ave[ind[1]], horizontalalignment='right',verticalalignment='center', transform=ax[1].transAxes)

    ax[2].text(0, 1.06, r'\textbf{c)}', horizontalalignment='left',verticalalignment='center', transform=ax[2].transAxes)
    ax[2].text(1, 1.06, r'$\langle|\phi_j|\rangle=%.2f$ eV' % shap_ave[ind[2]], horizontalalignment='right',verticalalignment='center', transform=ax[2].transAxes)

    ax[3].text(0, 1.06, r'\textbf{d)}', horizontalalignment='left',verticalalignment='center', transform=ax[3].transAxes)
    ax[3].text(1, 1.06, r'$\langle|\phi_j|\rangle=%.2f$ eV' % shap_ave[ind[3]], horizontalalignment='right',verticalalignment='center', transform=ax[3].transAxes)

    plt.subplots_adjust(wspace=0.175, bottom=0.13)
    
    cax = fig.add_axes([.89,.128,.04,.754])
    m = cm.ScalarMappable(cmap=colors.red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000,cax=cax)
    cb.set_ticklabels(['Low','High'])
    cb.set_label('Datapoint density', labelpad=0)
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 8)

    plt.savefig('%s/shap_dependence.pdf' % DATA_PATH)

    if(all):
        fig, ax = plt.subplots(5,5,figsize=(24,24))
        ax=ax.flatten()

        xdata = [[] for i in range(25)]
        ydata = [[] for i in range(25)]

        for split in np.arange(0,10):
            for pos, i in enumerate(ind_shap):
                for value in x[split][:,i]:
                    xdata[pos].append(value)
                for value in shap_values[split][:,i]:
                    ydata[pos].append(value)

        for pos, i in enumerate(ind_shap):
            xtmp = np.array(xdata[pos])
            ytmp = np.array(ydata[pos])
            mask = ~np.isnan(xtmp)
            xtmp = xtmp[mask]
            ytmp = ytmp[mask]
            xy = np.vstack([xtmp,ytmp])
            z = gaussian_kde(xy).logpdf(xy)
            idx = z.argsort()
            xi, yi, zi = xtmp[idx], ytmp[idx], z[idx]
            ax[pos].scatter(xi, yi, c=zi, s=36, edgecolor='', cmap=colors.red_blue,alpha=0.5)
            #ax[pos].plot(x[split][:,i],shap_values[split][:,i],'o',
            #    color=colors.blue_rgb,alpha=0.5,markeredgewidth=0)
            ax[pos].minorticks_on()
            ax[pos].tick_params(which='both',direction='in',top=True,right=True)
            ax[pos].set_xlabel("%s (%s)" % (featureSym[i],funit[i]))
            ax[pos].yaxis.set_major_formatter(FormatStrFormatter('$%.2f$'))
            ax[pos].text(1, 1.06, r'$\langle|\phi_j|\rangle=%.2f$ eV' % shap_ave[i], horizontalalignment='right',verticalalignment='center', transform=ax[pos].transAxes, fontsize=14)
        plt.subplots_adjust(hspace=0.4,wspace=0.35)
        
        plt.savefig('%s/shap_dependence_all.pdf' % DATA_PATH)