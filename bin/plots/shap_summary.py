import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.cm as cm
import os
from shap.plots import colors
from . import settings

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../data"))
OUT_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../"))

def plot():

    # Function for plotting SHAP analysis results
    
    settings.rcparams()
    featureSym = settings.flabels()

    # Loop over splits and load data
    shap_values = []
    x = []
    shap_i = []
    shap_c = []
    for k in np.arange(1,11):
        shap_tmp = np.loadtxt('%s/shap-split_%s.out' % (DATA_PATH,k))
        shap_values.append(shap_tmp)
        x_tmp = np.loadtxt('%s/features-split_%s.out' % (DATA_PATH,k))
        x.append(x_tmp)
        shap_corrs = []
        for i, feature in enumerate(x_tmp[0]):
            shap_corrs.append(np.corrcoef(x_tmp[:,i],shap_tmp[:,i])[1,0])
        shap_i.append(np.mean(abs(shap_tmp),axis=0))
        shap_c.append(shap_corrs)

    shap_ave = np.mean(shap_i,axis=0)
    corr_ave = np.mean(shap_c,axis=0)
    shap_std = np.std(shap_i,axis=0,ddof=1)
    ind_shap = np.argsort(shap_ave)

    # Plot global feature importances including the correlation
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    for pos, i in enumerate(ind_shap[-10:]):
        ax[0].axhline(y=pos,color='k',alpha=0.5,lw=1,dashes=(1,5),zorder=-1)
        if corr_ave[i] >= 0:
            ax[0].barh(featureSym[i],shap_ave[i],xerr=shap_std[i],capsize=4,color=colors.red_rgb)
        else:
            ax[0].barh(featureSym[i],shap_ave[i],xerr=shap_std[i],capsize=4,color=colors.blue_rgb)

    ax[0].minorticks_on()
    ax[0].tick_params(which='both',direction='in',top=True,left=False)
    ax[0].tick_params(axis='y',labelsize=16)
    ax[0].set_xlabel(r'$\langle|\phi_j|\rangle$ (eV)')
    ax[0].set_xticks([0,0.02,0.04,0.06,0.08])
    ax[0].set_xlim(left=0)
    red_patch = mpa.Patch(color=colors.red_rgb,label=r'$r_{ij}\geq0$')
    blue_patch = mpa.Patch(color=colors.blue_rgb,label=r'$r_{ij}<0$')
    ax[0].legend(handles=[red_patch,blue_patch],frameon=False,loc='lower right')

    # SHAP summary plot
    cmap = colors.red_blue
    for split in np.arange(0,10):
        for pos, i in enumerate(ind_shap[-10:]):
            if split == 0:
                ax[1].axhline(y=pos, color='k', alpha=0.5, lw=1, dashes=(1,5), zorder=-1)
            shaps = shap_values[split][:,i]
            values = x[split][:, i]
            inds=np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            values = np.array(values, dtype=np.float64)
            quant = np.round(100 * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(len(shaps)) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(len(shaps))
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (0.4 / np.max(ys + 1))
            
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                       vmax=vmax, s=16, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                       cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                       c=cvals, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.set_ticklabels(['Low','High'])
    cb.set_label('Feature value', labelpad=0)
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 10)

    ax[1].minorticks_on()
    ax[1].tick_params(which='both',direction='in',top=True,left=False,labelleft=False)
    ax[1].set_xlabel(r'$\phi_j$ (eV)')
    ax[1].set_xticks([-1.6,-1.2,-0.8,-0.4,0.0,0.4])
    plt.subplots_adjust(left=0.17,bottom=0.13,wspace=0.05)

    plt.savefig('%s/shap_summary.pdf' % OUT_PATH)
