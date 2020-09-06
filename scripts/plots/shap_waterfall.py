import numpy as np
import matplotlib.pyplot as plt
import os, subprocess
from shap.plots import colors
from . import settings
import matplotlib

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH, "../../data/gga"))

settings.rcparams()
featureSym = settings.flabels()
funit = settings.funit()

def plot():

    # Loop over CV splits and load data
    shap_values = []
    x = []
    expected=[]
    for k in np.arange(1,11):
        exp_tmp=subprocess.check_output("grep Base %s/shap-split_%s.out \
         | awk '{print $9}'" % (DATA_PATH,k), shell=True).decode()
        expected.append(float(exp_tmp.strip(')\n')))
        shap_tmp = np.loadtxt('%s/shap-split_%s.out' % (DATA_PATH,k), delimiter=',')
        shap_values.append(shap_tmp)
        x_tmp = np.loadtxt('%s/features-split_%s.out' % (DATA_PATH,k), delimiter=',')
        x.append(x_tmp)

    # Which prediction should be explained?
    sampleid = 1 #54, 124, 567, 578
    cv_split = 9
    expected_value = expected[0]
    shap = shap_values[cv_split][sampleid,:]
    features = x[cv_split][sampleid,:]

    #for i in np.arange(0,len(shap_values[cv_split][:,0])):
    #    if expected_value + shap_values[cv_split][i,:].sum() > 0:
    #        print(i,expected_value + shap_values[cv_split][i,:].sum())

    upper_bounds = None
    lower_bounds = None

    # init variables we use for tracking the plot locations
    num_features = min(10, len(shap))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(shap))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = expected_value + shap.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    # size the plot based on how many features we are plotting
    fig,ax = plt.subplots(figsize=(8, num_features * row_height + 1.5))

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(shap):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = shap[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="k",alpha=0.5, linewidth=1, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = featureSym[order[i]]
        else:
            yticklabels[rng[i]] = featureSym[order[i]] + " $=" + "{:.2f}".format(features[order[i]]) + "$ " + funit[order[i]] 
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(shap):
        yticklabels[0] = "%d other features" % (len(shap) - num_features + 1)
        remaining_impact = expected_value - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)
    
    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw, left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw  if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw, left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)
    
    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb, width=bar_width,
            head_width=bar_width
        )
        
        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i], 
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=colors.light_red_rgb
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], "{:.2f}".format(pos_widths[i]),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=14
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], "{:.2f}".format(pos_widths[i]),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=14
            )
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i], 
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=colors.light_blue_rgb
            )
        
        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], "{:.2f}".format(neg_widths[i]),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=14
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], "{:.2f}".format(neg_widths[i]),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=14
            )

    # draw the y-ticks
    plt.yticks(range(num_features), yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=16)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color='k', alpha=0.5, lw=1, dashes=(1, 5), zorder=-1)

    # draw the E[f(X)] tick mark
    xmin,xmax = ax.get_xlim()
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([expected_value,expected_value+shap.sum()])
    ax2.set_xticklabels(['$\mathbb{E}[\hat{f}(x)]=%s$ eV' % "{:.2f}".format(expected_value),
        '$\hat{f}(x)=%s$ eV' % "{:.2f}".format(expected_value+shap.sum())])
    ax2.tick_params(labelsize=16)

    ax.minorticks_on()
    ax.tick_params(which='both',direction='in',top=True,left=False)
    ax.tick_params(axis='y',labelsize=16)
    plt.subplots_adjust(left=0.26,bottom=0.12)
    ax.set_xlabel(r'Model output (eV)')
    
    plt.savefig('%s/shap_waterfall_4.pdf' % DATA_PATH)