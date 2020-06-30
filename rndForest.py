
"""
Random Forest ML for H adsorption on NCNTs
Includes options for cross-validation, hyperparameter 
gridsearch, calculation of SHAP feature importances 
and plotting

author: Rasmus Kronberg
rasmus.kronberg@aalto.fi
"""

# Load necessary packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,learning_curve
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MaxAbsScaler
import shap
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import warnings

def lcurve(rf,x,y,strat):

    # Function for calculating learning curve (10-CV)

    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=rnd)

    train_sizes,train_scores,test_scores = learning_curve(rf,x,y,cv=kf.split(x,strat),
        train_sizes=np.linspace(0.1,1,9),shuffle=True,random_state=rnd,n_jobs=-1)

    return train_sizes, train_scores, test_scores


def doSHAP(rf,x_train):

    # Function for calculating feature importances based on Shapley values
    # and correlation coefficients for the signed impacts on the model output

    i = 0
    shap_corrs = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shap_values = shap.TreeExplainer(rf).shap_values(x_train)

    shap_imps = np.mean(abs(shap_values),axis=0)

    for feature in x_train[0]:
        shap_corrs.append(np.corrcoef(shap_values[:,i],x_train[:,i])[1,0])
        i += 1

    return shap_imps, shap_corrs


def plotSHAP(shap_i,shap_c,featureNames):

    # Function for outputting (cross-validated) SHAP feature importances

    shap_ave = np.mean(shap_i,axis=0)
    corr_ave = np.mean(shap_c,axis=0)
    shap_std = np.std(shap_i,axis=0,ddof=1)
    ind_shap = np.argsort(shap_ave)
    
    featureSym=[r'$x_\mathrm{V}$',r'$x_\mathrm{N}$',r'$x_\mathrm{H}$',r'$Z$',r'RMSD',r'RmaxSD',
    r'$\min\{d_\mathrm{NS}\}$',r'$\langle d_\mathrm{NS}\rangle$',r'$\min\{d_\mathrm{HS}\}$',
    r'$\langle d_\mathrm{HS}\rangle$',r'$M$',r'$\chi$',r'$q$',r'$\mu$',r'$E_g$',
    r'$\mathrm{CN}_\mathrm{N}$',r'$\Delta\mathrm{CN}_\mathrm{N}$',r'$\mathrm{CN}_\mathrm{S}$',
    r'$\Delta\mathrm{CN}_\mathrm{S}$',r'$\min\{\varphi_\mathrm{S}\}$',r'$\max\{\varphi_\mathrm{S}\}$',
    r'$\min\{\varphi_\mathrm{N}\}$',r'$\max\{\varphi_\mathrm{N}\}$',r'$\alpha_\mathrm{N}$',
    r'$\alpha_\mathrm{H}$']

    print('Feature importances (SHAP, positive or negative correlation):')
    fig,ax=plt.subplots(figsize=(7,6))
    for i in ind_shap[-10:]:
        if corr_ave[i] >= 0:
            print('%s: %s (+)' % (featureNames[i],shap_ave[i]))
            plt.barh(featureSym[i],shap_ave[i],xerr=shap_std[i],capsize=4,color='C3')
        else:
            print('%s: %s (-)' % (featureNames[i],shap_ave[i]))
            plt.barh(featureSym[i],shap_ave[i],xerr=shap_std[i],capsize=4,color='C9')

    plt.minorticks_on()
    ax.tick_params(which='both',direction='in',top=True,left=False)
    ax.tick_params(axis='y',labelsize=16)
    plt.xlabel(r'$\langle|\phi_j|\rangle$')
    plt.subplots_adjust(left=0.17,bottom=0.13)
    plt.xticks([0,0.02,0.04,0.06,0.08])
    plt.xlim(left=0)
    red_patch = mpa.Patch(color='C3',label=r'$r_{ij}\geq0$')
    blue_patch = mpa.Patch(color='C9',label=r'$r_{ij}<0$')
    plt.legend(handles=[red_patch,blue_patch],frameon=False,loc='lower right')


def crossval(x,y,model,strat):

    # Function for cross-validated results

    n_splits=10
    error_test = []
    error_train = []
    error_R2_test = []
    error_R2_train = []
    shap_i = []
    shap_c = []

    kf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=rnd)

    k=1
    for train, test in kf.split(x,strat):
        print('Cross-validation split %s/%s' % (k,n_splits))
        model.fit(x[train], y[train])
        y_pred_test = model.predict(x[test])
        y_pred_train = model.predict(x[train])
        error_test.append(np.sqrt(MSE(y[test],y_pred_test)))
        error_train.append(np.sqrt(MSE(y[train],y_pred_train)))
        error_R2_test.append(model.score(x[test], y[test]))
        error_R2_train.append(model.score(x[train], y[train]))

        # SHAP analysis
        if(SHAP):
            shap_imps, shap_corrs = doSHAP(model,x[train])
            shap_i.append(shap_imps)
            shap_c.append(shap_corrs)

        k+=1

    score_test = np.mean(error_test)
    score_test_std = np.std(error_test, ddof=1)
    score_train = np.mean(error_train)
    score_train_std = np.std(error_train, ddof=1)

    score_R2_test = np.mean(error_R2_test)
    score_R2_test_std = np.std(error_R2_test, ddof=1)
    score_R2_train = np.mean(error_R2_train)
    score_R2_train_std = np.std(error_R2_train, ddof=1)

    line()

    print('%s-fold cross-validation (training data: %s, test data: %s)' % 
        (n_splits,len(train),len(test)))
    print('R2 score (Training set): %.4f +- %.4f' % (score_R2_train,score_R2_train_std))
    print('R2 score (Test set): %.4f +- %.4f' % (score_R2_test,score_R2_test_std))
    print('RMSE (Training set): %.4f +- %.4f eV' % (score_train,score_train_std))
    print('RMSE (Test set): %.4f +- %.4f eV' % (score_test,score_test_std))

    return y[train],y_pred_train,y[test],y_pred_test,shap_i,shap_c


def gridsearch(x,y,strat):

    # Function for performing hyperparameter grid search

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,
        stratify=strat,shuffle=True,random_state=rnd)
    rf=GridSearchCV(RandomForestRegressor(oob_score=True,random_state=rnd),cv=10,
        param_grid={"n_estimators": np.linspace(100,500,5,dtype=int),
        "max_features": np.linspace(1,len(x[0,:]),len(x[0,:])+1,dtype=int)},n_jobs=-1)
    rf.fit(x_train,y_train)
    print('Best parameters from gridsearch: %s' % rf.best_params_)


def plot(y,y_train,y_pred_train,y_test,y_pred_test,train_sizes,train_scores,test_scores):

    # Function for plotting predictions vs. DFT data and learning curve

    fig,ax=plt.subplots(figsize=(7,6))
    plt.plot([-3,3],[-3,3],'k--')
    plt.plot(y_train,y_pred_train,'C9o',fillstyle='none',markeredgewidth=1.5,
        label=r'Training set')
    plt.plot(y_test,y_pred_test,'C3x',markeredgewidth=1.5,label=r'Test set')
    plt.xlim(np.amin(y)-0.1,np.amax(y)+0.1)
    plt.ylim(np.amin(y)-0.1,np.amax(y)+0.1)
    plt.minorticks_on()
    ax.tick_params(which='both',direction='in',right=True,top=True)
    plt.ylabel(r'$\Delta E_\mathrm{RF}$ (eV)')
    plt.xlabel(r'$\Delta E_\mathrm{DFT}$ (eV)')
    plt.subplots_adjust(left=0.17,bottom=0.13)
    plt.legend(frameon=False,handletextpad=0.1,loc='upper left')

    if(LC):
        train_mean = np.mean(train_scores,axis=1)
        test_mean = np.mean(test_scores,axis=1)
        train_std = np.std(train_scores,axis=1,ddof=1)
        test_std = np.std(test_scores,axis=1,ddof=1)

        ax2 = fig.add_axes([0.60, 0.22, 0.27, 0.26])
        ax2.plot(train_sizes,train_mean,'C9o-',fillstyle='none',lw=1.5,
            markeredgewidth=1.5)
        ax2.plot(train_sizes,test_mean,'C3x-',lw=1.5,markeredgewidth=1.5)
        ax2.fill_between(train_sizes,train_mean - train_std,
            train_mean + train_std,alpha=0.2,color='C9',lw=0)
        ax2.fill_between(train_sizes,test_mean - test_std,
            test_mean + test_std,alpha=0.2,color='C3',lw=0)
        ax2.minorticks_on()
        ax2.tick_params(which='both',direction='in',top=True,right=True,
            labelsize=14)
        ax2.set_xlabel(r'Training set size',fontsize=14)
        ax2.set_ylabel(r'$R^2$',fontsize=14)
        ax2.set_xticks([1000,2000,3000,4000])
        ax2.set_xticklabels(['1k','2k','3k','4k'])
        ax2.xaxis.set_label_coords(0.5,-0.15)

    plt.show()


def line():
    print('\n========================\n')


def main():

    # Get the data
    line()
    data = pd.read_csv(args['input'])
    print("Data types in dataframe:")
    print(data.dtypes)

    print('Finished reading data, length of data: %s' % len(data))

    # Replace missing values with -1
    data = data.apply(pd.to_numeric, errors='coerce').fillna(-1, downcast='infer')

    # Select features to test
    featureNames=['cV','cN','cH','Z','rmsd','rmaxsd','dminNS','daveNS','dminHS','daveHS','mult','chir','q',
    'mu','Egap','cnN','dcnN','cnS','dcnS','aminS','amaxS','aminN','amaxN','adispN','adispH']

    # Describe the data before scaling
    line()
    print('Metadata:')
    print(data.describe())

    # Scale features by maximum absolute values
    transf = MaxAbsScaler().fit(data[data.columns.drop('Ead')])
    data[pd.Index(featureNames)] = transf.transform(data[pd.Index(featureNames)])

    # Get matrix of features and target variable vector
    x=data[pd.Index(featureNames)].values
    y=data['Ead'].values

    # Stratify based on adsorption energies for balanced train-test splits
    strat=np.around(y)

    # Search best parameters for n_estimators, max_features and quit?
    if(GS):
        gridsearch(x,y,strat)
        quit()

    # Predicting numerical values - Random forest regression model
    # Train and test
    line()
    print('RANDOM FOREST REGRESSOR')
    print('Predicting numerical values for training and test set:')
    rf = RandomForestRegressor(n_estimators=200, max_features=10,
        oob_score=True,random_state=rnd,n_jobs=-1)

    # Cross-validation
    y_train,y_pred_train,y_test,y_pred_test,shap_i,shap_c = crossval(x,y,rf,strat)

    # Learning curve
    (train_sizes,train_scores) = (0,0)
    if(LC):
        train_sizes, train_scores, test_scores = lcurve(rf,x,y,strat)

    # Plot SHAP importances
    if(SHAP):
        line()
        plotSHAP(shap_i,shap_c,featureNames)

    line()

    # Plot predictions vs. DFT data and learning curve if chosen
    if(Plot):
        plot(y,y_train,y_pred_train,y_test,y_pred_test,train_sizes,train_scores,test_scores)

if __name__ == '__main__':
    rnd = 111
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=24)
    plt.rc('axes', linewidth=2)
    plt.rc('lines', linewidth=2)
    plt.rc('xtick.major',width=2,size=7)
    plt.rc('xtick.minor',width=1,size=4)
    plt.rc('ytick.major',width=2,size=7)
    plt.rc('ytick.minor',width=1,size=4)
    parser = argparse.ArgumentParser(description='Random forest ML model for H adsorption on NCNTs')
    parser.add_argument('-i','--input',required=True,help='Input data')
    parser.add_argument('-s','--shap',action='store_true',help='Do SHAP analysis')
    parser.add_argument('-p','--plot',action='store_true',help='Plot predictions vs. DFT data')
    parser.add_argument('-l','--lcurve',action='store_true',help='Plot learning curve')
    parser.add_argument('-g','--gridsearch',action='store_true',help='Do hyperparameter gridsearch')
    args = vars(parser.parse_args())
    SHAP = args['shap']
    Plot = args['plot']
    LC = args['lcurve']
    GS = args['gridsearch']
    main()