'''
Random Forest ML implementation for H adsorption on NCNTs.
Includes options for randomized hyperparameter search, calculation
of SHAP values and learning/validation curve generation.

Optimal hyperparameters based on OOB scores
GGA dataset: ntrees 500, nfeatures 12
Hybrid dataset: ntrees 500, nfeatures 25

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from joblib import Parallel, delayed
from time import time
import pandas as pd
import numpy as np
import os

from crossval import CrossValidate
from utils import Utilities


def parse():

    # Parse command line arguments

    parser = ArgumentParser(
        description='Random forest ML model for H adsorption on NCNTs')
    parser.add_argument('-i', '--input', required=True, help='Input data')
    parser.add_argument('-sh', '--shap', type=int,
                        help='Run SHAP (negative value includes interactions')
    parser.add_argument('-lc', '--ntrain', type=int,
                        help='Number of learning curve training set sizes')
    parser.add_argument('-rs', '--rsiter', type=int,
                        help='Number of random parameter search iterations')
    parser.add_argument('-cv', '--cvfolds', default=10, type=int,
                        help='Number of CV folds')
    parser.add_argument('-nt', '--ntrees', default=100, type=int,
                        help='Number of trees')
    parser.add_argument('-nf', '--nfeatures', default=10, type=int,
                        help='Number of features to consider at each split')
    parser.add_argument('-v', '--validation', type=str,
                        help='Generate validation curve wrt. given argument')

    return parser.parse_args()


def line():

    print('\n========================\n')


def main():

    args = parse()
    inp = args.input
    ntrees = args.ntrees
    nfeatures = args.nfeatures
    nfolds = args.cvfolds
    rsiter = args.rsiter
    ntrain = args.ntrain
    doshap = args.shap
    name = args.validation

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.normpath(
        os.path.join(CURRENT_PATH, os.path.dirname(inp)))

    line()
    print('RANDOM FOREST REGRESSOR')
    print('Current directory: %s' % CURRENT_PATH)
    print('Output directory: %s' % DATA_PATH)

    # Get the data
    line()
    data = pd.read_csv(inp)
    size = len(data)
    print('Data types in dataframe:')
    print(data.dtypes)
    print('Finished reading data, length of data: %s' % size)

    # Describe the data
    line()
    print('Metadata:')
    print(data.describe())

    # Impute missing values with -999
    data = data.apply(
        pd.to_numeric, errors='coerce').fillna(-999, downcast='infer')

    # Select features to test
    feature_names = ['cV', 'cN', 'cH', 'Z', 'rmsd', 'rmaxsd', 'dminNS',
                     'dminHS', 'mult', 'chir', 'q', 'mu', 'Egap', 'cnN',
                     'dcnN', 'cnS', 'dcnS', 'aminS', 'amaxS', 'aminN', 'amaxN',
                     'adispN', 'adispH', 'lmin', 'lmax']

    # Get matrix of features and target variable vector
    x = data[pd.Index(feature_names)].values
    y = data['Ead'].values

    # Stratify based on adsorption energies for balanced train-test folds
    strat = np.around(y)

    # Initialize utility methods
    u = Utilities(x, y, strat, nfolds, rnd)

    # Sample optimal hyperparameters and override defaults
    if rsiter is not None:
        line()
        dist = dict(n_estimators=np.arange(100, 600, 100),
                    max_features=np.arange(1, 26))
        u.random_search(dist, rsiter)
        ntrees = u.best_params['n_estimators']
        nfeatures = u.best_params['max_features']
        print(u.best_params)
        print('\nOptimal number of trees: %s' % ntrees)
        print('Optimal number of features: %s' % nfeatures)
        print('Lowest OOB error: %.4f' % (1-u.best_score))

    # Initialize random forest regressor with given/optimized parameters
    rf = RandomForestRegressor(n_estimators=ntrees, max_features=nfeatures,
                               random_state=rnd, n_jobs=-1)

    # Generate learning curve
    if ntrain is not None:
        line()
        u.learning_curve(rf, lcsize=ntrain)
        header = 'Train sizes, Train mean, Train std, Test mean, Test std'
        np.savetxt('%s/learning_curve.out' % DATA_PATH,
                   np.c_[u.train_sizes, u.lc_train_mean, u.lc_train_std,
                         u.lc_test_mean, u.lc_test_std],
                   header=header, delimiter=',')

    # Generate validation curve
    if name is not None:
        line()
        if name == 'max_features':
            alt_name = 'n_estimators'
            alt_param = ntrees
            grid = np.linspace(1, 25, 25, dtype=int)
        elif name == 'n_estimators':
            alt_name = 'max_features'
            alt_param = nfeatures
            grid = np.logspace(1, 3, 49, dtype=int)
        else:
            print('Parameter %s not implemented.' % name)
            print('Specify max_features or n_estimators (or hack the code).')
            quit()

        alt = {alt_name: alt_param}
        u.validation_curve(name, grid, alt)
        header = 'Parameters, Train mean, Train std, Test mean, Test std'
        np.savetxt('%s/validation_curve_%s.out' % (DATA_PATH, name),
                   np.c_[grid, u.vc_train_mean, u.vc_train_std,
                         u.vc_test_mean, u.vc_test_std],
                   header=header, delimiter=',')

    line()

    # Train, test model and perform SHAP analysis with k-fold stratifed CV
    print('Predicting numerical values for training and test set:')
    print('%s-fold cross-validation (training data: %s, test data: %s)' %
          (nfolds, size-round(size/nfolds), round(size/nfolds)))

    cv = CrossValidate(nfolds, rnd)
    gen = cv.skf.split(x, strat)
    results = Parallel(n_jobs=cv.ncpu)(
              delayed(cv.run)(k, train, test, x, y, rf,
                              DATA_PATH, doshap=doshap)
              for k, (train, test) in enumerate(gen))

    mean_scores = np.mean(results, axis=0)
    std_scores = np.std(results, axis=0, ddof=1)

    print('\nTraining set scoring:')
    print('RMSE (Train): %.4f +/- %.4f eV' % (mean_scores[0], std_scores[0]))
    print('MAE (Train): %.4f +/- %.4f eV' % (mean_scores[1], std_scores[1]))
    print('R2 (Train): %.4f +/- %.4f' % (mean_scores[2], std_scores[2]))
    print('Test set scoring:')
    print('RMSE (Test): %.4f +/- %.4f eV' % (mean_scores[3], std_scores[3]))
    print('MAE (Test): %.4f +/- %.4f eV' % (mean_scores[4], std_scores[4]))
    print('R2 (Test): %.4f +/- %.4f' % (mean_scores[5], std_scores[5]))

    print('\nScript executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    rnd = 123
    main()
