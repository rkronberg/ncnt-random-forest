
"""
Random Forest ML implementation for H adsorption on NCNTs.
Includes options for randomized hyperparameter search,
calculation of SHAP values and learning curve generation.

Optimal hyperparameters
GGA dataset: ntrees 500, nfeatures 13
Hybrid dataset: ntrees 200, nfeatures 16

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
"""

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
    featureNames = ['cV', 'cN', 'cH', 'Z', 'rmsd', 'rmaxsd', 'dminNS',
                    'daveNS', 'dminHS', 'daveHS', 'mult', 'chir', 'q', 'mu',
                    'Egap', 'cnN', 'dcnN', 'cnS', 'dcnS', 'aminS', 'amaxS',
                    'aminN', 'amaxN', 'adispN', 'adispH', 'lmin', 'lmax',
                    'lave']

    # Get matrix of features and target variable vector
    x = data[pd.Index(featureNames)].values
    y = data['Ead'].values

    # Stratify based on adsorption energies for balanced train-test folds
    strat = np.around(y)

    # Initialize utility methods
    u = Utilities(x, y, strat, nfolds, rnd)

    # Sample optimal hyperparameters and override defaults
    if rsiter is not None:
        line()
        print('Performing randomized search of optimal hyperparameters... ',
              end='')
        dist = dict(n_estimators=np.arange(100, 600, 100),
                    max_features=np.arange(1, 21))
        u.random_search(dist, rsiter=rsiter)
        ntrees = u.best_params['n_estimators']
        nfeatures = u.best_params['max_features']
        print('Done!')
        print('Optimal number of trees: %s' % ntrees)
        print('Optimal number of features: %s' % nfeatures)
        print('Best R2 score: %.4f' % u.best_score)

    # Initialize random forest regressor with given/optimized parameters
    rf = RandomForestRegressor(n_estimators=ntrees, max_features=nfeatures,
                               oob_score=True, random_state=rnd, n_jobs=-1)

    # Generate learning curve
    if ntrain is not None:
        line()
        print('Generating learning curve... ', end='')
        u.learning_curve(rf, lcsize=ntrain)
        header = 'Train sizes, Train mean, Train std, Test mean, Test std'
        np.savetxt('%s/learning_curve.out' % DATA_PATH,
                   np.c_[u.train_sizes, u.train_mean, u.train_std,
                         u.test_mean, u.test_std],
                   header=header, delimiter=',')
        print('Done!')

    line()

    # Train, test model and perform SHAP analysis with k-fold stratifed CV
    print('Predicting numerical values for training and test set:')
    print('%s-fold cross-validation (training data: %s, test data: %s)' %
          (nfolds, size-round(size/nfolds), round(size/nfolds)))

    cv = CrossValidate(nfolds, rnd)
    results = Parallel(n_jobs=cv.ncores)(delayed(cv.run)(k, train, test, x, y,
             rf, DATA_PATH, doshap=doshap)
             for k, (train, test) in enumerate(cv.skf.split(x, strat)))

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

    print('\nExecuted in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    rnd = 123
    main()
