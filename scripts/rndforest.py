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
    parser.add_argument('-nt', '--ntrees', default=100, type=int,
                        help='Number of trees')
    parser.add_argument('-nf', '--nfeatures', default=10, type=int,
                        help='Number of features considered at each split')
    parser.add_argument('-cv', '--cvfolds', default=10, type=int,
                        help='Number of CV folds')
    parser.add_argument('-sh', '--shap', type=int,
                        help='Run SHAP (negative value includes interactions)')
    parser.add_argument('-rs', '--rsiter', type=int,
                        help='Perform this many parameter search iterations')
    parser.add_argument('-vc', '--valname', type=str,
                        help='Generate validation curve for given parameter')
    parser.add_argument('-lc', '--lcsize', type=int,
                        help='Generate learning curve with given train sizes')

    return parser.parse_args()


def line():

    print('\n========================\n')


def main():

    args = parse()
    inp = args.input
    ntrees = args.ntrees
    nfeatures = args.nfeatures
    k = args.cvfolds
    doshap = args.shap
    rsiter = args.rsiter
    name = args.valname
    lcsize = args.lcsize

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
    cvsize = np.around(size/k)

    print('Data types in data frame:')
    print(data.dtypes)
    print('Finished reading data with %s rows, %s columns' % data.shape)

    # Describe the data
    line()
    print('Metadata:')
    print(data.describe(include='all'))

    # Impute missing values with -999 (nan not understood by sklearn)
    data = data.apply(pd.to_numeric,
                      errors='coerce').fillna(-999, downcast='infer')

    # Select features to test (drop conf, id, Ead)
    # Get matrix of features and target variable vector
    x = data.drop(columns=['conf', 'id', 'Ead']).to_numpy()
    y = data['Ead'].to_numpy()

    # Stratify based on adsorption energies for balanced train-test folds
    strat = np.around(y)

    # Initialize RF regressor with given/default hyperparameters
    rf = RandomForestRegressor(n_estimators=ntrees, max_features=nfeatures,
                               oob_score=True, random_state=rnd, n_jobs=-1)

    # Initialize utility methods
    u = Utilities(x, y, strat, k, rnd, DATA_PATH)

    # Sample optimal hyperparameters and override defaults
    if rsiter is not None:
        line()
        grid = {'n_estimators': np.arange(100, 600, 100),
                'max_features': np.arange(1, 26)}
        u.random_search(rf, grid, rsiter)
        rf.set_params(**u.best_pars)

        print('\nOptimal number of trees: %s' % u.best_pars['n_estimators'])
        print('Optimal number of features: %s' % u.best_pars['max_features'])
        print('OOB error: %.4f' % (1-u.best_score))

    # Generate validation curve with respect to given parameter
    if name is not None:
        line()
        if name == 'max_features':
            grid = np.linspace(1, 25, 25, dtype=int)
        elif name == 'n_estimators':
            grid = np.logspace(1, 3, 49, dtype=int)
        else:
            print('Parameter %s not implemented.' % name)
            print('Specify max_features or n_estimators (or hack the code).')
            quit()

        u.validation_curve(rf, name, grid)

    # Generate learning curve
    if lcsize is not None:
        line()
        grid = np.linspace(0.0858, 1, lcsize)
        u.learning_curve(rf, grid)

    line()

    # Train, test model and perform SHAP analysis with k-fold stratifed CV
    print('Predicting numerical values for training and test set:')
    print('%s-fold cross-validation (training data: %.0f, test data: %.0f)' %
          (k, size-cvsize, cvsize))

    cv = CrossValidate(k, rnd)
    results = Parallel(n_jobs=cv.ncpu)(
              delayed(cv.run)(i, train, test, rf, x, y, DATA_PATH, doshap)
              for i, (train, test) in enumerate(cv.skf.split(x, strat)))

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
