
"""
Random Forest ML implementation for H adsorption on NCNTs.
Includes options for randomized hyperparameter search,
calculation of SHAP values and learning curve generation.

GGA: ntrees 500, nfeatures 10
Hybrid: ntrees 100, nfeatures 12

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
"""

# Load necessary packages
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os

from crossval import CrossValidate
from utils import Utilities


def line():

    print('\n========================\n')


def parse():

    # Parse input arguments

    parser = ArgumentParser(description='Random forest ML model for H \
        adsorption on NCNTs')
    parser.add_argument('-i', '--input', required=True, help='Input data')
    parser.add_argument('-sh', '--shap', action='store_true', help='Do SHAP \
     analysis')
    parser.add_argument('-lc', '--ntrain', type=int, help='Number of evenly \
        spaced training set sizes for learning curve generation')
    parser.add_argument('-rs', '--rsiter', type=int, help='Number of \
        iterations for randomized hyperparameter search')
    parser.add_argument('-cv', '--cvfolds', default=10, type=int,
                        help='Number of CV folds')
    parser.add_argument('-nt', '--ntrees', default=100, type=int,
                        help='Number of trees in RF')
    parser.add_argument('-nf', '--nfeatures', default=10, type=int,
                        help='Number of features considered for each tree')

    return vars(parser.parse_args())


def main():

    args = parse()
    inp = args['input']
    ntrees = args['ntrees']
    nfeatures = args['nfeatures']
    nfolds = args['cvfolds']
    rsiter = args['rsiter']
    ntrain = args['ntrain']
    doshap = args['shap']

    # RNG seed
    rnd = 11111

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.normpath(os.path.join(CURRENT_PATH,
                                              os.path.dirname(inp)))

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

    # Select features to test
    featureNames = ['cV', 'cN', 'cH', 'Z', 'rmsd', 'rmaxsd', 'dminNS',
                    'daveNS', 'dminHS', 'daveHS', 'mult', 'chir', 'q', 'mu',
                    'Egap', 'cnN', 'dcnN', 'cnS', 'dcnS', 'aminS', 'amaxS',
                    'aminN', 'amaxN', 'adispN', 'adispH']

    # Describe the data
    line()
    print('Metadata:')
    print(data.describe())

    # Impute missing values with -999
    data = data.apply(pd.to_numeric,
                      errors='coerce').fillna(-999, downcast='infer')

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
                    max_features=np.arange(10, 15))
        u.random_search(dist, rsiter)
        ntrees = u.best_params['n_estimators']
        nfeatures = u.best_params['max_features']
        print('Done!')
        print('Optimal number of trees: %s' % ntrees)
        print('Optimal number of features: %s' % nfeatures)
        print('Best R2 score: %.4f' % u.best_score)

    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=ntrees, max_features=nfeatures,
                               oob_score=True, random_state=rnd, n_jobs=-1)

    # Generate learning curve
    if ntrain is not None:
        line()
        print('Generating learning curve... ', end='')
        u.learning_curve(rf, ntrain)
        header = 'Train sizes, Train mean, Train std, Test mean, Test std'
        np.savetxt('%s/learning_curve.out' % DATA_PATH,
                   np.c_[u.train_sizes, u.train_mean, u.train_std,
                         u.test_mean, u.test_std],
                   header=header, delimiter=',')
        print('Done!')

    # Train, test model and perform SHAP analysis with k-fold stratifed CV
    line()
    print('Predicting numerical values for training and test set:')
    print('%s-fold cross-validation (training data: %s, test data: %s)' %
          (nfolds, size-round(size/nfolds), round(size/nfolds)))

    cv = CrossValidate(nfolds)
    cv.run(x, y, rf, strat, rnd, DATA_PATH, doshap=doshap)

    print('\n')
    print('R2 score (Training set): %.4f +- %.4f' % (cv.score_R2_train,
          cv.score_R2_train_std))
    print('R2 score (Test set): %.4f +- %.4f' % (cv.score_R2_test,
          cv.score_R2_test_std))
    print('RMSE (Training set): %.4f +- %.4f eV' % (cv.score_train,
          cv.score_train_std))
    print('RMSE (Test set): %.4f +- %.4f eV' % (cv.score_test,
          cv.score_test_std))


if __name__ == '__main__':
    main()
