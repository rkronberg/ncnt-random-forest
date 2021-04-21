'''
Random Forest ML implementation for H adsorption on NCNTs.
Includes options for randomized hyperparameter search, calculation
of SHAP values and learning/validation curve generation.

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from argparse import ArgumentParser
from time import time
from pathlib import Path
from scipy import stats
from joblib import dump
from os import path

from crossval import CrossValidate


def parse():

    # Parse command line arguments
    parser = ArgumentParser(
        description='Random forest and SHAP analysis of H adsorption on NCNTs')
    parser.add_argument('--drop', default=[], nargs='+',
                        help='Columns not to consider as features')
    parser.add_argument('--n_estimators', default=100, type=int,
                        help='Number of estimators (decision trees)')
    parser.add_argument('--max_features', type=int,
                        help='Number of features considered at each split')
    parser.add_argument('--max_depth', type=int,
                        help='Max. depth of any tree')
    parser.add_argument('--min_samples_split', default=2, type=int,
                        help='Min. number of samples required to split node')
    parser.add_argument('--cv_folds', default=5, type=int,
                        help='Number of (outer) CV folds')
    parser.add_argument('--inner_folds', type=int,
                        help='Do nested CV with given number of inner folds')
    parser.add_argument('--shap', type=int,
                        help='Run SHAP (arg. < 0 includes interactions)')
    parser.add_argument('--random_search', action='store_true',
                        help='Do (non-nested) randomized parameter search')
    parser.add_argument('--n_iter', default=10, type=int,
                        help='Number of random parameter settings to test')

    required = parser.add_argument_group('Required named arguments')
    required.add_argument('-i', '--input', required=True,
                          help='Input DataFrame (.csv)')
    required.add_argument('-t', '--target', required=True,
                          help='Name of target column in input DataFrame')

    args = parser.parse_args()
    if args.shap and not args.inner_folds:
        parser.error('The following arguments are required: --inner_folds')

    return args


def line():

    print('\n========================\n')


def main():

    args = parse()
    inp = args.input
    target = args.target
    exclude = args.drop
    n_estimators = args.n_estimators
    max_features = args.max_features
    min_samples_split = args.min_samples_split
    max_depth = args.max_depth
    cv_folds = args.cv_folds
    do_shap = args.shap
    inner_folds = args.inner_folds
    random_search = args.random_search
    n_iter = args.n_iter

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(inp)))

    line()
    print('RANDOM FOREST REGRESSOR')
    print('Current directory: %s' % CURRENT_PATH)
    print('Output directory: %s' % DATA_PATH)

    # Get the data
    line()
    data = pd.read_csv(inp)

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
    x = data.drop(columns=exclude)
    y = data[target]

    # Stratify based on target variable for balanced train-test folds
    strat = np.around(y)

    # Initialize RF regressor with given/default hyperparameters
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_features=max_features,
        min_samples_split=min_samples_split, max_depth=max_depth,
        random_state=rnd, n_jobs=-1)

    # Initialize cross-validation methods
    cv = CrossValidate(x, y, data, cv_folds, inner_folds, strat, n_iter, rnd)

    # Sample optimal hyperparameters and override defaults
    if inner_folds is not None:
        line()
        grid = {'max_features': stats.randint(5, 26),
                'max_depth': stats.randint(10, 56),
                'min_samples_split': stats.randint(2, 4)}

        cv.nested_crossval(rf, grid, do_shap, DATA_PATH)

        print('Unbiased generalization performance estimation:\n')
        print('Training set:')
        print('MAE (Train): %.4f +/- %.4f'
              % (np.mean(cv.mae_train), np.std(cv.mae_train, ddof=1)))
        print('RMSE (Train): %.4f +/- %.4f'
              % (np.mean(cv.rmse_train), np.std(cv.rmse_train, ddof=1)))
        print('R2 (Train): %.4f +/- %.4f'
              % (np.mean(cv.r2_train), np.std(cv.r2_train, ddof=1)))
        print('Test set:')
        print('MAE (Test): %.4f +/- %.4f'
              % (np.mean(cv.mae_test), np.std(cv.mae_test, ddof=1)))
        print('RMSE (Test): %.4f +/- %.4f'
              % (np.mean(cv.rmse_test), np.std(cv.rmse_test, ddof=1)))
        print('R2 (Test): %.4f +/- %.4f'
              % (np.mean(cv.r2_test), np.std(cv.r2_test, ddof=1)))

    # Split data avoiding train_test_split n_splits > class members ValueError
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rnd)
    train, test = next(skf.split(x, strat))

    if random_search:
        line()
        grid = {'max_features': stats.randint(5, 26),
                'max_depth': stats.randint(10, 56),
                'min_samples_split': stats.randint(2, 4)}

        # Tune hyperparameters using training set and cross-validation
        cv.random_search(rf, train, grid)

        print('\nBest parameters: %s' % cv.best_pars)
        print('Best score: %.4f' % cv.best_score)

        rf.set_params(**cv.best_pars)

    # Train, test final model
    line()
    print('Train, test final model:')

    rf.fit(x.iloc[train], y.iloc[train])
    y_pred_test = rf.predict(x.iloc[test])
    y_pred_train = rf.predict(x.iloc[train])

    print('\nTraining set scoring:')
    print('MAE (Train): %.4f' % mae(y_pred_train, y.iloc[train]))
    print('RMSE (Train): %.4f' % np.sqrt(mse(y_pred_train, y.iloc[train])))
    print('R2 (Train): %.4f' % r2(y_pred_train, y.iloc[train]))
    print('Test set scoring:')
    print('MAE (Test): %.4f' % mae(y_pred_test, y.iloc[test]))
    print('RMSE (Test): %.4f' % np.sqrt(mse(y_pred_test, y.iloc[test])))
    print('R2 (Test): %.4f' % r2(y_pred_test, y.iloc[test]))

    # Pickle model
    dump(rf, '%s/model.pkl' % DATA_PATH)

    print('\nScript executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    rnd = np.random.RandomState(42)
    main()
