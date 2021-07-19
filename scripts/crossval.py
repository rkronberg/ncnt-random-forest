'''
Class for training and validating a machine learning model with nested
k*l-fold cross-validation. Includes an option for calculating Shapley
values for feature attribution as proposed by Lundberg & Lee (SHAP).

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
import shap
import numpy as np
import pandas as pd
from time import time
from joblib import Parallel, delayed, cpu_count, dump

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, \
    ParameterSampler


def line():

    print('\n========================\n')


class CrossValidate:
    def __init__(self, x, y, data, kout, kin, strat, n_iter, rnd):

        # Initialize
        self.x = x
        self.y = y
        self.data = data
        self.strat = strat
        self.rnd = rnd
        self.n_iter = n_iter
        self.outer = StratifiedKFold(
            n_splits=kout, shuffle=True, random_state=rnd)

        # If performing nested CV, initialize inner loop
        if kin is not None:
            self.inner = StratifiedKFold(
                n_splits=kin, shuffle=True, random_state=rnd)

    def random_search(self, model, train, grid):

        print('Performing random search of optimal hyperparameters...')
        cv = self.outer.split(
            self.x.iloc[train], self.strat.iloc[train])

        # Grid search hyperparameters on provided training set using CV
        rs = RandomizedSearchCV(model, grid, cv=cv, n_iter=self.n_iter,
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1, verbose=2, random_state=self.rnd)

        # Fit models
        rs.fit(self.x.iloc[train], self.y.iloc[train])

        # Get best score and corresponding parameters
        self.best_score = -rs.best_score_
        self.best_pars = rs.best_params_

    def loop_params(self, model, train, val, p):

        # Set model hyperparameters p
        model.set_params(**p)

        # Fit model on train, evaluate on val
        model.fit(self.x.iloc[train], self.y.iloc[train])
        y_pred = model.predict(self.x.iloc[val])

        return np.sqrt(mse(y_pred, self.y.iloc[val]))

    def nested_crossval(self, model, grid, do_shap, path):

        print('Performing nested cross-validation:')

        # Initialize
        n_inner = self.inner.get_n_splits()
        n_outer = self.outer.get_n_splits()

        print('%i outer folds, %i inner folds, %i candidates = %i models\n'
              % (n_outer, n_inner, self.n_iter, n_outer*n_inner*self.n_iter))

        self.rmse_test = np.zeros(n_outer)
        self.rmse_train = np.zeros(n_outer)
        self.mae_test = np.zeros(n_outer)
        self.mae_train = np.zeros(n_outer)
        self.r2_test = np.zeros(n_outer)
        self.r2_train = np.zeros(n_outer)

        ncpu = self.n_iter if self.n_iter < cpu_count() else cpu_count()
        outer_cv = self.outer.split(self.x, self.strat)

        # Loop over outer folds (trainval-test split)
        for i, (trainval, test) in enumerate(outer_cv):

            t_outer = time()
            scores = np.zeros((n_inner, self.n_iter))

            inner_cv = self.inner.split(
                self.x.iloc[trainval], self.strat.iloc[trainval])
            settings = list(ParameterSampler(
                grid, n_iter=self.n_iter, random_state=self.rnd))

            # For each trainval set, loop over inner folds (train-val split)
            for j, (train, val) in enumerate(inner_cv):

                t_inner = time()

                # For each train-val split, loop over all parameter settings
                scores[j, :] = Parallel(n_jobs=ncpu, verbose=10)(
                        delayed(self.loop_params)(model, train, val, p)
                        for p in settings)

                line()
                print('Processed INNER fold %i/%i (%.1f s)'
                      % (j+1, n_inner, time()-t_inner))
                line()

            # Average scores over the inner folds for each candidate
            averages = np.mean(scores, axis=0)
            stds = np.std(scores, axis=0, ddof=1)

            # Locate index corresponding to best hyperparameters (min loss)
            best_idx = np.argmin(averages)
            best_score = averages[best_idx]
            best_std = stds[best_idx]
            best_pars = settings[best_idx]

            # Set best hyperparameters, refit model and evaluate on test set
            model.set_params(**best_pars)
            model.fit(self.x.iloc[trainval], self.y.iloc[trainval])
            y_pred_test = model.predict(self.x.iloc[test])
            y_pred_train = model.predict(self.x.iloc[trainval])

            # Compute outer loop metrics
            self.rmse_test[i] = np.sqrt(mse(self.y.iloc[test], y_pred_test))
            self.rmse_train[i] = np.sqrt(mse(self.y.iloc[trainval],
                                             y_pred_train))
            self.mae_test[i] = mae(self.y.iloc[test], y_pred_test)
            self.mae_train[i] = mae(self.y.iloc[trainval], y_pred_train)
            self.r2_test[i] = r2(self.y.iloc[test], y_pred_test)
            self.r2_train[i] = r2(self.y.iloc[trainval], y_pred_train)

            # Dump true values and predictions
            train_out = pd.DataFrame({'y_train': self.y.iloc[trainval],
                                      'y_pred_train': y_pred_train})
            train_out.to_csv('%s/y-train-split_%s.csv' % (path, i+1))

            test_out = pd.DataFrame({'y_test': self.y.iloc[test],
                                     'y_pred_test': y_pred_test})
            test_out.to_csv('%s/y-test-split_%s.csv' % (path, i+1))

            print('Processed OUTER fold %i/%i (%.1f s)'
                  % (i+1, n_outer, time()-t_outer))
            print('Best hyperparameters: %s' % best_pars)
            print('Best inner score: %.4f +/- %.4f' % (best_score, best_std))
            print('Outer score: %.4f' % self.rmse_test[i])
            line()

            # Calculate observational SHAP (interaction) values
            if do_shap is not None:

                t_shap = time()
                print('Calculating SHAP values...')

                x_test = self.x.iloc[test]

                # Calculate SHAP values
                explainer = shap.explainers.GPUTree(model)
                shap_values = explainer(x_test)

                # Dump SHAP explanation object
                dump(shap_values, '%s/shap-split_%s.pkl' % (path, i+1))

                # Calculate SHAP interaction values
                if do_shap < 0:
                    intval = explainer.shap_interaction_values(x_test)

                    # Store 3D interaction matrix as 2D slices
                    with open('%s/int-split_%s.csv' % (path, i+1), 'w') as o:
                        o.write('# Array shape (%s, %s, %s)\n' % intval.shape)
                        for data_slice in intval:
                            pd.DataFrame(data_slice).to_csv(
                                o, mode='a', index=False, header=False)
                            o.write('# Next slice\n')

                print('Done! (%.1f s)' % (time()-t_shap))
                line()
