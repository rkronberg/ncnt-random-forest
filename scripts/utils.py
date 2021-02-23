'''
Utility class including methods for learning curve generation and randomized
hyperparameter optimization with stratified k-fold cross-validation.

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, \
    RandomizedSearchCV, learning_curve, validation_curve


class Utilities:
    def __init__(self, x, y, strat, nfolds, rnd):

        # Initialize

        self.skf = StratifiedKFold(n_splits=nfolds, shuffle=True,
                                   random_state=rnd)
        self.x = x
        self.y = y
        self.strat = strat
        self.rnd = rnd

    def learning_curve(self, model, lcsize=10):

        # Generate learning curve

        print('Generating learning curve... ')
        gen = self.skf.split(self.x, self.strat)
        lc = learning_curve(model, self.x, self.y, cv=gen, verbose=2,
                            train_sizes=np.linspace(0.1, 1, lcsize),
                            shuffle=True, random_state=self.rnd, n_jobs=-1)

        (self.train_sizes, train_scores, test_scores) = lc

        self.lc_train_mean = np.mean(train_scores, axis=1)
        self.lc_test_mean = np.mean(test_scores, axis=1)
        self.lc_train_std = np.std(train_scores, axis=1, ddof=1)
        self.lc_test_std = np.std(test_scores, axis=1, ddof=1)

    def validation_curve(self, name, grid, alt):

        # Generate validation curve

        print('Generating validation curve... ')
        model = RandomForestRegressor(oob_score=True, n_jobs=-1,
                                      random_state=self.rnd)
        model.set_params(**alt)
        gen = self.skf.split(self.x, self.strat)
        vc = validation_curve(model, self.x, self.y, cv=gen, n_jobs=-1,
                              param_range=grid, param_name=name, verbose=2,
                              scoring=self.oob)

        (train_scores, test_scores) = vc

        self.vc_train_mean = 1-np.mean(train_scores, axis=1)
        self.vc_test_mean = 1-np.mean(test_scores, axis=1)
        self.vc_train_std = np.std(train_scores, axis=1, ddof=1)
        self.vc_test_std = np.std(test_scores, axis=1, ddof=1)

    def random_search(self, dist, rsiter=20):

        '''
        Function for performing randomized hyperparameter search.
        OOB training samples are used as validation set to prevent
        leaking information into the test set.
        '''

        print('Performing grid search of optimal hyperparameters... ')

        estimator = RandomForestRegressor(oob_score=True, n_jobs=-1,
                                          random_state=self.rnd)
        gen = self.skf.split(self.x, self.strat)
        rf = RandomizedSearchCV(estimator, dist, n_iter=rsiter, cv=gen,
                                scoring=self.oob, random_state=self.rnd,
                                n_jobs=-1, verbose=2)
        rf.fit(self.x, self.y)
        self.best_score = rf.best_score_
        self.best_params = rf.best_params_

    def oob(self, estimator, x, y):

        return estimator.oob_score_
