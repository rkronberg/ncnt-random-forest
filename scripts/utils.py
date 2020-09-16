"""
Utility class including methods for learning curve generation and randomized
hyperparameter optimization with stratified k-fold cross-validation.

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, \
    RandomizedSearchCV, learning_curve


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

        iterator = self.skf.split(self.x, self.strat)
        lc = learning_curve(model, self.x, self.y, cv=iterator,
                            train_sizes=np.linspace(0.1, 1, lcsize),
                            shuffle=True, random_state=self.rnd, n_jobs=-1)

        (self.train_sizes, train_scores, test_scores) = lc

        self.train_mean = np.mean(train_scores, axis=1)
        self.test_mean = np.mean(test_scores, axis=1)
        self.train_std = np.std(train_scores, axis=1, ddof=1)
        self.test_std = np.std(test_scores, axis=1, ddof=1)

    def random_search(self, dist, rsiter=20):

        # Function for performing randomized hyperparameter search

        estimator = RandomForestRegressor(oob_score=True,
                                          random_state=self.rnd)
        iterator = self.skf.split(self.x, self.strat)
        rf = RandomizedSearchCV(estimator, dist, n_iter=rsiter, cv=iterator,
                                random_state=self.rnd, n_jobs=-1)
        search = rf.fit(self.x, self.y)
        self.best_score = search.best_score_
        self.best_params = search.best_params_
