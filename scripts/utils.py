'''
Utility class including methods for learning/validation
curve generation and randomized hyperparameter optimization
with stratified k-fold cross-validation.

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, \
     learning_curve, validation_curve


class Utilities:
    def __init__(self, x, y, strat, k, rnd, DATA_PATH):

        # Initialize
        self.x = x
        self.y = y
        self.strat = strat
        self.rnd = rnd
        self.path = DATA_PATH
        self.skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rnd)

    def oob(self, model, x, y):

        # OOB scorer
        return model.oob_score_

    def random_search(self, model, grid, rsiter):

        '''
        Function for performing randomized hyperparameter search.
        OOB training samples are used as validation set to keep
        the test set pure.
        '''

        print('Performing randomized search of optimal hyperparameters... ')

        gen = self.skf.split(self.x, self.strat)
        rs = RandomizedSearchCV(model, grid, n_iter=rsiter, cv=gen,
                                scoring=self.oob, random_state=self.rnd,
                                n_jobs=-1, verbose=2)

        rs.fit(self.x, self.y)
        self.best_score = rs.best_score_
        self.best_pars = rs.best_params_

    def validation_curve(self, model, name, grid):

        # Generate validation curve
        print('Generating validation curve... ')

        gen = self.skf.split(self.x, self.strat)
        vc = validation_curve(model, self.x, self.y, cv=gen, param_name=name,
                              param_range=grid, scoring=self.oob, n_jobs=-1,
                              verbose=2)

        train_scores, test_scores = vc
        train_mean = 1-np.mean(train_scores, axis=1)
        test_mean = 1-np.mean(test_scores, axis=1)
        train_std = np.std(train_scores, axis=1, ddof=1)
        test_std = np.std(test_scores, axis=1, ddof=1)

        np.savetxt('%s/validation_curve_%s.out' % (self.path, name),
                   np.c_[grid, train_mean, train_std, test_mean, test_std],
                   header='Param, Train mean, Train std, Test mean, Test std',
                   delimiter=',')

    def learning_curve(self, model, grid):

        # Generate learning curve
        print('Generating learning curve... ')

        gen = self.skf.split(self.x, self.strat)
        lc = learning_curve(model, self.x, self.y, cv=gen, shuffle=True,
                            train_sizes=grid, random_state=self.rnd, n_jobs=-1,
                            verbose=2)

        size, train_scores, test_scores = lc
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        train_std = np.std(train_scores, axis=1, ddof=1)
        test_std = np.std(test_scores, axis=1, ddof=1)

        np.savetxt('%s/learning_curve.out' % self.path,
                   np.c_[size, train_mean, train_std, test_mean, test_std],
                   header='Size, Train mean, Train std, Test mean, Test std',
                   delimiter=',')
