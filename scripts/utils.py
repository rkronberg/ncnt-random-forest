'''
Utility class including methods for learning and validation curve generation
with stratified k-fold cross-validation.

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, \
    learning_curve, validation_curve


class Utilities:
    def __init__(self, x, y, strat, nfolds, rnd, path):

        # Initialize

        self.skf = StratifiedKFold(n_splits=nfolds, shuffle=True,
                                   random_state=rnd)
        self.x = x
        self.y = y
        self.strat = strat
        self.rnd = rnd
        self.path = path

    def learning_curve(self, model, lcsize=10):

        # Generate learning curve

        print('Generating learning curve... ')
        gen = self.skf.split(self.x, self.strat)
        lc = learning_curve(model, self.x, self.y, cv=gen, shuffle=True,
                            train_sizes=np.logspace(-2, 0, lcsize),
                            scoring='neg_root_mean_squared_error', verbose=2,
                            random_state=self.rnd, n_jobs=-1)

        (train_sizes, train_scores, test_scores) = lc

        train_mean = np.mean(-train_scores, axis=1)
        test_mean = np.mean(-test_scores, axis=1)
        train_std = np.std(-train_scores, axis=1, ddof=1)
        test_std = np.std(-test_scores, axis=1, ddof=1)

        df = pd.DataFrame(
            np.c_[train_sizes, train_mean, train_std, test_mean, test_std],
            columns=['train_sizes', 'train_mean', 'train_std', 'test_mean',
                     'test_std'])
        df.to_csv('%s/learning_curve.csv' % self.path)

    def validation_curve(self, model, name, train):

        # Generate validation curve

        xdata = self.x.iloc[train]
        ydata = self.y.iloc[train]
        stratify = self.strat.iloc[train]

        print('Generating validation curve... ')
        if name == 'max_features':
            n_cols = xdata.shape[1]
            grid = np.linspace(1, n_cols, n_cols, dtype=int)
        elif name == 'n_estimators':
            grid = np.logspace(0, 3, 110, dtype=int)
        elif name == 'max_depth':
            grid = np.linspace(1, 50, 50, dtype=int)
        elif name == 'min_samples_split':
            grid = np.linspace(2, 25, 25, dtype=int)
        else:
            print('Parameter %s not implemented.' % name)
            quit()

        gen = self.skf.split(xdata, stratify)
        vc = validation_curve(model, xdata, ydata, cv=gen, param_range=grid,
                              param_name=name, verbose=2, n_jobs=-1,
                              scoring='neg_root_mean_squared_error')

        (train_scores, val_scores) = vc

        train_mean = np.mean(-train_scores, axis=1)
        val_mean = np.mean(-val_scores, axis=1)
        train_std = np.std(-train_scores, axis=1, ddof=1)
        val_std = np.std(-val_scores, axis=1, ddof=1)

        df = pd.DataFrame(
            np.c_[grid, train_mean, train_std, val_mean, val_std],
            columns=['parameters', 'train_mean', 'train_std', 'val_mean',
                     'val_std'])
        df.to_csv('%s/%s_validation_curve.csv' % (self.path, name))
