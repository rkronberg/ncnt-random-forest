"""
Class for training and testing the machine learning model
with k-fold cross-validation. Includes an option for calculating
cross-validated Shapley values following the method of Lundberg & Lee

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import shap
import warnings


class CrossValidate:
    def __init__(self, nfolds):

        # Initialize

        self.nfolds = nfolds
        self.error_test = np.empty(nfolds)
        self.error_train = np.empty(nfolds)
        self.error_R2_test = np.empty(nfolds)
        self.error_R2_train = np.empty(nfolds)

    def run(self, x, y, model, strat, rnd, path, doshap=False):

        # Cross-validate results using nfolds stratified splits

        skf = StratifiedKFold(n_splits=self.nfolds, shuffle=True,
                              random_state=rnd)

        for k, (train, test) in enumerate(skf.split(x, strat)):
            print('Cross-validation split %s/%s' % (k+1, self.nfolds),
                  end='\r')
            model.fit(x[train], y[train])
            y_pred_test = model.predict(x[test])
            y_pred_train = model.predict(x[train])
            self.error_test[k] = np.sqrt(mse(y[test], y_pred_test))
            self.error_train[k] = np.sqrt(mse(y[train], y_pred_train))
            self.error_R2_test[k] = model.score(x[test], y[test])
            self.error_R2_train[k] = model.score(x[train], y[train])

            # Calculate Shapley values following Lundberg & Lee (SHAP)
            if doshap:
                # Reset missing values to nan
                x_test = x[test]
                x_test[np.where(x_test == -999)] = np.nan

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = shap.TreeExplainer(model)
                    shap_base = explainer.expected_value[0]
                    shap_values = explainer.shap_values(x_test)

                np.savetxt('%s/shap-split_%s.out' % (path, k+1), shap_values,
                           header='SHAP values (CV split %s/%s, \
                           Base value %.6f)'
                           % (k+1, self.nfolds, shap_base), delimiter=',')
                np.savetxt('%s/features-split_%s.out' % (path, k+1), x_test,
                           header='Features (CV split %s/%s)'
                           % (k+1, self.nfolds), delimiter=',')

            np.savetxt('%s/y-train-split_%s.out' % (path, k+1),
                       np.c_[y[train], y_pred_train],
                       header='y_train, y_pred_train (CV split %s/%s)'
                       % (k+1, self.nfolds), delimiter=',')
            np.savetxt('%s/y-test-split_%s.out' % (path, k+1),
                       np.c_[y[test], y_pred_test],
                       header='y_test, y_pred_test (CV split %s/%s)'
                       % (k+1, self.nfolds), delimiter=',')

        # Calculate scores
        self.score_test = np.mean(self.error_test)
        self.score_test_std = np.std(self.error_test, ddof=1)
        self.score_train = np.mean(self.error_train)
        self.score_train_std = np.std(self.error_train, ddof=1)

        self.score_R2_test = np.mean(self.error_R2_test)
        self.score_R2_test_std = np.std(self.error_R2_test, ddof=1)
        self.score_R2_train = np.mean(self.error_R2_train)
        self.score_R2_train_std = np.std(self.error_R2_train, ddof=1)
