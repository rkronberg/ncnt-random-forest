'''
Class for training and testing the machine learning model with stratified
k-fold cross-validation. Includes an option for calculating Shapley values
for feature attribution as presented by Lundberg & Lee (SHAP).

author: Rasmus Kronberg
email: rasmus.kronberg@aalto.fi
'''

# Load necessary packages
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import multiprocessing as mp
import numpy as np
import shap


class CrossValidate:
    def __init__(self, nfolds, rnd):

        # Initialize

        self.nfolds = nfolds
        self.rnd = rnd
        self.skf = StratifiedKFold(n_splits=self.nfolds, shuffle=True,
                                   random_state=rnd)
        self.ncpu = nfolds if nfolds <= mp.cpu_count() else mp.cpu_count()
        print('Multiprocessing with %s CPUs\n' % self.ncpu)

    def run(self, k, train, test, x, y, model, path, doshap=None):

        # Cross-validate results using nfolds stratified splits

        print('Processing cross-validation split %s/%s' % (k+1, self.nfolds))
        model.fit(x[train], y[train])
        y_pred_test = model.predict(x[test])
        y_pred_train = model.predict(x[train])

        error_rmse_test = np.sqrt(mse(y[test], y_pred_test))
        error_rmse_train = np.sqrt(mse(y[train], y_pred_train))
        error_mae_test = mae(y[test], y_pred_test)
        error_mae_train = mae(y[train], y_pred_train)
        error_r2_test = model.score(x[test], y[test])
        error_r2_train = model.score(x[train], y[train])

        # Calculate Shapley values following Lundberg & Lee (SHAP)
        if doshap is not None:
            # Reset missing values to nan
            x_test = x[test]
            x_test[np.where(x_test == -999)] = np.nan

            # Calculate observational ('true to the data') SHAP values
            explainer = shap.explainers.Tree(model)
            shap_base = explainer.expected_value
            shap_values = explainer.shap_values(x_test)

            # Calculate SHAP interaction values
            if doshap < 0:
                interaction_values = explainer.shap_interaction_values(x_test)

                # Store 3D array as 2D slices
                with open('%s/interact-split_%s.out' % (path, k+1), 'w') as o:
                    o.write('# Interact. values, shape %s (CV split %s/%s)\n'
                            % (interaction_values.shape, k+1, self.nfolds))
                    for data_slice in interaction_values:
                        np.savetxt(o, data_slice, delimiter=',')
                        o.write('# Next slice\n')

            np.savetxt('%s/shap-split_%s.out' % (path, k+1), shap_values,
                       header='SHAP values (CV split %s/%s, Base value %.6f)'
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

        return error_rmse_train, error_mae_train, error_r2_train, \
            error_rmse_test, error_mae_test, error_r2_test
