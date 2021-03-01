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
    def __init__(self, k, rnd):

        # Initialize
        self.k = k
        self.skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rnd)
        self.ncpu = k if k <= mp.cpu_count() else mp.cpu_count()
        print('Multiprocessing with %s CPUs\n' % self.ncpu)

    def run(self, i, train, test, model, x, y, path, doshap=None):

        # Cross-validate results using k stratified splits
        print('Processing cross-validation split %s/%s' % (i+1, self.k))
        model.fit(x[train], y[train])
        y_pred_test = model.predict(x[test])
        y_pred_train = model.predict(x[train])

        rmse_test = np.sqrt(mse(y[test], y_pred_test))
        rmse_train = np.sqrt(mse(y[train], y_pred_train))
        mae_test = mae(y[test], y_pred_test)
        mae_train = mae(y[train], y_pred_train)
        r2_test = model.score(x[test], y[test])
        r2_train = model.score(x[train], y[train])

        # Calculate SHAP (interaction) values
        if doshap is not None:

            # Reset missing values to nan (understood by SHAP)
            x_test = x[test]
            x_test[np.where(x_test == -999)] = np.nan

            # Calculate observational ('true to the data') SHAP values
            explainer = shap.explainers.Tree(model)
            shap_base = explainer.expected_value
            shap_values = explainer.shap_values(x_test)

            # Calculate SHAP interaction values
            if doshap < 0:
                interaction_values = explainer.shap_interaction_values(x_test)

                # Store 3D interaction matrix as 2D slices
                with open('%s/interact-split_%s.out' % (path, i+1), 'w') as o:
                    o.write('# Interact. values, shape %s (CV split %s/%s)\n'
                            % (interaction_values.shape, i+1, self.k))
                    for data_slice in interaction_values:
                        np.savetxt(o, data_slice, delimiter=',')
                        o.write('# Next slice\n')

            np.savetxt('%s/shap-split_%s.out' % (path, i+1), shap_values,
                       header='SHAP values (CV split %s/%s, Base value %.6f)'
                       % (i+1, self.k, shap_base), delimiter=',')
            np.savetxt('%s/features-split_%s.out' % (path, i+1), x_test,
                       header='Features (CV split %s/%s)'
                       % (i+1, self.k), delimiter=',')

        np.savetxt('%s/y-train-split_%s.out' % (path, i+1),
                   np.c_[y[train], y_pred_train],
                   header='y_train, y_pred_train (CV split %s/%s)'
                   % (i+1, self.k), delimiter=',')
        np.savetxt('%s/y-test-split_%s.out' % (path, i+1),
                   np.c_[y[test], y_pred_test],
                   header='y_test, y_pred_test (CV split %s/%s)'
                   % (i+1, self.k), delimiter=',')

        return rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test
