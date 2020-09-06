from plots import predictions, shap_summary, shap_dependence, shap_waterfall
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Plot learning curve together with predictions?
    lc = True
    predictions.plot(lc)

    shap_summary.plot()

    # Plot dependence plots for all features
    plot_all = True
    shap_dependence.plot(plot_all)

    shap_waterfall.plot()