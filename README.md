# ncnt-random-forest
Random forest machine learning implementation written in Python for learning and analyzing hydrogen adsorption properties of defective nitrogen doped carbon nanotubes.

## Usage

```bash
$ python rndforest.py [-h] -i INPUT [-nt NTREES] [-nf NFEATURES] [-cv CVFOLDS] [-sh SHAP] [-rs RSITER] [-vc VALNAME] [-lc LCSIZE]
```

Optional flags for performing SHAP analysis ```-sh```, learning curve generation ```-lc```, validation curve generation ```-vc``` and randomized hyperparameter search ```-rs```. Number of cross-validation folds, decision trees in the random forest and random features considered at each node are provided using the ```-cv```, ```-nt``` and ```-nf``` flags, respectively. By default, the cross-validation routine is parallelized using ```CVFOLDS``` CPUs (or the maximum amount available). Integer ```SHAP``` arguments activates calculation of cross-validated SHAP values for the test set samples. Negative arguments trigger also the computation of SHAP interaction values.
