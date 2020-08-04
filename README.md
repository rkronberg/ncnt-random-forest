# rndForest
Random forest ML Python implementation for materials science data analysis, namely H adsorption on NCNTs

## Usage

```bash
usage: rndforest.py [-h] -i INPUT [-sh] [-lc NTRAIN] [-rs RSITER] [-cv CVFOLDS] [-nt NTREES] [-nf NFEATURES]
```

Optional flags for performing SHAP analysis ```-sh```, learning curve generation ```-lc``` and randomized hyperparameter search ```-rs```. Number of cross-validation folds, decision trees in the random forest and maximum number of features considered in each tree are provided using the ```-cv```, ```-nt``` and ```-nf``` flags, respectively. Plot results by running ```plot.py```.

Repository includes also an *ad hoc* ```feature_parser.py``` script for preprocessing data from CP2K output
