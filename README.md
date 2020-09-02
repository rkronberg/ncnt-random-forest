# ncnt-random-forest
Random forest machine learning implementation written in Python for learning and analyzing hydrogen adsorption properties of defective nitrogen doped carbon nanotubes.

## Usage

```bash
$ python rndforest.py [-h] -i INPUT [-sh] [-lc NTRAIN] [-rs RSITER] [-cv CVFOLDS] [-nt NTREES] [-nf NFEATURES]
```

Optional flags for performing SHAP analysis ```-sh```, learning curve generation ```-lc``` and randomized hyperparameter search ```-rs```. Number of cross-validation folds, decision trees in the random forest and features considered for each tree are provided using the ```-cv```, ```-nt``` and ```-nf``` flags, respectively.
