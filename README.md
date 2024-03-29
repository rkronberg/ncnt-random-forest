# ncnt-random-forest

[![DOI:10.1021/acs.jpcc.1c03858](http://img.shields.io/badge/DOI-10.1021/acs.jpcc.1c03858-blue.svg)](https://doi.org/10.1021/acs.jpcc.1c03858)

Random forest (RF) and Shapley additive explanations (SHAP) implementation written in Python for machine learning and analysis of hydrogen adsorption on defective nitrogen doped carbon nanotubes (NCNT). The script can be easily applied for other datasets and regression tasks as well.

## Usage

```console
$ python rndforest.py --help
usage: rndforest.py [-h] [--drop DROP [DROP ...]] [--n_estimators N_ESTIMATORS]
                    [--max_features MAX_FEATURES] [--max_depth MAX_DEPTH]
                    [--min_samples_split MIN_SAMPLES_SPLIT] [--cv_folds CV_FOLDS]
                    [--inner_folds INNER_FOLDS] [--shap SHAP] [--random_search]
                    [--n_iter N_ITER] [--learning_curve LEARNING_CURVE]
                    [--validation VALIDATION] -i INPUT -t TARGET

Random forest and SHAP analysis of H adsorption on NCNTs

optional arguments:
  -h, --help            show this help message and exit
  --drop DROP [DROP ...]
                        Columns not to consider as features
  --n_estimators N_ESTIMATORS
                        Number of estimators (decision trees)
  --max_features MAX_FEATURES
                        Number of features considered at each split
  --max_depth MAX_DEPTH
                        Max. depth of any tree
  --min_samples_split MIN_SAMPLES_SPLIT
                        Min. number of samples required to split node
  --cv_folds CV_FOLDS   Number of (outer) CV folds
  --inner_folds INNER_FOLDS
                        Do nested CV with given number of inner folds
  --shap SHAP           Run SHAP (arg. < 0 includes interactions)
  --random_search       Do (non-nested) randomized parameter search
  --n_iter N_ITER       Number of random parameter settings to test
  --learning_curve LEARNING_CURVE
                        Number of learning curve training set sizes
  --validation VALIDATION
                        Generate validation curve for given argument

Required named arguments:
  -i INPUT, --input INPUT
                        Input DataFrame (.csv)
  -t TARGET, --target TARGET
                        Name of target column in input DataFrame
```

Input data is expected to be a pandas ```DataFrame```. The script implements a nested cross-validation (CV) procedure for exhaustive analysis and algorithm stability estimation (see figure below). Randomized hyperparameter search is conducted within the inner CV loop if ```--inner_folds``` is specified. For a non-nested (only outer CV) variant yielding a single optimized model, the ```--random_search``` flag should be given. In both cases, the number of parameter iterations is provided using the ```--n_iter``` flag. Integer ```--shap``` arguments activates the calculation of SHAP values for all outer CV test set samples, requiring that also ```--inner_folds``` is given. Computation of SHAP interaction values is triggered by negative ```--shap``` arguments. Learning and validation curves can be computed using the ```--learning_curve``` and ```--validation``` flags with appropriate arguments.

To improve the computational performance, hyperparameter optimization within the inner CV loop is parallelized using ```--n_iter``` CPUs, or the maximum amount available. The SHAP analysis is accelerated by using the GPU version of the TreeSHAP algorithm, requiring a source build of the [```shap```](https://shap.readthedocs.io/) package with CUDA available and the ```CUDA_PATH``` environment variable defined.

A conceptual visualization of the nested cross-validation workflow is illustrated below.

![Conceptual illustration of nested cross-validation](./nested_cv.svg)
