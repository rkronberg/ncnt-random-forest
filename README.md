# rndForest
Random forest ML Python implementation for materials science data analysis, namely H adsorption on NCNTs

## Usage

```bash
python3 rndForest.py [-h] [-i INPUT] [-s] [-p] [-l] [-g]
```

Optional flags for performing SHAP analysis (```bash -s```), plotting (```bash -l```), generating a learning curve (```bash -l```)  and hyperparameter gridsearch (```bash -g```)

Repository includes also an ad hoc ```bash dataCollector.py``` script for preprocessing data from CP2K output
