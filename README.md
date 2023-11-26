# Bayesian Optimization for Metallophotocatalysis Formulations

[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

This repository comprises scripts used in the research paper titled 
'Sequential Closed-Loop Bayesian Optimization as a Guide for Organic 
Molecular Metallophotocatalyst Formulations Discovery'.

## Code Structure

### Overview
The Bayesian Optimization instance ``BayesOptimizer``, available in 
``src.BayesOpt.optimizer``, is employed for the optimization of organic 
catalysts and the reaction conditions. 

The Gaussian processes instance ``GaussianProcessRegressor``, created in 
``src.BayesOpt.GPR``, is used as the surrogate model of the optimization 
of reaction conditions. 

### Major functions 

| Function                                          | Description                                                      |
|---------------------------------------------------|------------------------------------------------------------------|
| ``src.data_pretreatment.ks_selection``            | The Kennard-Stone algorithm used for selecting represent subset. |
| ``src.data_pretreatment.cal_fingerprints``        | Calculating the Fingerprints of given chemical SMILES.           |
| ``src.BayesOpt.BayesOptimizer.ask``               | Query one point at which objective should be evaluated.          |
| ``src.BayesOpt.BayesOptimizer.parallel_ask``      | Query several points at which objective should be evaluated.     |
| ``src.BayesOpt.BayesOptimizer.tell``              | Recording evaluated points of the objective function.            |
| ``src.BayesOpt.acquisition_function``             | Computing the acquisition function.                              |
| ``src.BayesOpt.GaussianProcessRegressor.fit``     | Fit Gaussian process regression model.                           |
| ``src.BayesOpt.GaussianProcessRegressor.predict`` | Predict using the Gaussian process regression model.             |

## Author
Yu Che

