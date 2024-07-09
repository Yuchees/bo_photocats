# Bayesian Optimization for Metallophotocatalysis Formulations

[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41557--024--01546--5%20-blue)](https://doi.org/10.1038/s41557-024-01546-5)

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

The Jupyter notebooks in `workflows` are the original record of the optimization
of CNPs and reaction conditions. The kernel matrix of designed reaction 
conditions need to be built before running the optimization. 

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

