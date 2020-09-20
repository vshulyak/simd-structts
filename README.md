# simd-structts
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Using PyTorch](https://img.shields.io/badge/PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/vshulyak/torch-es/blob/master/LICENSE)

Multivariate forecasting using StructTS/Unobserved Components model without MLE param estimation.

## 🤦🏾‍ Motivation

I love structts model and Kalman filters for forecasting. Sometimes you just want a model that works out of the box
without *designing* a model with a Kalman filter, especially if you need to use long seasonalites and exog variables.
Defining all these state space matrices gets tedious pretty quickly...

The code in this repo is an attempt to bring a familiar API to multivariate StructTS model, currently with the simdkalman library as a backend.

## 👩🏾‍🚀 Installation

      pip install simd-structts


## 📋 WIP:
- [x] Statsmodels and simdkalman backend implementation.
- [x] Equal filtered/smoothed/predicted states for level/trend models.
- [ ] Proper testing for multiple python versions.
- [ ] Equal filtered/smoothed/predicted states for exog components.
- [ ] Equal filtered/smoothed/predicted states for long seasonal fourier components.
- [ ] Passing tests for statsmodels-like initialization of model.
- [ ] Pretty API with ABC and stuff.
