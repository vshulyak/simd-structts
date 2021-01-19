# simd-structts
[![pypi](https://img.shields.io/pypi/v/simd-structts)](https://pypi.org/project/simd-structts/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/simd-structts)
[![Build Status](https://travis-ci.org/vshulyak/simd-structts.svg?branch=master)](https://travis-ci.org/vshulyak/simd-structts)
[![codecov](https://codecov.io/github/vshulyak/simd-structts/branch/master/graph/badge.svg)](https://codecov.io/github/vshulyak/simd-structts)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/github/license/vshulyak/simd_structts)](https://github.com/vshulyak/simd-structts/blob/master/LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/vshulyak/simd-structts/issues)
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
- [x] Proper testing for multiple python versions.
- [x] Equal filtered/smoothed/predicted states for exog components.
- [x] Equal filtered/smoothed/predicted states for long seasonal fourier components.
- [x] Passing tests for statsmodels-like initialization of model.
- [ ] Pretty API with ABC and stuff.
- [ ] Example notebook
- [ ] Gradient methods for finding optimal params
