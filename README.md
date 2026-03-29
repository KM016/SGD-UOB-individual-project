# Stochastic Approximation: From Robbins-Monro to Variance-Reduced Proximal Gradient Methods

**University of Bristol** | Level 6 Mathematics Project  
**Author:** Keyaan Miah  
**Grade:** 92% (First-Class)

## Overview
This repository contains the full dissertation and Python simulation code for my final-year mathematics project. The research bridges the gap between classical 1950s statistical theory and modern large-scale machine learning optimisation. 

By framing modern stochastic gradient algorithms as instances of **Robbins-Monro Stochastic Approximation**, this project isolates the exact mathematical mechanism that separates classical methods from modern variance-reduced schemes: **the asymptotic behaviour of the noise.**

While standard Stochastic Gradient Descent (SGD) introduces a persistent variance floor that prevents exact convergence under constant step sizes, variance-reduced algorithms like **SVRG** and **Prox-SVRG** exploit finite-sum structures to create control-variate estimators. This project mathematically proves and empirically demonstrates how this modification forces the estimator variance to collapse to zero, enabling linear (geometric) convergence under strong convexity.

## 📂 Repository Structure
```
├── figures/                         # Figures used in the report
│   ├── ill-conditioned/             # Results for ill-conditioned experiments
│   ├── well-conditioned/            # Results for well-conditioned experiments
│   ├── prox_step_visualisation.png
│   └── variance_reduction_trajectory.png
│
├── notebooks/
│   └── experiment_final.ipynb       # Notebook for well-conditioned experiment
│
├── scripts/                        # Scripts used                
│   ├── experiment.py
│   ├── ill_conditioned_experiment.py
│   ├── prox_step_visualization.py
│   └── variance_reduction_trajectory.py
│
├── MiahK Individual Report.pdf      # Final Project Report
└── .gitignore
```
## 🚀 Usage & Replication
To replicate the convergence experiments from the report:
1. Clone this repository.
2. Install the required dependencies.
3. Run the main simulation notebook/script to generate the well-conditioned ($\kappa \approx 1.8$) and ill-conditioned ($\kappa \approx 5 \times 10^3$) mechanism plots.

## 📚 Core References
1. **Herbert Robbins and Sutton Monro (1951).** *A stochastic approximation method*. The annals of mathematical statistics, 400-407.
2. **Clément W. Royer (2023).** *Lecture notes on stochastic gradient methods*. LAMSADE, Université Paris-Dauphine.
3. **Guillaume Garrigos and Robert M. Gower (2023).** *Handbook of convergence theorems for (stochastic) gradient methods*.
4. **Léon Bottou, Frank E Curtis, and Jorge Nocedal (2018).** *Optimization methods for large-scale machine learning*. Siam Review, 60(2):223-311.
5. **Lin Xiao and Tong Zhang (2014).** *A proximal stochastic gradient method with progressive variance reduction*. SIAM Journal on Optimization, 24(4):2057-2075.
6. **Rie Johnson and Tong Zhang (2013).** *Accelerating stochastic gradient descent using predictive variance reduction*. Advances in Neural Information Processing Systems, 26.
7. **Nicolas Roux, Mark Schmidt, and Francis Bach (2012).** *A stochastic gradient method with an exponential convergence rate for finite training sets*. Advances in Neural Information Processing Systems, 25.
8. **Herbert Robbins and David Siegmund (1971).** *A convergence theorem for non negative almost supermartingales and some applications*. In Optimizing methods in statistics, 233-257. Elsevier.
9. **Eric Moulines and Francis Bach (2011).** *Non-asymptotic analysis of stochastic approximation algorithms for machine learning*. Advances in Neural Information Processing Systems, 24.
10. **Robert M Gower, Mark Schmidt, Francis Bach, and Peter Richtárik (2020).** *Variance-reduced methods for machine learning*. Proceedings of the IEEE, 108(11):1968-1983.
11. **Umut Şimşekli, Levent Sagun, and Mert Gürbüzbalaban (2019).** *A tail-index analysis of stochastic gradient noise in deep neural networks*. International Conference on Machine Learning, 5827-5837.
12. **Qianxiao Li, Cheng Tai, and Weinan E (2017).** *Stochastic modified equations and adaptive stochastic gradient algorithms*. International Conference on Machine Learning, 2101-2110.
13. **Stephan Mandt, Matthew D Hoffman, and David M Blei (2017).** *Stochastic gradient descent as approximate bayesian inference*. The Journal of Machine Learning Research, 18(1):4873-4907.
