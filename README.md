# Vectorized Regressions

Linear and kernel regression, with and without regularization, built from scratch in NumPy. These notebooks follow **6.S088 (2023)**, taught by Prof. Adit Radhakrishnan at MIT.

## Notebooks

- **`lin_reg.ipynb`**: Linear regression via the Moore–Penrose pseudoinverse, evaluated with MSE and R², and compared against scikit-learn's `LinearRegression`.
- **`kernel_reg.ipynb`**: Kernel regression with the Laplace kernel:
  - a naive loop-based implementation and a vectorized one (pairwise Euclidean distances via $\|x - y\|^2 = \|x\|^2 - 2x^\top y + \|y\|^2$)
  - how the bandwidth $L$ affects the fit
  - regularized (ridge) kernel regression, $\alpha = (K + \lambda I)^{-1} y$, on noisy data

## Requirements

```bash
pip install numpy matplotlib scikit-learn jupyter
```

## Acknowledgments

Many thanks to Prof. Adit Radhakrishnan and the 6.S088 course materials, which this work follows.
