#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ml.model module
# ******************************************************************************

import numpy as np

import warnings
warnings.simplefilter("ignore")

__version__ = '0.0.5'

def Mymodel(X_train, y_train, X_pool):
    """
    Perform User-defined model.
    
    Example: Bayesian inference of linear regression.
    
    - Model: y = Xw
    - Prior: p(w) = N(0, ∞)
    - Likelihood: p(y|X, w) = N(Xw, I)
      -> Posterior: p(w|y, X) = N((X^TX)^{-1}X^Ty, (X^TX)^{-1})
      -> Prediction distribution: p(y_new|X_new) = N(X_new*(X^TX)^{-1}X^Ty, X_new*(X^TX)^{-1}*X_new + I)  
    
    Parameters:
    - X_train: array-like, shape (n_samples, n_features)
        The training input samples.
    - y_train: array-like, shape (n_samples,)
        The target values.
    - X_pool: array-like, shape (n_samples, n_features)
        The input samples for which predictions are needed.
    Returns:
    - m: array-like, shape (n_samples,)
        The mean of the predicted target values.
    - s: array-like, shape (n_samples,)
        The standard deviation of the predicted target values.
    """
    sigma2 = 1
    XTX_inv = np.linalg.inv(X_train.T @ X_train)
    beta_hat = XTX_inv @ X_train.T @ y_train
    posterior_cov = sigma2 * XTX_inv
    m = X_pool @ beta_hat
    v = np.array([X_pool[i] @ posterior_cov @ X_pool[i].T for i in range(X_pool.shape[0])]) + sigma2
    return np.squeeze(m), np.squeeze(np.sqrt(v))

def sklearn_GP(X_train, y_train, X_pool):
    """
    Perform Gaussian Process regression using scikit-learn.

    Parameters:
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values for training.
    - X_pool (array-like): Input samples for which predictions are required.

    Returns:
    - m (array-like): Predicted mean values for X_pool.
    - s (array-like): Predicted standard deviation values for X_pool.
    """

    from sklearn.gaussian_process import GaussianProcessRegressor
    # Rest of the function code goes here
    from sklearn.gaussian_process.kernels import (WhiteKernel,
                                                  RBF,
                                                  ConstantKernel)

    model = GaussianProcessRegressor(
        kernel=(ConstantKernel() * RBF() + WhiteKernel()),
        alpha=0,
        normalize_y=True
    )
    model.fit(X_train, y_train)
    m, s = model.predict(X_pool, return_std=True)
    return m, s

def GPy_GP(X_train, y_train, X_pool):
    """
    Perform Gaussian Process regression using GPy library.

    Args:
        X_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training target data.
        X_pool (numpy.ndarray): Input data for prediction.

    Returns:
        tuple: A tuple containing the mean and standard deviation
        of the predicted values.
    """

    import GPy
    model = GPy.models.GPRegression(
        X_train,
        y_train.values,
        GPy.kern.RBF(X_train.shape[1])
    )
    model.optimize()
    m, s = model.predict(X_pool)
    return np.squeeze(m), np.squeeze(np.sqrt(s))


def gpytorch_GP(X_train, y_train, X_pool):
    """
    Perform Gaussian Process regression using the GPyTorch library.

    Args:
        X_train (numpy.ndarray): The training input data.
        y_train (numpy.ndarray): The training target data.
        X_pool (numpy.ndarray): The input data for which
        predictions are to be made.

    Returns:
        tuple: A tuple containing the mean and standard
        deviation of the predicted values.
    """

    import torch
    import gpytorch

    y_m = y_train.values.reshape(-1,).mean()
    y_s = y_train.values.reshape(-1,).std()
    y_sc = (y_train.values.reshape(-1,) - y_m)/y_s

    train_x = torch.from_numpy(X_train.astype(np.float32)).clone()
    pool_x = torch.from_numpy(X_pool.astype(np.float32)).clone()
    train_y = torch.from_numpy(y_sc.astype(np.float32)).clone()

    class GP(gpytorch.models.ExactGP):
        """
        Gaussian Process model.

        Args:
            train_x (torch.Tensor): The training input data.
            train_y (torch.Tensor): The training target data.
            likelihood (gpytorch.likelihoods.Likelihood):
            The likelihood function.

        Attributes:
            mean_module (gpytorch.means.Mean):
            The mean module for the GP model.

            covar_module (gpytorch.kernels.Kernel):
            The covariance module for the GP model.
        """

        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            """
            Forward pass of the GP model.

            Args:
                x (torch.Tensor): The input data.

            Returns:
                gpytorch.distributions.MultivariateNormal:
                The predicted distribution.
            """
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(train_x, train_y, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 50
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        test_y_pred = likelihood(model(pool_x))

    m = np.array(test_y_pred.mean)*y_s+y_m
    s = np.array(test_y_pred.stddev)*y_s
    return m, s
