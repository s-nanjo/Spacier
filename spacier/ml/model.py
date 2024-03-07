#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ml.model module
# ******************************************************************************

import numpy as np

import warnings
warnings.simplefilter("ignore")

__version__ = '0.0.3'


def sklearn_GP(X_train, y_train, X_pool):
    from sklearn.gaussian_process import GaussianProcessRegressor
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


def sklearn_GP_st(X_train, y_train, X_pool):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (WhiteKernel,
                                                  RBF,
                                                  ConstantKernel)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=ConstantKernel()*RBF()+WhiteKernel(),
            alpha=0,
            normalize_y=True
        )
    )
    model.fit(X_train, y_train)
    m, s = model.predict(X_pool, return_std=True)
    return m, s


def GPy_GP(X_train, y_train, X_pool):
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
    import torch
    import gpytorch

    y_m = y_train.values.reshape(-1,).mean()
    y_s = y_train.values.reshape(-1,).std()
    y_sc = (y_train.values.reshape(-1,) - y_m)/y_s

    train_x = torch.from_numpy(X_train.astype(np.float32)).clone()
    pool_x = torch.from_numpy(X_pool.astype(np.float32)).clone()
    train_y = torch.from_numpy(y_sc.astype(np.float32)).clone()

    class GP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
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
