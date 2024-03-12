#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ml.spacier module
# ******************************************************************************

import numpy as np
import pandas as pd

from scipy import stats
from . import model

import warnings
warnings.simplefilter("ignore")

__version__ = '0.0.3'


class Random():
    """
    A class that represents a random sampling strategy.

    Parameters:
    -----------
    df_X : pandas.DataFrame
        The feature matrix of the labeled data.
    df_pool_X : pandas.DataFrame
        The feature matrix of the unlabeled data pool.
    df : pandas.DataFrame
        The labeled data.
    df_pool : pandas.DataFrame
        The unlabeled data pool.

    Attributes:
    -----------
    index_pool : list
        A list of indices of the unlabeled data pool.
    df_X : pandas.DataFrame
        The feature matrix of the labeled data.
    df_pool_X : pandas.DataFrame
        The feature matrix of the unlabeled data pool.
    df : pandas.DataFrame
        The labeled data.
    df_pool : pandas.DataFrame
        The unlabeled data pool.

    Methods:
    -----------
    sample(query_n)
        Randomly samples a specified number of indices
        from the unlabeled data pool.

    """
    def __init__(self, df_X, df_pool_X, df, df_pool):
        index_pool = list(df_pool.index)
        print("Number of candidates : ", len(df_pool))

        self.index_pool = index_pool
        self.df_X = check(df_X)
        self.df_pool_X = check(df_pool_X)
        self.df = df
        self.df_pool = df_pool

    def sample(self, query_n):
        """
        Randomly samples a specified number of indices
        from the unlabeled data pool.

        Parameters:
        -----------
        query_n : int
            The number of indices to sample.

        Returns:
        -----------
        list
            A list of randomly sampled indices from the unlabeled data pool.

        """
        new_index = list(self.df_pool.sample(query_n).index)
        return new_index


class BO():
    """
    Bayesian Optimization class for performing optimization
    using Bayesian methods.

    Parameters:
    - df_X (DataFrame): Input features for training data.
    - df_pool_X (DataFrame): Input features for candidate data.
    - df (DataFrame): Training data.
    - df_pool (DataFrame): Candidate data.
    - model_name (str): Name of the model to be used for optimization.
    - target (list): List of target variables.
    - standardization (bool): Flag indicating whether to
    perform standardization on the data.
    """
    def __init__(
            self,
            df_X,
            df_pool_X,
            df, df_pool,
            model_name,
            target,
            standardization=False
    ):
        q = ""
        for _ in target:
            q += "(df['" + str(_) + "'] != -9999) &"

        index_train = list(df[eval(q[:-1])].index)
        index_pool = list(df_pool.index)
        print("Number of training data : ", len(index_train))
        print("Number of candidates : ", len(df_pool))

        self.index_train = index_train
        self.index_pool = index_pool
        self.df_X = check(df_X)
        self.df_pool_X = check(df_pool_X)
        self.df = df
        self.df_pool = df_pool
        self.model_name = model_name
        self.target = target

        mu = np.zeros((len(self.df_pool_X), len(self.target)))
        sigma = np.zeros((len(self.df_pool_X), len(self.target)))

        if standardization:
            df_tmp = self.df[self.target].iloc[self.index_train, :]
            self.df_sc = df_tmp.apply(stats.zscore, axis=0)
            for _ in range(len(self.target)):
                method_to_call = getattr(model, self.model_name)
                mu[:, _], sigma[:, _] = method_to_call(
                    self.df_X[self.index_train, :],
                    self.df_sc[[self.target[_]]],
                    self.df_pool_X
                )
        else:
            for _ in range(len(self.target)):
                method_to_call = getattr(model, self.model_name)
                mu[:, _], sigma[:, _] = method_to_call(
                    self.df_X[self.index_train, :],
                    self.df[[self.target[_]]].iloc[self.index_train],
                    self.df_pool_X
                )
        self.mu = mu
        self.sigma = sigma

    def uncertainty(self, query_n):
        """
        Returns the indices of the candidates with the highest uncertainty.

        Parameters:
        - query_n (int): Number of candidates to select.

        Returns:
        - new_index (list): List of indices of the selected candidates.
        """
        new_index = list(np.argpartition(-self.sigma[:, 0], query_n)[:query_n])
        return new_index

    def EI(self, query_n):
        """
        Returns the indices of the candidates with the highest
        Expected Improvement (EI).

        Parameters:
        - query_n (int): Number of candidates to select.

        Returns:
        - new_index (list): List of indices of the selected candidates.
        """
        ei = np.array([
            EI_integral(
                self.mu[_, 0],
                self.sigma[_, 0],
                self.df[self.target[0]].max()
            )
            for _ in self.index_pool
        ])
        new_index = list(np.argpartition(-ei, query_n)[:query_n])
        return new_index

    def UCB(self, query_n):
        """
        Returns the indices of the candidates with the highest
        Upper Confidence Bound (UCB).

        Parameters:
        - query_n (int): Number of candidates to select.

        Returns:
        - test_new_idx (list): List of indices of the selected candidates.
        """
        kappa = np.sqrt(np.log(len(self.index_train))/len(self.index_train))
        ucb = self.mu[:, 0] + kappa*self.sigma[:, 0]
        test_new_idx = list(np.argpartition(-ucb, query_n)[:query_n])
        return test_new_idx

    def PI(self, target_range, query_n):
        """
        Returns the indices of the candidates with the highest
        Probability of Improvement (PI).

        Parameters:
        - target_range (list): List of target ranges.
        - query_n (int): Number of candidates to select.

        Returns:
        - new_index (list): List of indices of the selected candidates.
        """
        pi = np.ones(len(self.df_pool_X))
        for _ in range(len(self.target)):
            pi *= PI_integral(self.mu[:, _], self.sigma[:, _], target_range[_])
        new_index = list(np.argpartition(-pi, query_n)[:query_n])
        return new_index

    def EHVI(self, query_n):
        """
        Returns the indices of the candidates with the highest
        Expected Hypervolume Improvement (EHVI).

        Parameters:
        - query_n (int): Number of candidates to select.

        Returns:
        - new_index (list): List of indices of the selected candidates.
        """
        PF = np.array(PF_max(self.df_sc.values[:, 0], self.df_sc.values[:, 1]))
        r = [-5, -5]
        ehvi = np.array([
            sum_EHVI(
                PF,
                r,
                np.array([self.mu[_, 0], self.mu[_, 1]]),
                np.array([self.sigma[_, 0], self.sigma[_, 1]]))[0]
            for _ in range(len(self.df_pool))
        ])
        new_index = list(np.argpartition(-ehvi, query_n)[:query_n])
        return new_index


def check(df_X):
    """
    Convert the input dataframe or numpy array to a numpy array.

    Parameters:
    df_X (pd.core.frame.DataFrame or np.ndarray):
    The input dataframe or numpy array.

    Returns:
    np.ndarray: The converted numpy array.
    """
    if isinstance(df_X, pd.core.frame.DataFrame):
        return df_X.values
    if isinstance(df_X, np.ndarray):
        return df_X


def PI_integral(mu, sigma, target_range):
    """
    Calculate the probability of a random variable following
    a normal distribution falling within a given target range.

    Parameters:
    mu (float): The mean of the normal distribution.
    sigma (float): The standard deviation of the normal distribution.
    target_range (tuple): A tuple representing the target range
    (lower bound, upper bound).

    Returns:
    float: The probability of the random variable falling within
    the target range.
    """
    return (
        stats.norm.cdf(target_range[1] - mu, scale=sigma) -
        stats.norm.cdf(target_range[0] - mu, scale=sigma)
    )


def EI_integral(mu, sigma, y_max):
    """
    Calculate the Expected Improvement (EI) integral.

    Parameters:
    - mu (float): The mean of the Gaussian distribution.
    - sigma (float): The standard deviation of the Gaussian distribution.
    - y_max (float): The maximum observed value.

    Returns:
    - ei (float): The value of the EI integral.
    """
    if sigma == 0:
        ei = 0
    else:
        t = (mu - y_max) / sigma
        ei = (
            (mu - y_max) * stats.norm.cdf(t) +
            sigma * stats.norm.pdf(t)
        )
    return ei


def PF_max(f1, f2):
    """
    Find the Pareto front for two objective functions.

    Parameters:
    - f1 (numpy.ndarray): Array representing
    the first objective function values.
    - f2 (numpy.ndarray): Array representing
    the second objective function values.

    Returns:
    - numpy.ndarray: Array representing the Pareto front,
    sorted by the second objective function values.
    """

    f1 = f1.reshape(-1, 1)
    f2 = f2.reshape(-1, 1)
    f = np.hstack((f1, f2))
    f_sorted_by_f2 = f[np.argsort(-f[:, 0])]

    pareto_front_ = [f_sorted_by_f2[0]]
    for pair in f_sorted_by_f2[1:]:
        if pair[1] > pareto_front_[-1][1]:
            pareto_front_ = np.vstack((pareto_front_, pair))
    return pareto_front_


def sum_EHVI(PF, r, mu, sigma):
    """
    Calculate the sum of Expected Hypervolume Improvement
    (EHVI) given the Pareto Front (PF), reference point (r),
    mean (mu), and standard deviation (sigma).

    Parameters:
    - PF (numpy.ndarray): The Pareto Front points.
    - r (list): The reference point.
    - mu (numpy.ndarray): The mean values.
    - sigma (numpy.ndarray): The standard deviation values.

    Returns:
    - EHVI (float): The sum of Expected Hypervolume Improvement.
    """
    S1 = np.array([r[0], np.inf])
    S1 = S1.reshape(1, -1)
    Send = np.array([np.inf, r[1]])
    Send = Send.reshape(1, -1)
    index = np.argsort(PF[:, 0])

    S = PF[index, :]
    S = np.concatenate((S1, S, Send), axis=0)
    # n = S.shape[0]
    y1 = S[:, 0]
    y2 = S[:, 1]
    y1 = y1.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    mu = mu.reshape(1, -1)
    sigma = sigma.reshape(1, -1)

    sum_total1 = 0
    sum_total2 = 0
    for i in range(1, len(S)):
        t = (y1[i] - mu[0][0]) / sigma[0][0]
        if 1 - stats.norm.cdf(t) == 0:
            sum_total1 += 0
        else:
            sum_total1 += (y1[i] - y1[i-1]) * (1 - stats.norm.cdf(t)) * \
                psi_calc(y2[i], y2[i], mu[0][1], sigma[0][1])

        sum_total2 += (psi_calc(y1[i-1], y1[i-1], mu[0][0], sigma[0][0]) -
                       psi_calc(y1[i-1], y1[i], mu[0][0], sigma[0][0])) * \
            psi_calc(y2[i], y2[i], mu[0][1], sigma[0][1])
    EHVI = sum_total1 + sum_total2
    return EHVI


def psi_calc(a, b, m, s):
    """
    Calculate the psi value using the given parameters.

    Parameters:
    a (float): The lower bound of the range.
    b (float): The upper bound of the range.
    m (float): The mean value.
    s (float): The standard deviation.

    Returns:
    float: The calculated psi value.
    """

    t = (b - m)/s
    return s * stats.norm.pdf(t) + (m - a) * (1 - stats.norm.cdf(t))
