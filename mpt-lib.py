#!/usr/bin/python3


from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Abbreviations and definitions:
# er: expected return
# mv: minimum variance
# port_p: the risky portfolio consisting of two risky assets
# port_c: the complete portfolio consisting of a risky portfolio (i.e., port_p) and a risk-free asset.
# rf: return of a risk-free asset
# sr: Sharpe
#
# Notes:
# 1. Correlation and covariance are related but different concepts. 𝜌(rho) is the greek alphabet
#    used to represent correlation.
# 2. The definitions of port_p and port_c are the same as those on Prof. Benz's slides.


def get_utility_score(expected_returns: List[float],
                      risk_aversion_coefficients: List[float],
                      standard_deviations_aka_risks: List[float]) -> List[float]:

    assert len(expected_returns) == len(risk_aversion_coefficients)

    σ_squared = []
    for i in range(len(standard_deviations_aka_risks)):
        σ_squared.append(standard_deviations_aka_risks[i] ** 2)

    utility_scores = [0] * len(expected_returns)

    for i in range(len(utility_scores)):
        utility_scores[i] = expected_returns[i] - 0.5 * risk_aversion_coefficients[i] * σ_squared[i]
      #  print(i, utility_scores[i], expected_returns[i], risk_aversion_coefficients[i], σ_squared[i])
    return utility_scores


def get_portfolio_sigma(weights_based_on_market_price: List[float],
                        sigmas: List[float], rho: float) -> float:

    assert len(weights_based_on_market_price) == 2
    assert len(sigmas) == 2
    assert sum(weights_based_on_market_price) == 1

    σs = sigmas
    w = weights_based_on_market_price

    portfolio_var = w[0] ** 2 * σs[0] ** 2 + w[1] ** 2 * σs[1] ** 2 + 2 * w[0] * w[1] * σs[0] * σs[1] * rho

   # print(portfolio_var, w[0] ** 2 * σs[0] ** 2, w[1] ** 2 * σs[1] ** 2)
    portfolio_σ = portfolio_var ** 0.5
    return portfolio_σ

# cal: capital allocation line
def get_return_and_risk_of_portfolio_on_cal(
                                    weight_risky: float,
                                    weight_risk_free: float,
                                    rf: float,
                                    risky_portfolio_expected_return: float,
                                    risky_portfolio_sigma: float,
                                    variances_aka_standard_deviations_squared = None):

    assert 1 - (weight_risky + weight_risk_free) < 10 ** -10
    assert rf < 0.1 and risky_portfolio_expected_return < 0.5
    assert rf < risky_portfolio_expected_return
    assert variances_aka_standard_deviations_squared == None or risky_portfolio_sigma == None
    assert variances_aka_standard_deviations_squared != None or risky_portfolio_sigma != None

    σ_squared = variances_aka_standard_deviations_squared
    σ = risky_portfolio_sigma
    if σ == None:
        σ = σ_squared ** 0.5

    portfolio_expected_return = (1 - weight_risky) * rf
    portfolio_expected_return += weight_risky * risky_portfolio_expected_return


    portfolio_std_dev_aka_risk = weight_risky * σ

    return portfolio_expected_return, portfolio_std_dev_aka_risk


def plot_optimal_complete_portfolio(expected_returns: List[float], sigmas: List[float], rho: float,
                                    rf: float, risk_aversion_coefficient: float):

    assert len(expected_returns) == 2
    assert len(sigmas) == 2
    assert rf < 0.1

    A = risk_aversion_coefficient
    granularity = 10_000
    # If you just want to plot the graph, a very low granularity (e.g., 50) will suffice;
    # If you want the numbers to be accurate enough, you need to specify a much higher granularity.
    port_p_σs = []
    port_p_sr = -1
    port_p_returns = []
    port_p_w1 = 0
    port_p_σ = 0
    port_p_return = 0

    port_mv_w1 = 0
    port_mv_return = 0
    port_mv_σ = 6666

    for i in range(granularity + 1):
        w1 = i / granularity
        w2 = 1 - w1
        σ = get_portfolio_sigma(weights_based_on_market_price = [w1, w2], sigmas = sigmas, rho = rho)
        port_p_σs.append(σ)
        port_p_returns.append(w1 * expected_returns[0] + w2 * expected_returns[1])
        if σ != 0:
            sr = (port_p_returns[i] - rf) / σ
        else: # if two assets are perfectly negatively correlated, var and σ will become zero.
            sr = 2147483647

        if sr > port_p_sr:
            # Here we find the tangent portfolio
            port_p_sr = sr
            port_p_w1 = w1
            port_p_σ = σ
            port_p_return = port_p_returns[i]
        if σ < port_mv_σ:
            # Here we find the minimum variance portfolio
            port_mv_σ = σ
            port_mv_return = port_p_returns[i]
            port_mv_w1 = w1

    port_c_returns = []
    port_c_σs = []
    for i in range(granularity + 1):

        port_c_return, port_c_σ = get_return_and_risk_of_portfolio_on_cal(
                 weight_risky = (i / granularity),
                 weight_risk_free = 1 - (i / granularity),
                 rf = rf,
                 risky_portfolio_expected_return = port_p_return,
                 risky_portfolio_sigma = port_p_σ)
        port_c_returns.append(port_c_return)
        port_c_σs.append(port_c_σ)

    utils = get_utility_score(expected_returns=port_c_returns,
                              risk_aversion_coefficients=[A] * len(port_c_returns),
                              standard_deviations_aka_risks=port_c_σs)
    max_util = max(utils)
    returns_from_max_util = []
    for i in range(granularity + 1):
        returns_from_max_util.append(max_util + 0.5 * A * port_c_σs[i] ** 2)

    x = np.array(port_c_σs)
    f = np.array(port_c_returns)
    g = np.array(returns_from_max_util)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

    y_upper = max(expected_returns[0], expected_returns[1]) * 2

    plt.figure(figsize=(11, 11))
    plt.rcParams.update({'font.size': 13})
    plt.xlim(0, max(port_p_σs))
    plt.ylim(0, y_upper)
    plt.title('Optimal Complete Portfolio')
    plt.xlabel("Portfolio's Risk (measured in Standard Deviation)")
    plt.ylabel("Portfolio's Expected Return")

    # ===== Plot three curves =====
    plt.plot([0, max(port_p_σs)], [rf, max(port_p_σs) * (port_p_sr) + rf], color='b',
             label=f'Capital Allocation Line (Slope/Sharpe ratio: {port_p_sr:.3f})')
    plt.plot(port_p_σs, port_p_returns, color='r',
             label=f'Opportunity Set of Two Risky Assets (Corr: {rho})')
    plt.plot(port_c_σs, returns_from_max_util, color='g',
             label=f'Indifference Curve (Utility Score: {max_util:.3f})')

    # ===== Plot three points (i.e., portfolios) =====
    plt.plot(port_p_σ, port_p_return, 'bo',
             label='Optimal portfolio of two risky assets\n'
                   f'(asset1: {port_p_w1:.3f}, asset2: {1 - port_p_w1:.3f}, '
                   f'risk: {port_p_σ:.3f}, return: {port_p_return:.3f})')
    plt.plot(port_mv_σ, port_mv_return, 'ro',
             label='Minimum-Variance Portfolio\n'
                   f'(asset1: {port_mv_w1:.3f}, asset2: {1 - port_mv_w1:.3f}, '
                   f'risk: {port_mv_σ:.3f}, return: {port_mv_return:.3f})')
    plt.plot(x[idx], f[idx], 'go',
             label='Optimal complete portfolio\n'
                   f'(risky: {x[idx][0] / port_p_σ:.3f}, '
                   f'risk-free: {1 - x[idx][0] / port_p_σ:.3f}, '
                   f'risk: {x[idx][0]:.3f}, return: {f[idx][0]:.3f})')

    plt.legend(loc='upper left', framealpha=0.5)
    plt.show()


def plot_opportunity_set(expected_returns: List[float], sigmas: List[float], rho: float):

    assert len(expected_returns) == 2 and len(sigmas) == 2
    assert rho <= 1 and rho >= -1

    granularity = 200
    σ, σ_neg, σ_pos = [], [], []
    portfolio_return = []
    for i in range(0, granularity):
        weights = [i / granularity, 1 - i / granularity]
        σ.append(get_portfolio_sigma(weights_based_on_market_price=weights, sigmas=sigmas, rho=rho))
        σ_neg.append(get_portfolio_sigma(weights_based_on_market_price=weights, sigmas=sigmas, rho=-1))
        σ_pos.append(get_portfolio_sigma(weights_based_on_market_price=weights, sigmas=sigmas, rho=1))
        portfolio_return.append(weights[0] * expected_returns[0] + weights[1] * expected_returns[1])

    cutoff = σ.index(min(σ))

    plt.figure(figsize = (9, 9))
    plt.title('Opportunity Set')
    plt.xlabel("Portfolio's Standard Deviation (σ)")
    plt.ylabel("Portfolio's Expected Return")
    plt.grid() # it turns out that this graph looks better with a grid...
    plt.plot(σ[cutoff:], portfolio_return[cutoff:], color='blue',
             label=f'Efficient Frontier (rho == {rho})')
    plt.plot(σ[0:cutoff], portfolio_return[0:cutoff], color='magenta',
             label=f'INefficient Frontier (rho == {rho})')
    plt.plot(σ_neg, portfolio_return, color='red', label='Opportunity Set (if rho == -1)')
    plt.plot(σ_pos, portfolio_return, color='green', label='Opportunity Set (if rho == 1)')
    plt.legend(loc = 'best', framealpha = 0.5)
    plt.show()


def __plotting_cal_and_indifference_curve(
        upper_bound,
        portfolio_std_dev_aka_risks,
        portfolio_expected_returns,
        returns_for_highest_utility,
        returns_for_highest_utility_50percent,
        returns_for_highest_utility_150percent,
        max_util,
        A,
        utility_scores,
        util_50percent,
        util_150percent):

    # Definitions of c is the same as the lecture slides using by Prof. Benz
    port_c_σ = portfolio_std_dev_aka_risks
    port_c_ret = portfolio_expected_returns
    fig = plt.figure(figsize = (4 * 4, 4 * 3))

    fig.add_subplot(2, 2, 1)
    plt.title('Capital Allocation Line and Indifference Curve')
    plt.xlabel('Portfolio\'s Standard Deviation (aka Risk)')
    plt.ylabel('Portfolio\'s Return')
    plt.xlim(0, max(port_c_σ))
    plt.ylim(0, max(port_c_σ))
    plt.grid()

    plt.plot(port_c_σ, returns_for_highest_utility, color = 'b', label = 'IC (Utility Score: {:.4f})'.format(max_util))
    plt.plot(port_c_σ, port_c_ret, color = 'r',
             label = 'CAL (risky + risk-free assets)\n (Sharpe Ratio (i.e. slope): {:.4f})'.format((port_c_ret[int(upper_bound / 2)] - port_c_ret[0]) / port_c_σ[int(upper_bound / 2)]))

    plt.plot(port_c_σ[int(upper_bound / 2)], port_c_ret[int(upper_bound / 2)],'ro',
            label = '100% to risky assets\nRisk Aversion Coefficient (aka A): {}'.format(A))

    x = np.array(port_c_σ)
    f = np.array(port_c_ret)
    g = np.array(returns_for_highest_utility)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    plt.plot(x[idx], f[idx], 'bo', label = 'Optimal Portfolio\n(risky: {:.4f} + risk-free: {:.4f})'.format(port_c_σ.index(x[idx][0])*2/upper_bound, 1 - port_c_σ.index(x[idx][0])*2/upper_bound))
    plt.annotate('(Risk: {:.4f}, Return: {:.4f})'.format(x[idx][0], f[idx][0]),
                 xy=(x[idx][0], f[idx][0]), xytext=(x[idx][0] * 1.3, f[idx][0] * 0.7),
                 arrowprops=dict(facecolor='black', arrowstyle ='-|>'))
    plt.legend(loc = 'upper left', framealpha = 0.5)

    fig.add_subplot(2, 2, 2)
    plt.title('Utility Function')
    plt.xlabel('Allocation in Risky Assets (%)')
    plt.ylabel('Utility')
    plt.grid()
    upper_bound_x = [x / (upper_bound/100) for x in range(upper_bound)]
    plt.plot((upper_bound_x), utility_scores, color = 'r', linewidth = 1)

    fig.add_subplot(2, 2, 3)
    plt.title('CAL and IC (Wrong Example)')
    plt.xlabel('Portfolio Standard Deviation (aka Risk)')
    plt.ylabel('Portfolio Return')
    plt.grid()
#    plt.plot(portfolio_std_dev_aka_risks, returns_for_highest_utility, color = 'r', linewidth = 1)
    plt.plot(port_c_σ, port_c_ret, color = 'r', linewidth = 1, label = 'CAL')
    plt.plot(port_c_σ,
             returns_for_highest_utility_50percent,
             color = 'y', linewidth = 1, label = 'IC (Utility Score == {})'.format(round(util_50percent, 3)))
    plt.plot(port_c_σ,
             returns_for_highest_utility,
             color = 'b', linewidth = 1, label = 'IC (Utility Score == {})'.format(round(max_util, 3)))
    plt.plot(port_c_σ,
             returns_for_highest_utility_150percent,
             color = 'g', linewidth = 1, label = 'IC (Utility Score == {})'.format(round(util_150percent, 3)))

    x = np.array(port_c_σ)
    f = np.array(port_c_ret)
    g = np.array(returns_for_highest_utility_50percent)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    plt.plot(x[idx], f[idx], 'yo')

    x = np.array(port_c_σ)
    f = np.array(port_c_ret)
    g = np.array(returns_for_highest_utility)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    plt.plot(x[idx], f[idx], 'bo')

    plt.annotate("Blue and yellow points have the same\nsharpe ratio but different utility scores!",
                 xy=(x[idx][0], f[idx][0]),
                 xytext=(x[idx][0] * 1.3, f[idx][0] * 0.7),
                 arrowprops=dict(facecolor='black', arrowstyle ='-|>'))
    plt.annotate(' No allocation can achieve\nthis utility score!',
                 xy=(port_c_σ[int(upper_bound / 2)], returns_for_highest_utility_150percent[int(upper_bound / 2)]),
                 xytext=(port_c_σ[int(upper_bound / 2)] * 0.6, returns_for_highest_utility_150percent[int(upper_bound / 2)] * 1.3),
                 arrowprops=dict(facecolor='black', arrowstyle ='-|>'))
    plt.legend(loc = 'best', framealpha = 0.5)
    plt.show()


def plt_utility_and_indifference_curves(rf: float,
                        risky_portfolio_expected_return: float,
                        risky_portfolio_sigma: float,
                        risk_aversion_coefficient: float):

    A = risk_aversion_coefficient

    portfolio_expected_returns = []
    portfolio_std_dev_aka_risks = []
    returns_for_highest_utility = []
    returns_for_highest_utility_50percent = []
    returns_for_highest_utility_150percent = []

    upper_bound = 200000

    for i in range(upper_bound):

        portfolio_expected_return, portfolio_σ_aka_risk = get_return_and_risk_of_portfolio_on_cal(
                 weight_risky = (i / (upper_bound / 2)),
                 weight_risk_free = 1 - (i / (upper_bound / 2)),
                 rf = rf,
                 risky_portfolio_expected_return = risky_portfolio_expected_return,
                 risky_portfolio_sigma = risky_portfolio_sigma)
        portfolio_expected_returns.append(portfolio_expected_return)
        portfolio_std_dev_aka_risks.append(portfolio_σ_aka_risk)

    utility_scores = get_utility_score(
           expected_returns = portfolio_expected_returns,
           risk_aversion_coefficients = [A] * len(portfolio_expected_returns),
           standard_deviations_aka_risks = portfolio_std_dev_aka_risks)


    max_util = max(utility_scores)
    util_50percent = max_util * 0.5
    util_150percent = max_util * 1.5

    for i in range(upper_bound):
        returns_for_highest_utility.append(max_util + 0.5 * A * portfolio_std_dev_aka_risks[i] ** 2)
        returns_for_highest_utility_50percent.append(util_50percent + 0.5 * A * portfolio_std_dev_aka_risks[i] ** 2)
        returns_for_highest_utility_150percent.append(util_150percent + 0.5 * A * portfolio_std_dev_aka_risks[i] ** 2)



    optimal_allocation_in_risk_assets = (risky_portfolio_expected_return - rf) / (A * risky_portfolio_sigma ** 2)

    __plotting_cal_and_indifference_curve(
            upper_bound = upper_bound,
            portfolio_std_dev_aka_risks = portfolio_std_dev_aka_risks,
            portfolio_expected_returns = portfolio_expected_returns,
            returns_for_highest_utility = returns_for_highest_utility,
            returns_for_highest_utility_50percent = returns_for_highest_utility_50percent,
            returns_for_highest_utility_150percent = returns_for_highest_utility_150percent,
            max_util = max_util,
            A = A,
            utility_scores = utility_scores,
            util_50percent = util_50percent,
            util_150percent = util_150percent)

    return utility_scores, optimal_allocation_in_risk_assets
