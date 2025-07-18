#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:48:38 2024

@author: brandon
"""

## Simulations showing the log-normal-gamma distribution

mu_mock

sigma_mock

theta_fit

"""sweep k"""
mu = 6.25
sigma = 0.387
theta = 5433
k_arr = [0.1, 0.5, 1.5]
x = np.logspace(2, 5, 50)
prob_dens_arr = np.zeros((len(k_arr), len(x)))
for i in range(len(prob_dens_arr)):
    prob_dens_arr[i] = likelihood(x, mu, sigma, k_arr[i], theta, n_samples=1_000_000, n_bins=100)

plt.figure(figsize=(7,6))
reds = np.linspace(1, 0, len(prob_dens_arr))
greens = np.linspace(0, 1, len(prob_dens_arr))
blues = np.ones(len(prob_dens_arr))
for i in range(len(prob_dens_arr)):
    this_prob_dens = prob_dens_arr[i]
    plot_x = x[this_prob_dens > 5e-7]
    plot_prob = this_prob_dens[this_prob_dens > 5e-7]
    plt.plot(plot_x, plot_prob, linewidth=4, color=(reds[i], greens[i], blues[i]), label=r'$\alpha=$' + str(k_arr[i]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$x$', fontsize=fontsize)
plt.ylabel('$p(x)$', fontsize=fontsize)
plt.legend(fontsize=0.75 * fontsize)
ax = style_axes(plt.gca())

plt.savefig(r'/home/brandon/Documents/Code/diptericin-paper/figures/misc/example_log-normal-gamma_dists_sweep_alpha.pdf')

"""sweep theta"""
mu = 6.25
sigma = 0.387
theta_arr = [50, 500, 5000]
k = 0.5
x = np.logspace(2, 5, 50)
prob_dens_arr = np.zeros((len(theta_arr), len(x)))
for i in range(len(prob_dens_arr)):
    prob_dens_arr[i] = likelihood(x, mu, sigma, k, theta_arr[i], n_samples=1_000_000, n_bins=100)

plt.figure(figsize=(7,6))
reds = np.linspace(1, 0, len(prob_dens_arr))
greens = np.linspace(0, 1, len(prob_dens_arr))
blues = np.ones(len(prob_dens_arr))
for i in range(len(prob_dens_arr)):
    this_prob_dens = prob_dens_arr[i]
    plot_x = x[this_prob_dens > 5e-7]
    plot_prob = this_prob_dens[this_prob_dens > 5e-7]
    plt.plot(plot_x, plot_prob, linewidth=4, color=(reds[i], greens[i], blues[i]), label=r'$\beta=1/$' + str(theta_arr[i]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$x$', fontsize=fontsize)
plt.ylabel('$p(x)$', fontsize=fontsize)
plt.legend(fontsize=0.75 * fontsize)
ax = style_axes(plt.gca())

plt.savefig(r'/home/brandon/Documents/Code/diptericin-paper/figures/misc/example_log-normal-gamma_dists_sweep_beta.pdf')