#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:55:48 2025

@author: brandon
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def f(t, y, alpha, KD, tau, gamma, delta, r, K, mu):
    A, B = y
    dAdt = alpha * (B / (KD + B)) * (t >= tau) - gamma * A * B - delta * A
    dBdt = r * B * (1 - B / K) - mu * A * B
    return dAdt, dBdt

alpha = 10
KD = 1
tau = 1 
gamma = 0.1 
delta = 0.02
r = 0.5
K = 1000
mu = 0.3

args = alpha, KD, tau, gamma, delta, r, K, mu
y0 = np.array([0, 0.01 * K])
t_span = (0, 24)
t_eval = np.linspace(np.min(t_span), np.max(t_span), 1000)
sol = solve_ivp(f, y0=y0, t_span=t_span, t_eval=t_eval, args=args)

plt.figure()
plt.plot(sol.t, sol.y[0], color='g', label='A')
plt.ylabel('A', color='g')
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(sol.t, sol.y[1], color='m', label='B')
ax2.set_ylabel('B', color='m')
plt.xlabel('time')
plt.tight_layout()


"""loop over tau"""
tau_arr = np.linspace(0, 5, 100)
t_span = (0, 24)
A_arr = np.zeros_like(tau_arr)
B_arr = np.zeros_like(tau_arr)
for i in range(len(tau_arr)):
    args = alpha, KD, tau_arr[i], gamma, delta, r, K, mu
    sol = solve_ivp(f, y0=y0, t_span=t_span, t_eval=t_eval, args=args)
    A_arr[i] = sol.y[0, -1]
    B_arr[i] = sol.y[1, -1]

plt.figure()
plt.plot(tau_arr, B_arr, 'm-', linewidth=2)
plt.xlabel('tau')
plt.ylabel('final B')
plt.tight_layout()