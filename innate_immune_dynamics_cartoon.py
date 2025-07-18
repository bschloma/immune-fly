#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:46:40 2024

@author: brandon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


"""make cartoon plots of single-cell dynamics in innate immunity"""

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    
    return ax



t = np.linspace(0, 10, 1000)
color=(0, 150 / 255, 0)
linewidth=4
fontsize=24

plt.figure()

# oscillations
period = 2
x = np.sin(2 * np.pi * t / period) + 1
plt.subplot(311)
plt.plot(t, x, linewidth=linewidth, color=color)
plt.ylabel('nuclear TF \nconcentration', fontsize=fontsize)
plt.xlabel('time', fontsize=fontsize)
plt.title('oscillations (NF-$\kappa$B)', fontsize=fontsize, weight='normal')
plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax = style_axes(ax)

# amplification
lam = 0.5
x = np.exp(lam * t)
plt.subplot(312)
plt.plot(t, x, linewidth=linewidth, color=color)
plt.ylabel('cytokine \nconcentration', fontsize=fontsize)
plt.xlabel('time', fontsize=fontsize)
plt.title('amplification (TNF $\alpha$)', fontsize=fontsize, weight='normal')
plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax = style_axes(ax)

# stochasticity
x = np.zeros_like(t)
g = np.random.normal(size=len(x))
r = 1
gamma = 1
sigma = 3.0
dt = 0.01
for i in range(1, len(x)):
    x[i] = x[i - 1] + dt * (r - gamma * x[i - 1]) + np.sqrt(dt) * sigma * x[i - 1] * g[i]
    if x[i] < 0:
        x[i] = 0
        
plt.subplot(313)
plt.plot(t, x, linewidth=linewidth, color=color)
plt.ylabel('transcription \nrate', fontsize=fontsize)
plt.title('stochasticity (IFN $\beta$)', fontsize=fontsize, weight='normal')
plt.xticks([])
plt.yticks([])
plt.xlabel('time', fontsize=fontsize)
ax = plt.gca()
ax = style_axes(ax)

plt.tight_layout(h_pad=-.5)