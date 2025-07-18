#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:49:40 2024

@author: brandon
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

%matplotlib qt

"""plot style"""
linewidth = 4
mpl.rc('axes', linewidth=linewidth)
mpl.rc('font', family='Arial')
fontsize = 24


colors = {'no_inj': [0.8, 0.8, 0.8],
         'mock': [0.4, 0.4, 0.4],
         'e.coli': [0, 0.4, 0],
         'complete': [0, 0.8, 0],
         'cell2': [138 / 255, 43 / 255, 226 / 255]}

def style_axes(ax, fontsize=24):
    plt.minorticks_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    plt.tight_layout()
    
    return ax


def hill(x, A, n, KD, offset):
    return A * (x ** n) / (KD ** n + x ** n) + offset

def multiplicative_noise_model(r, gamma, sigma, Tmax, dt=0.01):
    """solve an sde using the Milstein method.
    dy = (r - gamma*y)dt + sigma * y *dWt"""
    
    t_arr = np.arange(0, Tmax, dt)
    y = np.zeros_like(t_arr)
    
    dWt = np.random.normal(scale=np.sqrt(dt), size=len(y))
    
    for i in range(1, len(y)):
        y[i] = y[i - 1] + dt * (r - gamma * y[i - 1]) + dWt[i] * sigma * y[i - 1] + 0.5 * sigma ** 2 * y[i - 1] * (dWt[i] ** 2 - dt)
        if y[i] < 0:
            y[i] = 0
    return y

"""example input-output function"""
A = 1
n = 12
KD = 0.5
offset = 0
x = np.linspace(0, 1, 1000)
y = hill(x, A, n, KD, offset)

plt.figure(figsize=(5.5, 5))
plt.plot(x, y, linewidth=8, color=colors['e.coli'])
plt.xlabel('\n[input microbial signal]', fontsize=fontsize)
plt.ylabel('output immune \ntranscription rate\n', fontsize=fontsize)
plt.ylim([-0.02, 1.05])
plt.xticks([])
plt.yticks([])
ax = style_axes(plt.gca())


plt.savefig(r'/media/brandon/Data1/Brandon/fly_immune/diptericin_paper/cartoons/example_input-output_function.pdf')

## plot a series of 1x2 plots for each of the 3 cases


"""case 1: different KDs"""

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10.5, 5))
A = 1
n = 4
offset = 0
x = np.linspace(0, 1, 1000)
linewidth = 4

#IO function
col = 0
ax = axes[col]
KD = 0.2
y = hill(x, A, n, KD, offset)
ax.plot(x, y, linewidth=linewidth, color=colors['cell2'], label='cell 1')
KD = 0.6
y = hill(x, A, n, KD, offset)
ax.plot(x, y, linewidth=linewidth, color=colors['e.coli'], label='cell 2')

x0 = 0.3
y = np.linspace(0, 1.2, 5)
ax.plot(x0 * np.ones_like(y), y, 'k--', linewidth=4, label='microbial \nload')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-0.02, 1.1])
ax.set_xlim([0, 1])
ax.set_xlabel('\n[input microbial signal]', fontsize=fontsize)
ax.set_ylabel('output immune \ntranscription rate\n', fontsize=fontsize)
ax = style_axes(ax)

ax.legend(fontsize=0.65*fontsize, loc='lower right')

# dynamics
col = 1
ax = axes[col]

r = 0.2
y = r * x
ax.plot(x, y, linewidth=linewidth, color=colors['e.coli'], label='cell 2')

r = 1
y = r * x
ax.plot(x, y, linewidth=linewidth, color=colors['cell2'], label='cell 1')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-0.02, 1.1])
ax.set_xlim([0, 1])
ax.set_xlabel('\ntime', fontsize=fontsize)
ax.set_ylabel('\noutput immune \ngene expression\n', fontsize=fontsize)
ax = style_axes(ax)


plt.savefig(r'/media/brandon/Data1/Brandon/fly_immune/diptericin_paper/cartoons/io_and_trace_different_ios.pdf')

"""case 2: different inputs"""

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10.5, 6))
A = 1
n = 4
offset = 0
x = np.linspace(0, 1, 1000)
linewidth = 4

#IO function
col = 0
ax = axes[col]
KD = 0.2
y = hill(x, A, n, KD, offset)
ax.plot(x, y, linewidth=linewidth, color='k', label='input-output \nfunction')

x0 = 0.2
y = np.linspace(0, 1.2, 5)
ax.plot(x0 * np.ones_like(y), y, '--', color=colors['e.coli'], linewidth=4, label='microbial load \nat cell 1')

x0 = 0.6
y = np.linspace(0, 1.2, 5)
ax.plot(x0 * np.ones_like(y), y, '--', color=colors['cell2'], linewidth=4, label='microbial load \nat cell 2')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-0.02, 1.1])
ax.set_xlim([0, 1])
ax.set_xlabel('\n[input microbial signal]', fontsize=fontsize)
ax.set_ylabel('output immune \ntranscription rate\n', fontsize=fontsize)
ax = style_axes(ax)

ax.legend(fontsize=0.65*fontsize, loc='upper left', bbox_to_anchor=(-0.5, 1.4), ncol=3, fancybox=False)

# dynamics
col = 1
ax = axes[col]

r = 0.2
y = r * x
ax.plot(x, y, linewidth=linewidth, color=colors['e.coli'], label='cell 2')

r = 1
y = r * x
ax.plot(x, y, linewidth=linewidth, color=colors['cell2'], label='cell 1')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-0.02, 1.1])
ax.set_xlim([0, 1])
ax.set_xlabel('\ntime', fontsize=fontsize)
ax.set_ylabel('\noutput immune \ngene expression\n', fontsize=fontsize)
ax = style_axes(ax)
ax.legend(fontsize=0.65*fontsize, loc='upper left')


plt.savefig(r'/media/brandon/Data1/Brandon/fly_immune/diptericin_paper/cartoons/io_and_trace_different_inputs.pdf')

"""case 3: different outputs"""

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
A = 1
n = 4
offset = 0
x = np.linspace(0, 1, 1000)
linewidth = 4

#IO function
col = 0
ax = axes[col]
KD = 0.2
y = hill(x, A, n, KD, offset)
ax.plot(x, y, linewidth=linewidth, color='k', label='input-output \nfunction')

x0 = 0.2
y = np.linspace(0, 1.2, 5)
ax.plot(x0 * np.ones_like(y), y, '--', color='k', linewidth=4, label='microbial load')

ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-0.02, 1.1])
ax.set_xlim([0, 1])
ax.set_xlabel('\n[input microbial signal]', fontsize=fontsize)
ax.set_ylabel('output immune \ntranscription rate\n', fontsize=fontsize)
ax = style_axes(ax)

ax.legend(fontsize=0.65*fontsize, loc='lower right')

# dynamics
col = 1
ax = axes[col]

r = 1
gamma = 1
sigma = 1
Tmax = 5
dt = 0.01
tarr = np.arange(0, Tmax, dt)
y = multiplicative_noise_model(r, gamma, sigma, Tmax, dt=dt)
ax.plot(tarr, y, linewidth=linewidth, color=colors['e.coli'], label='cell 1')

y = multiplicative_noise_model(r, gamma, sigma, Tmax, dt=dt)
ax.plot(tarr, y, linewidth=linewidth, color=colors['cell2'], label='cell 2')
ax.set_xticks([])
ax.set_yticks([])
#ax.set_ylim([-0.02, 1.1])
#ax.set_xlim([0, 1])
ax.set_xlabel('\ntime', fontsize=fontsize)
ax.set_ylabel('\noutput immune \ngene expression\n', fontsize=fontsize)
ax = style_axes(ax)
ax.legend(fontsize=0.65*fontsize, loc='upper left')

plt.savefig(r'/media/brandon/Data1/Brandon/fly_immune/diptericin_paper/cartoons/io_and_trace_different_outputs.pdf')

