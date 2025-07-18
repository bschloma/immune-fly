#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:07:53 2025

@author: brandon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

linewidth = 4
fontsize = 24
rc_params = {'font.family': 'Arial',
          'axes.linewidth': linewidth,
          'font.size': fontsize}
rcParams.update(rc_params)

x = np.linspace(0, 1, 100)

plt.figure()
"""rate"""
plt.subplot(121)
y1 = x
y2 = 2 * x
plt.plot(x, y1, linewidth=linewidth, color='m')
plt.plot(x, y2, linewidth=linewidth, color='g')
plt.xticks([])
plt.yticks([])
plt.xlim([0, 1])
plt.ylim([0, 2.1])
plt.xlabel('time', fontsize=fontsize)
plt.ylabel('single-cell DptA \nfluorescence intensity', fontsize=fontsize)
ax = style_axes(plt.gca())

"""delay"""
plt.subplot(122)
y1 = 2*x - 1
y2 = 2*x
plt.plot(x, y1, linewidth=linewidth, color='m')
plt.plot(x, y2, linewidth=linewidth, color='g')
plt.xticks([])
plt.yticks([])
plt.xlim([0, 1])
plt.ylim([0, 2.1])
plt.xlabel('time', fontsize=fontsize)
plt.ylabel('single-cell DptA \nfluorescence intensity', fontsize=fontsize)
ax = style_axes(plt.gca())
