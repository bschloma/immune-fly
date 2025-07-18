#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:49:14 2024

@author: brandon
"""

ys = df.y
counts, bins = np.histogram(ys / np.max(ys), bins=10)
plt.figure()
plt.plot(bins[:-1], counts, 'k-')

bins

counts / np.sum(counts)