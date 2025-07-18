#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:47:04 2024

@author: brandon
"""

bins = np.array([0.02731848, 0.12458663, 0.22185478, 0.31912293, 0.41639109,

       0.51365924, 0.61092739, 0.70819554, 0.8054637 , 0.90273185,

       1.        ])

​

true_fracs = np.array([0.09863946, 0.04846939, 0.07227891, 0.125     , 0.14115646,

       0.13520408, 0.12414966, 0.11989796, 0.10204082, 0.03316327])

​

# experiment 1

df['ap'] = df['y'] / df['y'].max()

df = bin_aps(df, bins, ap_col='ap')

ys = df[df.t == 0].y

ys = ys / np.max(ys)

​

counts, _ = np.histogram(ys, bins)

fracs = counts / np.sum(counts)

​

​

"""experiment 1"""

sampled_rates = []

sampled_ys = []

n_cells = len(df.particle.unique())

t_fit_max = 40

t_fit_min = 10

for i in range(len(bins) - 1):

    sub_df = df[df.binned_ap == i]

    counter = 0

    while counter < true_fracs[i] * n_cells:

        particle = np.random.choice(sub_df.particle.unique())

        sub_sub_df = sub_df[sub_df.particle == particle]

        sub_sub_df = sub_sub_df.sort_values(by='t')

        if any(sub_sub_df.groupby(by='t').size().values > 1):

            continue

        t = sub_sub_df.t

        dpt = sub_sub_df.mean_dpt

​

        dpt = dpt[t <= t_fit_max]

        t = t[t <= t_fit_max]

​

        dpt = dpt[t > t_fit_min]

        t = t[t > t_fit_min]

​

        if len(t) > 4:

            slope, intercept, r, p, se = linregress(minutes_per_time_point * t, dpt)

            sampled_rates.append(slope)

            sampled_ys.append(sub_sub_df.y.mean())

            counter += 1

​

    

plt.figure()

counts, _ = np.histogram(sampled_ys / np.max(sampled_ys), bins)

plt.plot(bins[:-1], counts / np.sum(counts), 'k-', linewidth=4)

plt.plot(bins[:-1], true_fracs, 'b-', linewidth=4)

[<matplotlib.lines.Line2D at 0x7f6fc4e096d0>]

