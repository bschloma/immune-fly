import numpy as np
import pandas as pd


def four_component(p):
    tvec = np.arange(0, p.Tmax, p.dt)
    num_time_points = len(tvec)
    receptor = np.zeros((num_time_points, 1))
    nfkb = np.zeros((num_time_points, 1))
    amp = np.zeros((num_time_points, 1))
    bacteria = np.zeros((num_time_points, 1))

    on_time = np.random.exponential(scale=1/p.encounter_rate)
    on_id = np.int32(np.floor(on_time / p.dt))
    bacteria[on_id] = 1
    remaining_time_points = np.where(tvec > on_time)

    for s in range(len(remaining_time_points)):

        receptor[s], nfkb[s], amp[s], bacteria[s] = update_four_component(p, receptor[s-1], nfkb[s-1], amp[s-1], bacteria[s-1])

    return receptor, nfkb, amp, bacteria


def update_four_component(p, r_, n_, a_, b_):
    r_out = np.max((r_ + p.dt * (p.production_rate_R * n_ - p.decay_rate_R * r_), 0))
    n_out = np.max((n_ + p.dt * (p.production_rate_N * r_ * b_ - p.decay_rate_N * n_), 0))
    a_out = np.max((a_ + p.dt * (p.production_rate_A * n_ - p.decay_rate_A * a_), 0))
    b_out = np.max((b_ + p.dt * (p.production_rate_B * b_ - p.decay_rate_B * a_ * b_), 0))

    return r_out, n_out, a_out, b_out


def initialize_params():
    p = pd.DataFrame

    # time in hours
    p.dt = 0.1
    p.Tmax = 10.0
    p.production_rate_R = 100.0
    p.production_rate_N = 100.0
    p.production_rate_A = 100.0
    p.production_rate_B = 0.1
    p.decay_rate_R = 10.0
    p.decay_rate_N = 10.0
    p.decay_rate_A = 10.0
    p.decay_rate_B = 1.0
    p.encounter_rate = 100.0

    return p

