import matplotlib.pyplot as plt
import numpy as np
from simulations import four_component, initialize_params


params = initialize_params()

R, N, A, B = four_component(params)

tvec = np.arange(0, params.Tmax, params.dt)
plt.figure()
plt.plot(tvec, R)
plt.plot(tvec, N)
plt.plot(tvec, A)
plt.plot(tvec, B)
