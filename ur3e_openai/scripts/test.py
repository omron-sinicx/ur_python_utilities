import glob
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.style.use('seaborn')


# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

import scipy.stats as stats


def force_signal():
    x = np.linspace(0, 1, 200)
    z = -1/(1 + np.exp(-x*15+5))

    plt.plot(x, z)
    plt.xlabel("Normalized Contact Force")
    plt.ylabel("Reward")
    plt.grid(                                                                                                                                                                                                               )

    plt.show()

def distance_signal():
    x = np.linspace(0, 1., 200)
    z = - np.tanh(5.0 * x)
    # z = -l1l2(x)

    plt.plot(x, z)
    plt.xlabel("Normalized Distance to target")
    plt.ylabel("Reward")
    plt.grid(                                                                                                                                                                                                               )

    plt.show()

def l1l2(dist):
    l1 = 1.0
    l2 = 1.0
    alpha = 1.0e-4
    norm = (1* (dist ** 2) * l2 +
            np.log(alpha + (dist ** 2)) * l1)
    return norm

def distance_velocity_reward():
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')

    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    # ax.plot3D(xline, yline, zline, 'gray')
    # ax.plot3D(x,x,x)

    # Distance and velocity reward
    def f(x, y):
        return (1-np.tanh(x*5.)) * (1-y) + (y*0.5)**2

    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Reward')
    plt.show()

def curriculum_pro():
    data = np.array([[0.1, 0.5],[0.1, 0.5],[0.1, 0.55],[0.1, 0.55],[0.1, 0.6],[0.1, 0.6],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.7],[0.1, 0.7],[0.1, 0.75],[0.1, 0.7],[0.1, 0.75],[0.1, 0.75],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.55],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.65],[0.6, 0.65],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.7],[0.6, 0.75],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.7],[0.6, 0.7],[0.6, 0.65],[0.6, 0.6],[0.6, 0.55],[0.6, 0.5],[0.6, 0.45],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.3],[0.6, 0.3],[0.6, 0.3],[0.6, 0.3],[0.6, 0.25],[0.6, 0.3],[0.6, 0.25],[0.6, 0.25],[0.6, 0.3],[0.6, 0.25],[0.6, 0.25],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.3],[0.5, 0.3],[0.5, 0.25],[0.5, 0.25],[0.5, 0.25],[0.5, 0.25],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.35],[0.4, 0.3],[0.4, 0.3],[0.4, 0.25],[0.4, 0.3],[0.4, 0.25],[0.4, 0.25],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.4],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.4],[0.3, 0.4],[0.3, 0.35],[0.3, 0.35],[0.3, 0.4],[0.3, 0.4],[0.3, 0.45],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.5],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.3, 0.75],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.65],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.65],[0.4, 0.6],[0.4, 0.6],[0.4, 0.55],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.55],[0.4, 0.6],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.5],[0.4, 0.55],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.45],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.55],[0.4, 0.6],[0.4, 0.55],[0.4, 0.6],[0.4, 0.55],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.65],[0.4, 0.6],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.7],[0.5, 0.7],[0.5, 0.7],[0.5, 0.7],[0.5, 0.7],[0.5, 0.7],[0.5, 0.7],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.65],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.75],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.65],[0.6, 0.65],[0.6, 0.65],[0.6, 0.6],[0.6, 0.6],[0.6, 0.6],[0.6, 0.65],[0.6, 0.6],[0.6, 0.65],[0.6, 0.65],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.75],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.65],[0.7, 0.6],[0.7, 0.65],[0.7, 0.6],[0.7, 0.6],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.55],[0.7, 0.55],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.4],[0.7, 0.35],[0.7, 0.3],[0.7, 0.25],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.3],[0.6, 0.3],[0.6, 0.25],[0.6, 0.3],[0.6, 0.25],[0.6, 0.3],[0.6, 0.3],[0.6, 0.35],[0.6, 0.35],[0.6, 0.4],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.4],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.6],[0.6, 0.6],[0.6, 0.55],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.65],[0.6, 0.7],[0.6, 0.75],[0.6, 0.75],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.65],[0.7, 0.65],[0.7, 0.7],[0.7, 0.7],[0.7, 0.75],[0.7, 0.75],[0.8, 0.5],[0.8, 0.55],[0.8, 0.5],[0.8, 0.55],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.6],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.5],[0.8, 0.55],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.5],[0.8, 0.5],[0.8, 0.55],[0.8, 0.5],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.6],[0.8, 0.6],[0.8, 0.55],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.45],[0.8, 0.4],[0.8, 0.4],[0.8, 0.35],[0.8, 0.35],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.25],[0.8, 0.25],[0.8, 0.25],[0.8, 0.3],[0.8, 0.3],[0.8, 0.25],[0.8, 0.25],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.25],[0.8, 0.25],[0.8, 0.25],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.45],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.35],[0.7, 0.35],[0.7, 0.3],[0.7, 0.3],[0.7, 0.25],[0.7, 0.25],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.4],[0.6, 0.4],[0.6, 0.4],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.4],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.45],[0.6, 0.45],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.6],[0.6, 0.65],[0.6, 0.65],[0.6, 0.65],[0.6, 0.7],[0.6, 0.65],[0.6, 0.65],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.75],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.55],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.65],[0.7, 0.6],[0.7, 0.65],[0.7, 0.65],[0.7, 0.7],[0.7, 0.7],[0.7, 0.75],[0.7, 0.75],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55]])
    data = data[:300]
    x = np.linspace(0, len(data), len(data))
    # plt.plot(x, data[:,1], label='Agent\'s performance', alpha=0.5, color='C3')
    plt.plot(x, data[:,0], label='Robot-Centric', color='C0')
    # data2 = np.array([[0.1, 0.5],[0.1, 0.5],[0.1, 0.55],[0.1, 0.55],[0.1, 0.6],[0.1, 0.6],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.7],[0.1, 0.7],[0.1, 0.75],[0.1, 0.75],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.55],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.35],[0.6, 0.35],[0.6, 0.3],[0.6, 0.35],[0.6, 0.3],[0.6, 0.3],[0.6, 0.25],[0.6, 0.25],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.3],[0.5, 0.3],[0.5, 0.25],[0.5, 0.25],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.45],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.35],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.35],[0.4, 0.3],[0.4, 0.25],[0.4, 0.25],[0.4, 0.25],[0.4, 0.25],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.45],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.4],[0.3, 0.4],[0.3, 0.4],[0.3, 0.4],[0.3, 0.45],[0.3, 0.5],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.5],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.6],[0.5, 0.65],[0.5, 0.7],[0.5, 0.65],[0.5, 0.6],[0.5, 0.55],[0.5, 0.55],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.3],[0.5, 0.25],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.4],[0.4, 0.4],[0.4, 0.45],[0.4, 0.4],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.4],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],])
    data2 = np.array([[0.1, 0.5],[0.1, 0.5],[0.1, 0.55],[0.1, 0.55],[0.1, 0.6],[0.1, 0.6],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.7],[0.1, 0.7],[0.1, 0.75],[0.1, 0.7],[0.1, 0.75],[0.1, 0.75],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.3, 0.75],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.7],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.65],[0.4, 0.65],[0.4, 0.6],[0.4, 0.6],[0.4, 0.55],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.25],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.4],[0.3, 0.35],[0.3, 0.35],[0.3, 0.3],[0.3, 0.3],[0.3, 0.3],[0.3, 0.35],[0.3, 0.3],[0.3, 0.3],[0.3, 0.3],[0.3, 0.3],[0.3, 0.25],[0.3, 0.3],[0.3, 0.25],[0.3, 0.25],[0.2, 0.5],[0.2, 0.55],[0.2, 0.5],[0.2, 0.5],[0.2, 0.5],[0.2, 0.5],[0.2, 0.5],[0.2, 0.5],[0.2, 0.45],[0.2, 0.45],[0.2, 0.4],[0.2, 0.45],[0.2, 0.4],[0.2, 0.4],[0.2, 0.4],[0.2, 0.45],[0.2, 0.45],[0.2, 0.45],[0.2, 0.4],[0.2, 0.4],[0.2, 0.35],[0.2, 0.3],[0.2, 0.3],[0.2, 0.35],[0.2, 0.3],[0.2, 0.3],[0.2, 0.25],[0.2, 0.25],[0.2, 0.25],[0.2, 0.3],[0.2, 0.35],[0.2, 0.35],[0.2, 0.4],[0.2, 0.45],[0.2, 0.4],[0.2, 0.35],[0.2, 0.35],[0.2, 0.4],[0.2, 0.4],[0.2, 0.4],[0.2, 0.45],[0.2, 0.5],[0.2, 0.5],[0.2, 0.45],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.55],[0.2, 0.55],[0.2, 0.5],[0.2, 0.5],[0.2, 0.45],[0.2, 0.45],[0.2, 0.4],[0.2, 0.4],[0.2, 0.45],[0.2, 0.45],[0.2, 0.45],[0.2, 0.5],[0.2, 0.55],[0.2, 0.5],[0.2, 0.5],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.55],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.55],[0.3, 0.55],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.55],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.3],[0.3, 0.3],[0.3, 0.3],[0.3, 0.35],[0.3, 0.35],[0.3, 0.35],[0.3, 0.4],[0.3, 0.45],[0.3, 0.5],[0.3, 0.55],[0.3, 0.6],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.6],[0.5, 0.55],[0.5, 0.55],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],])
    x = np.linspace(0, len(data2), len(data2))
    # plt.plot(x, data2[:,1], label='Agent\'s performance', alpha=0.5, color='C2')
    plt.plot(x, data2[:,0], label='Object-Centric', color='C1')
    plt.xlabel('Episodes', size='xx-large')
    plt.ylabel('Curriculum Level', size='xx-large')
    plt.ylim(0,1)
    # plt.xlim(0,250)
    plt.grid()
    plt.legend()
    plt.show()

def curriculum_pro_reward():
    data = np.array([[0.1, 0.5],[0.1, 0.5],[0.1, 0.55],[0.1, 0.55],[0.1, 0.6],[0.1, 0.6],[0.1, 0.65],[0.1, 0.65],[0.1, 0.7],[0.1, 0.7],[0.1, 0.75],[0.1, 0.75],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.6],[0.5, 0.65],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.3],[0.6, 0.3],[0.6, 0.25],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.35],[0.5, 0.4],[0.5, 0.35],[0.5, 0.3],[0.5, 0.35],[0.5, 0.35],[0.5, 0.35],[0.5, 0.4],[0.5, 0.45],[0.5, 0.5],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.55],[0.6, 0.6],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.65],[0.6, 0.65],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.75],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.65],[0.7, 0.65],[0.7, 0.65],[0.7, 0.65],[0.7, 0.7],[0.7, 0.7],[0.7, 0.75],[0.7, 0.75],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.6],[0.8, 0.6],[0.8, 0.65],[0.8, 0.65],[0.8, 0.65],[0.8, 0.6],[0.8, 0.65],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.65],[0.8, 0.6],[0.8, 0.6],[0.8, 0.55],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.6],[0.8, 0.55],[0.8, 0.55],[0.8, 0.6],[0.8, 0.65],[0.8, 0.65],[0.8, 0.7],[0.8, 0.7],[0.8, 0.7],[0.8, 0.75],[0.8, 0.75],[0.8, 0.75],[0.9, 0.5],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.55],[0.9, 0.5],[0.9, 0.55],[0.9, 0.5],[0.9, 0.55],[0.9, 0.5],[0.9, 0.55],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.65],[0.9, 0.6],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.7],[0.9, 0.7],[0.9, 0.7],[0.9, 0.75],[0.9, 0.7],[0.9, 0.7],[0.9, 0.7],[0.9, 0.7],[0.9, 0.65],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.55],[0.9, 0.5],[0.9, 0.5],[0.9, 0.55],[0.9, 0.55],[0.9, 0.5],[0.9, 0.5],[0.9, 0.45],[0.9, 0.45],[0.9, 0.45],[0.9, 0.45],[0.9, 0.4],[0.9, 0.4],[0.9, 0.45],[0.9, 0.4],[0.9, 0.4],[0.9, 0.45],[0.9, 0.5],[0.9, 0.5],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.55],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.6],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.65],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.65],[0.9, 0.65],[0.9, 0.7],[0.9, 0.7],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.6],[0.9, 0.55],[0.9, 0.6],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.6],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.55],[0.9, 0.6],[0.9, 0.55],[0.9, 0.5],[0.9, 0.45],[0.9, 0.4],[0.9, 0.4],[0.9, 0.45],[0.9, 0.4],[0.9, 0.45],[0.9, 0.4],[0.9, 0.35],[0.9, 0.35],[0.9, 0.3],[0.9, 0.25],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.45],[0.8, 0.5],[0.8, 0.5],[0.8, 0.55],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.6],[0.8, 0.55],[0.8, 0.6],[0.8, 0.6],[0.8, 0.65],[0.8, 0.65],[0.8, 0.7],[0.8, 0.7],[0.8, 0.75],[0.8, 0.7],[0.8, 0.75],[0.8, 0.75],[0.9, 0.5],[0.9, 0.55],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.65],[0.9, 0.65],[0.9, 0.65],[0.9, 0.6],[0.9, 0.6],[0.9, 0.55],[0.9, 0.6],[0.9, 0.6],[0.9, 0.65],[0.9, 0.65],[0.9, 0.7],[0.9, 0.7],[0.9, 0.75],[0.9, 0.7],[0.9, 0.7],[0.9, 0.65],[0.9, 0.6],[0.9, 0.55],[0.9, 0.5],[0.9, 0.5],[0.9, 0.45],[0.9, 0.45],[0.9, 0.45],[0.9, 0.5],[0.9, 0.5],[0.9, 0.5],[0.9, 0.45],[0.9, 0.4],[0.9, 0.35],[0.9, 0.35],[0.9, 0.3],[0.9, 0.25],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.6],[0.8, 0.6]])
    x = np.linspace(0, len(data), len(data))
    # plt.plot(x, data[:,1], label='Agent\'s performance', alpha=0.5, color='C3')
    plt.plot(x, data[:,0], label='Dynamic reward', color='C0')
    # data2 = np.array([[0.1, 0.5],[0.1, 0.5],[0.1, 0.55],[0.1, 0.55],[0.1, 0.6],[0.1, 0.6],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.65],[0.1, 0.7],[0.1, 0.7],[0.1, 0.75],[0.1, 0.75],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.55],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.35],[0.6, 0.35],[0.6, 0.3],[0.6, 0.35],[0.6, 0.3],[0.6, 0.3],[0.6, 0.25],[0.6, 0.25],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.3],[0.5, 0.3],[0.5, 0.25],[0.5, 0.25],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.45],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.35],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.3],[0.4, 0.35],[0.4, 0.3],[0.4, 0.25],[0.4, 0.25],[0.4, 0.25],[0.4, 0.25],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.55],[0.3, 0.5],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.4],[0.3, 0.45],[0.3, 0.45],[0.3, 0.5],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.45],[0.3, 0.4],[0.3, 0.4],[0.3, 0.4],[0.3, 0.4],[0.3, 0.4],[0.3, 0.45],[0.3, 0.5],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.5],[0.5, 0.5],[0.5, 0.55],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.6],[0.5, 0.65],[0.5, 0.7],[0.5, 0.65],[0.5, 0.6],[0.5, 0.55],[0.5, 0.55],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.3],[0.5, 0.25],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.35],[0.4, 0.4],[0.4, 0.4],[0.4, 0.45],[0.4, 0.4],[0.4, 0.45],[0.4, 0.4],[0.4, 0.4],[0.4, 0.4],[0.4, 0.45],[0.4, 0.45],[0.4, 0.5],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],])
    data2 = np.array([[0.1, 0.5],[0.1, 0.5],[0.1, 0.55],[0.1, 0.55],[0.1, 0.6],[0.1, 0.6],[0.1, 0.65],[0.1, 0.65],[0.1, 0.7],[0.1, 0.65],[0.1, 0.7],[0.1, 0.7],[0.1, 0.75],[0.1, 0.75],[0.2, 0.5],[0.2, 0.55],[0.2, 0.55],[0.2, 0.6],[0.2, 0.6],[0.2, 0.65],[0.2, 0.65],[0.2, 0.7],[0.2, 0.65],[0.2, 0.7],[0.2, 0.7],[0.2, 0.75],[0.2, 0.75],[0.3, 0.5],[0.3, 0.55],[0.3, 0.55],[0.3, 0.6],[0.3, 0.6],[0.3, 0.65],[0.3, 0.65],[0.3, 0.7],[0.3, 0.7],[0.3, 0.75],[0.3, 0.75],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.4, 0.75],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.55],[0.6, 0.55],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.35],[0.6, 0.35],[0.6, 0.35],[0.6, 0.3],[0.6, 0.25],[0.6, 0.25],[0.6, 0.3],[0.6, 0.3],[0.6, 0.35],[0.6, 0.3],[0.6, 0.3],[0.6, 0.35],[0.6, 0.3],[0.6, 0.25],[0.6, 0.25],[0.6, 0.25],[0.6, 0.25],[0.6, 0.25],[0.6, 0.25],[0.6, 0.25],[0.6, 0.25],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.45],[0.5, 0.45],[0.5, 0.4],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.35],[0.5, 0.4],[0.5, 0.35],[0.5, 0.35],[0.5, 0.3],[0.5, 0.3],[0.5, 0.25],[0.5, 0.25],[0.4, 0.5],[0.4, 0.5],[0.4, 0.45],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.55],[0.4, 0.5],[0.4, 0.55],[0.4, 0.55],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.6],[0.4, 0.65],[0.4, 0.65],[0.4, 0.7],[0.4, 0.7],[0.4, 0.75],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.55],[0.5, 0.55],[0.5, 0.6],[0.5, 0.6],[0.5, 0.65],[0.5, 0.65],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.7],[0.5, 0.75],[0.5, 0.7],[0.5, 0.7],[0.5, 0.7],[0.5, 0.75],[0.5, 0.75],[0.5, 0.75],[0.5, 0.75],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],[0.6, 0.4],[0.6, 0.4],[0.6, 0.45],[0.6, 0.4],[0.6, 0.45],[0.6, 0.45],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.65],[0.6, 0.65],[0.6, 0.65],[0.6, 0.65],[0.6, 0.65],[0.6, 0.6],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.6],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.55],[0.6, 0.6],[0.6, 0.6],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.55],[0.6, 0.6],[0.6, 0.65],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.7],[0.6, 0.75],[0.7, 0.5],[0.7, 0.55],[0.7, 0.5],[0.7, 0.55],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.65],[0.7, 0.65],[0.7, 0.7],[0.7, 0.7],[0.7, 0.7],[0.7, 0.7],[0.7, 0.75],[0.7, 0.75],[0.7, 0.75],[0.7, 0.75],[0.7, 0.75],[0.8, 0.5],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.55],[0.8, 0.5],[0.8, 0.5],[0.8, 0.45],[0.8, 0.45],[0.8, 0.4],[0.8, 0.4],[0.8, 0.35],[0.8, 0.35],[0.8, 0.3],[0.8, 0.35],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.35],[0.8, 0.35],[0.8, 0.35],[0.8, 0.35],[0.8, 0.35],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.35],[0.8, 0.35],[0.8, 0.3],[0.8, 0.3],[0.8, 0.3],[0.8, 0.25],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.45],[0.7, 0.4],[0.7, 0.4],[0.7, 0.35],[0.7, 0.4],[0.7, 0.35],[0.7, 0.35],[0.7, 0.35],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.5],[0.7, 0.45],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.6],[0.7, 0.6],[0.7, 0.65],[0.7, 0.6],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.6],[0.7, 0.6],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.55],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.35],[0.7, 0.35],[0.7, 0.4],[0.7, 0.35],[0.7, 0.35],[0.7, 0.4],[0.7, 0.4],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.6],[0.7, 0.6],[0.7, 0.55],[0.7, 0.5],[0.7, 0.45],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.45],[0.7, 0.45],[0.7, 0.4],[0.7, 0.45],[0.7, 0.45],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.5],[0.7, 0.45],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.45],[0.7, 0.4],[0.7, 0.35],[0.7, 0.35],[0.7, 0.35],[0.7, 0.35],[0.7, 0.3],[0.7, 0.35],[0.7, 0.3],[0.7, 0.3],[0.7, 0.25],[0.7, 0.25],[0.7, 0.25],[0.7, 0.3],[0.7, 0.25],[0.7, 0.3],[0.7, 0.3],[0.7, 0.35],[0.7, 0.4],[0.7, 0.4],[0.7, 0.4],[0.7, 0.45],[0.7, 0.5],[0.7, 0.45],[0.7, 0.5],[0.7, 0.5],[0.7, 0.55],[0.7, 0.55],[0.7, 0.6],[0.7, 0.65],[0.7, 0.7],[0.7, 0.75],[0.7, 0.75],[0.7, 0.75],[0.8, 0.5],[0.8, 0.5],[0.8, 0.5],[0.8, 0.55],[0.8, 0.5],[0.8, 0.5],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.45],[0.8, 0.4],[0.8, 0.4],[0.8, 0.35],[0.8, 0.35],[0.8, 0.3],[0.8, 0.3],[0.8, 0.25],[0.8, 0.25],[0.7, 0.5],[0.7, 0.5],[0.7, 0.45],[0.7, 0.45],[0.7, 0.4],[0.7, 0.4],[0.7, 0.35],[0.7, 0.35],[0.7, 0.3],[0.7, 0.35],[0.7, 0.3],[0.7, 0.3],[0.7, 0.25],[0.7, 0.25],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.5],[0.6, 0.45],[0.6, 0.45],[0.6, 0.4],[0.6, 0.4],[0.6, 0.35],])
    x = np.linspace(0, len(data2), len(data2))
    # plt.plot(x, data2[:,1], label='Agent\'s performance', alpha=0.5, color='C2')
    plt.plot(x, data2[:,0], label='Static reward', color='C1')
    plt.xlabel('Episodes', size='xx-large')
    plt.ylabel('Curriculum Level', size='xx-large')
    plt.ylim(0,1)
    # plt.xlim(0,250)
    plt.grid()
    plt.legend()
    plt.show()

def random_distributions():
    uni = np.random.uniform(size=200)
    x = np.linspace(0, 200, 200)
    # plt.plot(x, uni, label='uniform')
    mean = 0.9
    variance = 0.3
    uni = np.random.normal(mean,variance,size=200)
    uni = np.clip(uni, 0,1)
    plt.plot(x, uni, '.', label='normal')
    plt.legend()
    plt.ylim(0,1)
    plt.show()

def gaussian_distro():
    mu = 0
    variance = 10
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), '.')
    # plt.xlim(0, 1)
    plt.show()
    # mean = 0; std = 1; variance = np.square(std)
    # x = np.arange(-5,5,.01)
    # f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

    # plt.plot(x,f)
    # plt.ylabel('gaussian distribution')
    # plt.show()

# def my_gauss(x, sigma=1, h=1, mid=0):
#     from math import exp, pow
#     variance = pow(sdev, 2)
#     return h * exp(-pow(x-mid, 2)/(2*variance))

# gaussian_distro()
# random_distributions()

# force_signal()
distance_signal()
# curriculum_pro()
