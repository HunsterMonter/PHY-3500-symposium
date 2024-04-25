import matplotlib.pyplot as plt
import numpy as np


def a_dot_neg(u, a, alpha, beta, delta_v):
    a1_dot = -1j * beta * np.exp((delta_v+2j*alpha)*(u-1)/u) * a[1] / u**2
    a2_dot = -1j * beta * np.exp((delta_v-2j*alpha)*(u-1)/u) * a[0] / u**2

    return np.array([a1_dot, a2_dot])


def a_dot_pos(u, a, alpha, beta, delta_v):
    a1_dot = -1j * beta * np.exp((-delta_v+2j*alpha)*u/(1-u)) * a[1] / (1-u)**2
    a2_dot = -1j * beta * np.exp((-delta_v-2j*alpha)*u/(1-u)) * a[0] / (1-u)**2

    return np.array([a1_dot, a2_dot])


def RK2(u, a0, a_dot, alpha, beta, delta_v):
    # Setup
    # Taille et nombre de pas
    h = u[1]-u[0]
    steps = u.size

    # Initialisation de a
    a = np.zeros((steps, 2))
    a[0] = a0

    # Boucle principale d'intégration
    for i in range(1, steps):
        print(u[i-1])
        u_half = u[i-1] + h/2

        k1 = h * a_dot(u[i-1], a[i-1], alpha, beta, delta_v)
        k2 = h * a_dot(u[i-1]+h/2, a[i-1]+k1/2, alpha, beta, delta_v)

        a[i] = a[i-1] + k2

    return a


def RK4(u, a0, a_dot, alpha, beta, delta_v):
    # Setup
    # Taille et nombre de pas
    h = u[1]-u[0]
    steps = u.size

    # Initialisation de a
    a = np.zeros((steps, 2))
    a[0] = a0

    # Boucle principale d'intégration
    for i in range(1, steps):
        u_half = u[i-1] + h/2

        k1 = h * a_dot(u[i-1], a[i-1], alpha, beta, delta_v)
        k2 = h * a_dot(u_half, a[i-1]+k1/2, alpha, beta, delta_v)
        k3 = h * a_dot(u_half, a[i-1]+k2/2, alpha, beta, delta_v)
        k4 = h * a_dot(u[i], a[i-1]+k3/2, alpha, beta, delta_v)

        a[i] = a[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return a


def a_num(u, a0, alpha, beta, delta_v):
    a_neg = RK2(u, a0, a_dot_neg, alpha, beta, delta_v)
    a_pos = RK2(u, a_neg[-1], a_dot_pos, alpha, beta, delta_v)

    print(a_neg)
    print(a_pos)


u = np.linspace (0, 1, 100)
a_num(u, [1, 0], 1, 0.5, 1)
