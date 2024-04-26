import matplotlib.pyplot as plt
import numpy as np


def a_dot_neg(u, a, alpha, beta, delta_v):
    a1_dot = -1j * beta * np.exp((delta_v+2j*alpha)*u/(1+u)) * a[1] / (1+u)**2
    a2_dot = -1j * beta * np.exp((delta_v-2j*alpha)*u/(1+u)) * a[0] / (1+u)**2

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
    a = np.zeros((steps, 2), dtype=complex)
    a[0] = a0

    # Boucle principale d'intégration
    for i in range(1, steps):
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
    a = np.zeros((steps, 2), dtype=complex)
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


def a_num(steps, a0, alpha, beta, delta_v):
    u_neg = np.linspace (-1, 0, steps+1)
    u_pos = np.linspace (0, 1, steps+1)
    #a_neg_1 = 
    a_neg = RK4(u_neg[1:], a0, a_dot_neg, alpha, beta, delta_v)
    a_pos = RK4(u_pos[:-1], a_neg[-1], a_dot_pos, alpha, beta, delta_v)

    # Conversion de u à t
    t_neg = u_neg[1:] / (1+u_neg[1:])
    t_pos = u_pos[:-1] / (1-u_pos[:-1])

    a_neg = np.concatenate(([t_neg], a_neg.T))
    a_pos = np.concatenate(([t_pos], a_pos.T))
    a = np.concatenate((a_neg, a_pos), axis=1)

    # Transformation de a' à a
    a[1] *= np.exp(-1j*alpha*a[0])
    a[2] *= np.exp( 1j*alpha*a[0])

    return a


def main():
    a = a_num(100, [1, 0], 1, 0.5, 1)

    t = np.real(a[0])
    a1 = a[1]
    a2 = a[2]

    plt.style.use("ggplot")
    plt.figure(layout="constrained")
    plt.plot(t, np.real(a1), label=r"Re($a_1$)")
    plt.plot(t, np.imag(a1), label=r"Im($a_1$)")
    plt.plot(t, np.real(a2), label=r"Re($a_2$)")
    plt.plot(t, np.imag(a2), label=r"Im($a_2$)")
    plt.plot(t, np.abs(a1)**2+np.abs(a2)**2, color="k", label=r"$|a_1|^2 + |a_2|^2$")
    plt.xlabel("Temps [s]")
    plt.xlim(-11, 11)
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
