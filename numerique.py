from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


def a_dot_neg(u: float, a: NDArray, alpha: float|complex, beta: float, delta_v: float) -> NDArray:
    """
    Dérivée du système pour t<0
    
    Params:
        u: "Temps" auxquels calculer la dérivée
        a: Point où calculer la dérivée
        alpha: alpha
        beta: beta
        delta_v: delta*v
    
    Retourne:
        Array contenant a_1_dot et a_2_dot
    """
    a1_dot = -1j * beta * np.exp((delta_v+2j*alpha)*u/(1+u)) * a[1] / (1+u)**2
    a2_dot = -1j * beta * np.exp((delta_v-2j*alpha)*u/(1+u)) * a[0] / (1+u)**2

    return np.array([a1_dot, a2_dot])


def a_dot_pos(u: float, a: NDArray, alpha: float|complex, beta: float, delta_v: float) -> NDArray:
    """
    Dérivée du système pour t>0
    
    Params:
        u: "Temps" auquels calculer la dérivée
        a: Point où calculer la dérivée
        alpha: alpha
        beta: beta
        delta_v: delta*v
    
    Retourne:
        Array contenant a_1_dot et a_2_dot
    """
    a1_dot = -1j * beta * np.exp((-delta_v+2j*alpha)*u/(1-u)) * a[1] / (1-u)**2
    a2_dot = -1j * beta * np.exp((-delta_v-2j*alpha)*u/(1-u)) * a[0] / (1-u)**2

    return np.array([a1_dot, a2_dot])


def RK4(u: NDArray, a0: list, a_dot: callable, alpha: float|complex, beta: float, delta_v: float) -> NDArray:
    """
    Calcule numériquement la solution à l'équation différentielle a_dot pour
    les conditions initiales a0 avec la méthode RK4 aux points u

    Params:
        u: Points où calculer la solution
        a0: Conditions initiales
        a_dot: Équation différentielle à résoudre
        alpha: alpha
        beta: beta
        delta_v: delta*v

    Retourne:
        Array contenant les valeurs de a évalué à u
    """
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


def a_num(a0: list, alpha: float|complex, beta: float, delta_v: float, steps: int):
    """
    Résout pour a(t) avec 2*steps pas

    Params:
        a0: Array des conditions initiales
        alpha: alpha
        beta: beta
        delta_v: delta*v
        steps: Nombre de pas à effectuer pour chaque moitié de la solution

    Retourne:
        Array contenant le temps, a1 et a2
    """
    # Coordonnées de l'abscisse
    u_neg = np.linspace (-1, 0, steps+1)
    u_pos = np.linspace (0, 1, steps+1)

    # Le premier pas est effectué avec Euler implicite
    u = u_neg[1]
    h = u_neg[1] - u_neg[0]
    a1 = np.array(a0, dtype=complex)
    A12 = -1j * beta * np.exp((delta_v+2j*alpha)*u/(1+u)) / (1+u)**2
    A21 = -1j * beta * np.exp((delta_v-2j*alpha)*u/(1+u)) / (1+u)**2
    a1[0] -= A12 * a0[1]
    a1[1] -= A21 * a0[0]
    a1 /= 1 - A12*A21

    # Calcule a pour -1<u<1
    a_neg = RK4(u_neg[1:], a1, a_dot_neg, alpha, beta, delta_v)
    a_pos = RK4(u_pos[:-1], a_neg[-1], a_dot_pos, alpha, beta, delta_v)

    # Conversion de u à t
    t_neg = u_neg[1:] / (1+u_neg[1:])
    t_pos = u_pos[:-1] / (1-u_pos[:-1])

    # Crée la liste à retourner
    a_neg = np.concatenate(([t_neg], a_neg.T))
    a_pos = np.concatenate(([t_pos], a_pos.T))
    a = np.concatenate((a_neg, a_pos), axis=1)

    # Transformation de a' à a
    a[1] *= np.exp(-1j*alpha*a[0])
    a[2] *= np.exp( 1j*alpha*a[0])

    return a


def P2(a0: list, alpha: float|complex, beta: float, delta_v: NDArray, steps: int):
    """
    Calcule P2(infty) en fonction de v

    Params:
        a0: Array des conditions initiales
        alpha: alpha
        beta: beta
        delta_v: Array des vitesses auxquelles évaluer P2(infty)
        steps: Nombre de pas pour chaque moitié, t<0 et t>0

    Retourne:
        Array de P2(infty) évalué aux valeurs de delta_v
    """
    # Crée l'array à retourner
    p2 = np.zeros(delta_v.size)

    # Pour chaque valeur dans delta_v, calcule P2(infty)
    for i, v in enumerate(delta_v):
        a_sol = a_num(a0, alpha, beta, v, steps)
        a = np.array([a_sol[1][-1], a_sol[2][-1]])

        # Comme l'intégration numérique ne arrête un pas avant u=1 (t=infty), on
        # fait un pas RK2 (on ne peut pas faire RK4, car cela nécessite un calcul
        # à u+h, qui est à u=1, donc a_dot diverge
        h = 1/steps
        u = 1 - h

        k1 = h * a_dot_pos(u, a, alpha, beta, v)
        k2 = h * a_dot_pos(u, a+k1/2, alpha, beta, v)

        a += k2

        p2[i] = np.abs(a[1])**2

    return p2


def main():
    a = a_num([1, 0], 1, 0.5, 1, 100)

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
