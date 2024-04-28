from copy import deepcopy
from matplotlib.widgets import Slider, Button
from numpy.typing import NDArray
from scipy.special import gamma
from scipy.special import jv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def conditionsFinales(a0: list, alpha: float|complex, beta: float, delta_v: float, n_max: int) -> None:
    """
    Calcule les conditions finales à partir des conditions
    initiales et ajoute le résultat au vecteur a0

    Params:
        a0: Vecteur des conditions initales
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer

    Retourne:
        Rien, mais ajouter les conditions finales à a0
    """
    # Calcul pour t<0
    M_11 = M11(0, alpha, beta, delta_v, n_max)[0]
    M_12 = M12(0, alpha, beta, delta_v, n_max)[0]
    M_21 = M21(0, alpha, beta, delta_v, n_max)[0]
    M_22 = M22(0, alpha, beta, delta_v, n_max)[0]

    M_moins = np.array([[M_11, M_12], [M_21, M_22]])

    # Calcul pour t>0
    M_11 = M11(0, alpha, beta, -delta_v, n_max)[0]
    M_12 = M12(0, alpha, beta, -delta_v, n_max)[0]
    M_21 = M21(0, alpha, beta, -delta_v, n_max)[0]
    M_22 = M22(0, alpha, beta, -delta_v, n_max)[0]

    M_plus = np.array([[M_11, M_12], [M_21, M_22]])

    # Calcul des conditions finales
    M = np.matmul(np.linalg.inv(M_plus), M_moins)
    a_inf = np.matmul(M, a0)

    # Ajout des conditions finales à l'array a0
    a0.extend(a_inf)


def P2(a0: list, alpha: float|complex, beta: float, delta_v: NDArray, n_max: int) -> NDArray:
    """
    Calcule P2(infty) en fonction de v

    Params:
        a0: Array des conditions initiales/initiales et finales
        alpha: alpha
        beta: beta
        delta_v: Array des vitesses auxquelles évaluer P2(infty)
        n_max: Nombre de termes à sommer

    Retourne:
        Array de P2(infty) évalué aux valeurs dans delta_v
    """
    # Crée l'array à retourner
    p2 = np.zeros(delta_v.size)

    # Pour chaque valeur dans delta_v, calcule P2(infty)
    for i, v in enumerate(delta_v):
        # Copie a0, puisque les conditions finales dépendent de v
        a0_copy = deepcopy(a0)
        conditionsFinales(a0_copy, alpha, beta, v, n_max)
        p2[i] = np.abs(a0_copy[3])**2

    return p2


def M11(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int) -> float|NDArray:
    """
    Calcule l'élément 11 de la matrice M

    Params:
        t: Float ou array de float qui représente les temps où évaluer M11
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de M11 évalué aux valeurs de t
    """
    # Notation étrange pour fonctionner avec un array de temps
    n = np.arange(n_max)
    tn = np.outer(t, n)
    arr = np.matmul(np.exp(2*delta_v*tn), (-1)**n * beta**(2*n) / ((2*delta_v)**(2*n) * gamma(n+1) * gamma(n + 1/2 - 1j * alpha/delta_v)))
    
    return gamma(1/2 - 1j * alpha/delta_v) * arr


def M22(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int) -> float|NDArray:
    """
    Calcule l'élément 22 de la matrice M

    Params:
        t: Float ou array de float qui représente les temps où évaluer M22
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de M22 évalué aux valeurs de t
    """
    # Calcule M22 avec M11 en inversant a
    return M11(t, -alpha, beta, delta_v, n_max)


def M21(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int) -> float|NDArray:
    """
    Calcule l'élément 21 de la matrice M

    Params:
        t: Float ou array de float qui représente les temps où évaluer M21
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de M11 évalué aux valeurs de t
    """
    # Notation étrange pour fonctionner avec un array de temps
    n = np.arange(n_max)
    tn = np.outer(t, n)
    arr = np.matmul(np.exp(2*delta_v*tn), (-1)**n * beta**(2*n+1) / ((2*delta_v)**(2*n+1) * gamma(n+1) * gamma(n + 3/2 - 1j*alpha/delta_v)))
    
    return -1j * np.exp((delta_v-2j*alpha)*t) * gamma(1/2 - 1j*alpha/delta_v) * arr


def M12(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int) -> float|NDArray:
    """
    Calcule l'élément 12 de la matrice M

    Params:
        t: Float ou array de float qui représente les temps où évaluer M12
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de M12 évalué aux valeurs de t
    """
    # Calcule M12 avec M21 en inversant a
    return M21(t, -alpha, beta, delta_v, n_max)


def a1_prime(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int, a0: list) -> float|NDArray:
    """
    Calcule a_1' aux temps donnés par t

    Params:
        t: Float ou array de float qui représente les temps où évaluer a_1'
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer
        a0: Vecteur de 2 ou 4 composantes contenant les conditions
            initiales ou initiales et finales respectivement

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de a_1' évalué aux valeurs de t
    """
    # Si l'array contient seulement les conditions initiales, calculer les conditions finales
    if len(a0) == 2:
        conditionsFinales(a0, alpha, beta, delta_v, n_max)

    # Si t<0, on utilise delta_v, si t>0, on utilise -delta_v
    neg = M11(t[t<=0], alpha, beta, delta_v, n_max) * a0[0] + M12(t[t<=0], alpha, beta, delta_v, n_max) * a0[1],
    pos = M11(t[t>0], alpha, beta, -delta_v, n_max) * a0[2] + M12(t[t>0], alpha, beta, -delta_v, n_max) * a0[3]

    return np.append(neg, pos)


def a1(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int, a0: list) -> float|NDArray:
    """
    Calcule a_1 aux temps donnés par t

    Params:
        t: Float ou array de float qui représente les temps où évaluer a_1
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer
        a0: Vecteur de 2 ou 4 composantes contenant les conditions
            initiales ou initiales et finales respectivement

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de a_1 évalué aux valeurs de t
    """
    return np.exp(-1j*alpha*t) * a1_prime(t, alpha, beta, delta_v, n_max, a0)


def a2_prime(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int, a0: list) -> float|NDArray:
    """
    Calcule a_2' aux temps donnés par t

    Params:
        t: Float ou array de float qui représente les temps où évaluer a_2'
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer
        a0: Vecteur de 2 ou 4 composantes contenant les conditions
            initiales ou initiales et finales respectivement

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de a_2' évalué aux valeurs de t
    """
    # Si l'array contient seulement les conditions initiales, calculer les conditions finales
    if len(a0) == 2:
        conditionsFinales(a0, alpha, beta, delta_v, n_max)

    # Si t<0, on utilise delta_v, si t>0, on utilise -delta_v
    neg = M21(t[t<=0], alpha, beta, delta_v, n_max) * a0[0] + M22(t[t<=0], alpha, beta, delta_v, n_max) * a0[1]
    pos = M21(t[t>0], alpha, beta, -delta_v, n_max) * a0[2] + M22(t[t>0], alpha, beta, -delta_v, n_max) * a0[3]

    return np.append(neg, pos)


def a2(t: float|NDArray, alpha: float|complex, beta: float, delta_v: float, n_max: int, a0: list) -> float|NDArray:
    """
    Calcule a_2 aux temps donnés par t

    Params:
        t: Float ou array de float qui représente les temps où évaluer a_2
        alpha: alpha
        beta: beta
        delta_v: delta*v
        n_max: Nombre de termes à sommer
        a0: Vecteur de 2 ou 4 composantes contenant les conditions
            initiales ou initiales et finales respectivement

    Retourne:
        Si t est un float, un float, si t est un
        array, un array de a_2 évalué aux valeurs de t
    """
    return np.exp(1j*alpha*t) * a2_prime(t, alpha, beta, delta_v, n_max, a0)


def main():
    # Paramètres inititaux
    alpha = 1
    alpha_im = 0
    beta = 0.5
    delta_v = 1
    n_max = 10

    # Conditions initiales
    a01_abs = 1
    a01_arg = 0
    a02_arg = 0
    a01 = a01_abs * np.exp(1j * a01_arg)
    a02 = np.sqrt(1 - a01_abs**2) * np.exp(1j * a02_arg)
    a0 = [a01, a02]

    # Fonction initiale
    t = np.linspace(-11, 11, 5000)
    a_1 = a1(t, alpha+1j*alpha_im, beta, delta_v, n_max, a0)
    a_2 = a2(t, alpha+1j*alpha_im, beta, delta_v, n_max, a0)

    # Figure avec sliders ajustables
    plt.style.use("ggplot")
    fig = plt.figure(layout="constrained", figsize=(9.6, 7.2))
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[8, 1])
    gs10 = gs0[1].subgridspec(3, 2)

    # Affichage de a1, a2 et de |a1|^2 + |a2|^2
    ax = fig.add_subplot(gs0[0])
    a1_re, = ax.plot(t, np.real(a_1), label=r"Re($a_1$)")
    a1_im, = ax.plot(t, np.imag(a_1), label=r"Im($a_1$)")
    a2_re, = ax.plot(t, np.real(a_2), label=r"Re($a_2$)")
    a2_im, = ax.plot(t, np.imag(a_2), label=r"Im($a_2$)")
    amp, = ax.plot(t, np.abs(a_1)**2+np.abs(a_2)**2, color="k", label="$|a_1|^2 + |a_2|^2$")
    plt.xlabel("Temps [s]")
    plt.xlim(-11, 11)
    plt.ylim(-1.2, 1.2)

    # Sliders pour les 6 paramètres ajustables
    axalpha = fig.add_subplot(gs10[0, 0])
    alpha_slider = Slider(
        ax=axalpha,
        label=r"$\alpha$ [1/s]",
        valmin=-5,
        valmax=5,
        valinit=alpha
    )

    axbeta = fig.add_subplot(gs10[1, 0])
    beta_slider = Slider(
        ax=axbeta,
        label=r"$\beta$ [1/s]",
        valmin=-5,
        valmax=5,
        valinit=beta
    )

    axdeltav = fig.add_subplot(gs10[2, 0])
    deltav_slider = Slider(
        ax=axdeltav,
        label=r"$\delta v$ [1/m]",
        valmin=0,
        valmax=10,
        valinit=delta_v
    )

    axa01abs = fig.add_subplot(gs10[0, 1])
    a01abs_slider = Slider(
        ax=axa01abs,
        label=r"$|a_1|$",
        valmin=0,
        valmax=1,
        valinit=a01_abs,
    )

    axa01arg = fig.add_subplot(gs10[1, 1])
    a01arg_slider = Slider(
        ax=axa01arg,
        label=r"$\mathrm{arg}(a_1)$",
        valmin=0,
        valmax=2*np.pi,
        valinit=a01_arg
    )

    axa02arg = fig.add_subplot(gs10[2, 1])
    a02arg_slider = Slider(
        ax=axa02arg,
        label=r"$\mathrm{arg}(a_2)$",
        valmin=0,
        valmax=2*np.pi,
        valinit=a02_arg
    )

    # Fonction qui update le graphique lors d'un changement de slider
    def update(val):
        alpha = alpha_slider.val
        beta = beta_slider.val
        delta_v = deltav_slider.val

        a01 = a01abs_slider.val * np.exp(1j * a01arg_slider.val)
        a02 = np.sqrt(1 - a01abs_slider.val**2) * np.exp(1j * a02arg_slider.val)
        a0 = [a01, a02]

        a_1 = a1(t, alpha+1j*alpha_im, beta, delta_v, n_max, a0)
        a_2 = a2(t, alpha+1j*alpha_im, beta, delta_v, n_max, a0)

        a1_re.set_ydata(np.real(a_1))
        a1_im.set_ydata(np.imag(a_1))
        a2_re.set_ydata(np.real(a_2))
        a2_im.set_ydata(np.imag(a_2))
        amp.set_ydata(np.abs(a_1)**2 + np.abs(a_2)**2)

        fig.canvas.draw_idle()

    # Invoque update lorsqu'un slider est changé
    alpha_slider.on_changed(update)
    beta_slider.on_changed(update)
    deltav_slider.on_changed(update)
    a01abs_slider.on_changed(update)
    a01arg_slider.on_changed(update)
    a02arg_slider.on_changed(update)

    # Affiche la légende et le graphique
    ax.legend(loc="upper right")
    plt.show()


    # Tests inutilisés
    """
    a_1dot = np.gradient(a_1, t[1]-t[0])
    a_2dot = np.gradient(a_2, t[1]-t[0])

    res1 = 1j*a_1dot - alpha*a_1 - beta*np.exp(-delta_v*np.abs(t))*a_2
    res2 = 1j*a_2dot - beta*np.exp(-delta_v*np.abs(t))*a_1 + alpha*a_2

    #plt.figure(layout="constrained")
    #plt.style.use("ggplot")

    plt.plot(t, np.abs(a_1)**2, label="P(a1)")
    plt.plot(t, np.abs(a_2)**2, label="P(a2)")
    plt.plot(t, np.abs(a_1)**2+np.abs(a_2)**2)

    plt.plot(t, np.real(a_1), label="Re(a1)")
    plt.plot(t, np.imag(a_1), label="Im(a1)")
    plt.plot(t, np.real(a_2), label="Re(a2)")
    plt.plot(t, np.imag(a_2), label="Im(a2)")

    plt.plot(t, np.real(exp1), label="Re(exp1)")
    plt.plot(t, np.imag(exp1), label="Im(exp1)")
    plt.plot(t, np.real(exp2), label="Re(exp2)")
    plt.plot(t, np.imag(exp2), label="Im(exp2)")

    plt.plot(t, np.real(a_1dot), label="a1, real")
    plt.plot(t, np.imag(a_1dot), label="a1, imag")
    plt.plot(t, np.real(a_2dot), label="a2, real")
    plt.plot(t, np.imag(a_2dot), label="a2, imag")

    plt.plot(t, np.real(res1), label="Re(res1)")
    plt.plot(t, np.imag(res1), label="Im(res1)")
    plt.plot(t, np.real(res2), label="Re(res2)")
    plt.plot(t, np.imag(res2), label="Im(res2)")

    plt.legend()
    plt.show()
    """


if __name__ == "__main__":
    main()
