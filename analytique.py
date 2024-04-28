from copy import deepcopy
from matplotlib.widgets import Slider, Button
from scipy.special import gamma
from scipy.special import jv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def conditionsFinales(a0, alpha, beta, delta_v, n_max):
    M_11 = M11(0, alpha, beta, delta_v, n_max)[0]
    M_12 = M12(0, alpha, beta, delta_v, n_max)[0]
    M_21 = M21(0, alpha, beta, delta_v, n_max)[0]
    M_22 = M22(0, alpha, beta, delta_v, n_max)[0]

    M_moins = np.array([[M_11, M_12], [M_21, M_22]])

    M_11 = M11(0, alpha, beta, -delta_v, n_max)[0]
    M_12 = M12(0, alpha, beta, -delta_v, n_max)[0]
    M_21 = M21(0, alpha, beta, -delta_v, n_max)[0]
    M_22 = M22(0, alpha, beta, -delta_v, n_max)[0]

    M_plus = np.array([[M_11, M_12], [M_21, M_22]])

    M = np.matmul(np.linalg.inv(M_plus), M_moins)
    a_inf = np.matmul(M, a0)

    a0.extend(a_inf)


def P2(a0, alpha, beta, delta_v, n_max):
    p2 = np.zeros(delta_v.size)

    for i, v in enumerate(delta_v):
        a0_copy = deepcopy(a0)
        conditionsFinales(a0_copy, alpha, beta, v, n_max)
        p2[i] = np.abs(a0_copy[3])**2

    return p2


def M11(t, alpha, beta, delta_v, n_max):
    n = np.arange(n_max)
    tn = np.outer(t, n)
    arr = np.matmul(np.exp(2*delta_v*tn), (-1)**n * beta**(2*n) / ((2*delta_v)**(2*n) * gamma(n+1) * gamma(n + 1/2 - 1j * alpha/delta_v)))
    
    return gamma(1/2 - 1j * alpha/delta_v) * arr


def M22(t, alpha, beta, delta_v, n_max):
    return M11(t, -alpha, beta, delta_v, n_max)


def M21(t, alpha, beta, delta_v, n_max):
    n = np.arange(n_max)
    tn = np.outer(t, n)
    arr = np.matmul(np.exp(2*delta_v*tn), (-1)**n * beta**(2*n+1) / ((2*delta_v)**(2*n+1) * gamma(n+1) * gamma(n + 3/2 - 1j*alpha/delta_v)))
    
    return -1j * np.exp((delta_v-2j*alpha)*t) * gamma(1/2 - 1j*alpha/delta_v) * arr


def M12(t, alpha, beta, delta_v, n_max):
    return M21(t, -alpha, beta, delta_v, n_max)


def a1_prime(t, alpha, beta, delta_v, n_max, a0):
    if len(a0) == 2:
        conditionsFinales(a0, alpha, beta, delta_v, n_max)

    neg = M11(t[t<=0], alpha, beta, delta_v, n_max) * a0[0] + M12(t[t<=0], alpha, beta, delta_v, n_max) * a0[1],
    pos = M11(t[t>0], alpha, beta, -delta_v, n_max) * a0[2] + M12(t[t>0], alpha, beta, -delta_v, n_max) * a0[3]

    return np.append(neg, pos)


def a1(t, alpha, beta, delta_v, n_max, a0):
    return np.exp(-1j*alpha*t) * a1_prime(t, alpha, beta, delta_v, n_max, a0)


def a2_prime(t, alpha, beta, delta_v, n_max, a0):
    if len(a0) == 2:
        conditionsFinales(a0, alpha, beta, delta_v, n_max)

    neg = M21(t[t<=0], alpha, beta, delta_v, n_max) * a0[0] + M22(t[t<=0], alpha, beta, delta_v, n_max) * a0[1]
    pos = M21(t[t>0], alpha, beta, -delta_v, n_max) * a0[2] + M22(t[t>0], alpha, beta, -delta_v, n_max) * a0[3]

    return np.append(neg, pos)


def a2(t, alpha, beta, delta_v, n_max, a0):
    return np.exp(1j*alpha*t) * a2_prime(t, alpha, beta, delta_v, n_max, a0)


def a1_analytique(t, beta, delta_v, a0):
    z = np.where(t <= 0, -beta * np.exp(delta_v*t) / delta_v, beta * (np.exp(-delta_v * t) - 2) / delta_v)

    return np.cos(z) * a0[0] + 1j * np.sin(z) * a0[1]


def a2_analytique(t, beta, delta_v, a0):
    z = np.where(t <= 0, -beta * np.exp(delta_v*t) / delta_v, beta * (np.exp(-delta_v * t) - 2) / delta_v)

    return 1j * np.sin(z) * a0[0] + np.cos(z) * a0[1]


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
