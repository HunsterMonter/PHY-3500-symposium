from matplotlib.widgets import Slider, Button
import analytique as an
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numerique as num
import numpy as np


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
    a0_num = [a01, a02]

    # Fonction analytique initiale
    t_an = np.linspace(-11, 11, 5000)
    a1_an = an.a1(t_an, alpha+1j*alpha_im, beta, delta_v, n_max, a0)
    a2_an = an.a2(t_an, alpha+1j*alpha_im, beta, delta_v, n_max, a0)

    # Fonction numérique initiale
    steps = [25, 50, 100]
    t_num = []
    a1_num = []
    a2_num = []
    a1_delta = []
    a2_delta = []

    for step in steps:
        a_num = num.a_num(a0_num, alpha, beta, delta_v, step)
        t_num.append(np.real(a_num[0]))
        a1_num.append(a_num[1])
        a2_num.append(a_num[2])

        a1_an_delta = an.a1(t_num[-1], alpha+1j*alpha_im, beta, delta_v, n_max, a0)
        a2_an_delta = an.a2(t_num[-1], alpha+1j*alpha_im, beta, delta_v, n_max, a0)
        a1_delta.append(a1_num[-1] - a1_an_delta)
        a2_delta.append(a2_num[-1] - a2_an_delta)

    # Figure avec sliders ajustables
    plt.style.use("ggplot")
    #plt.style.use("https://raw.githubusercontent.com/HunsterMonter/ggplot-dark/main/ggplot_dark.mplstyle")
    fig = plt.figure(layout="constrained", figsize=(9.6, 7.2))

    # Division du graphique
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[8, 1])
    gs00 = gs0[0].subgridspec(2, 2)
    gs10 = gs0[1].subgridspec(3, 2)
    gs000 = gs00[0].subgridspec(2, 1, height_ratios=[2, 1])
    gs001 = gs00[1].subgridspec(2, 1, height_ratios=[2, 1])
    gs002 = gs00[2].subgridspec(2, 1, height_ratios=[2, 1])
    gs003 = gs00[3].subgridspec(2, 1, height_ratios=[2, 1])

    # Création des axes
    ax00 = fig.add_subplot(gs000[0, 0])
    ax01 = fig.add_subplot(gs000[1, 0])
    ax10 = fig.add_subplot(gs001[0, 0])
    ax11 = fig.add_subplot(gs001[1, 0])
    ax20 = fig.add_subplot(gs002[0, 0])
    ax21 = fig.add_subplot(gs002[1, 0])
    ax30 = fig.add_subplot(gs003[0, 0])
    ax31 = fig.add_subplot(gs003[1, 0])

    axalpha  = fig.add_subplot(gs10[0, 0])
    axbeta   = fig.add_subplot(gs10[1, 0])
    axdeltav = fig.add_subplot(gs10[2, 0])
    axa01abs = fig.add_subplot(gs10[0, 1])
    axa01arg = fig.add_subplot(gs10[1, 1])
    axa02arg = fig.add_subplot(gs10[2, 1])

    # Affichage des fonctions analytiques
    a1_re_an, = ax00.plot(t_an, np.real(a1_an))
    a1_im_an, = ax10.plot(t_an, np.imag(a1_an), label="Analytique")
    a2_re_an, = ax20.plot(t_an, np.real(a2_an))
    a2_im_an, = ax30.plot(t_an, np.imag(a2_an))

    # Affichage des probabilités totales
    amp0, = ax00.plot(t_an, np.abs(a1_an)**2+np.abs(a2_an)**2, color="k")
    amp1, = ax10.plot(t_an, np.abs(a1_an)**2+np.abs(a2_an)**2, color="k", label=r"$|a_1|^2 + |a_2|^2$")
    amp2, = ax20.plot(t_an, np.abs(a1_an)**2+np.abs(a2_an)**2, color="k")
    amp3, = ax30.plot(t_an, np.abs(a1_an)**2+np.abs(a2_an)**2, color="k")

    # Solution numérique
    a1_re_num = [ax00.plot(t, np.real(a1))[0] for t, a1 in zip(t_num, a1_num)]
    a1_im_num = [ax10.plot(t, np.imag(a1), label=f"{2*step} pas")[0] for t, a1, step in zip(t_num, a1_num, steps)]
    a2_re_num = [ax20.plot(t, np.real(a2))[0] for t, a2 in zip(t_num, a2_num)]
    a2_im_num = [ax30.plot(t, np.imag(a2))[0] for t, a2 in zip(t_num, a2_num)]

    # Erreurs sur les solutions numériques
    a1_re_delta = [ax01.plot(t, np.real(a1))[0] for t, a1 in zip(t_num, a1_delta)]
    a1_im_delta = [ax11.plot(t, np.imag(a1), label=f"{2*step} pas - analytique")[0] for t, a1, step in zip(t_num, a1_delta, steps)]
    a2_re_delta = [ax21.plot(t, np.real(a2))[0] for t, a2 in zip(t_num, a2_delta)]
    a2_im_delta = [ax31.plot(t, np.imag(a2))[0] for t, a2 in zip(t_num, a2_delta)]

    # Ajuste la légende et les axes
    ax00.set_title(r"Re($a_1$)")
    ax10.set_title(r"Im($a_1$)")
    ax20.set_title(r"Re($a_2$)")
    ax30.set_title(r"Im($a_2$)")

    ax01.set_xlabel("Temps [s]")
    ax11.set_xlabel("Temps [s]")
    ax21.set_xlabel("Temps [s]")
    ax31.set_xlabel("Temps [s]")

    ax00.set_xlim(-11, 11)
    ax10.set_xlim(-11, 11)
    ax20.set_xlim(-11, 11)
    ax30.set_xlim(-11, 11)

    ax00.set_ylim(-1.2, 1.2)
    ax10.set_ylim(-1.2, 1.2)
    ax20.set_ylim(-1.2, 1.2)
    ax30.set_ylim(-1.2, 1.2)

    ax01.set_xlim(-11, 11)
    ax11.set_xlim(-11, 11)
    ax21.set_xlim(-11, 11)
    ax31.set_xlim(-11, 11)

    max_delta_v = [0, 0]
    max_delta_v[0] = np.max([np.max(np.abs(delta)) for delta in a1_delta])
    max_delta_v[1] = np.max([np.max(np.abs(delta)) for delta in a2_delta])
    max_delta = np.max(max_delta_v)

    ax01.set_ylim(-1.1*max_delta, 1.1*max_delta)
    ax11.set_ylim(-1.1*max_delta, 1.1*max_delta)
    ax21.set_ylim(-1.1*max_delta, 1.1*max_delta)
    ax31.set_ylim(-1.1*max_delta, 1.1*max_delta)

    # Enlever les graduations entre les graphiques de a et de l'erreur
    ax00.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax10.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax20.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax30.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.suptitle("Erreur de la solution numérique sur la solution analytique.", wrap=True)
    ax10.legend(loc="upper right")
    ax11.legend(loc="upper right")

    # Sliders pour les 6 paramètres ajustables
    alpha_slider  = Slider(ax=axalpha, label=r"$\alpha$ [1/s]", valmin=-5, valmax=5, valinit=alpha)
    beta_slider   = Slider(ax=axbeta, label=r"$\beta$ [1/s]", valmin=-5, valmax=5, valinit=beta)
    deltav_slider = Slider(ax=axdeltav, label=r"$\delta v$ [1/m]", valmin=0, valmax=10, valinit=delta_v)
    a01abs_slider = Slider(ax=axa01abs, label=r"$|a_1|$", valmin=0, valmax=1, valinit=a01_abs)
    a01arg_slider = Slider(ax=axa01arg, label=r"$\mathrm{arg}(a_1)$", valmin=0, valmax=2*np.pi, valinit=a01_arg)
    a02arg_slider = Slider(ax=axa02arg, label=r"$\mathrm{arg}(a_2)$", valmin=0, valmax=2*np.pi, valinit=a02_arg)

    # Fonction qui update le graphique lors d'un changement de slider
    def update(val):
        alpha = alpha_slider.val
        beta = beta_slider.val
        delta_v = deltav_slider.val

        a01 = a01abs_slider.val * np.exp(1j * a01arg_slider.val)
        a02 = np.sqrt(1 - a01abs_slider.val**2) * np.exp(1j * a02arg_slider.val)
        a0 = [a01, a02]
        a0_num = [a01, a02]

        a1_an = an.a1(t_an, alpha+1j*alpha_im, beta, delta_v, n_max, a0)
        a2_an = an.a2(t_an, alpha+1j*alpha_im, beta, delta_v, n_max, a0)

        for i, step in enumerate(steps):
            a_num = num.a_num(a0_num, alpha, beta, delta_v, step)
            a1_num[i] = a_num[1]
            a2_num[i] = a_num[2]

            a1_an_delta = an.a1(t_num[i], alpha+1j*alpha_im, beta, delta_v, n_max, a0)
            a2_an_delta = an.a2(t_num[i], alpha+1j*alpha_im, beta, delta_v, n_max, a0)
            a1_delta[i] = a1_num[i] - a1_an_delta
            a2_delta[i] = a2_num[i] - a2_an_delta

        # Mise à jour des fonctions
        a1_re_an.set_ydata(np.real(a1_an))
        a1_im_an.set_ydata(np.imag(a1_an))
        a2_re_an.set_ydata(np.real(a2_an))
        a2_im_an.set_ydata(np.imag(a2_an))
        
        amp0.set_ydata(np.abs(a1_an)**2 + np.abs(a2_an)**2)
        amp1.set_ydata(np.abs(a1_an)**2 + np.abs(a2_an)**2)
        amp2.set_ydata(np.abs(a1_an)**2 + np.abs(a2_an)**2)
        amp3.set_ydata(np.abs(a1_an)**2 + np.abs(a2_an)**2)

        [a1_re.set_ydata(np.real(a1)) for a1_re, a1 in zip(a1_re_num, a1_num)]
        [a1_im.set_ydata(np.imag(a1)) for a1_im, a1 in zip(a1_im_num, a1_num)]
        [a2_re.set_ydata(np.real(a2)) for a2_re, a2 in zip(a2_re_num, a2_num)]
        [a2_im.set_ydata(np.imag(a2)) for a2_im, a2 in zip(a2_im_num, a2_num)]

        # Mise à jour des écarts
        [a1_re.set_ydata(np.real(a1)) for a1_re, a1 in zip(a1_re_delta, a1_delta)]
        [a1_im.set_ydata(np.imag(a1)) for a1_im, a1 in zip(a1_im_delta, a1_delta)]
        [a2_re.set_ydata(np.real(a2)) for a2_re, a2 in zip(a2_re_delta, a2_delta)]
        [a2_im.set_ydata(np.imag(a2)) for a2_im, a2 in zip(a2_im_delta, a2_delta)]

        max_delta_v = [0, 0]
        max_delta_v[0] = np.max([np.max(np.abs(delta)) for delta in a1_delta])
        max_delta_v[1] = np.max([np.max(np.abs(delta)) for delta in a2_delta])
        max_delta = np.max(max_delta_v)

        ax01.set_ylim(-1.1*max_delta, 1.1*max_delta)
        ax11.set_ylim(-1.1*max_delta, 1.1*max_delta)
        ax21.set_ylim(-1.1*max_delta, 1.1*max_delta)
        ax31.set_ylim(-1.1*max_delta, 1.1*max_delta)

        fig.canvas.draw_idle()

    # Invoque update lorsqu'un slider est changé
    alpha_slider.on_changed(update)
    beta_slider.on_changed(update)
    deltav_slider.on_changed(update)
    a01abs_slider.on_changed(update)
    a01arg_slider.on_changed(update)
    a02arg_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
