import analytique as an
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# Paramètres
alpha_re = 1
alpha_im_1 = -0.05
alpha_im_2 = 0.5
beta = 0.5
delta_v = 1
n_max = 10

# Conditions initiales
a01_abs = 1
a01_arg = 0
a02_arg = 0
a01 = a01_abs * np.exp(1j * a01_arg)
a02 = np.sqrt(1 - a01_abs**2) * np.exp(1j * a02_arg)
a0_1_re = [a01, a02]
a0_2_re = [a01, a02]
a0_1_im = [a01, a02]
a0_2_im = [a01, a02]

# Temps en abscisse
t_1 = np.linspace(-7, 39, 5000)
t_2 = np.linspace(-18, 13, 5000)

# Partie imaginaire de alpha nulle
a1_1_re = np.abs(an.a1(t_1, alpha_re, beta, delta_v, n_max, a0_1_re))**2
a2_1_re = np.abs(an.a2(t_1, alpha_re, beta, delta_v, n_max, a0_1_re))**2
a1_2_re = np.abs(an.a1(t_2, alpha_re, beta, delta_v, n_max, a0_2_re))**2
a2_2_re = np.abs(an.a2(t_2, alpha_re, beta, delta_v, n_max, a0_2_re))**2

# Partie imaginaire de alpha non-nulle
a1_1_im = np.abs(an.a1(t_1, alpha_re+1j*alpha_im_1, beta, delta_v, n_max, a0_1_im))**2
a2_1_im = np.abs(an.a2(t_1, alpha_re+1j*alpha_im_1, beta, delta_v, n_max, a0_1_im))**2
a1_2_im = np.abs(an.a1(t_2, alpha_re+1j*alpha_im_2, beta, delta_v, n_max, a0_2_im))**2
a2_2_im = np.abs(an.a2(t_2, alpha_re+1j*alpha_im_2, beta, delta_v, n_max, a0_2_im))**2

plt.style.use("ggplot")
#plt.style.use("https://raw.githubusercontent.com/HunsterMonter/ggplot-dark/main/ggplot_dark.mplstyle")
fig = plt.figure(layout="constrained", figsize=(9.6, 4.8))

# Division du graphique et création des axes
gs0 = gridspec.GridSpec(1, 2, figure=fig)
ax0 = fig.add_subplot(gs0[0])
ax1 = fig.add_subplot(gs0[1])

# Affichage des fonctions
a1, = ax0.plot(t_1, a1_1_re, "--")
a2, = ax0.plot(t_1, a2_1_re, "--")
ax1.plot(t_2, a1_2_re, "--")
ax1.plot(t_2, a2_2_re, "--")

ax0.plot(t_1, a1_1_im, color=a1.get_color())
ax0.plot(t_1, a2_1_im, color=a2.get_color())
ax1.plot(t_2, a1_2_im, color=a1.get_color(), label=r"P_1(t)")
ax1.plot(t_2, a2_2_im, color=a2.get_color(), label=r"P_2(t)")

# Affichage des probabilités totales
ax0.plot(t_1, a1_1_re + a2_1_re, "--", color="#777777")
ax1.plot(t_2, a1_2_re + a2_2_re, "--", color="#777777")

ax0.plot(t_1, a1_1_im + a2_1_im,  color="#777777")
ax1.plot(t_2, a1_2_im + a2_2_im,  color="#777777", label=r"$P(t)$")

# Ajuste la légende et les axes
ax0.set_xlabel("Temps [s]")
ax1.set_xlabel("Temps [s]")

ax0.set_ylabel("Amplitude")
ax1.set_ylabel("Amplitude")

ax0.set_xlim(-7, 39)
ax1.set_xlim(-18, 13)

ax0.set_ylim(-0.15, 1.15)
ax1.set_ylim(-0.15, 1.15)

ax1.legend(loc="upper right")
plt.show()
