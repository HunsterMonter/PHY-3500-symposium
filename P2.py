import analytique as an
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numerique as num
import numpy as np


# Vitesses en abscisses
delta_v_an = np.linspace(0.01, 10, 1000)
delta_v_num = np.linspace(0.01, 10, 100)

# Calcule P2(infty) pour différents alpha
P2_an_1 = an.P2([1, 0], 0.75, 0.5, delta_v_an, 10)
P2_num_1 = num.P2([1, 0], 0.75, 0.5, delta_v_num, 100)
P2_an_2 = an.P2([1, 0], 1, 0.5, delta_v_an, 10)
P2_num_2 = num.P2([1, 0], 1, 0.5, delta_v_num, 100)
P2_an_3 = an.P2([1, 0], 1.25, 0.5, delta_v_an, 10)
P2_num_3 = num.P2([1, 0], 1.25, 0.5, delta_v_num, 100)

# Calcule P2(infty) pour différents beta
P2_an_4 = an.P2([1, 0], 1, 0.25, delta_v_an, 10)
P2_num_4 = num.P2([1, 0], 1, 0.25, delta_v_num, 100)
P2_an_5 = an.P2([1, 0], 1, 0.5, delta_v_an, 10)
P2_num_5 = num.P2([1, 0], 1, 0.5, delta_v_num, 100)
P2_an_6 = an.P2([1, 0], 1, 0.75, delta_v_an, 10)
P2_num_6 = num.P2([1, 0], 1, 0.75, delta_v_num, 100)

#plt.style.use("https://raw.githubusercontent.com/HunsterMonter/ggplot-dark/main/ggplot_dark.mplstyle")
plt.style.use("ggplot")
fig = plt.figure(layout="constrained", figsize=(9.6, 4.8))

# Divise le graphique et crée les axes
gs = gridspec.GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Affiche les courbes pour différents alpha
ax1.plot(delta_v_an, P2_an_1, label=r"Analytique, $\alpha = 0.75\,\mathrm{s}^{-1}$")
ax1.plot(delta_v_num, P2_num_1, "--", label=r"Numérique, $\alpha = 0.75\,\mathrm{s}^{-1}$")
ax1.plot(delta_v_an, P2_an_2, label=r"Analytique, $\alpha = 1\,\mathrm{s}^{-1}$")
ax1.plot(delta_v_num, P2_num_2, "--", label=r"Numérique, $\alpha = 1\,\mathrm{s}^{-1}$")
ax1.plot(delta_v_an, P2_an_3, label=r"Analytique, $\alpha = 1.25\,\mathrm{s}^{-1}$")
ax1.plot(delta_v_num, P2_num_3, "--", label=r"Numérique, $\alpha = 1.25\,\mathrm{s}^{-1}$")

# Ajuste la légende et les axes
ax1.set_ylim(-0.001, 0.125)
ax1.set_xlabel(r"$v \, [\mathrm{m/s}]$")
ax1.set_ylabel(r"$P_2(\infty)$")
ax1.legend()

# Affiche les courbes pour différents alpha
ax2.plot(delta_v_an, P2_an_4, label=r"Analytique, $\beta = 0.25\,\mathrm{s}^{-1}$")
ax2.plot(delta_v_num, P2_num_4, "--", label=r"Numérique, $\beta = 0.25\,\mathrm{s}^{-1}$")
ax2.plot(delta_v_an, P2_an_5, label=r"Analytique, $\beta = 0.5\,\mathrm{s}^{-1}$")
ax2.plot(delta_v_num, P2_num_5, "--", label=r"Numérique, $\beta = 0.5\,\mathrm{s}^{-1}$")
ax2.plot(delta_v_an, P2_an_6, label=r"Analytique, $\beta = 0.75\,\mathrm{s}^{-1}$")
ax2.plot(delta_v_num, P2_num_6, "--", label=r"Numérique, $\beta = 0.75\,\mathrm{s}^{-1}$")

# Ajuste la légende et les axes
ax2.set_ylim(-0.001, 0.125)
ax2.set_xlabel(r"$v \, [\mathrm{m/s}]$")
ax2.set_ylabel(r"$P_2(\infty)$")
ax2.legend()

plt.show()
