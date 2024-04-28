import analytique as an
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numerique as num
import numpy as np


# Conditions initiales
a01_abs = 1
a01_arg = 0
a02_arg = 0
a01 = a01_abs * np.exp(1j * a01_arg)
a02 = np.sqrt(1 - a01_abs**2) * np.exp(1j * a02_arg)
a0_1 = [a01, a02]
a0_2 = [a01, a02]
a0_num = [a01, a02]

# Fonction analytique initiale
t_an = np.linspace(-11, 11, 5000)
a1_an = np.abs(an.a2(t_an, 1, 0.5, 1, 10, a0_1))**2
a2_an = np.abs(an.a2(t_an, 1.25, 0.75, 3, 10, a0_2))**2

# Fonction numérique initiale
steps = [25, 50, 100]
t_num = []
a1_num = []
a2_num = []
a1_delta = []
a2_delta = []

for step in steps:
    a_num = num.a_num(a0_num, 1, 0.5, 1, step)
    t_num.append(np.real(a_num[0]))
    a1_num.append(np.abs(a_num[2])**2)

    a_num = num.a_num(a0_num, 1.25, 0.75, 3, step)
    a2_num.append(np.abs(a_num[2])**2)

    a1_an_delta = np.abs(an.a2(t_num[-1], 1, 0.5, 1, 10, a0_1))**2
    a2_an_delta = np.abs(an.a2(t_num[-1], 1.25, 0.75, 3, 10, a0_2))**2
    a1_delta.append(a1_num[-1] - a1_an_delta)
    a2_delta.append(a2_num[-1] - a2_an_delta)

plt.style.use("ggplot")
#plt.style.use("https://raw.githubusercontent.com/HunsterMonter/ggplot-dark/main/ggplot_dark.mplstyle")
fig = plt.figure(layout="constrained", figsize=(9.6, 4.8))

# Division du graphique
gs0 = gridspec.GridSpec(1, 2, figure=fig)
gs00 = gs0[0].subgridspec(2, 1, height_ratios=[2, 1])
gs01 = gs0[1].subgridspec(2, 1, height_ratios=[2, 1])

# Création des axes
ax00 = fig.add_subplot(gs00[0, 0])
ax01 = fig.add_subplot(gs00[1, 0])
ax10 = fig.add_subplot(gs01[0, 0])
ax11 = fig.add_subplot(gs01[1, 0])

# Affichage des fonctions analytiques
ax00.plot(t_an, a1_an, label="Analytique")
ax10.plot(t_an, a2_an)

# Affichage des probabilités totales
#amp0, = ax00.plot(t_an, np.abs(a1_an)**2+np.abs(a2_an)**2, color="k", label=r"$|a_1|^2 + |a_2|^2$")
#amp1, = ax10.plot(t_an, np.abs(a1_an)**2+np.abs(a2_an)**2, color="k")

# Solution numérique
[ax00.plot(t, a1, label=f"{2*step} pas")[0] for t, a1, step in zip(t_num, a1_num, steps)]
[ax10.plot(t, a2)[0] for t, a2 in zip(t_num, a2_num)]

# Erreurs sur les solutions numériques
[ax01.plot(t, a1, label=f"{2*step} pas")[0] for t, a1, step in zip(t_num, a1_delta, steps)]
[ax11.plot(t, a2)[0] for t, a2 in zip(t_num, a2_delta)]

# Ajuste la légende et les axes
ax01.set_xlabel("Temps [s]")
ax11.set_xlabel("Temps [s]")

ax00.set_ylabel(r"$P_2(t)$")
ax10.set_ylabel(r"$P_2(t)$")

ax01.set_ylabel("Erreur")
ax11.set_ylabel("Erreur")

ax00.set_xlim(-5.5, 5.5)
ax10.set_xlim(-5.5, 5.5)

ax00.set_ylim(-0.01, 0.11)
ax10.set_ylim(-0.01, 0.11)

ax01.set_xlim(-5.5, 5.5)
ax11.set_xlim(-5.5, 5.5)

max_delta_0 = np.max([np.max(np.abs(delta)) for delta in a1_delta])
max_delta_1 = np.max([np.max(np.abs(delta)) for delta in a2_delta])
max_delta = max(max_delta_0, max_delta_1)

ax01.set_ylim(-1.2*max_delta, 1.2*max_delta)
ax11.set_ylim(-1.2*max_delta, 1.2*max_delta)

# Enlever les graduations entre les graphiques de a et de l'erreur
ax00.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax10.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax00.legend(loc="upper left")
ax01.legend(loc="upper left")
plt.show()
