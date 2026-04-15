"""
Simulation MEMS capacitif — Méthode de relaxation 2D (Laplace)
================================================================
PHY-1007 · Projet capteur MEMS capacitif · Université Laval

GÉOMÉTRIE SIMULÉE
-----------------
On regarde une coupe transversale (plan xz) d'une paire de doigts :

        stator (+V0)      rotor (0 V)
             |                |
    z ▲  ----+----    gap    ----+----
      |      |    <--- d --->    |
      |      |                   |
      +---> x  (direction du mouvement)

  x = direction du déplacement de la masse
  z = épaisseur des doigts (t)
  y = longueur des doigts L  ← hors plan, multiplicateur après

La longueur L est prise en compte en multipliant C/longueur × L.

DÉPENDANCES
-----------
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import convolve

# ================================================================
#  PARAMÈTRES — modifie uniquement cette section
# ================================================================

# --- Géométrie des doigts [µm] ---
d0    = 2.0    # gap nominal entre les faces au repos        [µm]
t     = 3.0    # épaisseur (hauteur dans la coupe) des doigts [µm]
L     = 200.0  # longueur des doigts                         [µm]

# --- Peigne interdigité ---
N_pairs = 50   # nombre de paires de doigts (rotor/stator)

# --- Circuit ---
V0 = 3.0       # tension appliquée entre stator et rotor     [V]

# --- Mécanique ---
m = 1.0e-9     # masse d'épreuve  [kg]  (ex: 1 µg = 1e-9 kg)
k = 1.0        # constante de rappel [N/m]

# --- Accélérations à évaluer ---
a_min, a_max, n_pts = -40.0, 40.0, 17   # [g]

# --- Paramètres numériques ---
dx     = 0.25    # résolution de la grille        [µm/pixel]
                 # ↑ diminuer pour plus de précision (plus lent)
margin = 6.0     # marge vide autour des doigts   [µm]
n_iter = 3000    # itérations de relaxation pour la visualisation
n_iter_fast = 1500  # itérations pour la courbe C(a) (plus rapide)

# ================================================================
#  CONSTANTES PHYSIQUES
# ================================================================
eps0 = 8.854e-12   # permittivité du vide [F/m]
g_si = 9.81        # [m/s²]

# ================================================================
#  FONCTIONS
# ================================================================

def build_grid(d_um):
    """
    Construit la grille 2D pour un gap d_um (µm).

    Retourne
    --------
    V        : ndarray (nz, nx)  potentiel initial (0 partout sauf conducteurs)
    mask_pos : ndarray bool      pixels du stator  (+V0)
    mask_neg : ndarray bool      pixels du rotor   ( 0 V)
    nx, nz   : int               dimensions de la grille
    """
    Lx = d_um + 2 * margin   # largeur du domaine [µm]
    Lz = t    + 2 * margin   # hauteur du domaine [µm]

    nx = max(4, int(round(Lx / dx)))
    nz = max(4, int(round(Lz / dx)))

    V = np.zeros((nz, nx))

    # Positions en pixels
    ix_s = int(round(margin       / dx))   # colonne stator
    ix_r = int(round((margin+d_um)/ dx))   # colonne rotor
    iz_b = int(round(margin       / dx))   # bas  du doigt
    iz_t = int(round((margin + t) / dx))   # haut du doigt

    # Masques
    mask_pos = np.zeros((nz, nx), dtype=bool)
    mask_neg = np.zeros((nz, nx), dtype=bool)
    mask_pos[iz_b:iz_t+1, ix_s] = True
    mask_neg[iz_b:iz_t+1, ix_r] = True

    V[mask_pos] = +V0
    V[mask_neg] =  0.0

    return V, mask_pos, mask_neg, nx, nz


def relax_laplace(V, mask_pos, mask_neg, n_iter):
    """
    Résout ∇²V = 0 par la méthode de relaxation.

    À chaque itération :
      1. On convolve V avec le noyau K = [[0,1,0],[1,0,1],[0,1,0]] / 4
         → chaque pixel devient la moyenne de ses 4 voisins
         → c'est exactement la discrétisation de ∇²V = 0 sur grille carrée
      2. On réimpose les conditions aux limites (conducteurs fixes)

    Convergence : les variations entre itérations → 0 quand V satisfait Laplace.
    """
    K = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=float) / 4.0

    for _ in range(n_iter):
        V = convolve(V, K, mode='nearest')  # Neumann BC aux bords du domaine
        V[mask_pos] = +V0                   # stator maintenu à +V0
        V[mask_neg] =  0.0                  # rotor maintenu à 0 V

    return V


def capacitance_energy(V, dx):
    """
    Calcule la capacité totale via la méthode de l'énergie électrostatique.

    Principe
    --------
    L'énergie stockée dans le champ électrique (par unité de longueur en y) :
        u = (ε₀/2) ∫∫ |E|² dx dz      [J/m]

    Pour un condensateur :  u = C_per_length × V0² / 2
    Donc :  C_per_length = ε₀ ∫∫ |E|² dx dz / V0²   [F/m]

    Avantage : inclut automatiquement les effets de frange (champ qui déborde
    aux bouts des doigts), contrairement à la formule ε₀A/d.

    Retourne C_total [F] pour N_pairs paires de longueur L.
    """
    dx_m = dx * 1e-6        # µm → m

    # Champ électrique : E = -∇V (différences centrées par np.gradient)
    Ex = -np.gradient(V, dx_m, axis=1)   # composante x  [V/m]
    Ez = -np.gradient(V, dx_m, axis=0)   # composante z  [V/m]

    E_sq = Ex**2 + Ez**2                 # |E|²  [V²/m²]

    # Intégrale numérique sur le domaine (surface élémentaire = dx_m²)
    u = 0.5 * eps0 * np.sum(E_sq) * dx_m**2    # [J/m]

    C_per_length = 2 * u / V0**2               # [F/m]
    C_one_pair   = C_per_length * L * 1e-6     # [F]  (× longueur L en m)

    return N_pairs * C_one_pair                 # [F]  (× nombre de paires)


def compute_C_vs_a(a_g_list, n_it):
    """
    Calcule C(a) pour une liste d'accélérations [g].
    Retourne deux tableaux : a_valid [g] et C_valid [F].
    """
    a_out, C_out = [], []
    for a_g in a_g_list:
        x    = m * (a_g * g_si) / k        # déplacement [m]
        d_m  = d0 * 1e-6 - x               # gap réel    [m]
        d_um = d_m * 1e6                   # gap réel    [µm]

        if d_um <= dx * 0.6:
            print(f"  {a_g:+6.1f} g → collision (d = {d_um*1000:.0f} nm < résolution)")
            continue

        V, mp, mn, *_ = build_grid(d_um)
        V = relax_laplace(V, mp, mn, n_it)
        C = capacitance_energy(V, dx)

        a_out.append(a_g)
        C_out.append(C)
        print(f"  {a_g:+6.1f} g  d = {d_um:.3f} µm  C = {C*1e15:.3f} fF")

    return np.array(a_out), np.array(C_out)


# ================================================================
#  SIMULATION PRINCIPALE
# ================================================================

print("=" * 60)
print(" Simulation MEMS — méthode de relaxation 2D (Laplace)")
print("=" * 60)
print(f"\nParamètres :")
print(f"  gap nominal d0 = {d0} µm   ({int(round(d0/dx))} pixels dans le gap)")
print(f"  épaisseur t    = {t} µm")
print(f"  longueur L     = {L} µm")
print(f"  paires N       = {N_pairs}")
print(f"  résolution dx  = {dx} µm\n")

# --- 1. Simulation au repos (a = 0) ---
print("--- Simulation au repos (a = 0) ---")
V_init, mp0, mn0, nx0, nz0 = build_grid(d0)[:5]
V_solved = relax_laplace(V_init, mp0, mn0, n_iter)

C0_sim = capacitance_energy(V_solved, dx)
C0_ana = N_pairs * eps0 * (L * 1e-6) * (t * 1e-6) / (d0 * 1e-6)

print(f"  C0 simulation  = {C0_sim * 1e15:.3f} fF")
print(f"  C0 analytique  = {C0_ana * 1e15:.3f} fF")
print(f"  Écart relatif  = {abs(C0_sim - C0_ana) / C0_ana * 100:.1f}%")
print(f"  (l'écart reflète les effets de frange captés par la simulation)\n")

# --- 2. Sensibilité analytique ---
dCda_ana = N_pairs * eps0 * (L*1e-6) * (t*1e-6) * m / (k * (d0*1e-6)**2)
print(f"  Sensibilité analytique dC/da = {dCda_ana * 1e15:.4f} fF·s²/m")
print(f"                               = {dCda_ana * 1e15 / g_si:.4f} fF/g\n")

# --- 3. Courbe C(a) ---
print("--- Calcul C(a) ---")
a_list = np.linspace(a_min, a_max, n_pts)
a_valid, C_valid = compute_C_vs_a(a_list, n_iter_fast)

# ΔV si le condensateur est déconnecté après charge à V0
# Q = C0 * V0  →  V(a) = Q / C(a) = V0 * C0 / C(a)
# ΔV = V(a) - V0 = V0 * (C0/C(a) - 1)
dV_valid = V0 * (C0_sim / C_valid - 1)

# Sensibilité numérique autour de a = 0
idx0 = np.argmin(np.abs(a_valid))
da_m = (a_valid[1] - a_valid[0]) * g_si if len(a_valid) > 1 else 1
if idx0 > 0 and idx0 < len(C_valid) - 1:
    dCda_num = (C_valid[idx0+1] - C_valid[idx0-1]) / (2 * da_m)
    print(f"\n  Sensibilité numérique dC/da  = {dCda_num * 1e15:.4f} fF·s²/m")
    print(f"                               = {dCda_num * 1e15 / g_si:.4f} fF/g")

# ================================================================
#  FIGURES
# ================================================================
fig = plt.figure(figsize=(16, 5))

# ---- Figure 1 : Potentiel V(x,z) ----
ax1 = fig.add_subplot(1, 3, 1)

Lx_um = d0 + 2 * margin
Lz_um = t  + 2 * margin
extent = [0, Lx_um, 0, Lz_um]

im = ax1.imshow(
    V_solved, origin='lower', extent=extent,
    cmap='RdBu_r', norm=TwoSlopeNorm(vmin=0, vcenter=V0/2, vmax=V0),
    interpolation='bilinear'
)
plt.colorbar(im, ax=ax1, label='Potentiel V [V]', shrink=0.85)

# Lignes équipotentielles
X_vec = np.linspace(0, Lx_um, nx0)
Z_vec = np.linspace(0, Lz_um, nz0)
XX, ZZ = np.meshgrid(X_vec, Z_vec)
ax1.contour(XX, ZZ, V_solved, levels=12, colors='white',
            linewidths=0.6, alpha=0.7)

# Lignes de champ électrique (streamlines)
Ex_plot = -np.gradient(V_solved, axis=1)
Ez_plot = -np.gradient(V_solved, axis=0)
ax1.streamplot(XX, ZZ, Ex_plot, Ez_plot, color='yellow',
               linewidth=0.6, density=1.0, arrowsize=0.8)

# Doigts
frac_bot = margin / Lz_um
frac_top = (margin + t) / Lz_um
ax1.axvline(margin,      ymin=frac_bot, ymax=frac_top, color='#5599FF', lw=3,
            label=f'Stator (+{V0} V)')
ax1.axvline(margin + d0, ymin=frac_bot, ymax=frac_top, color='#FF5555', lw=3,
            label='Rotor (0 V)')

ax1.set_xlabel('x [µm]  (direction du mouvement)')
ax1.set_ylabel('z [µm]  (épaisseur)')
ax1.set_title(
    f'Potentiel V(x,z) au repos\n'
    f'd₀={d0} µm, t={t} µm, {n_iter} itérations'
)
ax1.legend(fontsize=8, loc='upper right')

# ---- Figure 2 : C(a) simulation vs analytique ----
ax2 = fig.add_subplot(1, 3, 2)

a_fine = np.linspace(a_valid[0], a_valid[-1], 300)
x_fine = m * a_fine * g_si / k
d_fine = d0 * 1e-6 - x_fine
C_ana_fine = N_pairs * eps0 * (L*1e-6) * (t*1e-6) / d_fine

ax2.plot(a_fine, C_ana_fine * 1e15, 'k--', lw=1.8,
         label='Analytique  ε₀A/d')
ax2.plot(a_valid, C_valid * 1e15, 'o-', color='steelblue',
         ms=5, lw=1.5, label='Simulation (Laplace)')
ax2.axhline(C0_sim * 1e15, color='gray', lw=0.8, ls=':', alpha=0.7,
            label=f'C₀ = {C0_sim*1e15:.2f} fF')
ax2.axvline(0, color='gray', lw=0.8, ls=':', alpha=0.5)

ax2.set_xlabel('Accélération [g]')
ax2.set_ylabel('Capacité [fF]')
ax2.set_title('C(a) — simulation vs analytique')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ---- Figure 3 : ΔV(a) ----
ax3 = fig.add_subplot(1, 3, 3)

dV_ana_fine = V0 * (C0_ana / C_ana_fine - 1)

ax3.plot(a_fine,  dV_ana_fine  * 1e3, 'k--', lw=1.8, label='Analytique')
ax3.plot(a_valid, dV_valid     * 1e3, 'o-',  color='darkorange',
         ms=5, lw=1.5, label='Simulation')
ax3.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax3.axvline(0, color='gray', lw=0.8, alpha=0.5)

ax3.set_xlabel('Accélération [g]')
ax3.set_ylabel('ΔV [mV]')
ax3.set_title(
    f'ΔV(a) — condensateur déconnecté\n'
    f'(chargé à V₀ = {V0} V, Q = C₀V₀ constant)'
)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mems_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure sauvegardée : mems_simulation.png")
print("\nRésumé :")
print(f"  C0           = {C0_sim*1e15:.3f} fF  (simulation)")
print(f"  C0           = {C0_ana*1e15:.3f} fF  (analytique ε₀A/d)")
print(f"  dC/da        = {dCda_ana*1e15/g_si:.4f} fF/g  (analytique)")
print(f"  ΔV max (±{a_valid[-1]:.0f}g) = {np.nanmax(np.abs(dV_valid))*1e3:.1f} mV")