import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fast')

# ================================================================
#  CONSTANTES ---- en µm et unités SI
# ================================================================
N_pairs  = 4      # nombre de paires de doigts
Y_gap    = 4.0    # gap entre doigts [µm]
X_width  = 2.0    # épaisseur des doigts [µm]
L_finger = 20.0   # longueur des doigts [µm]
m_mass   = 0.00002# masse d'épreuve [kg]   ← nommé m_mass pour éviter conflit
k_spring = 2.0    # constante du ressort [N/m]
V0       = 3.0    # tension stator [V]
epsilon_0 = 8.854e-12  # [F/m]
res      = 0.3    # résolution [µm/pixel]
iters    = 3000   # nombre d'itérations

# ================================================================
#  GÉOMÉTRIE SANS MARGE
#  
#  CHANGEMENT 1 : centres commence à Y_gap/2 (pas à margin)
#  → le premier doigt n'est pas collé au bord
# ================================================================
centres = Y_gap / 2 + np.arange(2 * N_pairs) * Y_gap
# centres[0] = Y_gap/2, centres[1] = 3*Y_gap/2, ...
# Alternance : pair=stator, impair=rotor

# CHANGEMENT 2 : Ly, Lx, Lz sans margin
# On ajoute Y_gap/2 de chaque côté pour symmétrie
Ly = centres[-1] + Y_gap / 2   # [µm]
Lx = X_width                    # [µm]  ← juste l'épaisseur du doigt
Lz = L_finger                   # [µm]  ← juste la longueur du doigt

ny = int(round(Ly / res))
nx = int(round(Lx / res))
nz = int(round(Lz / res))

print(f"Grille : nz={nz}, nx={nx}, ny={ny}  ({nz*nx*ny/1e6:.2f}M pixels)")
print(f"Domaine : Ly={Ly:.1f} µm, Lx={Lx:.1f} µm, Lz={Lz:.1f} µm")

# ================================================================
#  MASQUES
#
#  CHANGEMENT 3 : x_s=0, x_e=nx-1 (tout le domaine en x)
#  CHANGEMENT 4 : z_s=0, z_e=nz-1 (tout le domaine en z)
#  CHANGEMENT 5 : pas de plaques de base (elles causaient des
#                 problèmes aux bords avec np.roll périodique)
# ================================================================
vol_stator = np.zeros((nz, nx, ny), dtype=bool)
vol_rotor  = np.zeros((nz, nx, ny), dtype=bool)

slice_x = slice(0, nx)   # tout le domaine en x
# Doigts s'étendent sur toute la hauteur z (pas de plaques de base)
z_s = 0
z_e = nz

for i, y_pos in enumerate(centres):
    y_idx = int(round(y_pos / res))
    y_idx = np.clip(y_idx, 0, ny - 1)
    if i % 2 == 0:
        vol_stator[z_s:z_e, slice_x, y_idx] = True
    else:
        vol_rotor [z_s:z_e, slice_x, y_idx] = True

print(f"Pixels stator : {vol_stator.sum()}")
print(f"Pixels rotor  : {vol_rotor.sum()}")

# ================================================================
#  POTENTIEL ET RELAXATION
# ================================================================
potential = np.full((nz, nx, ny), V0 / 2.0)  # init à V0/2
potential[vol_stator] = V0
potential[vol_rotor]  = 0.0

is_free = ~(vol_stator | vol_rotor)

for i in range(iters):
    v_avg = (
        np.roll(potential,  1, axis=0)
      + np.roll(potential, -1, axis=0)
      + np.roll(potential,  1, axis=1)
      + np.roll(potential, -1, axis=1)
      + np.roll(potential,  1, axis=2)
      + np.roll(potential, -1, axis=2)
    ) / 6.0

    potential[is_free] = v_avg[is_free]

    # Conditions aux frontières Dirichlet en z
    potential[0,  :, :] = 0.0   # bas  → V = 0 V  (stator)
    potential[-1, :, :] = V0    # haut → V = V0    (rotor)

    # Les doigts (Dirichlet) sont déjà dans is_free=False,
    # donc ils ne sont pas écrasés par v_avg
    if i % 500 == 0:
        print(f'  Relaxation {i}/{iters}')

# ================================================================
#  CHAMP ÉLECTRIQUE 3D
# ================================================================
y_axis = np.linspace(0, Ly, ny)
z_axis = np.linspace(0, Lz, nz)
x_axis = np.linspace(0, Lx, nx)

grad_z, grad_x, grad_y = np.gradient(potential, z_axis, x_axis, y_axis)
Ex = -grad_x
Ey = -grad_y
Ez = -grad_z

# ================================================================
#  CAPACITÉ PAR INTÉGRALE DE FLUX (GAUSS)
# ================================================================
def compute_capacitance(potential_2d, vol_stator_2d, res_m, L_x_m, V0, epsilon_0):
    """
    C = epsilon_0 * (flux de E autour du stator) * L_x / V0

    On somme E·n sur chaque face de pixel voisin du stator.
    potential_2d : shape (nz, ny)
    vol_stator_2d : shape (nz, ny), bool
    """
    grad_z2, grad_y2 = np.gradient(potential_2d, res_m, res_m)
    Ey2 = -grad_y2
    Ez2 = -grad_z2

    flux = 0.0
    z_idxs, y_idxs = np.where(vol_stator_2d)

    for z, y in zip(z_idxs, y_idxs):
        nz2, ny2 = potential_2d.shape
        if z+1 < nz2 and not vol_stator_2d[z+1, y]:
            flux += Ez2[z+1, y] * res_m
        if z-1 >= 0  and not vol_stator_2d[z-1, y]:
            flux -= Ez2[z-1, y] * res_m
        if y+1 < ny2 and not vol_stator_2d[z, y+1]:
            flux += Ey2[z, y+1] * res_m
        if y-1 >= 0  and not vol_stator_2d[z, y-1]:
            flux -= Ey2[z, y-1] * res_m

    Q_per_length = epsilon_0 * flux          # [C/m]
    C = abs(Q_per_length) * L_x_m / V0      # [F]
    return C


# Example: Flux of E = <0, 0, 10> through a 1x1 flat surface in xy-plane
# Surface area element dA points in z-direction: <0, 0, dA>
def electric_field(x, y, z):
    return np.array([0, 0, 10])

# Discretize surface
x_coords = np.linspace(0, 1, 10)
y_coords = np.linspace(0, 1, 10)
dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]
da_scalar = dx * dy

total_flux = 0
for x in x_coords:
    for y in y_coords:
        # dA vector pointing in +z
        da_vec = np.array([0, 0, da_scalar])
        e_vec = electric_field(x, y, 0)
        total_flux += np.dot(e_vec, da_vec)

print(f"Total Electric Flux: {total_flux}")














x_slice   = nx // 2
V_2d      = potential[:, x_slice, :]
stator_2d = vol_stator[:, x_slice, :]

C0 = compute_capacitance(V_2d, stator_2d, res * 1e-6, Lx * 1e-6, V0, epsilon_0)
print(f"\nCapacité C0 (simulation) = {C0 * 1e15:.4f} fF")

# Comparaison analytique
# C_ana = N_pairs * epsilon_0 * (L_finger * 1e-6) * (X_width * 1e-6) / (Y_gap * 1e-6)
# print(f"Capacité C0 (analytique ε₀A/d) = {C_ana * 1e15:.4f} fF")
# print(f"Écart : {abs(C0 - C_ana)/C_ana * 100:.1f}%")

# ================================================================
#  dC/da PAR BALAYAGE D'ACCÉLÉRATIONS
# ================================================================
g_si   = 9.81          # [m/s²]
d0_m   = Y_gap * 1e-6  # gap nominal [m]

accs   = np.linspace(5, 5, 11)  # [g]  ← plage réduite pour test
caps   = []

print(f"\nCalcul C(a)...")
for a_g in accs:
    x_disp = m_mass * (a_g * g_si) / k_spring   # déplacement [m]
    d_new  = d0_m - x_disp                        # nouveau gap [m]

    if d_new <= res * 1e-6:
        print(f"  a={a_g:+.1f}g → collision (d={d_new*1e6:.2f} µm)")
        caps.append(np.nan)
        continue

    # Capacité analytique pour ce gap (rapide, sans re-simuler)
    C_a = N_pairs * epsilon_0 * (L_finger * 1e-6) * (X_width * 1e-6) / d_new
    caps.append(C_a)
    print(f"  a={a_g:+.1f}g  d={d_new*1e6:.2f}µm  C={C_a*1e15:.3f}fF")

caps = np.array(caps)

# Dérivée numérique par différence finie
dC_da = np.gradient(caps, accs * g_si)   # [F·s²/m]

idx0 = len(accs) // 2
print(f"\ndC/da à a=0 : {dC_da[idx0]*1e15:.4f} fF·s²/m")
print(f"           = {dC_da[idx0]*1e15/g_si:.4f} fF/g")

# Analytique pour vérification
dC_da_ana = N_pairs * epsilon_0 * (L_finger*1e-6) * (X_width*1e-6) * m_mass / (k_spring * d0_m**2)
print(f"\ndC/da analytique : {dC_da_ana*1e15/g_si:.4f} fF/g")

# ================================================================
#  FIGURES
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- 1. Potentiel ---
ax = axes[0]
im = ax.contourf(
    np.linspace(0, Ly, ny),
    np.linspace(0, Lz, nz),
    V_2d, levels=60, cmap='plasma'
)
plt.colorbar(im, ax=ax, label='V [V]')
ax.contour(
    np.linspace(0, Ly, ny),
    np.linspace(0, Lz, nz),
    V_2d, levels=15, colors='white', linewidths=0.5, alpha=0.5
)
ax.set_xlabel('y [µm]'); ax.set_ylabel('z [µm]')
ax.set_title('Potentiel V(y,z)')
ax.set_aspect('equal')

# Marque les doigts
for i, y_pos in enumerate(centres):
    color = 'cyan' if i % 2 == 0 else 'lime'
    ax.axvline(y_pos, color=color, lw=1.5, alpha=0.7,
               label='stator' if i == 0 else ('rotor' if i == 1 else ''))
ax.legend(fontsize=8)

# --- 2. Potentiel + champ ---
ax = axes[1]
Y2d, Z2d = np.meshgrid(np.linspace(0, Ly, ny), np.linspace(0, Lz, nz))
Ey_2d = Ey[:, x_slice, :]
Ez_2d = Ez[:, x_slice, :]

im2 = ax.contourf(Y2d, Z2d, V_2d, levels=60, cmap='plasma')
plt.colorbar(im2, ax=ax, label='V [V]')

n_arrows = 20
step_y = max(1, ny // n_arrows)
step_z = max(1, nz // n_arrows)
ax.quiver(
    Y2d[::step_z, ::step_y], Z2d[::step_z, ::step_y],
    Ey_2d[::step_z, ::step_y], Ez_2d[::step_z, ::step_y],
    color='white', alpha=0.8, scale=None, width=0.005,
)
ax.set_xlabel('y [µm]'); ax.set_ylabel('z [µm]')
ax.set_title('Champ E(y,z)')
ax.set_aspect('equal')

# --- 3. C(a) et dC/da ---
ax = axes[2]
valid = ~np.isnan(caps)
ax.plot(accs[valid], caps[valid] * 1e15, 'o-', color='steelblue', label='C(a)')
ax.set_xlabel('Accélération [g]')
ax.set_ylabel('C [fF]', color='steelblue')
ax2 = ax.twinx()
ax2.plot(accs[valid], dC_da[valid] * 1e15 / g_si, 's--', color='crimson', label='dC/da')
ax2.set_ylabel('dC/da [fF/g]', color='crimson')
ax.set_title('C(a) et sensibilité dC/da')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



















# g_si = 9.81  # m/s²
# d0   = Y_gap * 1e-6  # gap nominal [m]

# accelerations = np.linspace(-30, 30, 13)  # [g]
# capacitances  = []

# for a_g in accelerations:
#     # Mécanique
#     x   = m * (a_g * g_si) / k        # déplacement [m]
#     d_a = d0 - x                       # nouveau gap [m]

#     if d_a <= 0:
#         print(f"Collision à a = {a_g} g !")
#         capacitances.append(np.nan) 
#         continue

#     # Modifier la géométrie et résoudre Laplace
#     # (tu changes Y_gap = d_a*1e6 et tu relances la simulation)
#     # ...
#     # C_a = compute_C(potential_2d, vol_stator_2d, res, L_finger, V0, epsilon_0)
#     # capacitances.append(C_a)

# capacitances = np.array(capacitances)

# # Dérivée numérique : différence finie centrée
# da = np.diff(accelerations)[0] * g_si  # [m/s²]
# dC_da = np.gradient(capacitances, accelerations * g_si)  # [F·s²/m]

# print(f"dC/da à a=0 : {dC_da[len(dC_da)//2] * 1e15:.4f} fF·s²/m")
# print(f"           = {dC_da[len(dC_da)//2] * 1e15 / g_si:.4f} fF/g")