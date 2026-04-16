import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fast")  # fait en sorte que ça devrait moins bugger


# --------------- CONSTANTES ---------------- en µm
# N_pairs = 20 # nombre de paires de doigts [µm] 50
N_pairs = 4  # nombre de paires de doigts [µm] 50
Y_gap = 4  # gap size entre les doigts [µm] (d)
X_width = 2  # largeur des doigts [µm] (h)
L_finger = 100  # longueur des doigts [µm] (l)
m = 0.002  # masse d'épreuve [kg]
k = 2  # constante du ressort [N/m]
epsilon_0 = 8.854e-12
# res = 2  # résolution [µm/pixel]
res = 0.3  # résolution [µm/pixel]
iters = 2000  # nombre d'itérations voulues
V0 = 3  # potentiel stator arbitraire


# ------------------- GÉOMÉTRIE -------------------------
centres = Y_gap / 2 + np.arange(2 * N_pairs) * Y_gap
Ly = centres[-1] + Y_gap / 2
ny = int(round(Ly / res))
nx = max(int(round(X_width / res)), 2) 
nz = int(round(L_finger / res))  

vol_stator = np.zeros((nz, nx, ny), dtype=bool)  # genre de matrice en 3D (tenseur)
vol_rotor = np.zeros((nz, nx, ny), dtype=bool)

slice_x = slice(0, nx)
z_0 = 0
z_e = nz

def modify_center(delta_d):
    vol_stator[:] = False  # reset 
    vol_rotor[:]  = False
    for i, y_pos in enumerate(centres):
        if i % 2 == 0:
            y_idx = np.clip(int(round(y_pos / res)), 0, ny - 1)  
            vol_stator[z_0:z_e, slice_x, y_idx] = True
        else:
            y_idx = np.clip(int(round((y_pos + delta_d) / res)), 0, ny - 1)  
            vol_rotor[z_0:z_e, slice_x, y_idx] = True
    
def make_pot(show_iters=True):
    potential = np.full((nz, nx, ny), V0 / 2.0)
    potential[vol_stator] = V0
    potential[vol_rotor] = 0

    # Masque des points où le potentiel peut varier (le vide)
    is_free = ~(vol_stator | vol_rotor)

    # --- 5. Boucle de Relaxation ---
    for i in range(iters):  # Nombre d'itérations
        v_avg = (
            np.roll(potential, 1, axis=0)
            + np.roll(potential, -1, axis=0)
            + np.roll(potential, 1, axis=1)
            + np.roll(potential, -1, axis=1)
            + np.roll(potential, 1, axis=2)
            + np.roll(potential, -1, axis=2)
        ) / 6.0

        potential[is_free] = v_avg[is_free]

        potential[0, :, :] = 0
        potential[-1, :, :] = V0

        if i % 5 == 0 and show_iters:
            print(f"{i}/{iters}")

    return potential

def make_elec(potential):
    # on met les variables en mètres
    z_axis_m = np.linspace(0, L_finger * 1e-6, nz)
    x_axis_m = np.linspace(0, X_width * 1e-6, nx)
    y_axis_m = np.linspace(0, Ly * 1e-6, ny)

    # Champ E en V/m
    grad_z, grad_x, grad_y = np.gradient(potential, z_axis_m, x_axis_m, y_axis_m)
    Ex = -grad_x  # shape (nz, nx, ny)
    Ey = -grad_y
    Ez = -grad_z

    return (Ex, Ey, Ez)

def find_capa(E):
    Ex, Ey, Ez = E
    dA = (res * 1e-6) ** 2  # aire d'une face [m²]

    free = ~vol_stator  # True = vide

    flux_pz = np.sum(Ez[1:,  :,  :] * dA * ( vol_stator[:-1, :,  :] & free[1:,  :,  :]))
    flux_mz = np.sum(Ez[:-1, :,  :] * dA * ( vol_stator[1:,  :,  :] & free[:-1, :,  :]))
    flux_px = np.sum(Ex[:,  1:,  :] * dA * ( vol_stator[:,  :-1, :] & free[:,  1:,  :]))
    flux_mx = np.sum(Ex[:, :-1,  :] * dA * ( vol_stator[:,  1:,  :] & free[:,  :-1, :]))
    flux_py = np.sum(Ey[:,  :,  1:] * dA * ( vol_stator[:, :, :-1] & free[:,  :,  1:]))
    flux_my = np.sum(Ey[:,  :, :-1] * dA * ( vol_stator[:, :,  1:] & free[:,  :, :-1]))

    # np.dot pour combiner les 6 composantes en un scalaire
    flux_vec = np.array([flux_pz, flux_mz, flux_px, flux_mx, flux_py, flux_my])
    signs    = np.array([1,       -1,       1,       -1,       1,       -1      ])
    flux_dot = np.dot(signs, flux_vec)  # ← np.dot ici

    Q = epsilon_0 * flux_dot
    C = abs(Q) / V0

    return C

def find_sensibility(delta_d):
    modify_center(delta_d)             # 1. On déplace le rotor en +delta_d
    pot_plus = make_pot()              # 2. On recalcule le potentiel
    E_plus = make_elec(pot_plus)       # 3. On recalcule le champ E
    C_plus = find_capa(E_plus)         # 4. On trouve la capacité C(+)

    modify_center(-delta_d)            # 1. On déplace le rotor en -delta_d
    pot_minus = make_pot()             # 2. On recalcule le potentiel
    E_minus = make_elec(pot_minus)     # 3. On recalcule le champ E
    C_minus = find_capa(E_minus)       # 4. On trouve la capacité C(-)

    modify_center(0)

    delta_d_meters = delta_d * 1e-6 
    
    # méthode dérivée numérique par différence finie
    dC_da = (C_plus - C_minus) / (2 * delta_d_meters)

    return dC_da

def main():
    # ==================== SIMULATION ===================== 
    print(f'dC/da : {find_sensibility(delta_d=1)}')
    potential = make_pot(show_iters=False)
    Ex, Ey, Ez = make_elec(potential)
    # ===================================================== 


    # ---------------   AFFICHAGE plaques 3d -------------------------- 
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")

    voxels = vol_stator | vol_rotor
    colors = np.empty(voxels.shape, dtype=object)
    colors[vol_stator] = "red"
    colors[vol_rotor] = "blue"

    ax3d.voxels(voxels, facecolors=colors, edgecolor=None, linewidth=0, alpha=1)
    ax3d.set_title("Géométrie 3D : Plaques et Doigts Interdigités")
    ax3d.set_xlabel("Z (Longueur)")
    ax3d.set_ylabel("X (Épaisseur)")
    ax3d.set_zlabel("Y (Déploiement)")
    # ax3d.set_box_aspect([nz, nx, ny])  # fait des carrés


    # ---------------   AFFICHAGE potentiel -------------------------- 
    x_slice = 1 # courbe de niveau
    V_2d = potential[:, x_slice, :]

    plt.figure(figsize=(10, 4))
    plt.imshow(
        V_2d,  # shape (ny, nz)
        cmap="plasma",
        origin="lower",
        extent=[0, Ly, 0, L_finger],
        aspect="equal",  # auto
    )
    plt.colorbar(label="Potentiel V")
    plt.xlabel("y [µm]")
    plt.ylabel("z [µm]")
    plt.title(f"Potentiel — coupe x = {x_slice}")
    plt.gca().set_aspect("equal", adjustable="box")  # pour faire des carrés


    # ---------------   AFFICHAGE potentiel + gradient -------------------------- 
    V_2d = potential[:, x_slice, :]  # (nz, ny)
    Ey_2d = Ey[:, x_slice, :]
    Ez_2d = Ez[:, x_slice, :]

    y_axis = np.linspace(0, Ly, ny)
    z_axis = np.linspace(0, L_finger, nz)

    Y, Z = np.meshgrid(y_axis, z_axis)

    # --- Figure : potentiel + champ superposés ---
    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.contourf(Y, Z, V_2d, levels=60, cmap="plasma")
    plt.colorbar(im, ax=ax, label="Potentiel V [V]")
    plt.gca().set_aspect("equal", adjustable="box")  # pour faire des carrés

    ax.contour(Y, Z, V_2d, levels=15, colors="white", linewidths=0.5, alpha=0.6)

    n_arrows = 100  # densité + flèches
    step_y = max(1, ny // n_arrows)
    step_z = max(1, nz // n_arrows)

    ax.quiver(
        Y[::step_z, ::step_y],
        Z[::step_z, ::step_y],
        Ey_2d[::step_z, ::step_y],
        Ez_2d[::step_z, ::step_y],
        color="white",
        alpha=0.8,
        scale=None,
        width=0.003,
        headwidth=2,
        headlength=4,
    )

    ax.set_xlabel("y [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title(f"Potentiel et champ E — coupe x = {x_slice}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()