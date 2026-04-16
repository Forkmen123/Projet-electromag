import numpy as np
import matplotlib.pyplot as plt

plt.style.use("fast")

# --------------- CONSTANTES ---------------- en µm
N_pairs = 4
Y_gap = 4
X_width = 2
L_finger = 100
m = 0.002
k = 2
epsilon_0 = 8.854e-12
res = 2
iters = 2000
V0 = 3
delta_d = 2

# ------------------- GÉOMÉTRIE -------------------------
centres = Y_gap / 2 + np.arange(2 * N_pairs) * Y_gap
Ly = centres[-1] + Y_gap / 2
ny = int(round(Ly / res))
nx = max(int(round(X_width / res)), 2)  # ✅ minimum 2 pour np.gradient
nz = int(round(L_finger / res))

vol_stator = np.zeros((nz, nx, ny), dtype=bool)
vol_rotor  = np.zeros((nz, nx, ny), dtype=bool)

slice_x = slice(0, nx)
z_0 = 0
z_e = nz

def modify_center(delta_d):
    vol_stator[:] = False  # ✅ reset avant remplissage
    vol_rotor[:]  = False
    for i, y_pos in enumerate(centres):
        if i % 2 == 0:
            y_idx = np.clip(int(round(y_pos / res)), 0, ny - 1)  # ✅ assigné avant usage
            vol_stator[z_0:z_e, slice_x, y_idx] = True
        else:
            y_idx = np.clip(int(round((y_pos + delta_d) / res)), 0, ny - 1)  # ✅
            vol_rotor[z_0:z_e, slice_x, y_idx] = True

def make_pot():
    potential = np.full((nz, nx, ny), V0 / 2.0)
    potential[vol_stator] = V0
    potential[vol_rotor]  = 0
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
        potential[0,  :, :] = 0
        potential[-1, :, :] = V0
        if i % 100 == 0:
            print(f"{i}/{iters}")

    return potential  # ✅ hors de la boucle

def make_elec(potential):  # ✅ reçoit le potentiel déjà calculé
    z_axis_m = np.linspace(0, L_finger * 1e-6, nz)
    x_axis_m = np.linspace(0, X_width * 1e-6, nx)
    y_axis_m = np.linspace(0, Ly * 1e-6, ny)

    grad_z, grad_x, grad_y = np.gradient(potential, z_axis_m, x_axis_m, y_axis_m)
    Ex = -grad_x
    Ey = -grad_y
    Ez = -grad_z
    return Ex, Ey, Ez

def find_capa(E):
    Ex, Ey, Ez = E
    dA   = (res * 1e-6) ** 2
    free = ~vol_stator

    flux_pz = np.sum(Ez[1:,  :,  :] * dA * (vol_stator[:-1, :,  :] & free[1:,  :,  :]))
    flux_mz = np.sum(Ez[:-1, :,  :] * dA * (vol_stator[1:,  :,  :] & free[:-1, :,  :]))
    flux_px = np.sum(Ex[:,  1:,  :] * dA * (vol_stator[:,  :-1, :] & free[:,  1:,  :]))
    flux_mx = np.sum(Ex[:, :-1,  :] * dA * (vol_stator[:,  1:,  :] & free[:,  :-1, :]))
    flux_py = np.sum(Ey[:,  :,  1:] * dA * (vol_stator[:, :, :-1] & free[:,  :,  1:]))
    flux_my = np.sum(Ey[:,  :, :-1] * dA * (vol_stator[:, :,  1:] & free[:,  :, :-1]))

    flux_vec = np.array([flux_pz, flux_mz, flux_px, flux_mx, flux_py, flux_my])
    signs    = np.array([1, -1, 1, -1, 1, -1])
    flux_dot = np.dot(signs, flux_vec)

    Q = epsilon_0 * flux_dot
    C = abs(Q) / V0
    print(f"Flux = {flux_dot:.4e} V·m | Q = {Q:.4e} C | C = {C:.4e} F")
    return C

def display():
    modify_center(delta_d)          # ✅ appelée en premier

    potential = make_pot()          # ✅ un seul appel
    Ex, Ey, Ez = make_elec(potential)  # ✅ on passe le potentiel
    C = find_capa((Ex, Ey, Ez))

    # --- 3D ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    voxels = vol_stator | vol_rotor
    colors = np.empty(voxels.shape, dtype=object)
    colors[vol_stator] = "red"
    colors[vol_rotor]  = "blue"
    ax.voxels(voxels, facecolors=colors, edgecolor=None, linewidth=0, alpha=1)  # ✅ hors du for
    ax.set_title("Géométrie 3D")
    ax.set_xlabel("Z (Longueur)")
    ax.set_ylabel("X (Épaisseur)")
    ax.set_zlabel("Y (Déploiement)")
    ax.set_box_aspect([nz, nx, ny])

    # --- Coupe 2D potentiel + champ ---
    x_slice = nx // 2
    V_2d  = potential[:, x_slice, :]
    Ey_2d = Ey[:, x_slice, :]
    Ez_2d = Ez[:, x_slice, :]

    y_axis = np.linspace(0, Ly, ny)
    z_axis = np.linspace(0, L_finger, nz)
    Y, Z   = np.meshgrid(y_axis, z_axis)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.contourf(Y, Z, V_2d, levels=60, cmap="plasma")
    plt.colorbar(im, ax=ax, label="Potentiel V [V]")
    ax.contour(Y, Z, V_2d, levels=15, colors="white", linewidths=0.5, alpha=0.6)

    step_y = max(1, ny // 20)
    step_z = max(1, nz // 20)
    ax.quiver(
        Y[::step_z, ::step_y], Z[::step_z, ::step_y],
        Ey_2d[::step_z, ::step_y], Ez_2d[::step_z, ::step_y],
        color="white", alpha=0.8, scale=None, width=0.003,
    )
    ax.set_xlabel("y [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title(f"Potentiel et champ E — coupe x = {x_slice}")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display()