import numpy as np
import matplotlib.pyplot as plt

# --- cosntantes ---
N_pairs = 2
L_finger = 40.0
X_width = 10.0
Y_gap = 26.0
res = 2 # résolution 
margin = 1

centres = (
    margin + np.arange(2 * N_pairs) * Y_gap
)  # position en y des centres des doigts
Ly, Lx, Lz = centres[-1] + margin, X_width + 2 * margin, L_finger + 2 * margin
ny, nx, nz = (
    int(Ly / res),
    int(Lx / res),
    int(Lz / res),
)  # conversion en nombres de poixels

vol_stator = np.zeros((nz, nx, ny), dtype=bool) # genre de matrice en 3D (tenseur)
vol_rotor = np.zeros((nz, nx, ny), dtype=bool)

# indices limites
z_s, z_e = int(margin / res), int((margin + L_finger) / res)
x_s, x_e = int(margin / res), int((margin + X_width) / res)
y_s, y_e = int(margin / res), int(centres[-1] / res)

slice_z = slice(z_s, z_e)
slice_x = slice(x_s, x_e)
slice_y = slice(y_s, y_e + 1)

# --- 2. Ajout des plaques de base (z=0 et z=L) ---
# Plaque Stator (Rouge) à la base z_s
vol_stator[z_s, slice_x, slice_y] = True

# Plaque Rotor (Bleue) à l'extrémité z_e
vol_rotor[z_e, slice_x, slice_y] = True

# --- 3. Ajout des doigts (lames minces) ---
for i, y_pos in enumerate(centres):
    y_idx = int(y_pos / res)
    if i % 2 == 0:
        vol_stator[z_s:z_e, slice_x, y_idx] = True  # Part du bas
    else:
        vol_rotor[z_s + 1 : z_e + 1, slice_x, y_idx] = True  # Part du haut

    # --- 4. Initialisation du Potentiel ---
V_stator, V_rotor = 100.0, 0.0
potential = np.zeros((nz, nx, ny))
potential[vol_stator] = V_stator
potential[vol_rotor] = V_rotor

# Masque des points où le potentiel peut varier (le vide)
is_free = ~(vol_stator | vol_rotor)

# --- 5. Boucle de Relaxation ---
for i in range(150):  # Nombre d'itérations
    v_avg = (
        np.roll(potential, 1, axis=0)
        + np.roll(potential, -1, axis=0)
        + np.roll(potential, 1, axis=1)
        + np.roll(potential, -1, axis=1)
        + np.roll(potential, 1, axis=2)
        + np.roll(potential, -1, axis=2)
    ) / 6.0
    potential[is_free] = v_avg[is_free]

    # Note: On peut ajouter des conditions aux bords du domaine ici si nécessaire


# --- 3. Affichage corrigé ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# On combine les volumes pour l'affichage
voxels = vol_stator | vol_rotor
colors = np.empty(voxels.shape, dtype=object)
colors[vol_stator] = "red"
colors[vol_rotor] = "blue"

# L'appel se fait sur 'ax', pas sur 'fig'
ax.voxels(voxels, facecolors=colors, edgecolor="k", linewidth=0, alpha=1)

# Configuration des titres et labels
ax.set_title("Géométrie 3D : Plaques et Doigts Interdigités")
ax.set_xlabel("Z (Longueur)")
ax.set_ylabel("X (Épaisseur)")
ax.set_zlabel("Y (Déploiement)")

# On ajuste l'aspect pour ne pas que ce soit écrasé
ax.set_box_aspect([nz, nx, ny])

plt.tight_layout()
plt.show()
