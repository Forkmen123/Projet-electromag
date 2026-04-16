import numpy as np
import matplotlib.pyplot as plt

# --- cosntantes ---
N_pairs = 8
L_finger = 40.0
X_width = 10.0
Y_gap = 26.0
res = 1 # résolution 
margin = 1
iters = 200 # nombre d'itérations 

centres = (
    margin + np.arange(2 * N_pairs) * Y_gap
)  # position en y des centres des doigts
Ly, Lx, Lz = centres[-1] + margin, X_width + 2 * margin, L_finger + 2 * margin
ny, nx, nz = (
    int(Ly / res),
    int(Lx / res),
    int(Lz / res),
)  # conversion en nombres de pixels

vol_stator = np.zeros((nz, nx, ny), dtype=bool) # genre de matrice en 3D (tenseur)
vol_rotor = np.zeros((nz, nx, ny), dtype=bool)

# indices limites 
x_s, x_e = int(margin / res), int((margin + X_width) / res)
y_s, y_e = int(margin / res), int(centres[-1] / res)
z_s, z_e = int(margin / res), int((margin + L_finger) / res) # indices en z start, z end...

slice_z = slice(z_s, z_e) # on prend une coupe en z 
slice_x = slice(x_s, x_e)
slice_y = slice(y_s, y_e + 1)

# base stator en rouge 
vol_stator[z_s, slice_x, slice_y] = True # on ajoute une plaque à la base
# base rotor en bleu 
vol_rotor[z_e, slice_x, slice_y] = True

# --- 3. Ajout des doigts (lames minces) ---


for i, y_pos in enumerate(centres):
    y_idx = int(y_pos / res)
    if i % 2 == 0: # on alterne rouge / bleu
        vol_stator[z_s:z_e, slice_x, y_idx] = True  
    else:
        vol_rotor[z_s + 1 : z_e + 1, slice_x, y_idx] = True  


# pour le potentiel 

V_stator, V_rotor = 100, 0
potential = np.zeros((nz, nx, ny))
potential[vol_stator] = V_stator
potential[vol_rotor] = V_rotor

# Masque des points où le potentiel peut varier (le vide)
is_free = ~ (vol_stator | vol_rotor)

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

    if i % 5 == 0 :
        print(f'{i}/{iters}')

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
ax.set_box_aspect([nz, nx, ny])

# Choisir une couche en x (milieu du doigt par exemple)
x_slice = int(L_finger/ 2)

plt.figure(figsize=(10, 4))
plt.imshow(
    potential[x_slice, :, :],   # shape (ny, nz)
    cmap='RdBu_r',
    origin='lower',
    extent=[0, Lz, 0, 100],
    aspect='auto'
)
plt.colorbar(label="Potentiel V")
plt.xlabel("z [µm]")
plt.ylabel("y [µm]")
plt.title(f"Potentiel — coupe x = {x_slice}")
plt.tight_layout()
plt.show()