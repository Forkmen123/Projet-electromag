import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fast') # fait en sorte que ça devrait moins bugger

# Constantes ---- en µm
# N_pairs = 20 # nombre de paires de doigts [µm] 50
N_pairs = 4 # nombre de paires de doigts [µm] 50
Y_gap = 4 # gap size entre les doigts [µm] (d)
X_width = 2 # largeur des doigts [µm] (h)
L_finger = 100 # longueur des doigts [µm] (l)
m = 2 # masse d'épreuve [kg]
k = 2 # constante du ressort [N/m]
epsilon_0 = 8.854e-12 
# res = 2  # résolution [µm/pixel]
res = 0.3  # résolution [µm/pixel]
iters = 2000 # nombre d'itérations voulues 
margin = 0 #starting the graph at not zero


centres = (margin + np.arange(2 * N_pairs) * Y_gap)  # position en y des centres des doigts
Ly = centres[-1] + margin + Y_gap
Lx = X_width + 2 * margin + (1 * res if margin == 0 else 0)
Lz = L_finger + 2 * margin + (1 * res if margin == 0 else 0)
ny, nx, nz = (
    int(Ly / res),
    int(Lx / res),
    int(Lz / res),
)  # conversion en nombres de pixels

vol_stator = np.zeros((nz, nx, ny), dtype=bool) # genre de matrice en 3D (tenseur)
vol_rotor = np.zeros((nz, nx, ny), dtype=bool)

# indices limites 
x_s, x_e = 0, min(int(margin + X_width / res) , nx - 1)
y_s, y_e = 0, min(int(centres[-1] / res), ny - 1)
z_s, z_e = 0, min(int(margin + L_finger  / res), nz - 1) # indices en z start, z end...
# x_s, x_e = 0, nx - 1
# y_s, y_e = 0, ny - 1
# z_s, z_e = 0, nz # indices en z start, z end...

slice_x = slice(x_s, x_e + 1)
slice_y = slice(y_s, y_e + 1)

# base stator en rouge 
vol_stator[z_s, slice_x, slice_y] = True # on ajoute une plaque à la base
# base rotor en bleu 
vol_rotor[z_e, slice_x, slice_y] = True

for i, y_pos in enumerate(centres):
    y_idx = min(int(y_pos / res), ny - 1)
    if i % 2 == 0: # on alterne rouge / bleu
        vol_stator[z_s:z_e, slice_x, y_idx] = True  
    else:
        vol_rotor[z_s + 1 : z_e + 1, slice_x, y_idx] = True  


# pour le potentiel 

V_stator, V_rotor = 3, 0
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
for vol, color in [(vol_stator, 'red'), (vol_rotor, 'blue')]:
    z_idx, x_idx, y_idx = np.where(vol)
# ax.voxels(voxels, facecolors=colors, edgecolor=None, linewidth=0, alpha=1) 

# Configuration des titres et labels
ax.set_title("Géométrie 3D : Plaques et Doigts Interdigités")
ax.set_xlabel("Z (Longueur)")
ax.set_ylabel("X (Épaisseur)")
ax.set_zlabel("Y (Déploiement)")
ax.set_box_aspect([nz, nx, ny]) # fait des carrés 

# Choisir une couche en x (milieu du doigt par exemple)
x_slice = 1
V_2d = potential[:, x_slice, :]

plt.figure(figsize=(10, 4))
plt.imshow(
    V_2d,   # shape (ny, nz)
    cmap='plasma',
    origin='lower',
    extent=[0, Ly, 0, Lz],
    aspect='equal' # auto
)
plt.colorbar(label="Potentiel V")
plt.xlabel("y [µm]")
plt.ylabel("z [µm]")
plt.title(f"Potentiel — coupe x = {x_slice}")
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box') # pour faire des carrés 

 
# --- Champ électrique 3D ---
y_axis = np.linspace(0, Ly, ny)
z_axis = np.linspace(0, Lz, nz)
x_axis = np.linspace(0, Lx, nx)
 
# potential shape : (nz, nx, ny)
grad_z, grad_x, grad_y = np.gradient(potential, z_axis, x_axis, y_axis)
Ex = -grad_x
Ey = -grad_y
Ez = -grad_z
 
# --- Coupe 2D au milieu en x ---
x_slice = nx // 2
V_2d  = potential[:, x_slice, :]  # (nz, ny)
Ey_2d = Ey       [:, x_slice, :]
Ez_2d = Ez       [:, x_slice, :]
 
Y, Z = np.meshgrid(y_axis, z_axis)
 
# --- Figure : potentiel + champ superposés ---
fig, ax = plt.subplots(figsize=(12, 5))
 
im = ax.contourf(Y, Z, V_2d, levels=60, cmap='plasma')
plt.colorbar(im, ax=ax, label='Potentiel V [V]')
plt.gca().set_aspect('equal', adjustable='box') # pour faire des carrés 
 
ax.contour(Y, Z, V_2d, levels=15, colors='white', linewidths=0.5, alpha=0.6)
 
n_arrows = 100  # densité + flèches 
step_y = max(1, ny // n_arrows)
step_z = max(1, nz // n_arrows)

ax.quiver(
    Y [::step_z, ::step_y],
    Z [::step_z, ::step_y],
    Ey_2d[::step_z, ::step_y],
    Ez_2d[::step_z, ::step_y],
    color='white', alpha=0.8,
    scale=None, width=0.003,
    headwidth=2, headlength=4,
)
 
ax.set_xlabel('y [µm]')
ax.set_ylabel('z [µm]')
ax.set_title(f'Potentiel et champ E — coupe x = {x_slice}')
plt.tight_layout()
plt.show()
 