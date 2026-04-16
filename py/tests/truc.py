import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres ---
N_pairs = 2 
L_finger = 40.0       # Longueur (Z)
X_width = 10.0        # Largeur de la plaque (X)
Y_gap = 6.0           # Espacement entre lames (Y)
res = 1.0             
margin = 5.0

# --- 1. Calcul des dimensions ---
y_centers = margin + np.arange(2 * N_pairs) * Y_gap
Ly, Lx, Lz = y_centers[-1] + margin, X_width + 2*margin, L_finger + 2*margin
ny, nx, nz = int(Ly/res), int(Lx/res), int(Lz/res)

# --- 2. Création des volumes ---
vol_stator = np.zeros((nz, nx, ny), dtype=bool)
vol_rotor  = np.zeros((nz, nx, ny), dtype=bool)

# Définition des tranches pour les plaques (Z et X)
slice_z = slice(int(margin/res), int((margin + L_finger)/res))
slice_x = slice(int(margin/res), int((margin + X_width)/res))

# --- 3. Remplissage (Épaisseur Y = 1 pixel) ---
for i, y_pos in enumerate(y_centers):
    y_idx = int(y_pos / res)
    
    # On sélectionne le volume (Stator si i est pair, Rotor si impair)
    target_vol = vol_stator if i % 2 == 0 else vol_rotor
    
    # On remplit une surface (Z, X) à une position Y fixe
    target_vol[slice_z, slice_x, y_idx] = True

voxels = vol_stator | vol_rotor # | est OR en logique binaire
colors = np.empty(voxels.shape, dtype=object)
colors[vol_stator] = 'red'
colors[vol_rotor]  = 'blue'

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0)

ax.set_box_aspect([nz, nx, ny]) 
ax.set_xlabel('Z (Longueur)')
ax.set_ylabel('X (Largeur)')
ax.set_zlabel('Y (Positions des lames)')
plt.title("Lames infiniment minces (1 pixel d'épaisseur en Y)")
plt.show()