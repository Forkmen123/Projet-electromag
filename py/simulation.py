import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

# Constantes ---- en µm
N = 8 # nombre de paires de doigts [µm]
D = 4 # gap size entre les doigts [µm] 
h = 3 # largeur des doigts [µm]
l = 20 # longueur des doigts [µm]
m = 3 # masse d'épreuve [kg]
k = 2 # constante du ressort 
epsilon_0 = 8.854e-12 
resolution = 0.5  # résolution [µm/pixel]
iters = 20 # nombre d'itérations voulues 
margin = 0
grid = np.zeros((N, N, N)) + 0.5


n_total = 2 * N
y_positions = margin + np.arange(n_total) * gap

# 2. Dimensions de la grille
Ly = y_positions[-1] + margin
Lz = L_finger + 2 * margin
ny, nz = int(Ly/res), int(Lz/res)

# 3. Création des masques (Grilles de z_pixels par y_pixels)
mask_stator = np.zeros((nz, ny), dtype=bool)
mask_rotor  = np.zeros((nz, ny), dtype=bool)

# Limites verticales en pixels
z_start, z_end = int(margin/res), int((margin + L_finger)/res)

# 4. Remplissage des lignes
for i, y_pos in enumerate(y_positions):
    y_idx = int(y_pos / res)
    if i % 2 == 0: # Index pair = Stator
        mask_stator[z_start:z_end, y_idx] = True
    else:          # Index impair = Rotor
        mask_rotor[z_start:z_end, y_idx] = True

# --- Affichage ---
plt.figure(figsize=(10, 4))
extents = [0, Ly, 0, Lz]

# On affiche les deux sur le même graphe
# 'None' pour le fond, et on utilise une couleur solide pour les lignes
plt.imshow(mask_stator, cmap='Blues', origin='lower', extent=extents, label='Stator')
plt.imshow(mask_rotor, cmap='Oranges', origin='lower', extent=extents, alpha=0.6)

plt.title(f"Modèle filaire : {N} paires de doigts (Lignes)")
plt.xlabel("Y (µm)")
plt.ylabel("Z (µm)")
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

class Geometry3D:
    def __init__(self, d_gap, t_finger, L_finger, n, res, margin):
        """
        Géométrie dans le plan (y, z) :
        y : direction du mouvement / espacement entre doigts  → axe horizontal
        z : longueur des doigts                               → axe vertical
        x : épaisseur des doigts, sort du plan                → multiplicateur

        Les doigts sont des plaques dans le plan (x, z).
        Dans la coupe 2D (y, z), ils apparaissent comme
        des bandes verticales séparées par d_gap selon y.

        Array shape : (nz, ny)   →   [lignes=z, colonnes=y]
        """
        self.d_gap = d_gap
        self.t_finger = t_finger
        self.L_finger = L_finger
        self.n = n
        self.res = res
        self.margin = margin

        # Construction des doigts le long de y
        self.fingers = []
        y_pos = margin

        types = []
        while len(types) < n + n:
            if len([t for t in types if t == 0]) < n:
                types.append(0)   # 0 = stator (+V0)
            if len([t for t in types if t == 1]) < n:
                types.append(1)   # 1 = rotor  (0 V)

        for ftype in types:
            self.fingers.append({
                'type':    ftype,
                'y_start': y_pos,
                'y_end':   y_pos + t_finger,
            })
            y_pos += t_finger + d_gap

        # Domaine total
        self.Ly = y_pos + margin          # direction horizontale (espacement)
        self.Lz = L_finger + 2 * margin   # direction verticale   (longueur)

        # Grille : shape = (nz, ny)
        self.ny = int(round(self.Ly / res))
        self.nz = int(round(self.Lz / res))
        print(self.fingers)
    
    def get_masks(self):
        """
        Construit les masques stator et rotor.

        Shape : (nz, ny)
        - lignes   = axe z  (longueur des doigts, vertical)
        - colonnes = axe y  (espacement, horizontal)

        Chaque doigt est une bande verticale :
        z de margin à (margin + L_finger)   → toute la hauteur du doigt
        y de y_start à y_end               → épaisseur t_finger
        """
        mask_stator = np.zeros((self.nz, self.ny), dtype=bool)
        mask_rotor  = np.zeros((self.nz, self.ny), dtype=bool)

        # Étendue en z (axe vertical = lignes) — identique pour tous les doigts
        z_start_px = int(round(self.margin / self.res))
        z_end_px   = int(round((self.margin + self.L_finger) / self.res))

        for finger in self.fingers:
            # Étendue en y (axe horizontal = colonnes)
            y_start_px = int(round(finger['y_start'] / self.res))
            y_end_px   = int(round(finger['y_end']   / self.res))

            # Clamp pour rester dans la grille
            y_start_px = np.clip(y_start_px, 0, self.ny - 1)
            y_end_px   = np.clip(y_end_px,   0, self.ny - 1)
            z_start_px = np.clip(z_start_px, 0, self.nz - 1)
            z_end_px   = np.clip(z_end_px,   0, self.nz - 1)

            if finger['type'] == 0:
                mask_stator[z_start_px:z_end_px+1, y_start_px:y_end_px+1] = True
            else:
                mask_rotor [z_start_px:z_end_px+1, y_start_px:y_end_px+1] = True

        return mask_stator, mask_rotor

    def get_axes(self):
        """Axes physiques pour les graphes."""
        y_axis = np.linspace(0, self.Ly, self.ny)   # horizontal
        z_axis = np.linspace(0, self.Lz, self.nz)   # vertical
        return y_axis, z_axis
   

geometry = Geometry3D(D/2, h, l, N, resolution, margin=0)
print(geometry.get_masks()[0])
print()
print(geometry.get_masks()[1])
# print(geometry.get_axes())
print()

mask_stator, mask_rotor = geometry.get_masks()  # appel ici une fois

# plt.figure(figsize=(6, 4))

plt.imshow(mask_stator, cmap='gray', origin='lower')
# plt.imshow(mask_rotor, cmap='gray', origin='lower')
plt.xlabel("Distance horizontale (μm)")
plt.ylabel("Distance horizontale (μm)")

plt.grid(color='lightgray', linestyle='-', linewidth=0.1)

plt.show()






# grid[30:70,30:70,20] = 1
# grid[30:70,30:70,80] = 0
# mask_pos = grid==1
# mask_neg = grid==0

# yv, xv, zv = np.meshgrid(np.arange(N),np.arange(N),np.arange(N))
# grid = 1-zv/100 # premier guess

# noyau = (float(1) / 6) * np.array(
#     [
#         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
#         [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#     ]
# )

# print(noyau)

# def neumann(a):
#     a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
#     a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
#     a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]
#     return a

# iters = 2000
# for i in range(iters):
#     grid_updated = convolve(grid,kern, mode='constant')
#     # Boundary conditions (neumann)
#     grid_updated = neumann(grid_updated)
#     # Boundary conditions (dirchlett)
#     grid_updated[mask_pos] = 1
#     grid_updated[mask_neg] = 0
#     # See what error is between consecutive arrays
#     grid = grid_updated


# x = np.linspace(0, 1, grid.shape[1])
# y = np.linspace(0, 1, grid.shape[0])

# slc = 40

# plt.figure(figsize=(6,5))

# CS = plt.contourf(x, y, grid[slc], levels=100, cmap='viridis')
# plt.colorbar(CS)

# plt.xlabel('y/y0')
# plt.ylabel('z/z0')

# plt.axvline(0.2, ymin=0.3, ymax=0.7, color='r')
# plt.axvline(0.8, ymin=0.3, ymax=0.7, color='g')

# plt.show()