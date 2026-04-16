import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

# Constantes
N = 8
D = 4e-6
h = 3e-6
l = 5e-5
m = 3e-6
k = 2
epsilon_0 = 8.854e-12
resolution = 0.25  # résolution [µm/pixel]
iters = 20

N = 100
grid = np.zeros((N,N,N))+0.5

grid[30:70,30:70,20] = 1
grid[30:70,30:70,80] = 0
mask_pos = grid==1
mask_neg = grid==0

yv, xv, zv = np.meshgrid(np.arange(N),np.arange(N),np.arange(N))
grid = 1-zv/100

kern = generate_binary_structure(3,1).astype(float)/6
kern[1,1,1] = 0

def neumann(a):
    a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
    a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
    a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]
    return a


class Geometry2D:
    def __init__(self, d_gap, t_finger, L_finger, n_stator, n_rotor, res, margin):
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
        self.d_gap    = d_gap
        self.t_finger = t_finger
        self.L_finger = L_finger
        self.n_stator = n_stator
        self.n_rotor  = n_rotor
        self.res      = res
        self.margin   = margin

        # Construction des doigts le long de y
        self.fingers = []
        y_pos = margin

        types = []
        while len(types) < n_stator + n_rotor:
            if len([t for t in types if t == 0]) < n_stator:
                types.append(0)   # 0 = stator (+V0)
            if len([t for t in types if t == 1]) < n_rotor:
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

        print(f"Géométrie MEMS 2D  —  plan (y, z)")
        print(f"─" * 50)
        print(f"  y : espacement / mouvement   [horizontal]")
        print(f"  z : longueur des doigts      [vertical]")
        print(f"  x : épaisseur, hors-plan     [multiplicateur]")
        print(f"  Stator  : {n_stator} doigt(s)  (+V₀, fixes)")
        print(f"  Rotor   : {n_rotor} doigt(s)  (0 V, se déplace selon y)")
        print(f"  d_gap   : {d_gap:.2f} μm")
        print(f"  t_finger: {t_finger:.2f} μm")
        print(f"  L_finger: {L_finger:.2f} μm")
        print(f"  Domaine : Ly={self.Ly:.1f} μm × Lz={self.Lz:.1f} μm")
        print(f"  Grille  : ny={self.ny} × nz={self.nz}  pixels")
        print(f"  Résol.  : {res:.2f} μm/pixel")
        print()

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



y_axis, z_axis = geom.get_axes()

plt.contourf(y_axis, z_axis, V_solution, levels=50, cmap='RdBu_r')
plt.xlabel('y [μm]  ← direction du mouvement')
plt.ylabel('z [μm]  ← longueur des doigts')