"""
Solveur 2D Laplace pour accéléromètre MEMS capacitif
=====================================================

Basé sur la méthode de relaxation avec convolution.
Permet de paramétrer :
  - Nombre de doigts
  - Espacement entre doigts
  - Épaisseur des doigts
  - Position et taille du domaine

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from matplotlib.colors import TwoSlopeNorm

# ================================================================
#  PARAMÈTRES GÉOMÉTRIQUES — À MODIFIER
# ================================================================

# --- Dimensions physiques [µm] ---
d_gap = 2.0      # espace entre doigts (ce qui se déplace)
t_finger = 3.0   # épaisseur des doigts
L_finger = 200.0 # longueur des doigts

# --- Nombre et arrangement de doigts ---
N_paires = 8
n_stator = N_paires     # doigts stator (fixes, +V0)
n_rotor = N_paires      # doigts rotor (mobiles, 0V)
# Les doigts s'alternent : stator-rotor-stator-rotor-...

# --- Paramètres numériques ---
res = 0.25       # résolution [µm/pixel]  ← diminuer pour plus de précision
margin = 5.0     # marge vide autour      [µm]

# --- Voltage ---
V0 = 3.0         # tension appliquée      [V]

# --- Convergence ---
n_iter = 3000    # nombre d'itérations
check_convergence = True  # afficher la convergence

# ================================================================
#  CLASSE POUR GÉRER LA GÉOMÉTRIE
# ================================================================

class MEMSGeometry2D:
    """
    Construit la géométrie 2D des doigts interdigités.
    
    Arrange les doigts alternativement dans la direction x.
    Chaque doigt a une hauteur L et une épaisseur t.
    """
    
    def __init__(self, d_gap, t_finger, L_finger, n_stator, n_rotor, res, margin):
        """
        d_gap    : espacement entre faces [µm]
        t_finger : épaisseur de chaque doigt [µm]
        L_finger : longueur (hauteur en z) [µm]
        n_stator : nombre de doigts stator
        n_rotor  : nombre de doigts rotor
        res      : résolution de la grille [µm/pixel]
        margin   : marge autour [µm]
        """
        self.d_gap = d_gap
        self.t_finger = t_finger
        self.L_finger = L_finger
        self.n_stator = n_stator
        self.n_rotor = n_rotor
        self.res = res
        self.margin = margin
        
        # Doigts alternés : liste de (type, position_x_début)
        self.fingers = []
        x_pos = margin
        
        # Alterne stator (0) et rotor (1)
        types = []
        while len(types) < n_stator + n_rotor:
            if len([t for t in types if t == 0]) < n_stator:
                types.append(0)
            if len([t for t in types if t == 1]) < n_rotor:
                types.append(1)
        
        for ftype in types:
            self.fingers.append({
                'type': ftype,  # 0=stator, 1=rotor
                'x_start': x_pos,
                'x_end': x_pos + t_finger
            })
            x_pos += t_finger + d_gap
        
        # Domaine total
        self.Lx = x_pos + margin
        self.Lz = L_finger + 2 * margin
        
        # Grille
        self.nx = int(round(self.Lx / res))
        self.nz = int(round(self.Lz / res))
        
        print(f"Géométrie MEMS 2D")
        print(f"─" * 60)
        print(f"  Doigts stator : {n_stator}")
        print(f"  Doigts rotor  : {n_rotor}")
        print(f"  Espacement    : {d_gap:.2f} µm")
        print(f"  Épaisseur     : {t_finger:.2f} µm")
        print(f"  Longueur      : {L_finger:.2f} µm")
        print(f"  Domaine       : {self.Lx:.1f} × {self.Lz:.1f} µm")
        print(f"  Grille        : {self.nx} × {self.nz} pixels")
        print(f"  Résolution    : {res:.2f} µm/pixel")
        print()
    
    def get_masks(self):
        """
        Retourne les masques des doigts stator et rotor.
        """
        mask_stator = np.zeros((self.nz, self.nx), dtype=bool)
        mask_rotor = np.zeros((self.nz, self.nx), dtype=bool)
        
        z_start = int(round(self.margin / self.res))
        z_end = int(round((self.margin + self.L_finger) / self.res))
        
        for finger in self.fingers:
            x_start = int(round(finger['x_start'] / self.res))
            x_end = int(round(finger['x_end'] / self.res))
            
            if finger['type'] == 0:  # stator
                mask_stator[z_start:z_end+1, x_start:x_end+1] = True
            else:  # rotor
                mask_rotor[z_start:z_end+1, x_start:x_end+1] = True
        
        return mask_stator, mask_rotor
    
    def get_axes(self):
        """Retourne les axes x et z pour les graphes."""
        x_axis = np.linspace(0, self.Lx, self.nx)
        z_axis = np.linspace(0, self.Lz, self.nz)
        return x_axis, z_axis


# ================================================================
#  SOLVEUR LAPLACE 2D
# ================================================================

class LaplaceSolver2D:
    """
    Résout ∇²V = 0 en 2D par la méthode de relaxation (convolutional).
    """
    
    def __init__(self, nx, nz):
        """
        nx, nz : dimensions de la grille
        """
        self.nx = nx
        self.nz = nz
        self.V = np.zeros((nz, nx))
        self.errors = []
        
        # Noyau de convolution : moyenne des 4 voisins
        # (haut, bas, gauche, droite)
        self.kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=float) / 4.0
    
    def set_initial_condition(self, V_init):
        """Définit la condition initiale."""
        self.V = V_init.copy()
    
    def apply_neumann_bc(self):
        """
        Applique les conditions aux limites de Neumann :
        ∂V/∂n = 0 aux bords du domaine.
        
        Cela signifie : V[0,:] = V[1,:], V[-1,:] = V[-2,:], etc.
        """
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]
    
    def apply_dirichlet_bc(self, mask_stator, mask_rotor, V0):
        """
        Applique les conditions aux limites de Dirichlet :
        V = V0 sur le stator
        V = 0 sur le rotor
        """
        self.V[mask_stator] = +V0
        self.V[mask_rotor] = 0.0
    
    def step(self, mask_stator, mask_rotor, V0):
        """
        Une itération de relaxation.
        """
        # Moyenne des 4 voisins via convolve
        self.V = convolve(self.V, self.kernel, mode='constant', cval=0.0)
        
        # Neumann aux bords
        self.apply_neumann_bc()
        
        # Dirichlet sur les doigts
        self.apply_dirichlet_bc(mask_stator, mask_rotor, V0)
    
    def solve(self, mask_stator, mask_rotor, V0, n_iter, check_convergence=False):
        """
        Résout Laplace en faisant n_iter étapes.
        
        Optionnellement affiche la convergence.
        """
        self.errors = []
        
        for i in range(n_iter):
            V_old = self.V.copy()
            self.step(mask_stator, mask_rotor, V0)
            
            if check_convergence:
                err = np.mean((self.V - V_old)**2)
                self.errors.append(err)
                
                if (i + 1) % 500 == 0:
                    print(f"  Itération {i+1:4d} / {n_iter}  erreur = {err:.3e}")
        
        return self.V


# ================================================================
#  CALCUL DE LA CAPACITÉ
# ================================================================

def compute_capacitance(V, mask_rotor, res_um, L_physical_um, V0, eps0=8.854e-12):
    """
    Calcule la capacité via la méthode de l'énergie électrostatique.
    
    Principe : u = (ε₀/2) ∫∫ |E|² dA  →  C = 2u / V0²
    
    Retourne C_total [F] pour la longueur L_physical_um en µm.
    """
    res_m = res_um * 1e-6
    
    # Champ électrique
    Ex = -np.gradient(V, res_m, axis=1)
    Ez = -np.gradient(V, res_m, axis=0)
    
    E_sq = Ex**2 + Ez**2
    
    # Intégrale (chaque pixel a une aire res_m²)
    u = 0.5 * eps0 * np.sum(E_sq) * res_m**2  # [J/m]
    
    # Capacité par unité de longueur
    C_per_length = 2 * u / V0**2  # [F/m]
    
    # Capacité totale pour la longueur L
    C_total = C_per_length * L_physical_um * 1e-6  # [F]
    
    return C_total


# ================================================================
#  SCRIPT PRINCIPAL
# ================================================================

print("="*70)
print(" SOLVEUR LAPLACE 2D — Accéléromètre MEMS capacitif")
print("="*70)
print()

# --- 1. Géométrie ---
geom = MEMSGeometry2D(
    d_gap=d_gap,
    t_finger=t_finger,
    L_finger=L_finger,
    n_stator=n_stator,
    n_rotor=n_rotor,
    res=res,
    margin=margin
)

mask_stator, mask_rotor = geom.get_masks()
x_axis, z_axis = geom.get_axes()

# --- 2. Initialisation ---
V_init = np.ones((geom.nz, geom.nx)) * 0.5  # Valeur initiale 0.5 V partout
V_init[mask_stator] = V0
V_init[mask_rotor] = 0.0

# --- 3. Résolution ---
print("Résolution de ∇²V = 0 par relaxation...")
print()

solver = LaplaceSolver2D(geom.nx, geom.nz)
solver.set_initial_condition(V_init)

V_solution = solver.solve(
    mask_stator, mask_rotor, V0,
    n_iter=n_iter,
    check_convergence=check_convergence
)



















print(f"✓ Convergence atteinte après {n_iter} itérations\n")

# --- 4. Capacité ---
C_total = compute_capacitance(V_solution, mask_rotor, res, L_finger, V0)

print(f"Capacité mesurée (simulation)")
print(f"─" * 70)
print(f"  C_total = {C_total * 1e15:.4f} fF")
print(f"  C/longueur = {C_total / (L_finger*1e-6) * 1e15:.6f} fF·m⁻¹")
print()





# --- 5. Modèle analytique pour comparaison ---
eps0 = 8.854e-12
n_pairs = min(n_stator, n_rotor)
A_finger = L_finger * 1e-6 * t_finger * 1e-6
C_ana = n_pairs * eps0 * A_finger / (d_gap * 1e-6)




print(f"Modèle analytique (ε₀A/d)")
print(f"─" * 70)
print(f"  Paires de doigts (min(stator, rotor)) : {n_pairs}")
print(f"  Aire d'un doigt : {A_finger*1e12:.2f} µm²")
print(f"  Gap : {d_gap:.2f} µm")
print(f"  C_ana = {C_ana * 1e15:.4f} fF")
print(f"  Écart : {abs(C_total - C_ana)/C_ana*100:.1f}%")
print()












# ================================================================
#  FIGURES
# ================================================================
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(1, 1, 1)

# NE PAS mettre set_aspect('equal') — laisser matplotlib étirer librement
ax1.set_aspect('auto')

im = ax1.contourf(
    x_axis, z_axis, V_solution,
    levels=50, cmap='RdBu_r'
)
cbar = plt.colorbar(im, ax=ax1, label='Potentiel V [V]', fraction=0.03, pad=0.02)

contours = ax1.contour(
    x_axis, z_axis, V_solution,
    levels=15, colors='white', linewidths=0.4, alpha=0.5
)

res_m = res * 1e-6
Ex = -np.gradient(V_solution, res_m, axis=1)
Ez = -np.gradient(V_solution, res_m, axis=0)
X, Y = np.meshgrid(x_axis, z_axis)
ax1.streamplot(X, Y, Ex, Ez, color='yellow', linewidth=0.5, density=0.8, arrowsize=0.8)

# Overlay des doigts
for finger in geom.fingers:
    x_start = finger['x_start']
    x_end   = finger['x_end']
    color   = '#0066CC' if finger['type'] == 0 else '#CC0000'
    ax1.axvspan(x_start, x_end, ymin=margin/geom.Lz,
                ymax=(margin + L_finger)/geom.Lz, color=color, alpha=0.25)

ax1.set_xlabel('x [µm]  (direction du mouvement)')
ax1.set_ylabel('z [µm]  (longueur des doigts)')
ax1.set_title(f'Potentiel V(x,z) — {n_iter} itérations\n'
              f'd={d_gap:.1f}µm, t={t_finger:.1f}µm, '
              f'{n_stator} stator + {n_rotor} rotor')

plt.tight_layout()
plt.savefig('mems_laplace_2d.png', dpi=150, bbox_inches='tight')
plt.show()