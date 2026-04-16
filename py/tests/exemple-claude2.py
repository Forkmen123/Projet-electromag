"""
PÉDAGOGIE : Comment obtenir dC/da à partir de la physique
===========================================================

PHY-1007 · Projet capteur MEMS capacitif · Université Laval

Cet exemple montre explicitement chaque étape de la chaîne physique :
  1. Accélération a [g]  →  2. Déplacement x [m]  →  3. Gap d(a) [m]
       ↓ (mécanique)               ↓ (géométrie)         ↓ (électrostatique)
  4. Potentiel V(x,z)  →  5. Champ E(x,z)  →  6. Charge Q  →  7. Capacité C(a)
       ↓ (Laplace)              ↓ (gradient)       ↓ (Gauss)       ↓
  8. Sensibilité dC/da

Le code calcule numériquement ce qu'il faut mettre dans le rapport.
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
#  ÉTAPE 1 & 2 : MÉCANIQUE (accélération → déplacement)
# ================================================================
print("="*70)
print(" CHAÎNE PHYSIQUE COMPLÈTE : a → x → d → V → E → Q → C → dC/da")
print("="*70)

# Paramètres
m = 1.0e-9     # masse d'épreuve [kg]
k = 1.0        # constante de rappel [N/m]
g_si = 9.81    # g en m/s² [m/s²]

# Équation du mouvement : F = -kx = ma   →   x = ma/k
print("\n" + "─"*70)
print(" 1. MÉCANIQUE (masse-ressort)")
print("─"*70)
print(f"\nÉquation : F = -kx = ma")
print(f"Solution : x(a) = ma/k")
print(f"\nParamètres :")
print(f"  m = {m*1e9:.2f} µg  [masse d'épreuve]")
print(f"  k = {k:.2f} N/m  [ressort]")

a_test = 10.0  # [g]
x_test = m * a_test * g_si / k  # [m]
print(f"\nExemple : si a = {a_test} g")
print(f"  x = {m*1e9:.2f}e-9 kg × {a_test} × {g_si} m/s² / {k} N/m")
print(f"    = {x_test*1e9:.3f} nm  [déplacement de la masse]")

# ================================================================
#  ÉTAPE 3 : GÉOMÉTRIE (déplacement → gap)
# ================================================================
print("\n" + "─"*70)
print(" 2. GÉOMÉTRIE (gap change)")
print("─"*70)

d0 = 2.0e-6    # gap nominal [m]
print(f"\nGap au repos : d₀ = {d0*1e6:.2f} µm")
print(f"\nLors d'une accélération a, la masse se déplace de x :")
print(f"  d(a) = d₀ − x(a) = d₀ − ma/k")
print(f"\nPour a = {a_test} g :")
d_test = d0 - x_test
print(f"  d = {d0*1e6:.2f} − {x_test*1e6:.3f} = {d_test*1e6:.3f} µm")

# ================================================================
#  ÉTAPE 4-6 : ÉLECTROSTATIQUE (V → E → Q)
# ================================================================
print("\n" + "─"*70)
print(" 3. ÉLECTROSTATIQUE (Laplace → Champ → Charge)")
print("─"*70)
print(f"\nDans l'espace entre les doigts :")
print(f"  ∇²V = 0   (aucune charge libre)")
print(f"  Conditions aux limites :")
print(f"    • Stator  : V = +V₀ = +3 V")
print(f"    • Rotor   : V = 0 V")
print(f"\nRésolution numérique (relaxation) → V(x,z) en chaque point")
print(f"  ↓")
print(f"Champ électrique : E(x,z) = −∇V")
print(f"  ↓")
print(f"Charge (théorème de Gauss) :")
print(f"  Q = ε₀ ∮ E·dA   (intégrale sur contour fermé autour du doigt)")
print(f"  ou via l'énergie :")
print(f"  u = (ε₀/2) ∫∫ |E|² dA  →  C = 2u/V₀²")

# ================================================================
#  ÉTAPE 7 : CAPACITÉ C(a)
# ================================================================
print("\n" + "─"*70)
print(" 4. CAPACITÉ C(a)")
print("─"*70)

eps0 = 8.854e-12  # [F/m]
L = 200e-6        # longueur des doigts [m]
t = 3e-6          # épaisseur [m]
N = 50            # nombre de paires
V0 = 3.0          # tension [V]

# Formule analytique (modèle simple sans effets de frange)
def C_analytical(a_g):
    """Capacité en utilisant le modèle ε₀A/d"""
    a_m = a_g * g_si
    x = m * a_m / k
    d = d0 - x
    
    if d <= 0:
        return np.nan  # collision
    
    return N * eps0 * L * t / d

# Calcul analytique
C0_ana = C_analytical(0)
C_test_ana = C_analytical(a_test)

print(f"\nModèle analytique (ε₀A/d) :")
print(f"  C(a) = N × ε₀ × L × t / d(a)")
print(f"       = {N} × {eps0:.3e} × {L*1e6:.0f}×10⁻⁶ × {t*1e6:.0f}×10⁻⁶ / (d₀ − ma/k)")
print(f"\nExemples :")
print(f"  C(0)    = {C0_ana*1e15:.3f} fF")
print(f"  C({a_test} g) = {C_test_ana*1e15:.3f} fF")
print(f"  ΔC = C({a_test} g) − C(0) = {(C_test_ana - C0_ana)*1e15:.3f} fF")

# ================================================================
#  ÉTAPE 8 : SENSIBILITÉ dC/da
# ================================================================
print("\n" + "─"*70)
print(" 5. SENSIBILITÉ dC/da")
print("─"*70)

print(f"\nMéthode 1 : ANALYTIQUE (dérivée formelle de ε₀A/d)")
print(f"─" * 70)

# Dérivation formelle
# C(a) = N ε₀ L t / (d₀ − ma/k)
# Posons u = d₀ − ma/k, alors C = cste/u, dC/du = −cste/u²
# dC/da = (dC/du) × (du/da) = −(−cste/u²) × (−m/k) = −cste × m/k / u²
# À a = 0 : dC/da = −cste × m/k / d₀² = N ε₀ L t m / (k d₀²)

dCda_ana = N * eps0 * L * t * m / (k * d0**2)

print(f"\nDérivation de C(a) = N ε₀ L t / (d₀ − ma/k) par rapport à a :")
print(f"\n  dC/da = N ε₀ L t × (m/k) / (d₀ − ma/k)²")
print(f"\nÀ a = 0 (linéarisation autour du repos) :")
print(f"\n  dC/da|ₐ₌₀ = N ε₀ L t × m / (k × d₀²)")
print(f"\nCalcul numérique :")
print(f"  = {N} × {eps0:.3e} × {L:.3e} × {t:.3e} × {m:.3e}")
print(f"    / ({k} × ({d0:.3e})²)")
print(f"  = {dCda_ana:.6e} F·s²/m")
print(f"  = {dCda_ana * 1e15:.4f} fF·s²/m  (par unité de longueur)")
print(f"  = {dCda_ana / g_si * 1e15:.4f} fF/g")

print(f"\nInterpétation :")
print(f"  • Chaque g d'accélération augmente la capacité de ~{dCda_ana/g_si*1e15:.3f} fF")
print(f"  • La réponse est quasi-linéaire pour |a| < ~20 g (tant que d(a) >> 0)")

print(f"\n" + "─"*70)
print(f"Méthode 2 : NUMÉRIQUE (différence finie)")
print(f"─"*70)

# Calcul de C(a) sur un balayage
a_sweep = np.linspace(-30, 30, 121)  # 121 points pour résolution fine
C_sweep = np.array([C_analytical(a) if not np.isnan(C_analytical(a)) else np.nan 
                    for a in a_sweep])

# Masque pour éviter les NaN (collisions)
mask = ~np.isnan(C_sweep)
a_valid = a_sweep[mask]
C_valid = C_sweep[mask]

# Différence finie centrée au point a ≈ 0
idx_zero = np.argmin(np.abs(a_valid))
if idx_zero > 0 and idx_zero < len(a_valid) - 1:
    da_g = a_valid[idx_zero + 1] - a_valid[idx_zero - 1]  # [g]
    dC_fF = (C_valid[idx_zero + 1] - C_valid[idx_zero - 1]) * 1e15  # [fF]
    dCda_num = dC_fF / da_g  # [fF/g]
    
    print(f"\nDifférence finie centrée autour de a ≈ 0 :")
    print(f"  a[{idx_zero-1}] = {a_valid[idx_zero-1]:+.1f} g,  C = {C_valid[idx_zero-1]*1e15:.3f} fF")
    print(f"  a[{idx_zero}]   = {a_valid[idx_zero]:+.1f} g,  C = {C_valid[idx_zero]*1e15:.3f} fF")
    print(f"  a[{idx_zero+1}] = {a_valid[idx_zero+1]:+.1f} g,  C = {C_valid[idx_zero+1]*1e15:.3f} fF")
    print(f"\n  dC/da ≈ (C[{idx_zero+1}] − C[{idx_zero-1}]) / (a[{idx_zero+1}] − a[{idx_zero-1}])")
    print(f"       ≈ ({C_valid[idx_zero+1]*1e15:.3f} − {C_valid[idx_zero-1]*1e15:.3f}) / {da_g:.1f}")
    print(f"       = {dCda_num:.4f} fF/g")

# Pente moyenne sur la plage linéaire
linear_mask = np.abs(a_valid) <= 15
if np.sum(linear_mask) > 2:
    dCda_linear = np.polyfit(a_valid[linear_mask], C_valid[linear_mask]*1e15, 1)[0]
    print(f"\nRégression linéaire sur |a| ≤ 15 g :")
    print(f"  dC/da = {dCda_linear:.4f} fF/g")
    print(f"  R² = {np.corrcoef(a_valid[linear_mask], C_valid[linear_mask]*1e15)[0,1]**2:.6f}")

# ================================================================
#  VISUALISATION
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- Graphe 1 : C(a) ---
ax = axes[0]
ax.plot(a_valid, C_valid * 1e15, 'o-', color='steelblue', ms=4, lw=2, label='C(a)')
ax.axhline(C0_ana * 1e15, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
ax.fill_between(a_valid[linear_mask], 0, np.max(C_valid)*1e15*1.1, 
                alpha=0.1, color='green', label='Région linéaire')
ax.set_xlabel('Accélération [g]')
ax.set_ylabel('Capacité [fF]')
ax.set_title(f'C(a)\nd₀={d0*1e6:.1f} µm, k={k} N/m, m={m*1e9:.1f} µg')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Graphe 2 : Gap d(a) ---
ax = axes[1]
d_sweep = d0 - m * a_valid * g_si / k
ax.plot(a_valid, d_sweep * 1e6, 'o-', color='darkorange', ms=4, lw=2)
ax.axhline(d0*1e6, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
ax.set_xlabel('Accélération [g]')
ax.set_ylabel('Gap [µm]')
ax.set_title(f'd(a) = d₀ − ma/k')
ax.grid(True, alpha=0.3)

# --- Graphe 3 : Sensibilité dC/da ---
ax = axes[2]
dCda_array = np.gradient(C_valid * 1e15, a_valid)  # dérivée numérique
ax.plot(a_valid, dCda_array, 'o-', color='crimson', ms=4, lw=2, label='dC/da numérique')
ax.axhline(dCda_ana / g_si * 1e15, color='black', ls='--', lw=2, label='dC/da analytique')
ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
ax.fill_between(a_valid[linear_mask], 0, np.max(dCda_array)*1.1, 
                alpha=0.1, color='green')
ax.set_xlabel('Accélération [g]')
ax.set_ylabel('dC/da [fF/g]')
ax.set_title('Sensibilité dC/da')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mems_pedagogie.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n\n" + "="*70)
print(" RÉSUMÉ POUR LE RAPPORT")
print("="*70)
print(f"\nQuestion 1 : Sensibilité dC/da")
print(f"─" * 70)
print(f"  Analytique (ε₀A/d) :  dC/da = {dCda_ana/g_si*1e15:.4f} fF/g")
if 'dCda_num' in locals():
    print(f"  Numérique (diff. finie) : dC/da = {dCda_num:.4f} fF/g")
print(f"\nCettes valeurs correspondent à l'équation linéarisée :")
print(f"  C(a) ≈ C₀ + (dC/da) × a   pour |a| petit")
print(f"\nVérification :")
print(f"  À a = {a_test} g → C ≈ {C0_ana*1e15:.3f} + {dCda_ana/g_si*1e15:.4f} × {a_test}")
print(f"                = {(C0_ana + dCda_ana/g_si*a_test)*1e15:.3f} fF")
print(f"  Valeur exacte :        {C_test_ana*1e15:.3f} fF")

# Mesure de voltage
dV = V0 * (C0_ana / C_test_ana - 1)
print(f"\nQuestion 1b : Mesure de ΔV")
print(f"─" * 70)
print(f"  Condensateur chargé à V₀ = {V0} V puis déconnecté")
print(f"  Q = C₀ V₀ (constant)")
print(f"  V(a) = Q / C(a) = V₀ × C₀/C(a)")
print(f"  ΔV(a) = V(a) − V₀ = V₀ × (C₀/C(a) − 1)")
print(f"\n  À a = {a_test} g :")
print(f"  ΔV = {V0} × ({C0_ana*1e15:.3f} / {C_test_ana*1e15:.3f} − 1)")
print(f"     = {dV*1e3:.2f} mV")
print(f"\nSensibilité ΔV/a ≈ {dV*1e3 / a_test:.2f} mV/g")

print("\nFigure sauvegardée : mems_pedagogie.png")