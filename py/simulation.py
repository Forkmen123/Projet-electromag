import matplotlib.pyplot as plt 
import numpy as np
import sympy as sp

# Constantes
N = 8
D = 4e-6
h = 3e-6
l = 50e-6
m = 3e-6
k = 2
epsilon_0 = 8.854e-12
a = sp.symbols('a')


Nx, Ny = 100, 100
V = np.zeros((Nx, Ny))


#ceci n'est qu'un test que j'ai essayé de faire avec claude -Anakin
V[0:3, 10:120] = 1.0
V[0:3, 20:130] = -1.0

for iteration in range(10000):
    V_old = V.copy()
    V[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] + 
                             V[1:-1, 2:] + V[1:-1, :-2])
    
    # re-enforce boundary conditions after each update
    V[0:3, 10:120] = 1.0
    V[0:3, 20:130] = -1.0
    
    # check convergence
    if np.max(np.abs(V - V_old)) < 1e-5:
        print(f"Converged at iteration {iteration}")
        break

plt.imshow(V, cmap='hot')
plt.colorbar(label='V')
plt.title('Potential field')
plt.show()







