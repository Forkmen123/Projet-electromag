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

# ri = np.linspace(0,1,500)

# liste = []
# C_tot = 0
# for i in range(int(N/2)): # on fait / 2 parce que c'est symétrique
#     for j in range(i, N):
#         liste.append((i, j))
#         distance = abs(j-i) * (D / 2)
#         delta_d = (m * a) / (k)
#         C_tot += (l * h * epsilon_0) / (distance - delta_d)
                    
# print(sp.simplify(C_tot))





