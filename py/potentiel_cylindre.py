from numpy import pi, cos, sin, cosh, sinh
import numpy as np
import pylab as plt

# for latex rendering
font = {'family' : 'monospace',
#         'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
#plt.rc('font', family='serif',size=22.)
# plt.rc('text', usetex=True)

ri = np.linspace(0,1,500)
re = np.linspace(1,3,500)
phi = np.linspace(0.,2.*pi,500)
Ri, PHI = np.meshgrid(ri,phi)
Re, PHI = np.meshgrid(re,phi)

def Vi(r, phi, it=range(300)):
   S = 0
   for i in it:
      ni = 4./pi*(-1.)**i/(2*i+1)*r**(2*i+1)*cos(phi*(2*i+1))
      S += ni
   return S

def Ve(r, phi, it=range(300)):
   S = 0
   for i in it:
      ni = 4./pi*(-1.)**i/(2*i+1)*(1./r)**(2*i+1)*cos(phi*(2*i+1))
      S += ni
   return S


Xi = Ri*cos(PHI)
Yi = Ri*sin(PHI)
Zi = Vi(Ri,PHI)
Xe = Re*cos(PHI)
Ye = Re*sin(PHI)
Ze = Ve(Re,PHI)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, Zi, rstride=10, cstride=10,cmap=plt.cm.BuPu)
#ax.plot_surface(Xe, Ye, Ze, rstride=10, cstride=10,cmap=plt.cm.BuPu)
ax.set_xlabel('x/b')
ax.set_ylabel('y/b')
ax.set_zlabel(r'$V/V_0$')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Xi, Yi, Zi, rstride=10, cstride=10,cmap=plt.cm.BuPu)
ax.plot_surface(Xe, Ye, Ze, rstride=10, cstride=10,cmap=plt.cm.BuPu)
ax.set_xlabel('x/b')
ax.set_ylabel('y/b')
ax.set_zlabel(r'$V/V_0$')
plt.show()
