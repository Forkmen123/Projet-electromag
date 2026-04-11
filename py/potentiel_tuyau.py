from numpy import pi, cos, sin, cosh, sinh
import numpy as np
import pylab as plt


# # for latex rendering
font = {'family' : 'monospace',
#         'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
#plt.rc('font', family='serif',size=22.)
# plt.rc('text', usetex=True)

#Y, X = np.mgrid[0.5:0.5:500j, -0.5:0.5:200j]
x = np.linspace(-0.5,0.5,500)
y = np.linspace(-0.5,0.5,500)
X,Y = np.meshgrid(x,y)

# V/V_0
# x = x/a
def V(x,y,s=range(200)):
   S = 0
   for i in s:
      ni = 4./pi*(-1)**i/(2*i+1)*cos( (2*i+1)*pi*x ) * cosh( (2*i+1)*pi*y) / cosh( (2*i+1)*pi/2.)
      S += ni
   return S


Z = V(X,Y)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# truncation in plot to avoid incomplete fourier artifacts
ax.plot_surface(X[5:-5,5:-5], Y[5:-5,5:-5], Z[5:-5,5:-5], rstride=10, cstride=10,cmap=plt.cm.BuPu)
ax.set_xlabel('x/a')
ax.set_ylabel('y/a')
ax.set_zlabel(r'$V/V_0$')
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10,cmap=plt.cm.bone)

plt.show()

#plt.plot(x,cos1,label=r'2n+1=1')
   



