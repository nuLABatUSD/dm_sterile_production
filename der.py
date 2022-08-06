#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb
from Francisco import gstar, dgSdx, dadx, dtda
from Emma import gamma, mixangle_medium, rho, l_m, active_dist, dfdt, anti_dfdt


# ## derivatives

# In[2]:


G_F = 1.1663787e-5*(1/1000**2)
m_pc = 1.22e22
m_Z = 91.187*1000
m_W = 80.379*1000
m = 0.511
riemannzeta3 = 1.2020569
A = (1/np.pi**2)*2*np.sqrt(2)*riemannzeta3*G_F   #constants on V_density
B = (7/8)*(np.pi**2/30)   #constants on rho_v_a-- energy density of active neutrinos

@nb.jit(nopython=True)
def trapezoid(x,y):
    N=len(x)
    summ = 0
    for i in range(N-1):
        I = 0.5*(y[i]+ y[i+1])*(x[i+1] - x[i])
        summ = summ + I
    return summ



#@nb.jit(nopython=True)
#def gstar(x, p): 
#    return p

#@nb.jit(nopython=True)
#def dgSdx(x, p): 
#    dgstarSdx = 0
#    return dgstarSdx

#@nb.jit(nopython=True)
#def dadx(x, y, p): 
#    return y[-1]/x - (y[-1]/(3*gstar(x,p[-7]))*dgSdx(x,p[-7]))
    
#@nb.jit(nopython=True)
#def dtda(x, y, p): 
#    return ((x**2)*m_pc)/(y[-1]*np.pi)*((8*np.pi*gstar(x,p[-6])/90)**(-1/2))


#   N = len(y)-3

@nb.jit(nopython=True)
def f(x, y, p):
    der = np.zeros(len(y))
    T = 1/x
    N = 0.5*(len(y)-3)
    N = int(N)
    T_cm = 1/y[-1]
    der[-1] = dadx(x, y, p)
    der[-2] = dtda(x, y, p)*der[-1]
    n_photon = 2*riemannzeta3/(2*np.pi**2)*T**3
    if p[-18] == 0:
        L = 2*y[-3] + p[-4]/(y[-1]**3*T**3) +p[-5]/(y[-1]**3*T**3) #electron p[-18]=0
    elif p[-18] == 1:
        L = 2*y[-3] + p[-3]/(y[-1]**3*T**3) +p[-5]/(y[-1]**3*T**3) #muon p[-18]=1
    else:
        L = 2*y[-3] + p[-3]/(y[-1]**3*T**3) +p[-4]/(y[-1]**3*T**3) #tau p[-18]=2
    r = rho(p[-16], T)
    mixangle_vacuum = p[-2]
    scattering_constant = p[-17]
    
    der[:N] = dfdt(x, y, p, mixangle_vacuum, scattering_constant, L, r)*der[-2]
    der[N:2*N] = anti_dfdt(x, y, p, mixangle_vacuum, scattering_constant, L, r)*der[-2]
    der[-3] = (-1)*(1/n_photon)*(T_cm**3/(2*np.pi**2))*(trapezoid(p[:N], p[:N]**2*der[:N])-trapezoid(p[:N], p[:N]**2*der[N:2*N])) - y[-3]*(3/y[-1]*der[-1] + 3*x*(-x**-2))
    return der

