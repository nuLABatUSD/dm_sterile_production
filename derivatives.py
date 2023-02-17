#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb 
from Francisco import dadx, dtda
import Emma3 as emma


# ## derivatives

# In[2]:


G_F = 1.1663787e-5*(1/1000**2)
m_pc = 1.22e22
m_Z = 91.187*1000
m_W = 80.379*1000
riemannzeta3 = 1.2020569
A = (1/np.pi**2)*2*np.sqrt(2)*riemannzeta3*G_F   #constants on V_density
B = (7/8)*(np.pi**2/30)   #constants on rho_v_a-- energy density of active neutrinos

@nb.jit(nopython=True)
def f(x, y, p):
    der = np.zeros(len(y))
    T = 1/x
    N = 0.5*(len(y)-5)
    N = int(N)
    T_cm = 1/y[-1]
    der[-1] = dadx(x, y, p)
    der[-2] = dtda(x, y, p)*der[-1]
    n_photon = 2*riemannzeta3/(2*np.pi**2)*T**3
    
    mixangv_tot = p[-2]
   
    scatterconst_e = 1.27
    scatterconst_mu = 1.27
    scatterconst_tau = 0.92
    
    L_e = 2*y[-3] + y[-4] +y[-5]
    L_mu = 2*y[-4] + y[-3] +y[-5]
    L_tau = 2*y[-5] + y[-3] +y[-4]
    
    r_e = emma.rho(0.511, T) 
    r_mu = emma.rho(105.658, T)
    r_tau = emma.rho(1777, T)
    
    
    if p[-16] == 0:
        dfdt_e = np.zeros(int(N))
        anti_dfdt_e = np.zeros(int(N))
    else: 
        dfdt_e = emma.dfdt(x, y, p, p[-16], scatterconst_e, L_e, r_e)
        anti_dfdt_e = emma.anti_dfdt(x, y, p, p[-16], scatterconst_e, L_e, r_e)
        
    if p[-17] == 0:
        dfdt_mu = np.zeros(int(N))
        anti_dfdt_mu = np.zeros(int(N))
    else: 
        dfdt_mu = emma.dfdt(x, y, p, p[-17], scatterconst_mu, L_mu, r_mu)
        anti_dfdt_mu = emma.anti_dfdt(x, y, p, p[-17], scatterconst_mu, L_mu, r_mu)
        
    if p[-18] == 0:
        dfdt_tau = np.zeros(int(N))
        anti_dfdt_tau = np.zeros(int(N))
    else:
        dfdt_tau = emma.dfdt(x, y, p, p[-18], scatterconst_tau, L_tau, r_tau)
        anti_dfdt_tau = emma.anti_dfdt(x, y, p, p[-18], scatterconst_tau, L_tau, r_tau)
    
    
    der[:N] = (dfdt_e + dfdt_mu + dfdt_tau)*der[-2] ##this is dfdx after chain rule
    der[N:2*N] = (anti_dfdt_e + anti_dfdt_mu + anti_dfdt_tau)*der[-2] #this is anti_dfdx after chain rule
    
    der[-3] = (-1)*(1/n_photon)*(T_cm**3/(2*np.pi**2))*(emma.trapezoid(p[:N], p[:N]**2*dfdt_e*der[-2])-emma.trapezoid(p[:N], p[:N]**2*anti_dfdt_e*der[-2])) - y[-3]*(3/y[-1]*der[-1] + 3*x*(-x**-2))
    
    der[-4] = (-1)*(1/n_photon)*(T_cm**3/(2*np.pi**2))*(emma.trapezoid(p[:N], p[:N]**2*dfdt_mu*der[-2])-emma.trapezoid(p[:N], p[:N]**2*anti_dfdt_mu*der[-2])) - y[-4]*(3/y[-1]*der[-1] + 3*x*(-x**-2))
    
    der[-5] = (-1)*(1/n_photon)*(T_cm**3/(2*np.pi**2))*(emma.trapezoid(p[:N], p[:N]**2*dfdt_tau*der[-2])-emma.trapezoid(p[:N], p[:N]**2*anti_dfdt_tau*der[-2])) - y[-5]*(3/y[-1]*der[-1] + 3*x*(-x**-2))
    
    return der

