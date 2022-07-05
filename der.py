#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numba as nb


# ## derivatives

# In[2]:


G_F = 1.1663787e-5*(1/1000**2)
m_pc = 1.22e22
m_Z = 91.187*1000
m_W = 80.379*1000
m =0.511
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

@nb.jit(nopython=True)
def gamma(eps, T_cm, T):
    return 1.27*G_F**2*eps*T_cm*T**4 

@nb.jit(nopython=True)
def rho_integrand(x, m, T): 
    return np.sqrt(x**2+(m/T)**2) * (x**2)/(np.exp(np.sqrt(x**2+(m/T)**2))+1) * np.exp(x)

x_lagauss, w_lagauss = np.polynomial.laguerre.laggauss(40)
@nb.jit(nopython=True)
def rho(m,T):
    rho_vals = T**4/np.pi**2 * rho_integrand(x_lagauss, m, T) * w_lagauss
    return np.sum(rho_vals)

@nb.jit(nopython=True)
def mixangle_medium(eps, T_cm, T, m_s, mixangle_vacuum, L, rho):
    delta = m_s**2/(2*eps*T_cm) 
    V_density = A*T**3*L
    rho_v_a = B*T**4
    rho_v_anti = B*T**4
    V_thermal = -1/(3*m_Z**2)*8*np.sqrt(2)*G_F*(eps*T_cm)*(rho_v_a + rho_v_anti) - 1/(3*m_W**2)*8*np.sqrt(2)*G_F*(eps*T_cm)*(2*rho)
    return delta**2*mixangle_vacuum /(delta**2*mixangle_vacuum + (delta*np.sqrt(1-mixangle_vacuum) - V_density - V_thermal)**2)


@nb.jit(nopython=True)
def l_m(eps, T_cm, T, m_s, mixangle_vacuum, L, rho):
    delta = m_s**2/(2*eps*T_cm)
    V_density = A*T**3*L
    rho_v_a = B*T**4
    rho_v_anti = B*T**4
    V_thermal = -1/(3*m_Z**2)*8*np.sqrt(2)*G_F*(eps*T_cm)*(rho_v_a + rho_v_anti) - 1/(3*m_W**2)*8*np.sqrt(2)*G_F*(eps*T_cm)*(2*rho)
    return (delta**2*mixangle_vacuum + (delta*np.sqrt(1-mixangle_vacuum) - V_density - V_thermal)**2)**(-1/2)


@nb.jit(nopython=True)
def active_dist(eps, T_cm, T):
    return 1/(np.exp(eps*T_cm/T)+1)

@nb.jit(nopython=True)
def gstar(x, p): 
    return p

@nb.jit(nopython=True)
def dgSdx(x, p): 
    dgstarSdx = 0
    return dgstarSdx

@nb.jit(nopython=True)
def dadx(x, y, p): 
    return y[-1]/x - (y[-1]/(3*gstar(x,p[-7]))*dgSdx(x,p[-7]))
    
@nb.jit(nopython=True)
def dtda(x, y, p): 
    return ((x**2)*m_pc)/(y[-1]*np.pi)*((8*np.pi*gstar(x,p[-6])/90)**(-1/2))

@nb.jit(nopython=True)
def f(x, y, p): 
    N = len(y)-3
    m_s = p[-1]
    mixangle_vacuum = p[-2]
    #L = 2*p[-3] + p[-4] + p[-5]
    L = 2*y[-3] + p[-4] + p[-5]
    T = 1/x
    T_cm = 1/y[-1]
    n_photon = 2*riemannzeta3/(2*np.pi**2)*T**3
    
    der = np.zeros(len(y))
    der[-1] = dadx(x, y, p)
    der[-2] = dtda(x, y, p)*der[-1]
    r = rho(m,T)
    
    for i in range(N):
        der[i] = (1/4)*gamma(p[i], T_cm, T)*mixangle_medium(p[i], T_cm, T, m_s, mixangle_vacuum, L, r)*(1+((1/2)*gamma(p[i], T_cm, T)*l_m(p[i], T_cm, T, m_s, mixangle_vacuum, L, r))**2)**(-1)*(active_dist(p[i], T_cm, T)-y[i])*der[-2]
    der[-3] = (-1)*((1/n_photon)*(T_cm**3/(2*np.pi**2))*trapezoid(p[:N], p[:N]**2*der[:N]) - y[-3]*(3/y[-1]*der[-1] + 3*x*(-x**-2)))
    return der

