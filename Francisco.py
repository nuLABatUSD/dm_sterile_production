#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb
 


# ## derivatives

# In[2]:


m_pc = 1.22e22

@nb.jit(nopython=True)
def dadx(x, y, p):
    return y[-1]/x - (y[-1]/(3*gstar(x,p[-15:-10]))*dgSdx(x,p[-15:-10])) ##changed the index of p[]

@nb.jit(nopython=True)
def dtda(x, y, p):
    return ((x**2)*m_pc)/(y[-1]*np.pi)*((8*np.pi*gstar(x,p[-10:-5])/90)**(-1/2)) ##changed the index of P[-6]

@nb.jit(nopython=True)
def gstar(x, p): 
    dx = (x - p[0]) 
    spline = p[1] + p[2]*dx + (1/2)*p[3]*(dx)**2 + (1/6)*p[4]*(dx)**3
    return spline

@nb.jit(nopython=True)
def dgSdx(x, p): 
    dx = (x - p[0])
    spline_diff = 0 + p[2] + p[3]*(dx) + (1/2)*p[4]*dx**2 
    return spline_diff

@nb.jit(nopython=True)
def f(x, y, p): 
    
    der = np.zeros(len(y))
    der[-1] = dadx(x, y, p)
    der[-2] = dtda(x, y, p) * der[-1]
    
    return der


# In[ ]:




