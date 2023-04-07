#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import ODEsolver_three_nu as solve
from Francisco import gstar
import numba as nb

# In[2]:
g = np.load("Relativistic_Degrees_of_Freedom.npz")

temp = g['T']
gs = g['g_star']
gss = g['g_star_s']

@nb.jit(nopython = True)
def with_spline_ODE(x0, y_0, dx0, p, xf): 
    end, x, y, dx = solve.ODEsolver(x0, y_0, dx0, p, 10000, 100, xf) 
    if end: 
        
        return (x[-1], y[-1,:], dx)
    else: 
        for i in range(100):
            end, x, y, dx = solve.ODEsolver(x[-1], y[-1,:], dx, p, 10000, 100, xf)
        
            if end: 
    
                return x[-1], y[-1,:], dx
        print('100 loops ran and if statement did not trigger!')

@nb.jit(nopython = True)
def steps_taken(x0, y_0, dx0, p, xf, index, index2):
    steps = index2 - index
    temp_array = np.zeros(steps + 1)
    y_array = np.zeros((steps + 1, len(y_0)))
    entropy_values = np.zeros(steps + 1)
    dx_values = np.zeros(steps + 1)
    
    for i in range(steps + 1):
        
        x0 = temp[index + i] 
        xf = temp[index + i + 1]

        p[-10:-5] = gs[index + i, :] 
        p[-15:-10] = gss[index + i, :] 
        
        x, y, dx = with_spline_ODE(x0, y_0, dx0, p, xf)
        y_0 = y
        dx0 = dx
        entropy = 2 * np.pi**2 / 45 * gstar(x,p[-15:-10]) * y[-1]**3 * x**-3
        entropy_values[i]  = entropy
        temp_array[i] = x
        y_array[i,:] = y
        dx_values[i] = dx 
        
    return temp_array, y_array, dx_values, entropy_values

