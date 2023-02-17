#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb
from derivatives import f


# In[2]:


a2 = 1/5
a3 = 3/10
a4 = 3/5
a5 = 1
a6 = 7/8

b21 = 1/5
b31 = 3/40
b41 = 3/10
b51 = -11/54
b61 = 1631/55296
b32 = 9/40
b42 = -9/10
b52 = 5/2
b62 = 175/512
b43 = 6/5
b53 = -70/27
b63 = 575/13824
b54 = 35/27
b64 = 44275/110592
b65 = 253/4096

c1 = 37/378
c2 = 0
c3 = 250/621
c4 = 125/594
c5 = 0
c6 = 512/1771

cstar1 = 2825/27648
cstar2 = 0
cstar3 = 18575/48384
cstar4 = 13525/55296
cstar5 = 277/14336
cstar6 = 1/4

eps = 10**(-6)
TINY = 10**(-40)
S = 0.9

@nb.jit(nopython=True)
def RungeKutta(x, y, dx, p): 
    k1 = dx * f(x, y, p)
    k2 = dx * f(x + a2*dx, y + b21*k1, p)
    k3 = dx * f(x + a3*dx, y + b31*k1 + b32*k2, p)
    k4 = dx * f(x + a4*dx, y + b41*k1 + b42*k2 +b43*k3, p)
    k5 = dx * f(x + a5*dx, y + b51*k1 + b52*k2 + b53*k3 + b54*k4, p)
    k6 = dx * f(x + a6*dx, y + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5, p)
    
    y5_next= y + c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6 
    y4_next= y + cstar1*k1 + cstar2*k2 + cstar3*k3 + cstar4*k4 + cstar5*k5 + cstar6*k6 
    x_stepped = x + dx
    
    return x_stepped, y5_next, y4_next

@nb.jit(nopython=True)
def step_adapt(x, y, dx, p):  

    for i in range(100):
        x_new, y5_next, y4_next =  RungeKutta(x, y, dx, p)
        delta1 = np.abs(y5_next - y4_next)                                  
        delta0 = eps*(np.abs(y) + np.abs(y5_next - y)) + TINY                                  
        
        if np.all(delta1 == 0):
            dx_new = 5*dx  
            return (x_new, y5_next, dx_new)                           

        else:
            counter = 0
            for i in range (len(delta1)):                               
                if (delta1[i] <= delta0[i]):                            
                    counter += 1                                        
            if (counter == len(delta1)):
                delta_max = np.max(delta1/delta0)                         
                dx_new = S*dx*(np.abs(1/delta_max))**0.2 
                if (dx_new < 5*dx):
                    return (x_new, y5_next, dx_new) 
                else:
                    return (x_new, y5_next, 5*dx) 
             
            else:                                                        
                delta_max = np.max(delta1/delta0)                       
                dx_new = S*dx*(np.abs(1/delta_max))**0.25
                dx = dx_new
                
    print ('error- no good dx was found')
    return (x_new, y5_next, dx_new)    

@nb.jit(nopython=True)
def ODEsolver(x_0, y_0, dx, p, N, dN, x_f):                   
    x_values = np.zeros(int(N/dN) + 1)                     
    y5_results = np.zeros((len(x_values), len(y_0)))        
    
    x_values[0] = x_0             
    y5_results[0,:] = y_0
    
    x = x_0                                               
    y = np.copy(y_0)                                       
    for i in range (1, N + 1):
         
        if (x+dx > x_f):
            dx = x_f - x
        
        x, y, dx = step_adapt(x, y, dx, p)
        
        if (x == x_f):
            j=int(np.ceil(i/dN)) 
            x_values[j] = x
            y5_results[j,:] = y
            return (True, x_values[:(j+1)], y5_results[:(j+1),:], dx)

        if (i%dN == 0):                                    
            j=int(i/dN) 
            x_values[j] = x
            y5_results[j,:] = y
   
    return (False, x_values, y5_results, dx)

@nb.jit(nopython=True)
def ODEsolver_files(x_0, y_0, dx, p, N_runs, N, dN, x_f, file_header = "file"):
    for i in range (N_runs):
        end, x, y, dx = Adaptive_ODEsolver(x_0, y_0, dx, p, N/N_runs, dN, x_f) 
        x_0 = x[len(x)-1]
        y_0 = y[len(x)-1, :]
        print(end, x)
        file = "{}{}".format(file_header, i)  
        np.savez(file, x=x, y=y, dx=dx)
        if end:
            print ("Reached x_f!")
            return (x, y, dx)
    print ("Did not reach x_f!")    
    return (x, y, dx)
