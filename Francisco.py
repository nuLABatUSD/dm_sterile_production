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

#@nb.jit(nopython=True)
def trapezoid(x,y):
    N=len(x)
    summ = 0
    for i in range(N-1):
        I = 0.5*(y[i]+ y[i+1])*(x[i+1] - x[i])
        summ = summ + I
    return summ

#@nb.jit(nopython=True)
def right_rectangle(x, y,new_epsilon, new_fe, steps):
    
    m = steps + 1
    area = trapezoid(new_epsilon[0:m],new_epsilon[0:m]**(2)*new_fe[0:m])
    y[1] = (area)/((x[1] - x[0])*x[2]**2)
    for i in range(1,len(x)-1):
        m = steps*i
        n = m + 5 
        area = trapezoid(new_epsilon[m:n],new_epsilon[m:n]**(2)*new_fe[m:n])
        y[i + 1] =(area)/((x[i + 1] - x[i])*x[i + 1]**2)
    return y


def factors(value, x):
    tmp_lst = []
    for i in range(1, value + 1):
        if value % i == 0:
            tmp_lst.append(i) 
    if x in tmp_lst: 
        return True
    else:
        
        return False
    
def xy_values_old(file_name, k): 
    
    dat = np.load(file_name)
    fe = dat['f_full']
    epsilon = dat['eps_full']
    
    new_epsilon = np.zeros(len(epsilon) + 1)
    new_epsilon[1:] = epsilon
    new_fe = np.zeros(len(fe) + 1)
    new_fe[1:] = fe
    
    value = epsilon.shape[0]
    
    a = factors(value, k)
    
    if a: 
        x = new_epsilon[::k]
        y = np.zeros(len(x))

        return x, right_rectangle(x, y, new_epsilon, new_fe, k), new_epsilon, new_fe, epsilon
        
    else:   
        raise Exception(str(k) +' is not a factor of ' + str(value))
                        

                        
def xy_values(file_name, k):
    
    dat = np.load('Neutrino Data/' + file_name)
    fe = dat['final_distribution']
    epsilon = dat['epsilon']
    
    new_epsilon = np.zeros(len(epsilon) + 1)
    new_epsilon[1:] = epsilon
    new_fe = np.zeros(len(fe) + 1)
    new_fe[1:] = fe
    
    value = epsilon.shape[0]
    
    a = factors(value, k)
    
    if a: 
        x = new_epsilon[::k]
        y = np.zeros(len(x))

        return x, right_rectangle(x, y, new_epsilon, new_fe, k), new_epsilon, new_fe, epsilon
        
    else:   
        raise Exception(str(k) +' is not a factor of ' + str(value))


def distributions(x,y,w,v):   
    distribution = np.zeros(len(w))
    total = trapezoid(w, (w**(2)*v))
    for i in range(0,len(distribution)-1):
        distribution[0] = 0
        sliced = trapezoid(w[0:i+2], (w[0:i+2]**(2)*v[0:i+2]))
        a  = sliced/total
        distribution[i + 1]  = a 

    rec_distribution = np.zeros(len(x))
    total2 = trapezoid(x, x**(2)*y)
    for i in range(0, len(rec_distribution)-1):
        distribution[0] = 0 
        sliced2 = trapezoid(x[0:i+2], x[0:i+2]**(2)*y[0:i+2])
        b = sliced2/total2 
        rec_distribution[i + 1] = b
    return distribution, rec_distribution


def ks_test(x,y,w,v,z): 
    
    deviation = []
    k_values = []
    value = z.shape[0]
    
    for i in range(1, value + 1):
            if value % i == 0:
                k_values.append(i)
                
    length = len(k_values)
    middle_index = length // 2
    
    first_half = k_values[:middle_index]
    second_half = k_values[middle_index:]
    
    for i in range(len(first_half)):
        current_k = first_half[i]
        ks_test = [] 
        x, y, w, v, z = xy_values('1x0.00049x3e-09-data.npz', current_k)
        ks_epsilon = w[::current_k]
        ks_x = w[::current_k]
        distribution = np.zeros(len(w))
        total = trapezoid(w, (w**(2)*v))

        for i in range(0,len(distribution)-1):
            distribution[0] = 0
            sliced = trapezoid(w[0:i+2], (w[0:i+2]**(2)*v[0:i+2]))

            a  = sliced/total
            distribution[i + 1]  = a

        for j in range(len(ks_x)): 

            eps_value = ks_epsilon[j]   
            x_value = ks_x[j]

            rec_distribution = np.zeros(len(x))
            total2 = trapezoid(x, x**(2)*y)

            for i in range(0, len(rec_distribution)-1):

                distribution[0] = 0 
                sliced2 = trapezoid(x[0:i+2], x[0:i+2]**(2)*y[0:i+2])
                b = sliced2/total2 
                rec_distribution[i + 1] = b

            diff = abs(rec_distribution - distribution[::current_k])
            diff_value = max(diff)

        deviation.append(diff_value)
        
    return k_values, first_half, second_half, deviation
   
# In[ ]:




