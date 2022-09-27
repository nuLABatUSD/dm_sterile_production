#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import multiprocessing as mp
import time
import parallel_run
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import ODEsolver_three_nu as solve
from run_three_nu import g, temp, gs, gss
from run_three_nu import with_spline_ODE, steps_taken
from Emma_three_nu import sterile_production


# In[2]:

def plot_contour(mixangle, L0): 
    N = 1000
    mass_s = 0.0071
    mixangv_e = mixangle
    mixangv_mu = mixangle
    mixangv_tau = mixangle
    Le0 = L0
    Lmu0 = L0
    Ltau0 = L0
    return sterile_production(N, mass_s, mixangv_e, mixangv_mu, mixangv_tau, Le0, Lmu0, Ltau0, make_plot=False, folder_name="ContourPlotData")

if __name__ == '__main__':
    mixang = np.linspace( 1e-10, 1e-9, 19)
    lep0 = np.linspace(1e-4, 1e-3, 19)
    
    run_list = []
    new_list = []
    for i in range(len(mixang)):
        for j in range(len(lep0)):
            run_list.append((mixang[i], lep0[j]))
            new_list.append((i,j))
        
    p = mp.Pool(4)
    new_start_time = time.time()
    
    res = p.starmap(plot_contour, run_list)
    
    p.close()
    p.join()
    
    print("Parallel, elapsed time = {} seconds".format(time.time()-new_start_time))
    print(res)
    
    np.savez("results", results = res, mixangle = mixang, L0 = lep0, index = new_list)
    

