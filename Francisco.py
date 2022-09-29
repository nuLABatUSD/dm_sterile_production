#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb
from classy import Class
import scipy.interpolate as sp
import os



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
        n = m + steps
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
    
    #dat = np.load('Neutrino Data/' + file_name)
    dat = np.load(file_name)
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

measured_omegacdm=.1188
h = 0.674

Sig_8 = 0.811
A_s = 2.1e-9

M_sol = 1.989e30 # solar mass in kg
Mpc = 3.086e22 # Mpc in m
kvec = np.logspace(-4,np.log10(100),100)

M0 = 1.77e12 / h # M_sol / h
M_min = 1e8 /h

LambdaCDM_settings = {
    'omega_b':0.023,
    'h':h,
    'output':'mPk',
    'P_k_max_1/Mpc':100.0,
    'ncdm_fluid_approximation':3,
    'background_verbose':0
}

def make_Class_dict(NH, k, pkNH):
    values_dict = {
                    'age': NH.age(),
                    'h': NH.h(),
                    'n_s': NH.n_s(),
                    'Neff': NH.Neff(),
                    'Omega0_cdm': NH.Omega0_cdm(),
                    'Omega0_k': NH.Omega0_k(),
                    'Omega0_m': NH.Omega0_m(),
                    'Omega_b': NH.Omega_b(),
                    'omega_b': NH.omega_b(),
                    'Omega_g': NH.Omega_g(),
                    'Omega_lambda': NH.Omega_Lambda(),
                    'Omega_m': NH.Omega_m(),
                    'Omega_r': NH.Omega_r(),
                    'rs_drag': NH.rs_drag(),
                    'Sigma8': NH.sigma8(),
                    'Sigma8_cb': NH.sigma8_cb(),
                    'T_cmb': NH.T_cmb(),
                    'tau_reio': NH.tau_reio(),
                    'theta_s_100': NH.theta_s_100(),
                    'theta_star_100': NH.theta_star_100(),
                    'pk': pkNH,
                    'k':kvec
                    }
    return values_dict

def make_CDM_Pk():
    cdm_settings = { 'omega_cdm': measured_omegacdm }
    
    NH = Class()
    NH.set(LambdaCDM_settings)
    NH.set(cdm_settings)
    NH.compute()
    
    pkNH = [] 
    
    for k in kvec:
        pkNH.append(NH.pk(k,0.))

    np.savez('LambdaCDM_results', Class_values = make_Class_dict(NH, k, pkNH))
#    np.save('Spec-Pknew',np.array(pkNH))
#    np.save('Spec-knew',kvec)
    
def trap(f,x):
    integral = 0
    for i in range(1,len(f)):
        integral += 0.5 * (f[i] + f[i-1]) * (x[i] - x[i-1])
    return integral

def R(M):
    c = 2.5
    G = 6.67e-11 # m^3 / kg / s^2
    H100 = 100 * (1000/Mpc) # km/s/Mpc -> 1/s
    omegah2 = measured_omegacdm

    rhobar = omegah2 * 3 * H100**2 / (8 * np.pi * G) / M_sol * Mpc**3 # kg / m^3 -> M_sol / Mpc^3
    return (3 * M / (4 * np.pi * rhobar * c**3))**(1/3) # Mpc

def W(k,R):
        if np.isscalar(k):
            if k * R > 1:
                return 0
            else:
                return 1
        else:
            result = np.zeros(len(k))
            for i in range(len(k)):
                if k[i] * R < 1 and k[i+1] * R > 1:
                    result[i] = (1 - k[i] * R)/(k[i+1]*R - k[i]*R)
                elif k[i] * R < 1:
                    result[i] = 1
            return result
        
def S(M, k_vals, Pk_vals):
        Rv = R(M)
        
        integrand = k_vals**2 * Pk_vals * W(k_vals, Rv)**2 / (2 * ( np.pi**2))
        return trap(integrand,k_vals)


def dNdlnM(M, k_vals, Pk_vals, P_spline):
    Rv = R(M)
    return 1 / 44.5 / (6 * np.pi**2) * (M0 / M) / Rv**3 / np.sqrt(2 * np.pi * ( S(M, k_vals, Pk_vals) - S(M0, k_vals, Pk_vals))) * P_spline(1/Rv)

def N_sh(k_vals, Pk_vals):
    P_spline  = sp.CubicSpline(k_vals,Pk_vals)

    lnM_vals = np.linspace(np.log(M_min),np.log(M0))
    
    sv = np.zeros(len(lnM_vals)-1)
    for i in range(len(sv)):
        sv[i] = S(np.exp(lnM_vals[i]), k_vals, Pk_vals)- S(np.exp(lnM_vals[-1]), k_vals, Pk_vals)
    
    integrand = np.zeros(len(lnM_vals)-1)

    for i in range(len(integrand)):
        integrand[i] = dNdlnM(np.exp(lnM_vals[i]), k_vals, Pk_vals, P_spline)
        
    integral = trap(integrand,lnM_vals[:-1])
                
    return integral, integrand, sv, lnM_vals

ncdm_settings = {          
          'N_ncdm':1,
          'use_ncdm_psd_files': 1,
          'm_ncdm': 7100,
          'T_ncdm':0.7,
          'ncdm_maximum_q':20
}
def ideal_sigma8 (spec_file, omega_h_h):
    
    NH = Class()
        #use method .set() to 
    NH.set(LambdaCDM_settings)
    othersettings = { 'ncdm_psd_filenames': spec_file,
                      'omega_cdm': measured_omegacdm - omega_h_h, 
                      'omega_ncdm': omega_h_h, 
                      'A_s': A_s,
                    }

    #use method .compute() to get data for my specific 'Spec-' file
    NH.set(ncdm_settings)
    NH.set(othersettings)
    NH.compute()

    Sigma8_value = NH.sigma8()
    #print("",Sigma8_value,"")
    
    ideal_value = ((Sig_8)/(Sigma8_value))**2*(A_s)
    
    NH.struct_cleanup()
    
    return ideal_value

def make_Pk(spec_file,omega_h_h, print_output=True, output_file = "CLASS_Values.npz"):    
    
    othersettings = { 'ncdm_psd_filenames': spec_file,
                      'omega_cdm': measured_omegacdm - omega_h_h, 
                      'omega_ncdm': omega_h_h, 
                      'A_s': ideal_sigma8(spec_file, omega_h_h),
                    }
    
    if print_output:
        for key, value in othersettings.items():
            print(key, ' : ', value)
        for key, value in LambdaCDM_settings.items():
            print(key, ' : ', value)
    
    NH = Class()
    NH.set(LambdaCDM_settings)
    NH.set(ncdm_settings)
    NH.set(othersettings)
    NH.compute()
    
    #new_sig8 = NH.sigma8()
    pkNH = [] 
    
        #MPk- Matter Power Spectrum
    for k in kvec:
        pkNH.append(NH.pk(k,0.))
    
    #this method clears the commonsettings out of the set method, so that we can load a new file 
    NH.struct_cleanup() 

    #saving the dictionary that we created above to a .npz file
    np.savez(output_file, Class_values = make_Class_dict(NH, kvec, pkNH))


    #np.save('Spec-Pknew',np.array(pkNH))
    #np.save('Spec-knew',kvec)
    
    #return new_sig8
    return kvec, pkNH

def Bella_2(file_name, k, print_output=True): ##.npz as first input and step size as the second
    
    a,b,c,d,e = xy_values(file_name, k)
    
    #dat = np.load('Neutrino Data/' + file_name)
    dat = np.load(file_name)
    omega_h_h = dat['omega_h2']
    
    if omega_h_h >= measured_omegacdm:
        print("Cannot find subhalos when Omega h^2 > ", measured_omegacdm)
        return
    
    spec_file  = file_name[:-4] + '-Spec'
    np.savetxt(spec_file, np.column_stack((a,b)))
    
    k_vals, Pk_vals = make_Pk(spec_file, omega_h_h, print_output=print_output, output_file = file_name[:-4] + "-CLASS.npz")
    
    os.remove(spec_file)
#    k_vals = np.load('Spec-knew.npy')
#    Pk_vals = np.load('Spec-Pknew.npy')
    
    return N_sh(k_vals, Pk_vals)

def Num_subhalos(filename, k):
    dat = np.load(filename)
    omega_h_h = dat['omega_h2']
    if omega_h_h >= measured_omegacdm:
        return np.nan
    
    xx = Bella_2(filename, k, print_output=False)
    return xx[0]



