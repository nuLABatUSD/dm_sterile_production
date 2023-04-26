#!/usr/bin/env python
# coding: utf-8

# In[54]:


import matplotlib.pyplot as plt
import numpy as np
from Emma3 import sterile_production
from classy import Class
from Francisco import N_sh

def f_exp_tail(e, e0, m, b):
    exponent = m * (e - e0) + b
    return np.exp(exponent)

def low_peak(e, T, N):
    return N/(np.exp(e/T)+1)

def gss(e,m,s,n):
    return np.exp(-(e-m)**2/(2*s**2))*n

def cdf(e, f):
    r = np.zeros(len(e))
    
    for i in range(1,len(r)):
        r[i] = np.trapz((e**2*f)[:i+1], e[:i+1])
        
    r /= np.trapz(e**2*f,e)
    
    return r

def fit_params(filename, show_plot=True):
    npz = np.load(filename)
    eps = npz['epsilon']
    fe = npz['final_distribution']
    
    mexp = np.log(fe[-10]/fe[-1]) / (eps[-10]-eps[-1])
    f0exp = np.log(fe[-10])
    eps_0 = eps[-10]
    
    diff = fe - f_exp_tail(eps, eps_0, mexp, f0exp)

    Temp = np.linspace(0.01,1,100)
    Normalize = np.linspace(0.01,1,100)

    msd = np.zeros((100,100))

    for i in range(len(Temp)):
        for j in range(len(Normalize)):
            msd[i,j] = np.sum((eps**2 * diff - eps**2 *low_peak(eps, Temp[i], Normalize[j]))**2)
    T_low, N_low = (Temp[np.where(msd==np.amin(msd))[0][0]], Normalize[np.where(msd==np.amin(msd))[1][0]])
    
    diff2 = diff - low_peak(eps, T_low, N_low)

    gm = np.linspace(1,10,100)
    sm = np.linspace(0.3,5,100)
    nm = np.linspace(0.00001,0.001,100)

    msdm = np.zeros((100,100,100))

    def gss(e,m,s,n):
        return np.exp(-(e-m)**2/(2*s**2))*n
    for i in range(100):
        for j in range(100):
            for k in range(100):
                msdm[i,j,k] = np.sum((eps**2 * diff2 - eps**2 * gss(eps, gm[i], sm[j], nm[k]))**2)

    w = np.where(msdm == np.amin(msdm))
    m0, s0, n0 = gm[w[0][0]], sm[w[1][0]], nm[w[2][0]]
    
    if show_plot:
        fapp = (f_exp_tail(eps, eps_0, mexp, f0exp)+low_peak(eps, T_low, N_low)+gss(eps, m0, s0, n0))
        
        plt.figure()
        plt.plot(eps, eps**2*fe, linestyle='--')
        plt.plot(eps, eps**2 * f_exp_tail(eps, eps_0, mexp, f0exp))
        plt.plot(eps, eps**2 * low_peak(eps, T_low, N_low))
        plt.plot(eps, eps**2 * gss(eps, m0, s0, n0))

        plt.figure()
        plt.plot(eps, eps**2*fe, linestyle='--')
        plt.plot(eps, eps**2*fapp)
        
        plt.figure()
        plt.plot(eps, cdf(eps,fe), linestyle='--')
        plt.plot(eps,cdf(eps, fapp))
        plt.show()
        
    return eps_0, mexp, f0exp, T_low, N_low, m0, s0, n0


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


measured_omegacdm=.1188
h = 0.674

Sig_8 = 0.811
A_s = 2.1e-9

kvec = np.logspace(-4,np.log10(100),100)


LambdaCDM_settings = {
    'omega_b':0.023,
    'h':h,
    'output':'mPk',
    'P_k_max_1/Mpc':100.0,
    'ncdm_fluid_approximation':3,
    'background_verbose':0
}
ncdm_settings = {          
          'N_ncdm':1,
          'use_ncdm_psd_files': 0,
          'm_ncdm': 7100,
          'T_ncdm':0.716*0.51,
          'ncdm_maximum_q':20
}

def ideal_sigma8 (othersettings):
    
    NH = Class()
        #use method .set() to 
    NH.set(LambdaCDM_settings)

    othersettings['A_s'] = A_s
    
    #use method .compute() to get data for my specific 'Spec-' file
    NH.set(ncdm_settings)
    NH.set(othersettings)
    NH.compute()

    Sigma8_value = NH.sigma8()
    #print("",Sigma8_value,"")
    
    ideal_value = ((Sig_8)/(Sigma8_value))**2*(A_s)
    
    NH.struct_cleanup()
    
    return ideal_value


def make_Pk(filename, make_plots=False):
    data = np.load(filename)
    
    omega_h_h = data['omega_h2']
    if omega_h_h > measured_omegacdm:
        return False
    
    output_file = filename[:-4] + "-CLASS.npz"
    params = fit_params(filename, make_plots)
    
    p = ""
    for i in range(len(params)-1):
        p += "{:.3f},".format(params[i])
    p += "{:.3f}".format(params[-1]*1e6)

    othersettings = { 
                      'omega_cdm': measured_omegacdm - omega_h_h, 
                      'omega_ncdm': omega_h_h, 
                      'ncdm_psd_parameters': p
                    }

    othersettings['A_s'] = ideal_sigma8(othersettings)
    NH = Class()
    NH.set(LambdaCDM_settings)
    NH.set(ncdm_settings)
    NH.set(othersettings)
    NH.compute()
    
    pkNH = [] 

    for k in kvec:
        pkNH.append(NH.pk(k,0.))

    d = make_Class_dict(NH, kvec, pkNH)
    
    NH.struct_cleanup() 

    np.savez(output_file, Class_values = d, other_settings = othersettings)
    
    return True

def num_subhalos(filename, make_plots=False, verbose=False):
    if not make_Pk(filename, make_plots):
        if not verbose:
            print("Cannot find number of subhalos when Omega_s h^2 > observed value.")
        return
    
    cd = np.load(filename[:-4]+"-CLASS.npz",allow_pickle=True)
    
    cl = cd['Class_values'].item()
    k = cl['k']
    pk = cl['pk']

    xx = N_sh(k,pk)
    
    return xx[0]


def solve(mixangv_e, mixangv_mu, mixangv_tau, Le0, Lmu0, Ltau0, folder_name, file_prefix, N=1000, make_plot=False):
    mass_s = 0.0071

    omega_sh2 = sterile_production(N, mass_s, mixangv_e, mixangv_mu, mixangv_tau, Le0, Lmu0, Ltau0, make_plot=make_plot, folder_name=folder_name, file_prefix=file_prefix)
    

    if omega_sh2 > measured_omegacdm:
        return omega_sh2, np.nan
    else:
        mixing_angle  = np.format_float_scientific(mixangv_e+mixangv_mu+mixangv_tau, precision = 2, unique=False)
        lepton_number = np.format_float_scientific(Le0, precision = 2, unique=False)

        fn = folder_name + "/" + file_prefix + 'x' + lepton_number +'x'+ mixing_angle + '.npz'
        
        try:
            return omega_sh2, num_subhalos(fn, make_plots=make_plot)
        except:
            return omega_sh2, -1
