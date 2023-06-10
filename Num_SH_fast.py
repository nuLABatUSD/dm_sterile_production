#!/usr/bin/env python
# coding: utf-8

# In[54]:


import matplotlib.pyplot as plt
import numpy as np
from Emma3 import sterile_production, create_full_filename
from classy import Class
from Francisco import N_sh
import numba as nb
import os

@nb.jit(nopython=True)
def cdf_faster(e,f):
    r = np.zeros(len(e))
    
    y = e**2 * f
    de = e[1] - e[0]
    
    #r[1] = 0.5 * (y[0] + y[1]) * de
    for i in range(1, len(r)):
        r[i] = r[i-1] + 0.5 * (y[i-1] + y[i]) * de
        
    r /= np.trapz(y,e)
    
    return r

@nb.jit(nopython=True)
def cdf_half(e,f):
    r = np.zeros(len(e)//2)
    
    y = e**2 * f
    de = e[1] - e[0]
    
    #r[1] = 0.5 * (y[0] + y[1]) * de
    for i in range(1, len(r)):
        r[i] = r[i-1] + 0.5 * (y[i-1] + y[i]) * de
        
    r /= np.trapz(y,e)
    
    return r

@nb.jit(nopython=True)
def exp_cube(eps, a, b, c):
    return 1/(np.exp(a*eps**2 + b*eps + c) + 1)

@nb.jit(nopython=True)
def f_exp_tail(e, e0, m, b):
    exponent = m * (e - e0) + b
    return np.exp(exponent)

@nb.jit(nopython=True)
def gss(e,m,s,n):
    return np.exp(-(e-m)**2/(2*s**2))*n


@nb.jit(nopython=True)
def new_approx(eps, eps_0, mexp, f0exp, aL, bL, cL, DL, m0, s0, n0):
    return DL * exp_cube(eps, aL, bL, cL) + f_exp_tail(eps, eps_0, mexp, f0exp) + gss(eps, m0, s0, n0)

@nb.jit(nopython=True)
def par3(eps, fe):
    cdf = cdf_faster(eps, fe)
    index_high = np.where(cdf > 0.95)[0][0]
    ind_sl = min(index_high + 10, len(eps) - 1)

    mexp = np.log(fe[index_high]/fe[ind_sl]) / (eps[index_high]-eps[ind_sl])
    f0exp = np.log(fe[index_high])
    eps_0 = eps[index_high]

    diff = fe - f_exp_tail(eps, eps_0, mexp, f0exp)

    index_fit_low_peak = np.where(eps < 2)[0][-1]

    am = np.linspace(-2,-0.01,200)
    am = np.linspace(0.01,2,200)
    bm = np.linspace(-4,4,100)
    cm = np.linspace(-10,10,100)

    cdfd = np.zeros((200,100,100))

    diff_lp = diff[:index_fit_low_peak]
    eps_lp = eps[:index_fit_low_peak]

    cdf_l = cdf_faster(eps_lp, diff_lp)

    for i in range(200):
        for j in range(100):
            for k in range(100):
                cdf_ijk = cdf_faster(eps_lp, exp_cube(eps_lp, am[i], bm[j], cm[k]))
                cdfd[i,j,k] = np.max(np.abs(cdf_l - cdf_ijk))

    w = np.where(cdfd == np.amin(cdfd))
    aL, bL, cL = am[w[0][0]], bm[w[1][0]], cm[w[2][0]]    

    D_est = np.max(eps_lp**2 * diff_lp) / np.max(eps_lp**2 * exp_cube(eps_lp, aL, bL, cL))
    Dm = np.linspace(0.9*D_est, 1.1*D_est, 100)
    msd = np.zeros_like(Dm)
    for i in range(100):
        msd[i] = np.sum((eps_lp**2 * diff_lp - Dm[i] * eps_lp**2 * exp_cube(eps_lp, aL, bL, cL))**2)

    DL = Dm[np.where(msd == np.min(msd))[0][0]]    

    gm = np.linspace(1,10,100)
    sm = np.linspace(0.3,5,100)
    nm = np.linspace(0.00001,0.001,100)

    cdfd = np.zeros((100,100,100))
    eps_half = eps[:len(eps)//2]
    fe_half = fe[:len(fe)//2]

    cdf_h = cdf_faster(eps_half, fe_half)

    for i in range(100):
        for j in range(100):
            for k in range(100):
                cdf_ijk = cdf_faster(eps_half, new_approx(eps_half, eps_0, mexp, f0exp, aL, bL, cL, DL, gm[i], sm[j], nm[k]))
                cdfd[i,j,k] = np.max(np.abs(cdf_h - cdf_ijk))

    w = np.where(cdfd == np.amin(cdfd))
    m0, s0, n0 = gm[w[0][0]], sm[w[1][0]], nm[w[2][0]]    
    
    return eps_0, mexp, f0exp, aL, bL, cL, DL, m0, s0, n0

def fit_params(filename, show_plot=True):
    npz = np.load(filename)
    eps = npz['epsilon']
    fe = npz['final_distribution']
    
    eps_0, mexp, f0exp, aL, bL, cL, DL, m0, s0, n0 = par3(eps, fe)
    if show_plot:
        fapp = new_approx(eps, eps_0, mexp, f0exp, aL, bL, cL, DL, m0, s0, n0)
        
        plt.figure()
        plt.plot(eps, eps**2*fe, linestyle='--')
        plt.plot(eps, eps**2 * f_exp_tail(eps, eps_0, mexp, f0exp))
        plt.plot(eps, eps**2 * DL * exp_cube(eps, aL, bL, cL))
        plt.plot(eps, eps**2 * gss(eps, m0, s0, n0))

        plt.figure()
        plt.plot(eps, eps**2*fe, linestyle='--')
        plt.plot(eps, eps**2*fapp)
        
        plt.figure()
        plt.plot(eps, cdf_faster(eps,fe), linestyle='--')
        plt.plot(eps,cdf_faster(eps, fapp))
        plt.show()
        
    return eps_0, mexp, f0exp, aL, bL, cL, DL, m0, s0, n0


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

def num_subhalos(filename, make_plots=False, verbose=False, run_again=True):
    pk_exists = False
    if not run_again:
        if os.path.exists(filename[:-4]+"-CLASS.npz"):
            pk_exists = True
            
    if not pk_exists:
        if not make_Pk(filename, make_plots):
            if not verbose:
                print("Cannot find number of subhalos when Omega_s h^2 > observed value.")
            return
        pk_exists = True
    
    cd = np.load(filename[:-4]+"-CLASS.npz",allow_pickle=True)
    
    cl = cd['Class_values'].item()
    k = cl['k']
    pk = cl['pk']

    xx = N_sh(k,pk)
    
    return xx[0]


def solve(mixangv_e, mixangv_mu, mixangv_tau, Le0, Lmu0, Ltau0, folder_name, file_prefix, N=1000, make_plot=False, run_sp_again=True, run_pk_again=True):
    mass_s = 0.0071

    omega_sh2 = -1
    if not run_sp_again:
        fn = create_full_filename(folder_name, file_prefix, Le0, Lmu0, Ltau0, mixangv_e, mixangv_mu, mixangv_tau) +".npz"
        if os.path.exists(fn):
            npz = np.load(fn, allow_pickle=True)
            omega_sh2 = npz['omega_h2'].item()
            
    if omega_sh2 == -1:
        omega_sh2 = sterile_production(N, mass_s, mixangv_e, mixangv_mu, mixangv_tau, Le0, Lmu0, Ltau0, make_plot=make_plot, folder_name=folder_name, file_prefix=file_prefix)
    
    if omega_sh2 > measured_omegacdm:
        return omega_sh2, np.nan
    else:
        fn = create_full_filename(folder_name, file_prefix, Le0, Lmu0, Ltau0, mixangv_e, mixangv_mu, mixangv_tau) +".npz"
        
        try:
            return omega_sh2, num_subhalos(fn, make_plots=make_plot, run_again=run_pk_again)
        except:
            return omega_sh2, -1
