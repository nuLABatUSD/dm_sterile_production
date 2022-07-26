import numpy as np
import numba as nb
import run
from Francisco import gstar
import der

G_F = 1.1663787e-5*(1/1000**2)
m_pc = 1.22e22
m_Z = 91.187*1000
m_W = 80.379*1000
m =0.511

riemannzeta3 = 1.2020569
A = (1/np.pi**2)*2*np.sqrt(2)*riemannzeta3*G_F   #constants on V_density
B = (7/8)*(np.pi**2/30)   #constants on rho_v_a-- energy density of active neutrinos

@nb.jit(nopython=True)
def gamma(scattering_rate, eps, T_cm, T):
    return scattering_rate*G_F**2*eps*T_cm*T**4 

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
def dfdt(x, y, p, mixangle_vacuum, scattering_constant, L, r):
    N = 0.5*(len(y)-3)
    N = int(N)
    m_s = p[-1]
    T = 1/x
    T_cm = 1/y[-1]
    dfdt_array = np.zeros(int(N))
    for i in range(int(N)):
        dfdt_array[i] = (1/4)*gamma(p[-17], p[i], T_cm, T)*mixangle_medium(p[i], T_cm, T, m_s, mixangle_vacuum, L, r)*(1+((1/2)*gamma(p[-17], p[i], T_cm, T)*l_m(p[i], T_cm, T, m_s, mixangle_vacuum, L, r))**2)**(-1)*(active_dist(p[i], T_cm, T) - y[i])
    return dfdt_array

@nb.jit(nopython=True)
def anti_dfdt(x, y, p, mixangle_vacuum, scattering_constant, L, r):
    N = 0.5*(len(y)-3)
    N = int(N)
    m_s = p[-1]
    T = 1/x
    T_cm = 1/y[-1]
    eps = p[:N] 
    anti_dfdt_array = np.zeros(int(N))
    for i in range(int(N)): 
        anti_dfdt_array[i] = (1/4)*gamma(p[-17], p[i], T_cm, T)*mixangle_medium(p[i], T_cm, T, m_s, mixangle_vacuum, -L, r)*(1+((1/2)*gamma(p[-17], p[i], T_cm, T)*l_m(p[i], T_cm, T, m_s, mixangle_vacuum, -L, r))**2)**(-1)*(active_dist(p[i], T_cm, T) - y[N+i])
    return anti_dfdt_array

@nb.jit(nopython=True)
def e_density(mass_s, eps, fe, anti_fe):
    m_pc = 1.22e22
    T_cmb = 2.369e-10
    index = np.where(run.temp < 1/2000)[0][-1]
    x0 = run.temp[index]
    gss_i = der.gstar(x0, run.gss[index,:])
    T_cm03 = (10.75/gss_i)*(4/11)*T_cmb**3 #this one needs to change p[-6] to the actual g*s,i
    c = 8*np.pi/(3*m_pc**2)*((1/2.13e-39)**2)*mass_s/(2*np.pi**2)*T_cm03
    oh2 = c*(der.trapezoid(eps, fe) + der.trapezoid(eps, anti_fe))
    return oh2