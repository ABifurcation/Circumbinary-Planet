import numpy as np
import matplotlib.pyplot as plt
import os,sys
from glob import glob
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import sympy as sp

def light_curve(lc_dir,look=False):
    data = np.genfromtxt(lc_dir,names="time,flux,et,ef")
    good = (data['ef'] == 0)
    time_diff = data['time'][good][1:] - data['time'][good][:-1]
    error_point = np.argmax(time_diff)
    time = np.append(data['time'][good][1:error_point-1],data['time'][good][error_point+2:-1])
    flux = np.append(data['flux'][good][1:error_point-1],data['flux'][good][error_point+2:-1])
    et = np.append(data['et'][good][1:error_point-1],data['et'][good][error_point+2:-1])

    norm_flux = flux/np.median(flux)
    norm_et = et/np.abs(np.median(flux))
    w = (norm_flux < 1.03) & (norm_flux > 0.75)

    if look == True:
        plt.figure(figsize=(10,5))
        plt.plot(time[w],norm_flux[w],'c')
        plt.plot(time[w],norm_flux[w],'r.')
        plt.xlabel('time');plt.ylabel('flux')
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        None

    return time[w],norm_flux[w],norm_et[w]




def mulit_light_curve(lc_dir,look=False):
    time = []; flux = []; et = []
    size = 0
    for i in range(10):
        test_dir = lc_dir[:-9] + "0%s" %i + "_LC.txt"
        if os.path.exists(test_dir) == True:
            t,f,e = light_curve(test_dir)
            if t == 'bad_data':
                None
            else:
                time.extend(t)
                flux.extend(f)
                et.extend(e)
                size += 1
        else:
            None

    for i in range(10):
        test_dir = lc_dir[:-9] + "1%s" %i + "_LC.txt"
        if os.path.exists(test_dir) == True:
            t,f,e = light_curve(test_dir)
            if t == 'bad_data':
                None
            else:
                time.extend(t)
                flux.extend(f)
                et.extend(e)
                size += 1
        else:
            None
            
    if size > 2:
        if look == True:
            plt.figure(figsize=(5*size,5))
            plt.plot(time,flux,'c')
            plt.plot(time,flux,'r.')
            plt.xlabel('time');plt.ylabel('flux')
            plt.grid()
            plt.show()
        else:
            None
    else:
        None

    return time, flux, et

def sec_poly(x,a,b,c):
    y = a*x**2+b*x+c
    return y

def fourth_poly(x,a,b,c,d,e):
    y = a*x**4+b*x**3+c*x**2+d*x+e
    return y

def root_4th_poly(a,b,c,d):
    x = sp.Symbol('x')
    f = 4*a*(x**3) + 3*b*(x**2) + 2*c*(x) + d
    root = sp.solve(f)
    r = [complex(root[0]).real,complex(root[1]).real,complex(root[2]).real]
    mid = r[np.argmin(np.abs(r))]
    return mid

def Phase_cycle(t,P,T_0):
    T_diff = np.array(t) - T_0
    I = np.round(T_diff/P)
    phi = (T_diff/P) - I
    return phi, I #phase and cycle


def sing_dc_func(guess_P,time,norm_flux,fir_eci,look=False):
    phi, I = Phase_cycle(time, guess_P, (fir_eci))
    phase_flux = np.stack((phi,norm_flux),axis=-1)
    phase_flux = np.array(sorted(phase_flux,key=lambda x:x[0]))
    phase = phase_flux[:,0]
    flux = phase_flux[:,1]

    if look == True:
        plt.figure(figsize=(15,5))
        plt.title('period = %s'%(guess_P))
        plt.plot(phase,flux,'c-')
        plt.plot(phase,flux,'r.')
        plt.xlabel('phase');plt.ylabel('flux')
        plt.show()        
    
    S = np.fabs(flux[-1] - flux[0]) + np.sum(np.fabs(flux[1:] - flux[:-1]))
    return S/47.-2.,phase,flux


tic00425935689 =  glob("/Users/laote/sdsu/Research/binary/curve_bin/3w_binary/tic00401926767*.txt")
tic00425935689 = sorted(tic00425935689, key=lambda name: int(name[-25:-14] + name[-10:-7]))
for tic in tic00425935689:
    print(tic)
    t200, f200, et200 = light_curve(tic,look=True)
# t200, f200, et200 = mulit_light_curve(tic00425935689[0],look=True)
# sing_dc_func(7.8187447,t200,f200,3325.79,look=True)


















# last line of code