import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from multiprocessing import Process, Array, Lock
import time


# basedir = "/Users/laote/sdsu/Research/binary/tess_keplerEBs"
# basedir2 = "/Users/laote/sdsu/Research/binary/tess_alldata/Curves"
# curvedir = "/Users/laote/sdsu/Research/binary/tess_alldata/Curves"

# lc_list = glob("%s/*.txt" %basedir)
# lc_list = sorted(lc_list, key=lambda name: int(name[52:-14]))
# lc_list = sorted(lc_list, key=lambda name: int(name[57:-14]))


basedir = "/home/zlin/research"
curvedir = "/home/zlin/research/Curves2"

lc_list = glob.glob("%s/*.txt" %curvedir)
lc_list = sorted(lc_list, key=lambda name: int(name[31:-14]))


def line_func(x, a, b):
    return a*x + b

def fit_med(datadir):
    data = np.genfromtxt(datadir,names="time,flux,et,ef")
    med = np.median(data['flux']) 
    std = np.std(data['flux'])
    c = (data['flux'] > (med-std)) & (data['flux'] < (med+std))
    para, epara = curve_fit(line_func, data['time'][c], data['flux'][c])
    slope = para[0]; inter = para[1]
    return slope, inter

def fit_peak(datadir, display=False):
    data = np.genfromtxt(datadir,names="time,flux,et,ef")
    slope, inter = fit_med(datadir)
    
    time_diff = data['time'][1:] - data['time'][:-1]
    error_point = np.argmax(time_diff)
    
    time = np.append(data['time'][20:error_point-40],data['time'][error_point+40:-20])
    flux = np.append(data['flux'][20:error_point-40],data['flux'][error_point+40:-20])
    
    norm_flux = flux - line_func(time,slope,inter)
    med = np.median(norm_flux) 
    std = np.std(norm_flux)
    peaks = find_peaks(-norm_flux, height=-(med-1*std))

    peaks_pos = peaks[0]
    height = -peaks[1]['peak_heights']
    
    if len(height) >= 1:
        sort_height = sorted(height)
        if sort_height[0] <= med-4.5*std:
            result = 'win'
        elif sort_height[0] >= med-2.5*std:
            result = 'lose'
        else:
            result = 'i dont know'
    else:
        result = 'lose'

#     fittime = []
#     for i in range(len(time[peaks_pos])):
#         fitx = [];fity = [];
#         for j in range(-2,3,1):
#             fitx.append(time[peaks_pos+j][i])
#             fity.append(norm_flux[peaks_pos+j][i])
#         z = np.polyfit(fitx,fity,2)
#         func = np.poly1d(z)
#         minimum = -z[1]/(2*z[0])
#         fittime.append(minimum)

    if display == True:

        L = len(norm_flux)
        yf = np.abs(np.fft.fft(norm_flux))[range(int(L/2))]
        xf = (time[2]-time[1])*np.arange(L)[range(int(L/2))]
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(data['time'],data['flux'],'c.')
        plt.plot([time[0],time[-1]],\
                 [(np.median(flux)),(np.median(flux))],'r-')
        plt.subplot(1,3,2)
        plt.plot(time,norm_flux,'c-')
#         plt.plot(fittime,height,'rx')
        plt.plot(time[peaks_pos],height,'rx')
        plt.plot([time[0],time[-1]],[(med),(med)],'r-')
        plt.plot([time[0],time[-1]],[(med-4.5*std),(med-4.5*std)],'r-')

        plt.subplot(1,3,3)
        plt.plot(xf,yf,'c-')
        plt.show()

    return result

def find_max_slope(datadir,loop=400):
    a = 0
    for i in range(loop):
        slope, inter = fit_med(datadir[i])
        if np.abs(slope) > a:
            a = np.abs(slope)
            w = i
    print(a,w)


def save_img(datadir,choise=0):
    data = np.genfromtxt(datadir,names="time,flux,et,ef")
    w = (data['ef'] == 0)
    data_time = data['time'][w]
    data_flux = data['flux'][w]

    time_diff = data_time[1:] - data_flux[:-1]
    error_point = np.argmax(time_diff)

    if (np.max(time_diff) < 2) & (np.sum(time_diff > 0.7) <2):
    
        time = np.append(data_time[20:error_point-30],data_time[error_point+30:-20])
        flux = np.append(data_flux[20:error_point-30],data_flux[error_point+30:-20])
        
        norm_flux = (flux - np.median(flux))/(np.max(flux) - np.min(flux))
        norm_flux = np.minimum(norm_flux, np.std(norm_flux))
        
        plt.figure(figsize=(5,5))
        plt.plot(time,norm_flux,'k-')
        plt.axis('off')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        # plt.savefig("/Users/laote/sdsu/Research/binary/tess_image/%s_%s.jpg" %(datadir[54:-7],choise))
        # plt.savefig("/Users/laote/sdsu/Research/binary/tess_image/%s_%s.jpg" %(datadir[49:-7],choise))
        plt.savefig("/home/zlin/research/curve_img/%s.jpg" %(datadir[31:-7]))
        print("ok")


procs = []
for i in range(2):
    p = Process(target = save_img, args = (lc_list[i],))
    p.start()

for p in procs:
    p.join()


# binary = np.genfromtxt("classification.txt",dtype=str)
# binary_lc_dir = []
# for i in range(len(binary)):
#     binary_lc_dir.append(curvedir + '/' + binary[i][0] + '_LC.txt')

# procs = []
# for i in range(len(binary_lc_dir)):
#     p = Process(target = save_img, args = (binary_lc_dir[i],))
#     p.start()

# for p in procs:
#     p.join()


# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1)
# plt.plot(test['time'],test['flux'],'c.')
# plt.plot([test['time'][0],test['time'][-1]],\
#          [(np.median(test['flux'])),(np.median(test['flux']))],'r-')
# plt.subplot(1,3,2)
# plt.plot(test['time'],norm_flux,'r-')
# plt.subplot(1,3,3)
# plt.plot(xf,yf,'c-')
# plt.show()



# for i in range(len(lc_list)):
#     test_curve = lc_list[i]
#     test = np.genfromtxt(test_curve,names="time,flux,et,ef")
#     flux = (test['flux'] - np.median(test['flux']))/(np.max(test['flux']) - np.min(test['flux']))
#     flux = np.minimum(flux, 0)
#     plt.figure(figsize=(2,2))
#     plt.plot(test['time'],flux,'k-')
#     plt.axis('off')
#     plt.gca().axes.get_yaxis().set_visible(False)
#     plt.gca().axes.get_xaxis().set_visible(False)
#     plt.savefig("/Users/laote/sdsu/Research/binary/tess_image/%s.jpg" %i)
#     print(i)
#     plt.show()







# last line of code