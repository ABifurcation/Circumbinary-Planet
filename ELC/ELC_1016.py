import numpy as np
import matplotlib.pyplot as plt

basedir = '/Users/laote/sdsu/Research/binary/ELC/ELC_rt'

def light_curve(lc_dir,xlim=(0,0),ylim=(0,0),look=False):
    lc = lc_dir + '/modelU_fit1.linear'
    res = lc_dir + '/ELCresidualsU_fit1.fold'
    data = np.genfromtxt(lc,names="time,flux")
    resdata = np.genfromtxt(res,names="time,flux")
    Time = data['time']
    flux = data['flux']
    resTime = resdata['time']
    resflux = resdata['flux']

    if look == True:
        plt.figure(figsize=(10,5))
        plt.plot(Time,flux,'c')
        plt.plot(resTime,resflux+1,'r.--')
        plt.xlabel('time');plt.ylabel('flux')
        if xlim[0] != 0:
            plt.xlim(xlim[0],xlim[1])
        if ylim[0] != 0:
            plt.ylim(ylim[0],ylim[1])
        plt.grid()
        plt.show()
    
    return 0


light_curve(basedir,look=True)