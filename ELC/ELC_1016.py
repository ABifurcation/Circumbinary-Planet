import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',size=10) 

basedir = '/Users/laote/sdsu/Research/binary/ELC/ELC_rt'

def light_curve(lc_dir,xlim=(0,0),ylim=(0,0),look=False):
    # lc = lc_dir + '/modelU_fit2.linear'
    lc = lc_dir +'/modelU_T.linear'
    # lc = lc_dir + '/ELCresidualsU_5675.fold'
    res = lc_dir + '/ELCdataU.fold'
    # res = lc_dir + '/ELCresidualsU_A183.fold'
    # res = lc_dir + '/ELCresidualsU_T93.fold'
    data = np.genfromtxt(lc,names="time,flux")
    resdata = np.genfromtxt(res,names="time,flux,error")
    Time = data['time']
    flux = data['flux']
    resTime = resdata['time']
    resflux = resdata['flux']

    w = np.abs(resflux) > 7
    print(resdata['time'][w])
    # f = open(basedir+'/res_100.txt','w')
    # for i in range(len(resdata['time'][w])):
    #     f.write('%.6f\n'%(resdata['time'][w][i]))
    # f.close()

    if look == True:
        plt.figure(figsize=(10,5))
        plt.plot(Time,flux,'r-')
        plt.plot(resTime,resflux,'b.-')
        # plt.errorbar(resTime,resflux, yerr=resdata['error'],fmt='.--',color='r',ecolor='c',capsize=4)
        plt.xlabel('time');plt.ylabel('flux')
        if xlim[0] != 0:
            plt.xlim(xlim[0],xlim[1])
        if ylim[0] != 0:
            plt.ylim(ylim[0],ylim[1])
        # plt.xlim(1129,1131.5)
        # plt.ylim(0.99,1.01)
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    return 0


light_curve(basedir,look=True)