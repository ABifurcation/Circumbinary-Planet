from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
import os
import glob
from io import StringIO

basedir = "/home/zlin/research"
# curvedir = basedir + '/Curves_all'
curvedir = '/home/cgoodteacher/Research/Curves2'

oridata_dir = glob.glob("%s/*.txt" %curvedir)

# lc_dir = sorted(lc_dir, key=lambda name: int(name[48:-13]))

def find_kmean(datadir):
    data = np.genfromtxt(datadir,names="time,flux,et,ef")
    use = (data['ef'] == 0)

    time_diff = data['time'][use][1:] - data['time'][use][:-1]
    error_point = np.argmax(time_diff)
    
    time = np.append(data['time'][use][20:error_point-50],data['time'][use][error_point+50:-20])
    flux = np.append(data['flux'][use][20:error_point-50],data['flux'][use][error_point+50:-20])

    med = np.median(flux) 
    mean = np.mean(flux) 
    std = np.std(flux)
    norm_flux = (flux - np.median(flux))/(np.max(flux) - np.min(flux))
    

    s1 = round(len(flux[(flux > mean)])/len(flux[(flux < mean)]),5)
    s2 = round(np.abs(np.min(norm_flux)),5)
    s3 = len(flux[flux < (med - 5*std)])
    s4 = len(flux[flux < (med - 4*std)])
    s5 = len(flux[flux < (med - 3*std)])
    
    return [s1,s2,s3,s4,s5]

kmean_data = []
for lc in oridata_dir:
    kmean_data.append(find_kmean(lc))

np.save("kmean_data.npy",kmean_data)
kmean_data = np.load("kmean_data.npy")

kmeans = KMeans(n_clusters=5, random_state=0).fit(kmean_data)

l1 = len(kmeans.labels_[kmeans.labels_ == 0])
l2 = len(kmeans.labels_[kmeans.labels_ == 1])
l3 = len(kmeans.labels_[kmeans.labels_ == 2])
l4 = len(kmeans.labels_[kmeans.labels_ == 3])
l5 = len(kmeans.labels_[kmeans.labels_ == 4])

nonbinary = np.argmax([l1,l2,l3,l4,l5])
binary = (kmeans.labels_ != nonbinary)

binary_list = oridata_dir[binary]

f = open("binary.txt", "w")
for i in range(len(binary_list)):
    f.write("%s \n"%(binary_list[i]))
f.close()










# last of line