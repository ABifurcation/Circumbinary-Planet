import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import glob
from io import StringIO

curvedir = "/home/zlin/research/Curves2"
basedir = "/home/zlin/research"
basedir2 = "/home/zlin/research"

lc_dir = glob.glob("%s/*.jpg" %basedir)
unknown_dir = glob.glob("%s/*.jpg" %curvedir)


lc_dir = sorted(lc_dir, key=lambda name: int(name[48:-13]))
unknown_dir = sorted(unknown_dir, key=lambda name: int(name[56:-13]))

def twod_array(lc_dir):
    img = Image.open(lc_dir)

    rgb_img = np.array(img)

    b = np.zeros((rgb_img.shape[0],rgb_img.shape[1]),dtype=rgb_img.dtype)
    b[:,:] = rgb_img[:,:,0]

    return b


def threed_array(lc_dir,count=True):
    counts = 0
    datacube = None
    for lc in lc_dir:
        data = twod_array(lc)
        
        # add to the stack
        image = np.expand_dims(data, axis = 0)
        if datacube is None:
            datacube = image
        else:
            datacube = np.vstack((datacube, image))
        counts += 1
        if count == True:
            print(counts)
        else:
            None
            
    return datacube


def stack_flux(lc_list):
    datacube = None
    for lc in lc_list:
        data = np.genfromtxt(lc,names="time,flux,et,ef")
        flux = (data['flux'] - np.median(data['flux']))/(np.max(data['flux']) - np.min(data['flux']))
        if flux.shape == (1161,):
            flux = flux[19:-19]
        else:
            None
        flux = np.minimum(flux, 0)
        # add to the stack
        image = np.expand_dims(flux, axis = 0)
        if datacube is None:
            datacube = image
        else:
            datacube = np.vstack((datacube, image))

            
    return datacube

def cas_model(lc_dir,model):
    L = len(lc_dir)
    loop = 100
    step = int(L/loop)
    extra = L - step*loop
    
    solution = []
    for i in range(step):
        new_dir = lc_dir[loop*i:(i+1)*loop]
        data = threed_array(new_dir,count=False)
        image = data/255.0
        answer = model.predict(image)
        for j in range(loop):
            solution.append([new_dir[j][53:-4],round(answer[j][0],3)])
    
    extra_dir = lc_dir[-extra:]
    extra_data = threed_array(extra_dir,count=False)
    extra_image = extra_data/255.0
    extra_answer = model.predict(extra_image)
    for k in range(extra):
            solution.append([extra_dir[k][53:-4],round(extra_answer[k][0],3)])
    
    return solution


new_model = keras.models.load_model('9953_model.h5')

print(len(unknown_dir))
solution = cas_model(unknown_dir,new_model)
print(len(solution))

sort_solu = sorted(solution, key=lambda x: -x[1])
f = open("classification.txt", "w")
for i in range(len(sort_solu)):
    if sort_solu[i][1] > 0.6:
        f.write("%23s %9.3f\n"%(sort_solu[i][0],sort_solu[i][1]))
f.close()




