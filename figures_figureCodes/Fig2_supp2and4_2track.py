#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:11:13 2025

@author: Rebecca Poon
"""
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from scipy import signal
import trackpy.predict

#%% set up
path_root = '/path/to/repo/' + 'glass_movement_results'
dt = 10 #s
umpp = 1.61 #microns per pixel

##################
# MAKE VELOCITY DATAFRAME FROM MATLAB SEGMENTATION RESULTS
#%% get filament CENTRES and LENGTHS
f_all = []
for frame in range(500):
    # Centroid X, Centroid Y, End 1 X, End 1 Y, End 2 X, End 2 Y, Length
    f = pd.read_csv(path_root+'frame%04d.csv'%frame,sep = ',' ,usecols= [0,1,6] ,header = None  )    
    f = f.rename(columns = {0:'x',1:'y',6:'length'})
    f = f[f['x']!=0]
    f['frame'] = np.ones((len(f)),dtype = np.int8)*frame
    f = f.set_index('frame',drop = False)
    f_all.append(f)
    
#%% link with velocity prediction
# for tracking with nearest velocity: need to have each frame as an entry in a tuple,
# instead of in one huge concatenated dataframe. 

f_iter = tuple(f_all) 
pred = tp.predict.NearestVelocityPredict()
t = pd.concat(pred.link_df_iter(f_iter,30,memory = 14)) #used (30,memory=14) for making velocitydf_centroid_spline

#%% filter stubs and make ids
t2  = tp.filter_stubs(t,13)
ids = np.unique(t2['particle'])

#%% filter by condition (eg max displacement)
tokeep = []
for n_ in tqdm(ids):
    ptrack = t.loc[t['particle'] == n_]
    r = ptrack[['x','y']].to_numpy()
    disp = r - r[0]
    maxd = np.sqrt(np.max(disp[:,0]**2+disp[:,1]**2))
    if maxd > 15:          
        tokeep.append(n_)
t3 = t[t['particle'].isin(tokeep)]
#plt.imshow(frames[2], cmap = 'gray')
#tp.plot_traj(t3)

#%% make velocity df

vdf = pd.DataFrame(columns = ['particle','x','y','vx','vy','frame','length','speed'])
#%% make velocities
howsmooth = 10
poly_order = 3
data_toconcat = []
ids = np.unique(t2['particle'])    
for n_ in tqdm(ids):
    ptrack = t2.loc[t2['particle'] == n_].copy()
    r = ptrack[['x','y']].to_numpy()   
    
    # make speeds using splining method
    t_x, c_x, k_x = interpolate.splrep(ptrack['frame'],r[:,0], s=0, k=4)
    spline_x = interpolate.BSpline(t_x, c_x, k_x, extrapolate=False)
    t_y, c_y, k_y = interpolate.splrep(ptrack['frame'],r[:,1], s=0, k=4)
    spline_y = interpolate.BSpline(t_y, c_y, k_y, extrapolate=False)
    
    #smooth fitted x and y pos splines
    pos_x_smooth = signal.savgol_filter(spline_x(ptrack['frame']), howsmooth, poly_order)
    pos_y_smooth = signal.savgol_filter(spline_y(ptrack['frame']), howsmooth, poly_order)
    v_x = np.gradient(pos_x_smooth)
    v_y = np.gradient(pos_y_smooth)
    v = np.sqrt(v_x**2+v_y**2) # this is in pix/frame

    # then update ptrack df
    ptrack['vx'] = v_x
    ptrack['vy'] = v_y
    ptrack['speed'] = v      
    # then add this ptrack to the master df
    data_toconcat.append(ptrack)
    
vdf = pd.concat(data_toconcat,axis = 0,sort = True)
    
#%% save vdf
vdf.to_pickle(path_root+'velocitydf_centroid_spline.pkl')
# velocitydf contains columns of:
# frame, filament length [pix], particle (ID), speed [pix/frame], velocity_x&y [pix/fr], centroid (x,y) position [pix]
    
    