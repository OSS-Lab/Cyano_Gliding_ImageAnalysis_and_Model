#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import trackpy.predict


#%% LOAD IN SAVED VDF

# NOTE: for glass data for paper: both saved velocitydf_centroid_spline have same umpp value of 1.61 
path_root = '/path/to/repo/'
# velocitydf contains columns of:
# frame, filament length [pix], particle (ID), speed [pix/frame], velocity_x&y [pix/fr], centroid (x,y) position [pix]
   
dt = 10 # frame interval in [s]
umpp = 1.61 # microns per pixel scaling 

vdf1 = pd.read_pickle(path_root+ '/data/glass_movement_results/velocitydf_centroid_spline_vid1.pkl')
vdf2 = pd.read_pickle(path_root+ '/data/glass_movement_results/velocitydf_centroid_spline_vid2.pkl')
vdf2['particle'] = vdf2['particle']+1+np.max(vdf1['particle'])

vdf = pd.concat([vdf1,vdf2],axis = 0)

ids = np.unique(vdf['particle'])
#%% make average v_array
# gives values averaged over each particle
# columns are [length, median speed, max speed, particleid]

vdf_mean = vdf.groupby(['particle'],as_index=False).median() # or median or max


#%% clean up by max track length and max displacement
# also filter so it has to be moving for more than half the track

t4 = tp.filter_stubs(vdf,300) #400
ids4 = np.unique(t4['particle'])
maxdispids = []
for n_ in range(len(ids4)):
    ptrack = t4.loc[t4['particle'] == ids4[n_]]    
    r = ptrack[['x','y']].values
    disp = r - r[0]
    maxd = np.sqrt(np.max(disp[:,0]**2+disp[:,1]**2))
    speed = ptrack[['speed']].to_numpy(dtype = np.float32)
    speed = signal.medfilt(speed[:,0],3)    
    if (maxd > 150) & (len(speed[speed>4]) > len(speed)/3): #maxd 200
        maxdispids.append(ids4[n_])    

#%% fill in dwell times etc


data_toconcat = []
toplot = np.linspace(0,len(maxdispids),40).astype(int)
for n in range(len(maxdispids)): #18
    ptrack = t4.loc[t4['particle'] == maxdispids[n]]
   # ptrack = ptrack[ptrack['frame']>300]    
    
    speed = ptrack[['speed']].to_numpy(dtype = np.float32)
    speed = speed[:,0]
 
    
    speed_threshold = 0.15*np.max(signal.medfilt(speed,3)[10:-10]); #0.2*dt/umpp
    
    speedthresh = speed > speed_threshold#0.15*dt/umpp
    speedthresh = speedthresh.astype(int)
    #speeddotthresh = speedthresh.copy()
    

    v = ptrack[['vx','vy','frame']].to_numpy(dtype = np.float32)
    v_norm = v.copy()
    v_norm[:,0:2] /=speed[:,np.newaxis] #normalise
    #v_norm  = v_norm[speed[:,0]>3] # excise frames when speed < 3pix/frame        
    v_norm  = v_norm[speed>speed_threshold] # excise frames when speed < 0.2um/s
    
    dotproduct = v_norm[:,1:].copy()
    dotproduct[1:,0] = np.sum(v_norm[1:,0:2]*v_norm[:-1,0:2],axis = 1) #dot product
    dotproduct[1:-1,1] = (dotproduct[1:-1,1]+ dotproduct[:-2,1]) /2 #time
    dotthresh = np.interp(v[:,2], dotproduct[:,1], dotproduct[:,0])
    thresholdvalue = 0.6# 0.4 for endpoints
    dotthresh[dotthresh>thresholdvalue] = 1
    dotthresh[dotthresh<thresholdvalue] = -1
    #speeddotthresh[np.logical_and(dotthresh == 1, speedthresh == 0)] = 2 
    #speeddotthresh -=1 
    # SPEEDDOTTHRESH is 0 during gliding, -1 in a reverse, +1 in a continue
    # if dot product is -ve then it reverses 
    cross = np.cross(v_norm[1:,:2],v_norm[:-1,:2])
    #allcross = np.concatenate((allcross,cross)) # angle of track over time
#%   
    
    thresh = speedthresh.astype(int) #THRESHOLD IS SPEED THRESHOLD
    diff = thresh[1:]-thresh[:-1]
    uppeaks = np.where(diff == 1)[0]     
    downpeaks = np.where(diff == -1)[0]

    #uppeaks, downpeaks, dwell times are in s, and frequencies is in Hz
    if len(uppeaks)>1:
        
        # edit data so track always start in 'gliding' mode. So first peak will be down and last will be up
        if uppeaks[0]<downpeaks[0]:
            uppeaks = uppeaks[1:]
        if downpeaks[-1]>uppeaks[-1]:
            downpeaks = downpeaks[:-1]
        uppeaks += 1
        
        t = ptrack['frame'].to_numpy(np.float16)*dt
        
        dwell_mids = ((downpeaks +uppeaks)/2).astype(int) #index of dwell midpoints
        dwell_mids[dwell_mids >= len(t)] = len(t)-1
        dwelltypes = dotthresh[dwell_mids]
        
        uppeaks_realunits = t[uppeaks]
        downpeaks_realunits = t[downpeaks]   
        down_lengths = uppeaks_realunits-downpeaks_realunits; # dwell times
        up_lengths = downpeaks_realunits[1:]-uppeaks_realunits[:-1]; # gliding times  
        
        
        thisparticle = pd.DataFrame(columns = ['particle','dwellstarts','dwellends','dwelltypes','dwelllengths'])
        thisparticle['dwellstarts'] = downpeaks # in index
        thisparticle['dwellends'] = uppeaks
        thisparticle['dwelltypes'] = dwelltypes #alldwelltypes = np.zeros((0))
        thisparticle['dwelllengths'] = down_lengths #alldwelltimes = np.zeros((0))
        thisparticle['particle'] = np.ones(len(downpeaks))*maxdispids[n]
        # filament length, speed, positions, t, N reversals/total time, we can get from 
        # remaking ptrack or using vdf.groupby
        

        data_toconcat.append(thisparticle)
        
dwelldf = pd.concat(data_toconcat,axis = 0,sort = False)

#%%
dwellids = np.unique(dwelldf['particle'])
vdf_dwells = vdf[vdf['particle'].isin(dwellids)]
#%% make averages for just the dwell tracks

dwelldf_med = vdf_dwells.groupby(['particle'],as_index=False).median() # or median or max
dwelldf_mean = vdf_dwells.groupby(['particle'],as_index=False).mean() # or median or max

alldwelllengths = dwelldf['dwelllengths'].values
reversedwells = alldwelllengths[dwelldf['dwelltypes'] == -1]
forwarddwells = alldwelllengths[dwelldf['dwelltypes'] == 1]


#%% plot Fig2_supplement2

plt.rc('font', size=12)
fig, ax = plt.subplots(1,2, figsize=(8,4), dpi=300,layout = 'constrained')

ax[0].text(-0.2, 1, 'A', transform=ax[0].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')
ax[0].scatter(np.array(dwelldf_mean['length'])*umpp,np.array(dwelldf_med['speed'])*umpp/dt,s = 50,color = 'gray')
ax[0].set_xlabel('Filament length [μm]')
ax[0].set_ylabel('Median speed [μm/s]')
ax[0].set_ylim([0, 2.0])
ax[0].set_xlim([11,900])
ax[0].set_xscale('log')
ax[0].legend(['N = %d'%(len(dwellids))])


ax[1].text(-0.1, 1, 'B', transform=ax[1].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')
ax[1].hist([ reversedwells,forwarddwells],color = ['tab:cyan','tab:orange'],bins = np.arange(0,1700,150),density = True)
ax[1].set_ylabel(r'probability density [s$^{-1}$]')
ax[1].set_xlabel('Dwell time [s]')
ax[1].set_yscale('log')
ax[1].legend(['%d filaments \n%d reversals'%(len(dwellids),len(reversedwells)), '%d stop-go events'%len(forwarddwells)])
ax[1].set_xlim([0,1600])


fig.savefig('Fig2_supplement2.pdf',dpi = 300,bbox_inches = 'tight')

#plt.xscale('log')
plt.show()


#%% get reversal/stopgo rates
reversenum = []
stopgonum = []
for id in dwellids:
    thisparticle = dwelldf[dwelldf['particle'] == id]
    reversenum.append(len(thisparticle[thisparticle['dwelltypes'] == -1]))
    stopgonum.append(len(thisparticle[thisparticle['dwelltypes'] == 1]))
trackdurations = vdf_dwells.groupby(['particle'],as_index=False).size()
reverserate = np.array(reversenum,dtype = float)/(trackdurations['size'].to_numpy(dtype = float)*dt)
stopgorate = np.array(stopgonum,dtype = float)/(trackdurations['size'].to_numpy(dtype = float)*dt)


#%% reversal frequency comparison with agar
agardata = pd.read_csv(path_root + '/data/0_all_data_summary.csv', delimiter = '\t')
agarreversalfreq = agardata['nReversals'].values/agardata['obsDuration[sec]'].values

#%% plot reversal/stopgo rates
temp = np.concatenate((reverserate[:,np.newaxis],stopgorate[:,np.newaxis]),axis = 1)


plt.rc('font', size=12)
fig, ax = plt.subplots(1,2, figsize=(8,4),layout = 'constrained')

ax[0].text(-0.2, 1, 'A', transform=ax[0].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')
ax[0].vlines(dwelldf_mean['length']*umpp,ymin = np.min(temp,axis = 1),ymax = np.max(temp,axis = 1),colors = [0.5,0.5,0.5,0.4],label='_nolegend_',linewidth = 0.7)
ax[0].scatter(dwelldf_mean['length']*umpp, reverserate,c = 'tab:cyan',s = 7)
ax[0].scatter(dwelldf_mean['length']*umpp, stopgorate, c ='tab:orange',s = 7)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlim([11,900])
ax[0].set_ylim([0.0001,0.05])
ax[0].set_xlabel('Filament length [μm]')
ax[0].set_ylabel(r'stopping frequency [s$^{-1}$]')
ax[0].legend(['%d filaments \nreversal frequency'%(len(dwellids)), 'stop-go frequency'])


ax[1].text(-0.2, 1, 'B', transform=ax[1].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')
ax[1].boxplot([reverserate,agarreversalfreq], tick_labels = ['glass data','agar data'],widths = 0.6)
        # ,color = ['tab:purple','tab:olive'] )
#plt.legend(['glass data','agar data'])
ax[1].set_ylabel(r'reversal frequency [s$^{-1}$]')
#plt.xlabel('probability density [s]')
#plt.ylim([-0.00005,0.01])
ax[1].set_yscale('log')
ax[1].set_ylim([0.0001,0.05])

plt.show()

fig.savefig('Fig2_supplement4.pdf',dpi = 300,bbox_inches = 'tight')
