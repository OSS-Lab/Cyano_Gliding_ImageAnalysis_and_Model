import numpy as np
import cv2
from skimage.morphology import skeletonize, closing, disk
from skimage import feature as sk_feature
import os
import skimage.filters as sk_fil
import skimage.measure as sk_mes
import matplotlib.pyplot as plt
import re
from scipy.ndimage import convolve
from scipy import interpolate
from scipy.stats import linregress
from scipy.spatial.distance import euclidean
import math

#helper function for grouping an array into chunks of sequential values
#from https://stackoverflow.com/questions/2154249/identify-groups-of-consecutive-numbers-in-a-list
def group(values):
    """return the first and last value of each continuous set in a list of sorted values"""

    values = sorted(values)
    first = last = values[0]

    for index in values[1:]:
        if index - last > 1:  # triggered if in a new group
            yield first, last

            first = index  # update first only if in a new group

        last = index  # update last on every iteration

    yield first, last  # this is needed to yield the last set of numbers

#helper function for getting midline
def neighborDat(mydat1,mydat2,endX,endY):
    minDist = 10000
    for i in range(0,len(mydat1),1):
        dist=np.sqrt((mydat1[i]-endX)**2+(mydat2[i]-endY)**2)
        if (dist<minDist): 
            minDist = dist
            neighbourID = i
    return neighbourID

#get midline from contour
def getMidLineOfCountour(x,y,peak1_S,peak2_S):
    #average using pairs from one end
    midLine_x = np.empty(shape=[0, 1])
    midLine_y = np.empty(shape=[0, 1])
    x1_cop = np.copy(x)
    y1_cop = np.copy(y)
    endIndex = np.min([peak1_S,peak2_S])
    endX = x[endIndex]
    endY = y[endIndex]
    x1_cop = np.delete(x1_cop, endIndex)
    y1_cop = np.delete(y1_cop, endIndex)
    while (len(x1_cop)>1):
        meNeighbour1 = neighborDat(x1_cop,y1_cop,endX,endY)
        meNeighbour1_x, meNeighbour1_y = x1_cop[meNeighbour1],y1_cop[meNeighbour1]
        x1_cop = np.delete(x1_cop, meNeighbour1)
        y1_cop = np.delete(y1_cop, meNeighbour1)
        meNeighbour2 = neighborDat(x1_cop,y1_cop,endX,endY)
        meNeighbour2_x, meNeighbour2_y = x1_cop[meNeighbour2],y1_cop[meNeighbour2]
        x1_cop = np.delete(x1_cop, meNeighbour2)
        y1_cop = np.delete(y1_cop, meNeighbour2)
        midLine_x = np.append(midLine_x,(meNeighbour1_x+meNeighbour2_x)/2)
        midLine_y = np.append(midLine_y,(meNeighbour1_y+meNeighbour2_y)/2)
    if len(x1_cop)==1:
        midLine_x = np.append(midLine_x,x1_cop[0])
        midLine_y = np.append(midLine_y,y1_cop[0])
    t_pts_mid = np.arange(0, len(midLine_x),1) #indices for each point
    t_sparse = np.arange(0, len(midLine_x),5) #indices for every 5th point
    if t_sparse[len(t_sparse)-1] != len(midLine_x)-1: t_sparse = np.append(t_sparse,len(midLine_x)-1)
    midspline_x = interpolate.splrep(t_sparse,midLine_x[t_sparse],s=0, k=1)
    midspline_y = interpolate.splrep(t_sparse,midLine_y[t_sparse],s=0, k=1)
    mid_x_smooth = interpolate.splev(t_pts_mid, midspline_x)
    mid_y_smooth = interpolate.splev(t_pts_mid, midspline_y)
    return mid_x_smooth, mid_y_smooth

#get midline from contour - only few key differences to previous function
def getMidLineOfCountour_better(x,y,peak1_S,peak2_S):
    #average using pairs from one end
    midLine_x = np.empty(shape=[0, 1])
    midLine_y = np.empty(shape=[0, 1])
    x1_cop = np.copy(x)
    y1_cop = np.copy(y)
    endIndex = np.min([peak1_S,peak2_S])
    endIndexToStop = np.max([peak1_S,peak2_S])
    endX = x[endIndex]
    endY = y[endIndex]
    x1_cop = np.delete(x1_cop, endIndex)
    y1_cop = np.delete(y1_cop, endIndex)
    midLine_x = np.append(midLine_x,endX)
    midLine_y = np.append(midLine_y,endY)
    stopMe = False
    while (stopMe==False):
        meNeighbour1 = neighborDat(x1_cop,y1_cop,endX,endY)
        meNeighbour1_x, meNeighbour1_y = x1_cop[meNeighbour1],y1_cop[meNeighbour1]
        x1_cop = np.delete(x1_cop, meNeighbour1)
        y1_cop = np.delete(y1_cop, meNeighbour1)
        meNeighbour2 = neighborDat(x1_cop,y1_cop,endX,endY)
        meNeighbour2_x, meNeighbour2_y = x1_cop[meNeighbour2],y1_cop[meNeighbour2]
        x1_cop = np.delete(x1_cop, meNeighbour2)
        y1_cop = np.delete(y1_cop, meNeighbour2)
        if len(x1_cop)<4 or (meNeighbour1_x == x[endIndexToStop] and meNeighbour1_y == y[endIndexToStop]) or (meNeighbour2_x == x[endIndexToStop] and meNeighbour2_y == y[endIndexToStop]): 
            stopMe = True
            break
        midLine_x = np.append(midLine_x,(meNeighbour1_x+meNeighbour2_x)/2)
        midLine_y = np.append(midLine_y,(meNeighbour1_y+meNeighbour2_y)/2)
        endX, endY = (meNeighbour1_x+meNeighbour2_x)/2, (meNeighbour1_y+meNeighbour2_y)/2    #update by setting end point as last used point
    #smooth only if enought data points found (which it should be!)
    if (len(midLine_x)>2):
        t_pts_mid = np.arange(0, len(midLine_x),1) #indices for each point
        t_sparse = np.arange(0, len(midLine_x),5) #indices for every 5th point
        if t_sparse[len(t_sparse)-1] != len(midLine_x)-1: t_sparse = np.append(t_sparse,len(midLine_x)-1)
        midspline_x = interpolate.splrep(t_sparse,midLine_x[t_sparse],s=0, k=1)
        midspline_y = interpolate.splrep(t_sparse,midLine_y[t_sparse],s=0, k=1)
        mid_x_smooth = interpolate.splev(t_pts_mid, midspline_x)
        mid_y_smooth = interpolate.splev(t_pts_mid, midspline_y)
    else:
        mid_x_smooth, mid_y_smooth = np.copy(midLine_x), np.copy(midLine_y)
    return mid_x_smooth, mid_y_smooth,midLine_x,midLine_y

#based on:https://stackoverflow.com/questions/35530634/detecting-the-centre-of-a-curved-shape-with-opencv
def getMidPointOfContour_OSS(x, y):
    cx = 0
    cy = 0
    minDist = 1000000
    half = int(len(x)/2)
    for i in range(0,half):
        dist = np.sqrt((x[i]-x[i+half])**2+(y[i]-y[i+half])**2)
        if (dist < minDist):
            cx = (x[i] + x[i+half]) / 2
            cy = (y[i] + y[i+half]) / 2
            minDist = dist
    minDist2 = 1000000
    half2 = int(len(x)/4)
    for i in range(0,half2):
        dist = np.sqrt((x[i]-x[i+half2])**2+(y[i]-y[i+half2])**2)
        if (dist < minDist2):
            cx2 = (x[i] + x[i+half2]) / 2
            cy2 = (y[i] + y[i+half2]) / 2
            minDist2 = dist
    return cx, cy, cx2, cy2

def getMidSplineFromCorners_OSS(x1, y1, cornerLow, cornerHigh):
    if (cornerLow==0): 
        xSideA_Ti, ySideA_Ti = x1[cornerLow:cornerHigh], y1[cornerLow:cornerHigh]
        xSideB_Ti, ySideB_Ti = x1[len(x1):cornerHigh:-1], y1[len(x1):cornerHigh:-1]
    if (cornerLow==len(x1)): 
        xSideA_Ti, ySideA_Ti = x1[cornerLow:cornerHigh:-1], y1[cornerLow:cornerHigh:-1]
        xSideB_Ti, ySideB_Ti = x1[0:cornerHigh], y1[0:cornerHigh]
    if (cornerLow!=len(x1) and cornerLow!=0):
        if (cornerLow>cornerHigh):
            xSideA_Ti, ySideA_Ti = x1[cornerLow:cornerHigh:-1], y1[cornerLow:cornerHigh:-1]
            xSideB_Ti = np.r_[x1[cornerLow:len(x1)], x1[0:cornerHigh]]
            ySideB_Ti = np.r_[y1[cornerLow:len(x1)], y1[0:cornerHigh]]
        else:
            xSideA_Ti, ySideA_Ti = x1[cornerLow:cornerHigh], y1[cornerLow:cornerHigh]
            xSideB_Ti = np.r_[x1[cornerLow-1:0:-1], x1[len(x1):cornerHigh:-1]]
            ySideB_Ti = np.r_[y1[cornerLow-1:0:-1], y1[len(x1):cornerHigh:-1]]
    return xSideA_Ti, ySideA_Ti, xSideB_Ti, ySideB_Ti

def getMidSplineFromContourSides_OSS(xSideA_Ti, ySideA_Ti, xSideB_Ti, ySideB_Ti, s, k, nest):
    #the pre-processing is needed to remove any duplicate points
    #see here for details: https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
    
    #remove any duplicate points from the sides: otherwise, interpolation throws an error!
    okay = np.where(np.abs(np.diff(xSideA_Ti)) + np.abs(np.diff(ySideA_Ti)) > 0)
    xSideA_Ti2 = xSideA_Ti[okay]
    ySideA_Ti2 = ySideA_Ti[okay]

    #remove any duplicate points from the sides: otherwise, interpolation throws an error!
    okay = np.where(np.abs(np.diff(xSideB_Ti)) + np.abs(np.diff(ySideB_Ti)) > 0)
    xSideB_Ti2 = xSideB_Ti[okay]
    ySideB_Ti2 = ySideB_Ti[okay]
    
    t_sideA, u_sideA = interpolate.splprep([xSideA_Ti2, ySideA_Ti2], s=s, k=k, nest=-1)
    t_sideB, u_sideB = interpolate.splprep([xSideB_Ti2, ySideB_Ti2], s=s, k=k, nest=-1)
    myXvalues = np.linspace(0, 1, 600)
    xn_sideA, yn_sideA = interpolate.splev(myXvalues, t_sideA)
    xn_sideB, yn_sideB = interpolate.splev(myXvalues, t_sideB)

    #get the average of the two splines from the two sides, so to get a midspline
    xn_M, yn_M = (xn_sideA+xn_sideB)/2, (yn_sideA+yn_sideB)/2
    midSpline_Ti = np.vstack((xn_M,yn_M)).T
    
    return xn_M, yn_M, midSpline_Ti

def get_indicesForCorners_OSS(cnt, extRight, extLeft, extTop, extBot):
    indicesforRightCorner1=np.where(cnt[:, 0, 0]==extRight[0])[0]
    index2ary=np.where(cnt[indicesforRightCorner1, 0, 1]==extRight[1])[0]
    indexforRightCorner = indicesforRightCorner1[index2ary][0]

    indicesforLeftCorner1=np.where(cnt[:, 0, 0]==extLeft[0])[0]
    index2ary=np.where(cnt[indicesforLeftCorner1, 0, 1]==extLeft[1])[0]
    indexforLeftCorner = indicesforLeftCorner1[index2ary][0]

    indicesforTopCorner1=np.where(cnt[:, 0, 1]==extTop[1])[0]
    index2ary=np.where(cnt[indicesforTopCorner1, 0, 0]==extTop[0])[0]
    indexforTopCorner = indicesforTopCorner1[index2ary][0]

    indicesforBottomCorner1=np.where(cnt[:, 0, 1]==extBot[1])[0]
    index2ary=np.where(cnt[indicesforBottomCorner1, 0, 0]==extBot[0])[0]
    indexforBottomCorner = indicesforBottomCorner1[index2ary][0]
    
    return indexforRightCorner, indexforLeftCorner, indexforTopCorner, indexforBottomCorner

def get_filamentLength_OSS(midSpline):
    filamentLength = 0
    for k in range(1,len(midSpline)):
        delta = np.linalg.norm(midSpline[k,:] - midSpline[k-1,:])
        filamentLength += delta
    return filamentLength

def sortPointsOut(points_x, points_y):
    #remove duplicates
    uniquePs = np.unique(np.array([points_x, points_y]).T, axis=0)
    points_x1 = uniquePs[:,0] 
    points_y1 = uniquePs[:,1]

    #get orders
    order_x = np.argsort(points_x1)
    order_y = np.argsort(points_y1)

    #use EW orientation
    points_x_EW = points_x1[order_x]
    points_y_EW = points_y1[order_x]

    #use NS orientation
    points_x_NS = points_x1[order_y]
    points_y_NS = points_y1[order_y]

    return points_x_EW, points_y_EW, points_x_NS, points_y_NS

#get extreme points using cumulative distance between corners
def getExtremePoints_fromCorners_contour(x1,y1,indexforRightCorner, indexforLeftCorner, indexforTopCorner, indexforBottomCorner):
    maxDist = 0
    my_ids = [indexforRightCorner, indexforLeftCorner, indexforTopCorner, indexforBottomCorner]
    for i in range(0,len(my_ids),1):
        P1 = my_ids[i]
        for j in range(0,len(my_ids),1):
            P2 = my_ids[j]
            cumdist = 0
            if (i>j):
                for k in range(j,i,1):
                    cumdist = cumdist + np.sqrt((x1[k]-x1[k+1])**2+(y1[k]-y1[k+1])**2)
                if (cumdist>maxDist):
                    maxDist = cumdist
                    myP1 = P1
                    myP2 = P2
            else:
                for k in range(i,j,1):
                    cumdist = cumdist + np.sqrt((x1[k]-x1[k+1])**2+(y1[k]-y1[k+1])**2)
                if (cumdist>maxDist):
                    maxDist = cumdist
                    myP1 = P1
                    myP2 = P2
    return myP1,myP2

def getExtremePoints_contour(x1,y1):
    maxDist = 0;
    for i in range(0,len(x1),1):
        for j in range(0,len(x1),1):
            dist_ij = np.sqrt((x1[i]-x1[j])**2+(y1[i]-y1[j])**2)
            if (dist_ij>maxDist):
                maxDist = dist_ij
                P1 = i
                P2 = j
    return P1,P2

def findExtremes_and_orderSkeleton(skeleton_line_x,skeleton_line_y):
    maxDist = 0
    P1, P2 = 0, 0
    for i in range(0,len(skeleton_line_x),1):
        for j in range(0,len(skeleton_line_x),1):
            dist_ij = np.sqrt((skeleton_line_x[i]-skeleton_line_x[j])**2+(skeleton_line_y[i]-skeleton_line_y[j])**2)
            if (dist_ij>maxDist):
                maxDist = dist_ij
                P1 = i
                P2 = j
    startP = np.min([P1,P2])
    x_skl_cop = np.copy(skeleton_line_x)
    y_skl_cop = np.copy(skeleton_line_y)
    skl_x = np.empty(shape=[0, 1])
    skl_y = np.empty(shape=[0, 1])
    skl_x = np.append(skl_x,x_skl_cop[startP])
    skl_y = np.append(skl_y,y_skl_cop[startP])
    x_skl_cop = np.delete(x_skl_cop, startP)
    y_skl_cop = np.delete(y_skl_cop, startP)
    while (len(x_skl_cop)>1):
        minDist = 1000
        for i in range(0,len(x_skl_cop),1):
            dist = np.sqrt((skeleton_line_x[startP]-x_skl_cop[i])**2+(skeleton_line_y[startP]-y_skl_cop[i])**2)
            if (dist<minDist):
                minDist = dist
                nextP = i
        skl_x = np.append(skl_x,x_skl_cop[nextP])
        skl_y = np.append(skl_y,y_skl_cop[nextP])
        x_skl_cop = np.delete(x_skl_cop, nextP)
        y_skl_cop = np.delete(y_skl_cop, nextP)
    if len(x_skl_cop)==1:
        skl_x = np.append(skl_x,x_skl_cop[0])
        skl_y = np.append(skl_y,y_skl_cop[0])
    return skl_x, skl_y

