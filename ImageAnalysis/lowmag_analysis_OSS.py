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


