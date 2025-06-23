#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    
INPUT:

OUTPUT: 

HISTORY:    
Created on Fri Mar 22 13:46:30 2024 From PRUEBA_THRESH.py (from Ana Maya Sevilla). Modified it to give as output the list borders(),   where each entry is a numpy array of row and col coordinate of the border pixels, in order. No positions are repeating in series. The first and last elements of the array are the same (closed loop)
@author: mpolin
"""

import numpy as np
from enum import IntEnum

class Directions(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


NORTH = Directions.NORTH
EAST = Directions.EAST
SOUTH = Directions.SOUTH
WEST = Directions.WEST

def trace_boundary(image):
    padded_img = np.pad(image, 1)

    img = padded_img[1:-1, 1:-1]
    img_north = padded_img[:-2, 1:-1]
    img_south = padded_img[2:, 1:-1]
    img_east = padded_img[1:-1, 2:]
    img_west = padded_img[1:-1, :-2]

    border = np.zeros((4, *padded_img.shape), dtype=np.intp)

    border[NORTH][1:-1, 1:-1] = (img == 1) & (img_north == 0)
    border[EAST][1:-1, 1:-1] = (img == 1) & (img_east == 0)
    border[SOUTH][1:-1, 1:-1] = (img == 1) & (img_south == 0)
    border[WEST][1:-1, 1:-1] = (img == 1) & (img_west == 0)

    adjacent = np.zeros((4, *image.shape), dtype=np.intp)
    adjacent[NORTH] = np.argmax(np.stack(
        (border[WEST][:-2, 2:],
         border[NORTH][1:-1, 2:],
         border[EAST][1:-1, 1:-1])
    ), axis=0)
    adjacent[EAST] = np.argmax(np.stack(
        (border[NORTH][2:, 2:],
         border[EAST][2:, 1:-1],
         border[SOUTH][1:-1, 1:-1])
    ), axis=0)
    adjacent[SOUTH] = np.argmax(np.stack(
        (border[EAST][2:, :-2],
         border[SOUTH][1:-1, :-2],
         border[WEST][1:-1, 1:-1])
    ), axis=0)
    adjacent[WEST] = np.argmax(np.stack(
        (border[SOUTH][:-2, :-2],
         border[WEST][:-2, 1:-1],
         border[NORTH][1:-1, 1:-1])
    ), axis=0)

    directions = np.zeros((len(Directions), *image.shape, 3, 3), dtype=np.intp)
    directions[NORTH][..., :] = [(3, -1, 1), (0, 0, 1), (1, 0, 0)]
    directions[EAST][..., :] = [(-1, 1, 1), (0, 1, 0), (1, 0, 0)]
    directions[SOUTH][..., :] = [(-1, 1, -1), (0, 0, -1), (1, 0, 0)]
    directions[WEST][..., :] = [(-1, -1, -1), (0, -1, 0), (-3, 0, 0)]

    proceding_edge = directions[
        np.arange(len(Directions))[:, np.newaxis, np.newaxis],
        np.arange(image.shape[0])[np.newaxis, :, np.newaxis],
        np.arange(image.shape[1])[np.newaxis, np.newaxis, :],
        adjacent
    ]

    unprocessed_border = border[:, 1:-1, 1:-1].copy()
    borders = list()
    for start_pos in zip(*np.nonzero(unprocessed_border)):
            if not unprocessed_border[start_pos]:
                continue
            idx = len(borders)
            borders.append(list())
            start_arr = np.array(start_pos, dtype=np.intp)
            current_pos = start_arr
            border_pos = current_pos[1:][np.newaxis,:]
            while True:
                unprocessed_border[tuple(current_pos)] = 0
                border_pos = np.concatenate((border_pos,current_pos[1:][np.newaxis,:]))
                current_pos += proceding_edge[tuple(current_pos)]
                if np.all(current_pos == np.array(start_pos)):
                    ww=np.asarray(np.any((border_pos[1:,:]-border_pos[:-1,:])!=0,axis=1))
                    border_pos=np.concatenate((border_pos[:-1,:][ww],border_pos[-1,:][np.newaxis,:]))
                    borders[idx]=border_pos
                    break

 
    return border_pos