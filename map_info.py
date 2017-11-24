#!/usr/bin/python

import math
import numpy as np

#======================================================================================#
# The following sections describe the behaviour of different maps that we are          #
# interested in studying. In particular, the first function of each section provides   #
# the next iterate of the map and the second function provides the jacobian. This      #
# module is used to aid in creation of map objects in the lib.py module.               #
#======================================================================================#


#======================================================================================#
#                           The Standard Map of the Torus                              #
#======================================================================================#

# K = a constant which alters the dynamics of the system
K = 1
Scale = 1

def std_map_next(pt):
    y = (pt[1] + K/(2*np.pi)*np.sin(2*np.pi*pt[0])) % Scale
    x = (pt[0] + y) % Scale
    return ((x, y))

def std_map_jacobian(pt):
    return (np.matrix([[1, K*np.cos(pt[1])], [1, 1 + K*np.cos(pt[1])]]))


#=======================================================================================#
#                               The Two-One-One Map                                     #
#=======================================================================================#

def two_one_map_next(pt):
    x_t, y_t = pt[0], pt[1]
    x_n = (2*x_t + y_t) % Scale
    y_n = (x_t + y_t) % Scale
    return ((x_n, y_n))

def two_one_one_map_jacobian(y):
    return (np.matrix('2 1; 1 1')

