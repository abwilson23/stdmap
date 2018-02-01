#!/usr/bin/python

import math
import numpy as np

#======================================================================================#
# The following sections describe the behaviour of different maps that we are          #
# interested in studying. In particular, the first function of each section provides   #
# the next iterate of the map and the second function provides the jacobian. This      #
# module is used to aid in creation of map objects in the lib.py module.               #
#======================================================================================#


#=================================================================================#
#                                Map Class                                        #
#=================================================================================#

# To add new maps, provide next_iterate and get_jacobian functions in the map_info.py
# module. 
class Map:

    def __init__(self, next_iterate, get_jacobian, p=0):
        self.next = next_iterate
        self.jacobian = get_jacobian
        self.p = p 


#======================================================================================#
#                           The Standard Map of the Torus                              #
#======================================================================================#

# K = a constant which alters the dynamics of the system
e = 0.1
K = 1.2
Scale = 1

# Assuming points in R^2
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

def two_one_map_jacobian(pt):
    return (np.matrix('2 1; 1 1'))

#=======================================================================================#
#                       The Two-One-One Map (with perturbation)                         #
#=======================================================================================#

def two_one_map_perturbed_next(pt):
    x_t, y_t = pt[0], pt[1]
    x_n = (2*x_t + y_t + e*np.cos(2*np.pi*x_t)) % Scale
    y_n = (x_t + y_t + e*np.cos(2*np.pi*x_t)) % Scale
    return ((x_n, y_n))

def two_one_map_perturbed_jacobian(pt):
    return (np.matrix([[2-2*np.pi*e*np.sin(2*np.pi*pt[0]), 1], [1-2*np.pi*e*np.sin(2*np.pi*pt[0]), 1]]))

#======================================================================================#
#                           Initialized Map Classes                                    #
#======================================================================================#

std = Map(std_map_next, std_map_jacobian)
t1 = Map(two_one_map_next, two_one_map_jacobian)
tp = Map(two_one_map_perturbed_next, two_one_map_perturbed_jacobian)

