#!/usr/bin/python

import math
import numpy as np

SCALE = 1

#======================================================================================#
#                                Map Class                                             #
#======================================================================================#

class Map: 

    def __init__(self, k=0):
        self.k = k

#======================================================================================#
#                           The Standard Map of the Torus                              #
#======================================================================================#

class StdMap(Map):

    def next(self, pt):
        x = (pt[0] + pt[1] + (self.k/(2*np.pi))*np.sin(2*np.pi*pt[0])) % SCALE
        y = (pt[1] + self.k/(2*np.pi)*np.sin(2*np.pi*pt[0])) % SCALE
        return (x, y)

    def jacobian(self, pt):
        return np.matrix([[1+self.k*np.cos(2*np.pi*pt[0]), 1], 
                        [self.k*np.cos(2*np.pi*pt[0]), 1]])

#=======================================================================================#
#                               The Two-One-One Map                                     #
#=======================================================================================#

class TwoOneMap(Map):

    def next(self, pt):
        x_t, y_t = pt[0], pt[1]
        x_n = (2*x_t + y_t) % SCALE
        y_n = (x_t + y_t) % SCALE
        return (x_n, y_n)

    def jacobian(self, pt):
        return np.matrix('2 1; 1 1')

#=======================================================================================#
#                       The Two-One-One Map (with perturbation)                         #
#=======================================================================================#

class TwoOnePertMap(Map):

    def next(self, pt):
        x_t, y_t = pt[0], pt[1]
        x_n = (2*x_t + y_t + self.k*np.sin(2*np.pi*x_t)) % SCALE
        y_n = (x_t + y_t + self.k*np.sin(2*np.pi*x_t)) % SCALE
        return (x_n, y_n)

    def jacobian(self, pt):
        return np.matrix([[2+2*np.pi*self.k*np.cos(2*np.pi*pt[0]), 1], 
                        [1+2*np.pi*self.k*np.cos(2*np.pi*pt[0]), 1]])

#=======================================================================================#
#                               Logistic Map  
#=======================================================================================#

class LogisticMap(Map):

    def next(self, pt):
        x = 4*pt[0]*(1-pt[0]) % SCALE
        y = pt[0]+pt[1] % SCALE
        return (x, y)

    def jacobian(self, pt):
        return np.matrix([[4-8*pt[0], 0], 
                        [1, 1]])

#======================================================================================#
#                           Initialized Map Classes                                    #
#======================================================================================#

std = StdMap(k=1.2)
t1 = TwoOneMap()
tp = TwoOnePertMap(k=0.01)

