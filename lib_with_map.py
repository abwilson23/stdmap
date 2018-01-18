#!/usr/bin/python

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import map_info as m

# N = the number of iterations used in most of the functions
K = 1.2
N = 12
Scale = 1


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

#=================================================================================#
#                                 Plotting                                        #
#=================================================================================#

# Computes orbit of a point under the given mapping Map
def get_orbit(Map, pt): 
    orbit = []
    tmp = (pt[0], pt[1])
    for n in range(100):
        orbit.append(tmp)
        tmp = Map.next(tmp)
    return orbit

# Generate a set of n random starting points. Use these as initial points in orbits.
def get_points(n):
    pts = []
    a = np.random.rand(n, 2)
    for i in range(n):
        pt = (a[i, 0], a[i, 1])
        pts.append(pt)
    return pts

# Plots the orbits of a set of randomly generated points under the given map
# Slow at the moment, need to optimize later
def plot(Map, n):
    pts = get_points(n)
    plt.axis([0, Scale, 0, Scale])
    plt.title('K = {}'.format(K))
    orbits = []
    for i in range(len(pts)):
        orbits = orbits + get_orbit(Map, pts[i])
    for pt in orbits:
        plt.scatter(pt[0], pt[1], c='black', s = 1)
    plt.savefig('./K={}.png'.format(K)) # Change this title later
    plt.axes().set_aspect('equal', 'datalim')
    plt.show(block=True)


#=================================================================================#
#                           Fast/Slow Spaces                                      #
#=================================================================================#

def get_jacobian_iterate(Map, pt, n):
    tmp = pt
    J = np.eye(2) # Assume we're dealing with 2x2's
    for i in range(0, n):
        J = J*Map.jacobian(tmp)
        tmp = Map.next(tmp)
    return J
    
# Choose an arbitrary (pt, v) \in Tangent manifold, compute J^N(pt), then apply 
# J^N to v to find the fast subspace
def find_fastspace(Map, pt, vec, n):
    v = np.matrix('{};{}'.format(vec[0], vec[1]))
    J = get_jacobian_iterate(Map, pt, n)
    return (normalize(J*v))

def find_slowspace(Map, pt, n):
    v = np.matrix('0;1')
    J = get_jacobian_iterate(Map, pt, n)
    return (normalize(np.linalg.solve(J, v)))

def get_info(Map, pt, vec, n):
    print ('Fast subspace: \n{}\n'.format(find_fastspace(Map, pt, vec, n)))
    print ('Slow subspace: \n{}\n'.format(find_slowspace(Map, pt, n)))
    L_exponents(Map, pt)

def normalize(vec):
    n = LA.norm(vec)
    v = np.matrix('{};{}'.format(vec[0,0]/n, vec[1,0]/n))
    return (v)

#=================================================================================#
#                           SVD and Lyapunov Exponents                            #
#=================================================================================#

# Compute J^N at pt, then take the SVD of J^N to find the Lyapunov exponents
def L_exponents(Map, pt):
    tmp = pt 
    J = np.eye(2) # Assume 2x2
    for i in range(0, N):
        J = J*Map.jacobian(tmp)
        tmp = Map.next(pt)
    (U,D,V) = compute_SVD(J)
    l1 = np.log(D[0,0])/float(N)
    l2 = np.log(D[1,1])/float(N)
    print ('Lambda_1 = {} \nLambda_2 = {}.'.format(l1, l2))

# Here we assume that det(A)=1 and that a_ij < 1000
# Computes the SVD decomposition of a matrix by looking at the eigenvectors of
# AA^T and A^TA. 
def compute_SVD(A):
    s, U = LA.eig(A*np.transpose(A))
    d, V = LA.eig(np.transpose(A)*A)
    s_vals = list(reversed(sorted([math.sqrt(x) for x in s])))
    D = np.diag((s_vals))
    return (U, D, np.transpose(V))

