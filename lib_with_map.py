#!/usr/bin/python

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# N = the number of iterations used in most of the functions
N = 100
Scale = 1

#=================================================================================#
#                                 Plotting                                        #
#=================================================================================#

# Computes orbit of a point under the given mapping Map
def get_orbit(Map, pt): 
    orbit = []
    tmp = (pt[0], pt[1])
    for n in range(N):
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

# Choose an arbitrary (pt, v) \in Tangent manifold, compute J^N(pt), then apply 
# J^N to v to find the fast subspace
def find_fast_subspace(Map, pt, vec):
    tmp = pt
    v = np.matrix('{};{}'.format(vec[0], vec[1]))
    J = np.eye(2) # Assume we're dealing with 2x2's
    for i in range(0, N):
        J = J*Map.jacobian(tmp[1])
        tmp = Map.next(pt)
    return J*v
#_______________________________________________________________________________#


# Compute J^N at pt, then take the SVD of J^N to find the Lyapunov exponents
def L_exponents(Map, pt):
    tmp = pt 
    J = np.eye(n) # Assume 2x2
    for i in range(0, N):
        J = J*Map.jacobian(tmp[1])
        tmp = Map.next(pt)
    (U,D,V) = compute_SVD(J)
    l1 = np.log(D[0,0])/float(N)
    l2 = (-1)*l1 
    print ('Lambda_1 is {}, and Lambda_2 is {}'.format(l1, l2))

# Here we assume that det(A)=1 and that a_ij < 1000
# Computes the SVD decomposition of a matrix by looking at the eigenvectors of
# AA^T and A^TA. 
def compute_SVD(A):
    s, U = LA.eig(A*np.transpose(A))
    d, V = LA.eig(np.transpose(A)*A)
    s_vals = list(reversed(sorted([math.sqrt(x) for x in s])))
    D = np.diag((s_vals))
    return (U, D, np.transpose(V))

# To add new maps, provide next_iterate and get_jacobian functions in the map_info.py
# module. 
class Map:

    def __init__(self, next_iterate, get_jacobian)
        self.next = next_iterate
        self.jacobian = get_jacobian
