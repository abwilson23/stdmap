#!/usr/bin/python

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# N = the number of iterations used in most of the functions
# K = a constant which alters the dynamics of the system
N = 100
K = 1.2
Scale = 1

# Computes orbit of a pt under the standard map T:[0,1]^2 -> [0,1]^2
def get_orbit(pt): 
    orbit = []
    x, y = pt[0], pt[1]
    for n in range(N):
        y = (y + K/(2*np.pi)*np.sin(2*np.pi*x)) % Scale
        x = (x + y) % Scale
        orbit.append((x,y))
    return orbit

# Used for Jacobian/subspace finding, but it basically does what get_orbit does
# Will amalgamate these later 
def get_next_iterate(pt):
    x, y = pt[0], pt[1]
    for n in range(N):
        y = (y + K/(2*np.pi)*np.sin(2*np.pi*x)) % Scale
        x = (x + y) % Scale
    return ((x, y))

# Plots the orbits of a set of randomly generated points under the standard map
# Slow at the moment, need to optimize later
def std_map_plot(n):
    pts = get_points(n)
    plt.axis([0, Scale, 0, Scale])
    plt.title('K = {}'.format(K))
    orbits = []
    for i in range(len(pts)):
        orbits = orbits + get_orbit(pts[i])
    for pt in orbits:
        plt.scatter(pt[0], pt[1], c='black', s = 1)
    plt.savefig('./K={}.png'.format(K))
    plt.axes().set_aspect('equal', 'datalim')
    plt.show(block=True)

# Add a function T which returns T(x,y) to refactor and tidy up
# def T(x,y):

# Generate a set of n random starting points. Use these as initial points in orbits.
def get_points(n):
    pts = []
    a = np.random.rand(n, 2)
    for i in range(n):
        pt = (a[i, 0], a[i, 1])
        pts.append(pt)
    return pts

# 
def compute_jacobian(pt):
    x, y = pt[0], pt[1]
    J = np.matrix([[1, K*np.cos(y)], [1, 1 + K*np.cos(y)]])
    for n in range(N):
        x = (x + K*np.sin(y)) % Pi
        y = (y + x) % Pi
        J = J*get_jacobian(y)
    print (J)

def get_jacobian(y):
    return (np.matrix([[1, K*np.cos(y)], [1, 1 + K*np.cos(y)]]))

# Choose an arbitrary (pt, v) \in Tangent manifold, compute J^N(pt), then apply 
# J^N to v to find the fast subspace
def find_fast_subspace(pt, vec):
    tmp = pt
    v = np.matrix('{};{}'.format(vec[0], vec[1]))
    J = get_jacobian(pt[1])
    for i in range(0, N):
        tmp = get_next_iterate(pt)
        J = J*get_jacobian(tmp[1])
    return J*v
    
# Compute J^N at pt, then take the SVD of J^N to find the Lyapunoov exponents
def L_exponents(pt):
    tmp = pt 
    J = get_jacobian(pt[1])
    for i in range(0,N):
        tmp = get_next_iterate(pt)
        J = J*get_jacobian(tmp[1])
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

#=======================================================================================#
# Two one one map section, need to amalgamate all of this rather than having it be a 
# separate set of similar functions.
#=======================================================================================#

def two_one_map_of(pt):
    x_t, y_t = pt[0], pt[1]
    x_n = (2*x_t + y_t) % Scale
    y_n = (x_t + y_t) % Scale
    return ((x_n, y_n))

def two_one_map_plot(n):
    pts = get_points(n)
    plt.axis([0, Scale, 0, Scale])
    plt.title('Two-one-one Map')
    orbits = []
    for i in range(len(pts)):
        orbits = orbits + get_two_one_orbit(pts[i])
    for pt in orbits:
        plt.scatter(pt[0], pt[1], c='black', s = 1)
    plt.savefig('Two-one-map, n = {}'.format(n))
    plt.axes().set_aspect('equal', 'datalim')
    plt.show(block=True)
    
def get_two_one_orbit(pt): 
    orbit = []
    tmp = pt
    for n in range(N):
        orbit.append(tmp)
        tmp = two_one_map_of(tmp)
    return orbit

def get_jacobian_two(y):
    return (np.matrix('2 1; 1 1')

def find_fast_subspace_two(pt, vec):
    tmp = pt
    v = np.matrix('{};{}'.format(vec[0], vec[1]))
    J = get_jacobian_two(pt[1])
    for i in range(0, N):
        J = J*J
    return J*v
    
# Compute J^N at pt, then take the SVD of J^N to find the Lyapunoov exponents
def L_exponents_two(pt):
    tmp = pt 
    J = get_jacobian_two(pt[1])
    for i in range(0,N):
        J = J*J
    (U,D,V) = compute_SVD(J)
    l1 = np.log(D[0,0])/float(N)
    l2 = (-1)*l1 
    print ('Lambda_1 is {}, and Lambda_2 is {}'.format(l1, l2))
