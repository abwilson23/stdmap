#!/usr/bin/python

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#K = np.random.rand(1)[0]
# 
N = 100
K = 1.6
Pi = 1

def get_orbit(pt): 
    orbit = []
    x, y = pt[0], pt[1]
    for n in range(1000):
        x = (x + (K/np.pi)*np.sin(y)) % Pi
        y = (y + x) % Pi
        orbit.append((x,y))
    return orbit

def get_next_iterate(pt):
    x, y = pt[0], pt[1]
    for n in range(1000):
        x = (x + K*np.sin(y)) % Pi
        y = (y + x) % Pi
    return ((x, y))

def generate_plot(n):
    pts = get_points(n)
    plt.axis([0, Pi, 0, Pi])
    plt.title('K = {}'.format(K))
    orbits = []
    for i in range(len(pts)):
        orbits = orbits + get_orbit(pts[i])
    for pt in orbits:
        plt.scatter(pt[1], pt[0], c='black', s = 1)
    plt.savefig('./orbits.png')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show(block=True)

# Generate a set of n random starting points. Use these as initial points in orbits.
def get_points(n):
    pts = []
    a = np.random.rand(n, 2)
    for i in range(n):
        pt = (a[i, 0]*Pi, a[i, 1]*Pi)
        pts.append(pt)
    return pts

# 
def compute_jacobian(pt):
    x, y = pt[0], pt[1]
    J = np.matrix([[1, K*np.cos(y)], [1, 1 + K*np.cos(y)]])
    for n in range(10000):
        x = (x + K*np.sin(y)) % Pi
        y = (y + x) % Pi
        J = J*get_jacobian(y)
    print (J)

def get_jacobian(y):
    return (np.matrix([[1, K*np.cos(y)], [1, 1 + K*np.cos(y)]]))

# Find Jacobian
def find_fast_subspace(pt):
    v = np.matrix('1;1')
    for i in range(0, 1000):
        J = get_jacobian(v.item((1,0)))
        J = J*v
    return v
    
def L_exponents(pt):
    v = pt
    J = get_jacobian(v[1])
    for i in range(0,N):
        v = get_next_iterate(v)
        J = get_jacobian(v[1])*J
    return J

# Here we assume that det(A)=1 and that a_ij < 1000
def compute_SVD(A):
    s, U = LA.eig(A*np.transpose(A))
    d, V = LA.eig(np.transpose(A)*A)
    s_vals = list(reversed(sorted([math.sqrt(x) for x in s])))
    D = np.diag((s_vals))
    return (U, D, np.transpose(V))
