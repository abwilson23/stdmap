#!/usr/bin/python

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import map_info as m

# N = the number of iterations used in most of the functions
K = 1.2
N = 10
Scale = 1
orbit_len = 100

#=================================================================================#
#                                 Plotting                                        #
#=================================================================================#

# smaller dot size

# Computes orbit of a point under the given mapping Map
def get_orbit(Map, pt, orbit_len): 
    orbit = []
    tmp = (pt[0], pt[1])
    for n in range(orbit_len):
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
def plot(Map, n, orbit_len):
    pts = get_points(n)
    plt.axis([0, Scale, 0, Scale])
    plt.title('K = {}'.format(K))
    orbits = []
    for i in range(len(pts)):
        orbits = orbits + get_orbit(Map, pts[i], orbit_len)
    for pt in orbits:
        plt.scatter(pt[0], pt[1], c='black', s = 1)
    plt.savefig('./K={}.png'.format(K)) # Change this title later
    plt.axes().set_aspect('equal', 'datalim')
    plt.show(block=True)

#def quiver_plot(Map):
#    pt = get_points(1)[0]
#    vec = (0.2, 0.5)
#    fast_v = find_fastspace(Map, pt, vec, 9)
#    fast_vec = (fast_v[0,0], fast_v[1,0])
#    fast = [fast_vec] * orbit_len
#    slow = []
#    orbit = get_orbit(Map, pt) 
#    for pt in orbit:
#        slow.append(find_slowspace(Map, pt, 10))
#    X, Y = zip(*orbit)
#    U, V = zip(*fast)
#    plt.figure()
#    #plt.quiver(X, Y, U, V, angles='xy', scale=30, color='r', headwidth='2', headlength='1')
#    for pt in orbit:
#        plt.scatter(pt[0], pt[1], c='black', s = 1)
#    plt.quiver(X, Y, U, V, width=0.001, headwidth='20', headlength='20', color='r')
#    plt.show()
    
def plot_all(Map, n, orbit_len):
    pts = get_points(n)
    plt.axis([0, Scale, 0, Scale])
    for pt in pts:
        plot_orbit(Map, pt, orbit_len, plt)
    plt.savefig('./arrows.png'.format(K)) 
    plt.axes().set_aspect('equal', 'datalim') # fix scaling when stretching windows
    plt.show()

def plot_orbit(Map, pt, orbit_len, plt):
    # Plots the points along the orbit
    orbit = get_orbit(Map, pt, orbit_len)
    for pt in orbit:
        plt.scatter(pt[0], pt[1], c='black', s = 0.5)

    plot_fastspace(Map, orbit, plt)
    plot_slowspace(Map, orbit, plt)

# Plot the fastspace at each point, only need fastspace at the first point of orbit
def plot_fastspace(Map, orbit, plt):
    vec = get_points(1)[0]
    fast = []
    for pt in orbit:
        tmp = find_fastspace(Map, pt, vec, 25)
        fast_v = (tmp[0,0], tmp[1,0])
        fast.append(fast_v)
    x,y = zip(*orbit)
    u,v = zip(*fast)
    plt.quiver(x, y, u, v, width=0.001, headwidth='10', headlength='10', color='r')

# Plot the slowspace at each point
def plot_slowspace(Map, orbit, plt):
    slow = []
    for pt in orbit:
        tmp = find_slowspace(Map, pt, 10)
        slow_v = (tmp[0,0], tmp[1,0])
        slow.append(slow_v)
    x,y = zip(*orbit)
    u,v = zip(*slow)
    plt.quiver(x, y, u, v, width=0.001, headwidth='10', headlength='10', color='b')

#=================================================================================#
#                           Fast/Slow Spaces                                      #
#=================================================================================#

def get_jacobian_iterate(Map, pt, n):
    tmp = pt
    J = np.eye(2) # Assume we're dealing with 2x2's
    for i in range(0, n):
        J = Map.jacobian(tmp)*J
        tmp = Map.next(tmp)
    return J
    
# Choose an arbitrary (pt, v) \in Tangent manifold, compute J^N(pt), then apply 
# J^N to v to find the fast subspace. Plot the result of find_fastspace over x_25 = T^25(x_0). 
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
    assert n != 0
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

