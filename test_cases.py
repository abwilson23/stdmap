import math 
import lib
import map_info
import numpy as np

def test_std_next():
    T = Map(std_map_next, std_map_jacobian)
    p_0 = (0,1)
    p_1 = (0,0)
    assert T.next(p_0) == p_1
    
def test_two_one_next():
    T = Map(two_one_map_next, two_one_map_jacobian)
    p_0 = (1,1)
    p_1 = (3,2)
    assert T.next(p_0) == p_1

def test_std_jacobian():
    T = Map(std_map_next, std_map_jacobian)

def test_two_one_jacobian():
    pt = (1,1)
    T = Map(two_one_map_next, two_one_map_jacobian)
    J = np.matrix('2 1; 1 1')
    assert J = T.jacobian(pt)
    
