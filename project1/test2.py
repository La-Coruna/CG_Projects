from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

vs=[
    [0.2, 0, 0.2         ,1,0,0],
    [-0.2, 0, 0.2        ,1,0,0],
    [-0.2, 0, -0.2       ,1,0,0],
    [0.2, 0, -0.2        ,1,0,0],
    [0.2, 0.2, 0.2       ,1,0,0],
    [-0.2, 0.2, 0.2      ,1,0,0],
    [-0.2, 0.2, -0.2     ,1,0,0],
    [0.2, 0.2, -0.2      ,1,0,0],
]
vertices = glm.array(np.concatenate([
    vs[0],vs[1],vs[2],vs[3],
    vs[0],vs[1],vs[5],vs[4],
    vs[0],vs[4],vs[7],vs[3],
    vs[6],vs[7],vs[4],vs[5],
    vs[6],vs[5],vs[1],vs[2],
    vs[6],vs[2],vs[3],vs[7],
],dtype=np.float32))
print(vertices)