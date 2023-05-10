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


## good pan
cam_direction = glm.normalize(cam_pos-cam_target) # actually opposite direction
cam_right = glm.normalize(glm.cross(up,cam_direction))
cam_up = glm.normalize(glm.cross(cam_direction,cam_right))
cam_pan = (cam_right*g_cam_move_right + cam_up*g_cam_move_up)

cam_pos += cam_pan
cam_target += cam_pan

V = glm.lookAt(cam_pos, cam_target, up)