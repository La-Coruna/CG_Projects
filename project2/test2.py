from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_obj_v = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
g_obj_vn = []
g_obj_f = [[(0, None, None), (1, None, None), (2, None, None), (3, None, None)], [(4, None, None), (7, None, None), (6, None, None), (5, None, None)], [(0, None, None), (4, None, None), (5, None, None), (1, None, None)], [(1, None, None), (5, None, None), (6, None, None), (2, None, None)], [(2, None, None), (6, None, None), (7, None, None), (3, None, None)], [(4, None, None), (0, None, None), (3, None, None), (7, None, None)]]


vertices = glm.array(glm.float32,
        # position      color
        -1 ,  1 ,  1 , 
         1 ,  1 ,  1 , 
         1 , -1 ,  1 ,  
        -1 , -1 ,  1 ,  
        -1 ,  1 , -1 , 
         1 ,  1 , -1 , 
         1 , -1 , -1 , 
        -1 , -1 , -1 ,  
)

    # prepare index data
    # 12 triangles
indices = glm.array(glm.uint32,
    0,2,1,
    0,3,2,
    4,5,6,
    4,6,7,
    0,1,5,
    0,5,4,
    3,6,2,
    3,7,6,
    1,2,6,
    1,6,5,
    0,7,3,
    0,4,7,
)

v = [               [-1 ,  1 ,  1 ], 
         [1 ,  1 ,  1 ], 
         [1 , -1 ,  1 ],  
        [-1 , -1 ,  1 ],  
        [-1 ,  1 , -1 ], 
         [1 ,  1 , -1 ], 
         [1 , -1 , -1 ], 
        [-1 , -1 , -1 ],  
    ]

vertices2 = glm.array(np.array(np.concatenate(v),glm.float32))
indices = glm.array(np.array(v,glm.float32))

#indices

print(vertices==vertices2)


