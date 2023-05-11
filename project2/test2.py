from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_obj_v = [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, -0.999999], [0.999999, 1.0, 1.000001], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]
g_obj_vn = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, -0.0, 0.0], [0.0, -0.0, 1.0], [-1.0, -0.0, -0.0], [0.0, 0.0, -1.0]]
#g_obj_f1 = [[(0, None, 0), (1, None, 0), (2, None, 0), (3, None, 0)], [(4, None, 1), (7, None, 1), (6, None, 1), (5, None, 1)], [(4, None, 2), (1, None, 2), (0, None, 2)], [(5, None, 3), (2, None, 3), (1, None, 3)], [(2, None, 4), (7, None, 4), (3, None, 4)], [(0, None, 5), (7, None, 5), (4, None, 5)], [(4, None, 2), (5, None, 2), (1, None, 2)], [(5, None, 3), (6, None, 3), (2, None, 3)], [(2, None, 4), (6, None, 4), (7, None, 4)], [(0, None, 5), (3, None, 5), (7, None, 5)]]
g_obj_f = [[[0, None, 0], [1, None, 0], [2, None, 0], [3, None, 0]], [[4, None, 1], [7, None, 1], [6, None, 1], [5, None, 1]], [[4, None, 2], [1, None, 2], [0, None, 2]], [[5, None, 3], [2, None, 3], [1, None, 3]], [[2, None, 4], [7, None, 4], [3, None, 4]], [[0, None, 5], [7, None, 5], [4, None, 5]], [[4, None, 2], [5, None, 2], [1, None, 2]], [[5, None, 3], [6, None, 3], [2, None, 3]], [[2, None, 4], [6, None, 4], [7, None, 4]], [[0, None, 5], [3, None, 5], [7, None, 5]]]
[[0, None, 0], [1, None, 0], [2, None, 0], [3, None, 0]]
face = [[4, None, 2], [1, None, 2], [0, None, 2]]
# indexed_vertex_infos = [4, None, 2]
# vertex_position = [1.0, -1.0, -1.0]
# vertex_normal = [0.0, -1.0, 0.0]
g_obj_vertices=[]
g_obj_vertices_with_normal=[]
for face in g_obj_f:
    indexed_vertex_infos_list = []
    if len(face) > 3:
        for i in range(len(face)-2):
            indexed_vertex_infos_list.extend(face[i:i+3])
    elif len(face) == 3:
        indexed_vertex_infos_list = face
    else:
        print("error: the number of face information is less than 3")
            
    for indexed_vertex_infos in indexed_vertex_infos_list:
        # vertex position
        vertex_info = g_obj_v[indexed_vertex_infos[0]]
        
        # vertex normal
        if indexed_vertex_infos[2]:
            vertex_info.extend(g_obj_vn[indexed_vertex_infos[2]])
            
            ## normal이 있다면 vertices_with_normal에 저장
            g_obj_vertices_with_normal.extend(vertex_info)
        else:
            g_obj_vertices.extend(vertex_info)

    
print(g_obj_vertices)

    

g_obj_vertices = glm.array(glm.float32,
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

print(g_obj_vertices==vertices2)
