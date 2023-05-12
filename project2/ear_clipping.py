from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_obj_v = [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, -0.999999], [0.999999, 1.0, 1.000001], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]
g_obj_vn = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, -0.0, 0.0], [0.0, -0.0, 1.0], [-1.0, -0.0, -0.0], [0.0, 0.0, -1.0]]
g_obj_f1 = [[(0, None, 0), (1, None, 0), (2, None, 0), (3, None, 0)], [(4, None, 1), (7, None, 1), (6, None, 1), (5, None, 1)], [(4, None, 2), (1, None, 2), (0, None, 2)], [(5, None, 3), (2, None, 3), (1, None, 3)], [(2, None, 4), (7, None, 4), (3, None, 4)], [(0, None, 5), (7, None, 5), (4, None, 5)], [(4, None, 2), (5, None, 2), (1, None, 2)], [(5, None, 3), (6, None, 3), (2, None, 3)], [(2, None, 4), (6, None, 4), (7, None, 4)], [(0, None, 5), (3, None, 5), (7, None, 5)]]
g_obj_f = [[[0, None, 0], [1, None, 0], [2, None, 0], [3, None, 0]], [[4, None, 1], [7, None, 1], [6, None, 1], [5, None, 1]], [[4, None, 2], [1, None, 2], [0, None, 2]], [[5, None, 3], [2, None, 3], [1, None, 3]], [[2, None, 4], [7, None, 4], [3, None, 4]], [[0, None, 5], [7, None, 5], [4, None, 5]], [[4, None, 2], [5, None, 2], [1, None, 2]], [[5, None, 3], [6, None, 3], [2, None, 3]], [[2, None, 4], [6, None, 4], [7, None, 4]], [[0, None, 5], [3, None, 5], [7, None, 5]]]


def triangulate_polygon(vertices):
    triangles = []
    polygon = vertices.copy()

    while len(polygon) >= 3:
        n = len(polygon)
        ear_index = -1

        for i in range(n):
            prev_index = (i - 1) % n
            curr_index = i
            next_index = (i + 1) % n

            prev_vertex = polygon[prev_index]
            curr_vertex = polygon[curr_index]
            next_vertex = polygon[next_index]

            if is_ear(prev_vertex, curr_vertex, next_vertex, polygon):
                ear_index = curr_index
                break

        if ear_index == -1:
            break

        triangle = [
            polygon[(ear_index - 1) % n],
            polygon[ear_index],
            polygon[(ear_index + 1) % n]
        ]
        triangles.append(triangle)

        del polygon[ear_index]

    return triangles

def is_ear(prev_vertex, curr_vertex, next_vertex, polygon):
    triangle = [prev_vertex, curr_vertex, next_vertex]

    for vertex in polygon:
        if vertex in triangle:
            continue

        if is_point_in_triangle(vertex, triangle):
            return False

    return True

def is_point_in_triangle(point, triangle):
    p1, p2, p3 = triangle

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, p1, p2)
    d2 = sign(point, p2, p3)
    d3 = sign(point, p3, p1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

# 다각형의 정점 정보를 파싱한 후에 다각형을 삼각형으로 분할
# faces 리스트에는 다각형의 각 face 정보가 포함되어 있다고 가정합니다.



triangles = []
for face in g_obj_f:
    if len(face) > 3:
        polygon_vertices = []
        for vertex_info_index in face:
            vertex_position = g_obj_v[vertex_info_index[0]]  
            #vertex_normal = g_obj_vn[vertex_info_index[1]] 
            polygon_vertices.append(vertex_position)
        triangles.extend(triangulate_polygon(polygon_vertices))

# 분할된 삼각형 정보를 출력
for triangle in triangles:
    print(triangle)

# print("------")
# triangles = []
# for face in g_obj_f:
#     polygon_vertices = []
#     for vertex_info_index in face:
#         vertex_position = g_obj_v[vertex_info_index[0]]  
#         vertex_normal = g_obj_vn[vertex_info_index[1]] 
#         polygon_vertices.append(vertex_position)
#     print(polygon_vertices)
