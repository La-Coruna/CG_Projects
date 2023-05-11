from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os



# for mouse control
g_mouse_button_left_toggle = False
g_mouse_button_right_toggle = False
g_cursor_last_xpos = 0
g_cursor_last_ypos = 0

# for orbit
g_cam_azimuth = 0. # z축으로부터의 각임.
g_cam_elevation = .1 # xz평면으로부터의 각임.
g_cam_orbit_direction_toggle = False # blender의 viewr와 같이, 처음 클릭했을 때의 elevation을 기준으로 azimuth의 회전 방향을 결정.

# for pan
g_cam_pan = glm.vec3(.0,.0,.0)

# camera's vector
g_cam_direction = glm.vec3(.0,.0,.0)    # w vector of camera
g_cam_right = glm.vec3(.0,.0,.0)        # u vector of camera
g_cam_up = glm.vec3(.0,.0,.0)           # v vector of camera

# for zoom
g_cam_distance = 3.6 # the distance between camera and target ## d가 0.2보다 작아지면 깨짐. 작을 수록 확대 됨.
g_ortho_mag = .15 # the magnification of lens in ortho ## 작을 수록 확대 됨.

# for viewport & projection
g_view_height = 10.
g_view_width = g_view_height * 800/800

# for projection
g_P = glm.perspective(np.deg2rad(45), g_view_width/g_view_height , 0.1, 100) # fov가 30도 ~ 60도 일 때 인간의 시야와 비슷.
g_P_toggle = True

# for obj file to be loaded
g_obj_file = None
g_obj_v = []
g_obj_vn = []
g_obj_f = []

g_obj_vertices=[]
g_obj_vertices_with_normal=[]
g_vao_obj = None

# for mode ( 0: single mesh rendering mode, 1: animating hierarchical model rendering mode )
# basic 
g_rendering_mode = 1

# ! for debug
g_debug_1 = 0

g_vertex_shader_src_lighting = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src_lighting = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 material_color;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,2,4);
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''

g_vertex_shader_src_color = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src_color = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def update_projection_matrix():
    global g_P
    if g_P_toggle:
        # perspective projection
        g_P = glm.perspective(np.deg2rad(45), g_view_width/g_view_height , 0.1, 100)
    else:
        # orthogonal projection
        d = g_ortho_mag # the magnification of lens, 작을 수록 화면이 확대됨.
        g_P = glm.ortho(-g_view_width*d, g_view_width*d, -g_view_height*d, g_view_height*d, -10, 10)

def framebuffer_size_callback(window, width, height):
    global g_view_width
    
    glViewport(0, 0, width, height)

    g_view_width = g_view_height * width/height
    update_projection_matrix()

def key_callback(window, key, scancode, action, mods):
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                global g_P_toggle
                g_P_toggle = not g_P_toggle
                update_projection_matrix()
                
            global g_debug_1
            if key==GLFW_KEY_1:
                if g_debug_1>0:
                    g_debug_1 -=1
                print(g_debug_1)
            if key==GLFW_KEY_2:
                g_debug_1 +=1
                print(g_debug_1)

def cursor_callback(window, xpos, ypos):
    global g_cursor_last_xpos, g_cursor_last_ypos # for cursor
    global g_mouse_button_left_toggle, g_cam_azimuth, g_cam_elevation # for orbit
    global g_mouse_button_right_toggle, g_cam_pan # for pan
    
    # orbit move
    if g_mouse_button_left_toggle:
        
        # check the cursor move
        xoffset = - (xpos - g_cursor_last_xpos)
        yoffset =  (ypos - g_cursor_last_ypos)
        g_cursor_last_xpos, g_cursor_last_ypos = xpos, ypos
        
        # set sensitivity
        sensitivity = 0.01

        # when ( 90 < elevation < 270 ), azimuth should change opposite direction
        if g_cam_orbit_direction_toggle:
            xoffset *= -1

        # update cam_angle and cam_height
        g_cam_azimuth += xoffset * sensitivity
        g_cam_elevation += yoffset * sensitivity
        
        # g_cam_angle must be in (0 ~ 2*pi)
        if g_cam_azimuth > 2 * np.pi:
            g_cam_azimuth -= 2 * np.pi
        elif g_cam_azimuth < 0:
            g_cam_azimuth += 2 * np.pi
        if g_cam_elevation > 2 * np.pi:
            g_cam_elevation -= 2 * np.pi
        elif g_cam_elevation < 0:
            g_cam_elevation += 2 * np.pi       
            
    # pan move
    elif g_mouse_button_right_toggle:
        
        # check the cursor move
        xoffset = - (xpos - g_cursor_last_xpos)
        yoffset = ypos - g_cursor_last_ypos
        g_cursor_last_xpos, g_cursor_last_ypos = xpos, ypos
        
        
        # set sensitivity
        sensitivity = 0.001
        xoffset*=sensitivity
        yoffset*=sensitivity
        
        # update move
        g_cam_pan += g_cam_right * xoffset + g_cam_up * yoffset
             
def button_callback(window, button, action, mod):
    global g_mouse_button_left_toggle, g_mouse_button_right_toggle
    global g_cursor_last_xpos, g_cursor_last_ypos
    global g_cam_orbit_direction_toggle
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            g_cursor_last_xpos, g_cursor_last_ypos = glfwGetCursorPos(window)
            # if elevation is in ( 90 ~ 270 ) degree, it will be True.
            g_cam_orbit_direction_toggle = True if np.rad2deg(g_cam_elevation) > 90 and np.rad2deg(g_cam_elevation) < 270 else False
            g_mouse_button_left_toggle = True
        elif action==GLFW_RELEASE:
            g_cursor_last_xpos, g_cursor_last_ypos = glfwGetCursorPos(window)
            g_mouse_button_left_toggle = False
    elif button==GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            g_cursor_last_xpos, g_cursor_last_ypos = glfwGetCursorPos(window)
            g_mouse_button_right_toggle = True
        elif action==GLFW_RELEASE:
            g_cursor_last_xpos, g_cursor_last_ypos = glfwGetCursorPos(window)
            g_mouse_button_right_toggle = False
     
def scroll_callback(window, xoffset, yoffset):
    global g_cam_distance, g_ortho_mag

    # set sensitivity
    sensitivity = 0.05
    
    new_cam_distance = g_cam_distance - yoffset * sensitivity
    if new_cam_distance >= 0.2:
        g_cam_distance = new_cam_distance
        
        # # projection이 ortho로 바뀌어도 비율이 유지 되도록 조정.
        # # ! 본 과제에서는 요구되지 않은 기능이기에 주석처리함. !
        # g_ortho_mag = g_cam_distance * 0.0417 
        # if not g_P_toggle:
        #     update_projection_matrix()

def drop_callback(window, paths):
    print(paths)
    global g_obj_file, g_obj_v, g_obj_vn, g_obj_f, g_rendering_mode
    global g_obj_vertices, g_obj_vertices_with_normal, g_vao_obj
    g_obj_file = open(paths[0], 'r')
    g_obj_v = []
    g_obj_vn = []
    g_obj_f = []
    
    g_obj_vertices=[]
    g_obj_vertices_with_normal=[]
    g_vao_obj = None
        
    # v : vertex positions
    # vn : vertex normals
    # f : face information
    # ex) f vertex_position_index / texture_coordinates_index / vertex_normal_index
    # ! all argument indices are 1 based indices !
    
    for line in g_obj_file:
        #print(line)
        fields = line.strip().split()
        if len(fields) == 0: # 빈 줄인 경우.
            continue
        elif fields[0] == 'v':
            g_obj_v.append( [float(x) for x in fields[1:]] )
        elif fields[0] == 'vn':
            g_obj_vn.append( [float(x) for x in fields[1:]] )
        elif fields[0] == 'f':
            face = []
            for vtx in fields[1:]:
                fields_of_vtx = vtx.split('/')
                
                # ex) vertex_position_index
                if len(fields_of_vtx) == 1:
                    v_idx = int(fields_of_vtx[0]) - 1
                    texture_idx = None
                    vn_idx = None
                    
                # ex) vertex_position_index / texture_coordinates_index
                elif len(fields_of_vtx) == 2:
                    v_idx = int(fields_of_vtx[0]) - 1
                    texture_idx = int(fields_of_vtx[1]) - 1 if fields_of_vtx[1] else None
                    vn_idx = None
                    
                # ex) vertex_position_index / texture_coordinates_index / vertex_normal_index
                elif len(fields_of_vtx) == 3:
                    v_idx = int(fields_of_vtx[0]) - 1
                    texture_idx = int(fields_of_vtx[1]) - 1 if fields_of_vtx[1] else None
                    vn_idx = int(fields_of_vtx[2]) - 1 if fields_of_vtx[2] else None
                    
                face.append( [v_idx, texture_idx, vn_idx] )
            g_obj_f.append( face )
                
    g_obj_file.close()
    
    # print obj file, print out the following information of the obj file to stdout
    print("Obj file name:", os.path.basename(paths[0]))
    print("\nTotal number of faces:", len(g_obj_f))
    print("\nNumber of faces with 3 vertices:", len([x for x in g_obj_f if len(x)==3]))
    print("\nNumber of faces with 4 vertices:", len([x for x in g_obj_f if len(x)==4]))
    print("\nNumber of faces with more than 4 vertices:", len([x for x in g_obj_f if len(x)>4]))
    
    # change mode to single mesh rendering mode
    g_rendering_mode = 0
    # print(g_obj_v)
    # print(g_obj_vn)
    # print(g_obj_f)
    # print("----")
    
    # #1. vertices array
    
    for face in g_obj_f:
        #print(face)
        indexed_vertex_infos_list = []
        if len(face) > 3:
            #print("polygon이 triangle이 아닐 경우")
            for i in range(1,(len(face)-1)):
                indexed_vertex_infos_list.extend(face[0:1]+face[i:i+2])
        elif len(face) == 3:
            indexed_vertex_infos_list = face
        else:
            print("error: the number of face information is less than 3")
        
        for indexed_vertex_infos in indexed_vertex_infos_list:
            vertex_info = []
            #print(indexed_vertex_infos)
            #print(g_obj_v)
            ## vertex position
            vertex_info.extend(g_obj_v[indexed_vertex_infos[0]])
            #print("<position>: ", g_obj_v[indexed_vertex_infos[0]])
            
            ## vertex normal
            if indexed_vertex_infos[2] != None:
                vertex_info.extend(g_obj_vn[indexed_vertex_infos[2]])
                
                #print("<normal>: ", g_obj_vn[indexed_vertex_infos[2]])
                ## normal이 있다면 vertices_with_normal에 저장
                g_obj_vertices_with_normal.extend(vertex_info)
            else:
                ## normal이 없다면 vertices에 저장
                #print(indexed_vertex_infos)
                g_obj_vertices.extend(vertex_info)
            #print(vertex_info)
            #print()

    
    g_vao_obj= prepare_vao_obj_with_normal()
    global g_debug_1
    g_debug_1 = len(g_obj_vertices_with_normal)/6
        

    
    

def prepare_vao_triangle():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 1.0, 1.0, # v0
         0.5, 0.0, 0.0,  1.0, 1.0, 1.0, # v1
         0.0, 0.5, 0.0,  1.0, 0.0, 0.0, # v2
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -2.0, 0.0, 0.0,  1.0, 1.0, 1.0, # x-axis start
         2.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, -2.0, 0.0,  1.0, 1.0, 1.0, # y-axis start
         0.0, 2.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -2.0,  1.0, 1.0, 1.0, # z-axis start
         0.0, 0.0, 2.0,  0.0, 0.0, 1.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_box():
    # prepare vertex data (in main memory)
    vs=[[], # for index to start from 1
        [0.2, 0, 0.2         ,1,1,0],
        [-0.2, 0, 0.2        ,1,1,0],
        [-0.2, 0, -0.2       ,1,1,0],
        [0.2, 0, -0.2        ,1,1,0],
        [0.2, 0.4, 0.2       ,0,1,1],
        [-0.2, 0.4, 0.2      ,0,1,1],
        [-0.2, 0.4, -0.2     ,0,1,1],
        [0.2, 0.4, -0.2      ,0,1,1],
    ]
    vertices = glm.array(np.concatenate([
        vs[3],
        vs[4],
        vs[2],
        vs[1],
        vs[5],
        vs[2],
        vs[6],
        vs[3],
        vs[7],
        vs[4],
        vs[8],
        vs[1],
        vs[5],
        vs[8],
        vs[6],
        vs[7]
    ],dtype=np.float32))

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    
    r,g,b = .5, .5, .5
    half_size, start, end, line_num =2, -2, 2, 21
    vertices_for_line_x = glm.array(np.concatenate([
        [
        -half_size, .0, np.round_(z,1), r, g, b,
        half_size, .0, np.round_(z,1), r, g, b
        ]
        for z in np.linspace(start, end, line_num)]
        , dtype=np.float32))
    
    vertices_for_line_z = glm.array(np.concatenate([
        [
        np.round_(x,1), .0, -half_size, r, g, b,
        np.round_(x,1), .0, half_size, r, g, b
        ]
        for x in np.linspace(start, end, line_num)]
        , dtype=np.float32))

    vertices = vertices_for_line_x.concat(vertices_for_line_z)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_obj_with_normal():

    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(np.array(g_obj_vertices_with_normal,glm.float32))

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)


    return VAO

def draw_frame(vao, MVP, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 84)

def draw_cube(vao, MVP, M, matcolor, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(unif_locs['M'], 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(unif_locs['material_color'], matcolor.r, matcolor.g, matcolor.b)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 16)

def draw_obj_with_normal(vao, MVP, M, matcolor, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(unif_locs['M'], 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(unif_locs['material_color'], matcolor.r, matcolor.g, matcolor.b)
    glBindVertexArray(g_vao_obj)
    #vertex_count = int(len(g_obj_vertices_with_normal)/6)
    vertex_count = int(g_debug_1)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'project2 2019019043 박종윤', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
    glfwSetDropCallback(window, drop_callback)
  
    # load shaders & get uniform locations
    shader_lighting = load_shaders(g_vertex_shader_src_lighting, g_fragment_shader_src_lighting)
    unif_names = ['MVP', 'M', 'view_pos', 'material_color']
    unif_locs_lighting = {}
    for name in unif_names:
        unif_locs_lighting[name] = glGetUniformLocation(shader_lighting, name)

    shader_color = load_shaders(g_vertex_shader_src_color, g_fragment_shader_src_color)
    unif_names = ['MVP']
    unif_locs_color = {}
    for name in unif_names:
        unif_locs_color[name] = glGetUniformLocation(shader_color, name)
    
    # prepare vaos
    vao_triangle = prepare_vao_triangle()
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()
    vao_box = prepare_vao_box()
    vao_obj = g_vao_obj

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        ## projection matrix
        P = g_P

        ## view matrix        
        cam_target = glm.vec3(0,0,0) + g_cam_pan
        cam_orbit = g_cam_distance * glm.vec3(
            np.cos(g_cam_elevation) * np.sin(g_cam_azimuth),
            np.sin(g_cam_elevation),
            np.cos(g_cam_elevation) * np.cos(g_cam_azimuth)
            ) # cam_orbit
        cam_pos = cam_orbit + g_cam_pan
        up = glm.vec3(.0, .1, .0) if np.rad2deg(g_cam_elevation) < 90 or np.rad2deg(g_cam_elevation) > 270 else glm.vec3(.0, -1, .0)
        
        # camera's vector
        global g_cam_direction,g_cam_right,g_cam_up
        g_cam_direction = glm.normalize(cam_pos-cam_target) # actually opposite direction
        g_cam_right = glm.normalize(glm.cross(up,g_cam_direction))
        g_cam_up = glm.normalize(glm.cross(g_cam_direction,g_cam_right))
        
        V = glm.lookAt(cam_pos, cam_target, up)
        
        ## drawing color
        glUseProgram(shader_color)
        
        #@ draw current frame
        draw_frame(vao_frame, P*V, unif_locs_color)
        
        #@ draw current grid
        draw_grid(vao_grid, P*V, unif_locs_color)
        
        ## drawing lighting
        glUseProgram(shader_lighting)
        
        #@ draw current box
        #draw_cube(vao_box, P*V, glm.mat4(), glm.vec3(0,0,1), unif_locs_lighting)
        
        #@ draw obj file
        if g_rendering_mode == 0:
            draw_obj_with_normal(vao_obj, P*V, glm.mat4(), glm.vec3(0,0,1), unif_locs_lighting)
        
        
        ## animating
        # t = glfwGetTime()

        # # tranlation
        # T = glm.translate(glm.vec3(np.sin(t), .2, 0.4))

        # M = T

        # # current frame: P*V*M
        # MVP = P*V*M
        # glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # # draw triangle w.r.t. the current frame
        # glBindVertexArray(vao_triangle)
        # glDrawArrays(GL_TRIANGLES, 0, 3)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
