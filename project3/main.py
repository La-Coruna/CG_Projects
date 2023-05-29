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
g_cam_distance = 15 # the distance between camera and target ## d가 0.2보다 작아지면 깨짐. 작을 수록 확대 됨.
g_ortho_mag = .6255 # the magnification of lens in ortho ## 작을 수록 확대 됨.

# for viewport & projection
g_view_height = 10.
g_view_width = g_view_height * 800/800

# for projection
g_P = glm.perspective(np.deg2rad(45), g_view_width/g_view_height , 0.1, 100) # fov가 30도 ~ 60도 일 때 인간의 시야와 비슷.
g_P_toggle = True

# for single mode obj file 
g_single_mode_obj_path = None
g_vao_single_mode_obj = None
g_vao_single_mode_obj_vertex_count = None

# for mode ( 0: single mesh rendering mode, 1: animating hierarchical model rendering mode )
# basic 
g_rendering_mode_toggle = 1
g_wireframe_solid_toggle = 1

# obj path
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
g_water_obj_path = os.path.join(fileDirectory,"obj_file","water.obj")
g_boat_obj_path = os.path.join(fileDirectory,"obj_file","boat.obj")
g_sail_obj_path = os.path.join(fileDirectory,"obj_file","sail.obj")
g_bird_obj_path = os.path.join(fileDirectory,"obj_file","bird.obj")
g_crab_obj_path = os.path.join(fileDirectory,"obj_file","crab.obj")
g_cat_obj_path = os.path.join(fileDirectory,"obj_file","cat.obj")
g_speaker_obj_path = os.path.join(fileDirectory,"obj_file","speaker.obj")
g_beachBall_obj_path = os.path.join(fileDirectory,"obj_file","beachBall.obj")
g_shark_obj_path = os.path.join(fileDirectory,"obj_file","shark.obj")
g_fish1_obj_path = os.path.join(fileDirectory,"obj_file","fish1.obj")
g_fish2_obj_path = os.path.join(fileDirectory,"obj_file","fish2.obj")

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
    vec3 light_pos = vec3(100,100,100);
    vec3 light_color = vec3(1,1,1);
    vec3 light_pos2 = vec3(-100,100,-100);
    vec3 light_color2 = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;
    vec3 light_ambient2 = 0.1*light_color2;
    vec3 light_diffuse2 = light_color2;
    vec3 light_specular2 = light_color2;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material
    vec3 material_specular2 = light_color2;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;
    vec3 ambient2 = light_ambient2 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;
    vec3 diffuse2 = diff2 * light_diffuse2 * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;
    vec3 specular2 = spec * light_specular2 * material_specular;

    vec3 color = ambient + ambient2 + diffuse + specular + diffuse2 + specular2 ;
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

class Node:
    def __init__(self, parent, scale, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.scale = scale
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_scale(self):
        return self.scale
    def get_color(self):
        return self.color

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
        g_P = glm.ortho(-g_view_width*d, g_view_width*d, -g_view_height*d, g_view_height*d, -1000, 1000)

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
            elif key==GLFW_KEY_1:
                global g_rendering_mode_toggle
                g_rendering_mode_toggle = 0
            elif key==GLFW_KEY_2:
                global g_rendering_mode_toggle
                g_rendering_mode_toggle = 1
            elif key==GLFW_KEY_SPACE:
                pass


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
        sensitivity = 0.01
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

# zooming
def scroll_callback(window, xoffset, yoffset):
    global g_cam_distance, g_ortho_mag

    # set sensitivity
    sensitivity = 0.3
    
    new_cam_distance = g_cam_distance - yoffset * sensitivity
    if new_cam_distance >= 0.2:
        g_cam_distance = new_cam_distance
        
        # projection이 ortho로 바뀌어도 비율이 유지 되도록 조정.
        g_ortho_mag = g_cam_distance * 0.0417 
        if not g_P_toggle:
            update_projection_matrix()

def parsing_face_info(polygon_face_info):
    parsing = []
    for vertex_info_with_slash in polygon_face_info:
        vertex_info_list = vertex_info_with_slash.split('/')
        
        # ex) vertex_position_index
        if len(vertex_info_list) == 1:
            v_idx = int(vertex_info_list[0]) - 1
            texture_idx = None
            vn_idx = None
            
        # ex) vertex_position_index / texture_coordinates_index
        elif len(vertex_info_list) == 2:
            v_idx = int(vertex_info_list[0]) - 1
            texture_idx = int(vertex_info_list[1]) - 1 if vertex_info_list[1] else None
            vn_idx = None
            
        # ex) vertex_position_index / texture_coordinates_index / vertex_normal_index
        elif len(vertex_info_list) == 3:
            v_idx = int(vertex_info_list[0]) - 1
            texture_idx = int(vertex_info_list[1]) - 1 if vertex_info_list[1] else None
            vn_idx = int(vertex_info_list[2]) - 1 if vertex_info_list[2] else None
            
        parsing.append( [v_idx, texture_idx, vn_idx] )
    return parsing

def open_and_parse_obj_file(path):
    obj_file = open(path, 'r')
    obj_v = []
    obj_vn = []
    obj_f = []
    
    obj_vertices=[]
    obj_vertices_with_normal=[]
    
    ## parsing
    # v : vertex positions
    # vn : vertex normals
    # f : face information
    # ex) f vertex_position_index / texture_coordinates_index / vertex_normal_index
    # ! all argument indices are 1 based indices !
    acc = []
    for line in obj_file:
        fields = line.strip().split()
        #print(fields)
        if len(fields) == 0: # 빈 줄인 경우.
            continue
        
        # @ If previous line ends with backslash, this line is connected to the previous line
        # @ then, this line is stored acc.
        if fields[-1]=='\\':
            acc.extend(fields[:-1])
            continue
        
        # @ If there is previous line
        if len(acc) != 0:
            acc.extend(fields)
            fields = acc
            acc = []                 
            print(fields)

        if fields[0] == 'v':
            obj_v.append( [float(x) for x in fields[1:]] )
        elif fields[0] == 'vn':
            obj_vn.append( [float(x) for x in fields[1:]] )
        elif fields[0] == 'f':
            obj_f.append( parsing_face_info(fields[1:]) ) 
    
    # #1. vertices array
    
    for face in obj_f:
        indexed_vertex_infos_list = []
        if len(face) > 3:
            for i in range(1,(len(face)-1)):
                indexed_vertex_infos_list.extend(face[0:1]+face[i:i+2])
        elif len(face) == 3:
            indexed_vertex_infos_list = face
        else:
            print("error: the number of face information is less than 3")
            print(face)
        
        for indexed_vertex_infos in indexed_vertex_infos_list:
            vertex_info = []
            ## vertex position
            vertex_info.extend(obj_v[indexed_vertex_infos[0]])
            
            ## vertex normal
            if indexed_vertex_infos[2] != None:
                vertex_info.extend(obj_vn[indexed_vertex_infos[2]])
                
                ## normal이 있다면 vertices_with_normal에 저장
                obj_vertices_with_normal.extend(vertex_info)
            else:
                ## normal이 없다면 vertices에 저장
                obj_vertices.extend(vertex_info)
                
        if (len(obj_vertices)!=0):
            print("The obj file has some face information without vertex normal. We should know vertex normal to do lighting.")

    return (obj_vertices_with_normal,int(len(obj_vertices_with_normal)/6),obj_f)

def drop_callback(window, paths):
    print(paths)
    global g_single_mode_obj_path, g_rendering_mode_toggle, g_vao_single_mode_obj, g_vao_single_mode_obj_vertex_count
    g_single_mode_obj_path=paths[0]
    
    g_vao_single_mode_obj, g_vao_single_mode_obj_vertex_count = prepare_vao_single_mode(g_single_mode_obj_path)
    
    # change mode to single mesh rendering mode
    g_rendering_mode_toggle = 0

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -10.0, 0.0, 0.0,  1.0, 1.0, 1.0, # x-axis start
         10.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, -10.0, 0.0,  1.0, 1.0, 1.0, # y-axis start
         0.0, 10.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -10.0,  1.0, 1.0, 1.0, # z-axis start
         0.0, 0.0, 10.0,  0.0, 0.0, 1.0, # z-axis end 
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

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    
    r,g,b = .5, .5, .5
    half_size, start, end, line_num =10, -10, 10, 21
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

def prepare_vao_single_mode(obj_file_path):

    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    v, v_count, obj_f = open_and_parse_obj_file(obj_file_path)
    
    print("Obj file name:", os.path.basename(g_single_mode_obj_path))
    print("\nTotal number of faces:", len(obj_f))
    print("\nNumber of faces with 3 vertices:", len([x for x in obj_f if len(x)==3]))
    print("\nNumber of faces with 4 vertices:", len([x for x in obj_f if len(x)==4]))
    print("\nNumber of faces with more than 4 vertices:", len([x for x in obj_f if len(x)>4]))
    
    ## v with normal
    #print(v,v_count)
    vertices = glm.array(np.array(v,glm.float32))

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

    return VAO, v_count

def prepare_vao_obj(obj_file_path):

    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    v, v_count, obj_f = open_and_parse_obj_file(obj_file_path)
    #print(v,v_count)
    vertices = glm.array(np.array(v,glm.float32))

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

    return VAO, v_count

def draw_frame(vao, MVP, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 84)

def draw_single_mode(vao, VP, M, matcolor, unif_locs, vertex_count):
    MVP = VP * M
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(unif_locs['M'], 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(unif_locs['material_color'], matcolor.r, matcolor.g, matcolor.b)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)
    
def draw_obj_node_with_normal_(vao, node, VP, unif_locs, vertex_count):
    M = node.get_global_transform() * glm.scale(node.get_scale())
    MVP = VP * M
    color = node.get_color()
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(unif_locs['M'], 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(unif_locs['material_color'], color.r, color.g, color.b)
    glBindVertexArray(vao)
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
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.79, 0.86, 0.91, 0)
        glEnable(GL_DEPTH_TEST)
        if g_wireframe_solid_toggle:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

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
        #@ set uniform "view_pos" for specular
        glUniform3f(unif_locs_lighting['view_pos'], cam_pos.x, cam_pos.y, cam_pos.z)
        
        #@ draw obj file
        global g_rendering_mode_toggle
        if g_rendering_mode_toggle == 0:
            if g_single_mode_obj_path == None:
                g_rendering_mode_toggle = 1
            else:
                M = glm.mat4()
                #TODO prepare_vao 바꾸기
                
                draw_single_mode(g_vao_single_mode_obj, P*V*M, M, glm.vec3(0,1,1), unif_locs_lighting, g_vao_single_mode_obj_vertex_count)
        
        #@ draw lab9
        else:
            t = glfwGetTime()
            M = glm.mat4()
            
            ## prepare vao and node
            #@ boat
            vao_boat_obj, vao_boat_obj_v_cnt = prepare_vao_obj(g_boat_obj_path)
            boat = Node(None, glm.vec3(1,1,1), glm.vec3(0.46, 0.31, 0.27))
            
            #@ sail
            vao_sail_obj, vao_sail_obj_v_cnt = prepare_vao_obj(g_sail_obj_path)
            sail = Node(boat, glm.vec3(1,1,1), glm.vec3(0.99, 0.84, 0.57))
            
            #@ cat
            vao_cat_obj, vao_cat_obj_v_cnt = prepare_vao_obj(g_cat_obj_path)
            cat = Node(boat, glm.vec3(1,1,1), glm.vec3(0.65, 0.61, 0.55))
            
            #@ crab
            vao_crab_obj, vao_crab_obj_v_cnt = prepare_vao_obj(g_crab_obj_path)
            crab = Node(sail, glm.vec3(.7,.7,.7), glm.vec3(0.99, 0.26, 0.39))
            
            #@ bird
            vao_bird_obj, vao_bird_obj_v_cnt = prepare_vao_obj(g_bird_obj_path)
            bird = Node(sail, glm.vec3(1,1,1), glm.vec3(1,1,1))
            
            #@ speaker
            vao_speaker_obj, vao_speaker_obj_v_cnt = prepare_vao_obj(g_speaker_obj_path)
            speaker = Node(cat, glm.vec3(1,1,1), glm.vec3(0.92, 0.45, 0.34))
            
            #@ beachBall
            vao_beachBall_obj, vao_beachBall_obj_v_cnt = prepare_vao_obj(g_beachBall_obj_path)
            beachBall = Node(cat, glm.vec3(1,1,1), glm.vec3(1, 0.73, 0.26))
            
            #@ shark
            vao_shark_obj, vao_shark_obj_v_cnt = prepare_vao_obj(g_shark_obj_path)
            shark = Node(boat, glm.vec3(1,1,1), glm.vec3(0.46, 0.66, 0.68))
            
            #@ fish1
            vao_fish1_obj, vao_fish1_obj_v_cnt = prepare_vao_obj(g_fish1_obj_path)
            fish1 = Node(shark, glm.vec3(.5,.5,.5), glm.vec3(0.66, 0.67, 0.82))
            
            #@ fish2
            vao_fish2_obj, vao_fish2_obj_v_cnt = prepare_vao_obj(g_fish2_obj_path)
            fish2 = Node(shark, glm.vec3(.5,.5,.5), glm.vec3(0.79, 0.65, 0.76))
            
            # move_like_eight = glm.translate(glm.vec3(-5,0,0)) * glm.rotate(-t, glm.vec3(0,1,0)) * glm.translate(glm.vec3(5,0,0))* glm.translate(glm.vec3(0,0.2+0.1*glm.sin(t),0)) if glm.sin(t/2) > 0 else  glm.rotate(glm.pi(), glm.vec3(0,1,0)) * glm.translate(glm.vec3(-5,0,0)) * glm.rotate(t, glm.vec3(0,1,0)) * glm.translate(glm.vec3(5,0,0))* glm.translate(glm.vec3(0,0.2+0.1*glm.sin(t),0)) * glm.rotate(glm.pi(), glm.vec3(0,1,0))
            
            ## set local transformations of each node
            boat.set_transform(
                glm.rotate(-t/10, glm.vec3(0,1,0)) * glm.translate(glm.vec3(5,0,0))*
                glm.translate(glm.vec3(0,0.2+0.1*glm.sin(t),0))
                )            
            sail.set_transform(glm.translate(glm.vec3(0,0,-1.25))*glm.rotate(0.3*glm.sin(t),glm.vec3(0,1,0)))
            cat.set_transform(
                glm.translate(glm.vec3(0,0,0.6+glm.sin(t)))
                * (glm.rotate(glm.pi(), glm.vec3(0,1,0)) if glm.cos(t) < 0 else glm.mat4() )
                * glm.translate(glm.vec3(0,0.05+0.05*glm.sin(t*6),0))
            )
            crab.set_transform((glm.translate(glm.vec3(.20,2.25,0)))
                               * (glm.translate(glm.vec3(0,glm.sin(t),0)))
                               * glm.rotate(-glm.pi()/2, glm.vec3(0,1,0))
                               * glm.rotate(0.3*glm.sin(t*6), glm.vec3(1,0,0))
                               )
            bird.set_transform((glm.translate(glm.vec3(0,4.3,-0.2)))* glm.translate(glm.vec3(0,0.1*(1+glm.sin(8*t)),0)) )
            speaker.set_transform(glm.translate(glm.vec3(0,0.7+0.1*glm.sin(t*6),0))*glm.rotate(0.3*glm.sin(t*3), glm.vec3(1,0,0)) * glm.rotate(glm.pi()/2, glm.vec3(0,1,0)))
            beachBall.set_transform(glm.translate(glm.vec3(0,.2,.6+.05*(glm.sin(8*t)))))
            shark.set_transform(glm.translate(glm.vec3(1,-1,0)) 
                                
                                *glm.rotate(0.3*glm.sin(t), glm.vec3(1,0,0))
                                
                                *glm.rotate(-t, glm.vec3(0,1,0))
                                *glm.translate(glm.vec3(4,0,0)) 
                                
                                * glm.rotate(-glm.pi()/2, glm.vec3(0,1,0)))
            fish1.set_transform(glm.translate(glm.vec3(1,0,0)) *glm.rotate(0.5*glm.sin(t*6), glm.vec3(1,0,0)) )
            fish2.set_transform(glm.rotate((t*3), glm.vec3(1,0,0)) * glm.translate(glm.vec3(0,0,1))* glm.rotate(-(t*3), glm.vec3(1,0,0))* glm.rotate(glm.pi()/2, glm.vec3(0,1,0)))
            
            ## recursively update global transformations of all nodes
            boat.update_tree_global_transform()

            ## draw nodes
            draw_obj_node_with_normal_(vao_boat_obj, boat, P*V, unif_locs_lighting, vao_boat_obj_v_cnt)
            draw_obj_node_with_normal_(vao_sail_obj, sail, P*V, unif_locs_lighting, vao_sail_obj_v_cnt)
            draw_obj_node_with_normal_(vao_cat_obj, cat, P*V, unif_locs_lighting, vao_cat_obj_v_cnt)
            draw_obj_node_with_normal_(vao_crab_obj, crab, P*V, unif_locs_lighting, vao_crab_obj_v_cnt)
            draw_obj_node_with_normal_(vao_bird_obj, bird, P*V, unif_locs_lighting, vao_bird_obj_v_cnt)
            draw_obj_node_with_normal_(vao_speaker_obj, speaker, P*V, unif_locs_lighting, vao_speaker_obj_v_cnt)
            draw_obj_node_with_normal_(vao_beachBall_obj, beachBall, P*V, unif_locs_lighting, vao_beachBall_obj_v_cnt)
            draw_obj_node_with_normal_(vao_shark_obj, shark, P*V, unif_locs_lighting, vao_shark_obj_v_cnt)
            draw_obj_node_with_normal_(vao_fish1_obj, fish1, P*V, unif_locs_lighting, vao_fish1_obj_v_cnt)
            draw_obj_node_with_normal_(vao_fish2_obj, fish2, P*V, unif_locs_lighting, vao_fish2_obj_v_cnt)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
