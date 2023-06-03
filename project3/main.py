from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os
import time


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

# for mode ( 0: single mesh rendering mode, 1: animating hierarchical model rendering mode )
g_rendering_mode_toggle = 1

# bvh info
g_node_list = None
g_max_offset_length = 1
g_min_offset_length = 0.05
g_channel_list = None
g_motion_data = None
g_motion_data_line_num = 0
g_motion_data_line_max = 0
g_frame_time = 1
g_animate_toggle = False
g_aux_toggle = True

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

class Channel:
    def __init__(self, axis, type):
        self.axis = axis
        self.type = type
    
    def set_value(self, value):
        self.value = value
        
    def get_axis(self):
        return self.axis
    def get_type(self):
        return self.type
    def get_value(self):
        return self.value
    
    def toString(self):
        return self.axis + self.type +": "+ str(self.value)

class Node:
    def __init__(self, parent, offset, color, name):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = glm.translate(offset)
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = glm.mat4()
        self.line_transform = glm.mat4()

        self.color = color
        
        self.name = name
        self.offset = offset

    def set_channels(self, channels):
        self.channels = channels
        
    def calculate_shape_transform(self):
        
        # # decide the child to shape with
        child_to_shape_with = None
        
        if len(self.children) == 1 or len(self.children) == 2:
            child_to_shape_with = self.children[0]
        elif len(self.children) > 2:
            is_exist_left = False
            is_exist_right = False
            non_direction_child = []
            for child in self.children:
                if child.name[0:4].upper() == "LEFT":
                    is_exist_left = True
                elif child.name[0:5].upper() == "RIGHT":
                    is_exist_right = True
                else:
                    non_direction_child.append(child)
                    
            if is_exist_left and is_exist_right and len(non_direction_child) != 0:
                child_to_shape_with = non_direction_child[0]
            else:
                child_to_shape_with = self.children[0]      
                   
        if child_to_shape_with != None:
            # cube의 굵기로 제일 작은 노드의 크기로 함
            # thickness = g_min_offset_length / 1
            thickness = g_max_offset_length / 15 # 최대 offset의 1/15로 두께 설정
            angle_between_polygon_and_direction = calculate_angle_between_vectors(child_to_shape_with.offset,glm.vec3(0,1,0))
            polygon_rotation = glm.rotate(angle_between_polygon_and_direction,glm.cross((0,1,0),child_to_shape_with.offset)) if angle_between_polygon_and_direction != .0 else glm.mat4()
            if angle_between_polygon_and_direction == .0:
                polygon_rotation = glm.mat4()
            elif angle_between_polygon_and_direction == np.pi:
                polygon_rotation = glm.scale((1,-1,1))
            else:
                polygon_rotation = glm.rotate(angle_between_polygon_and_direction,glm.cross((0,1,0),child_to_shape_with.offset))
            # polygon can be line or cube
            shape_magnitude = glm.length(child_to_shape_with.offset)
            cube_shape = glm.vec3(.05,shape_magnitude,.05)
            cube_shape = glm.vec3(thickness,shape_magnitude,thickness)
            line_shape = glm.vec3(0.1,shape_magnitude,0.1)

            self.shape_transform = polygon_rotation * glm.scale(cube_shape)         
            self.line_transform =  polygon_rotation * glm.scale(line_shape)
        # node의 child가 없는 경우, JOINT가 END SITE로 끝나지 않은 경우이다.
        else:
            print("error: the JOINT[",self.name,"] doesn't end with 'End Site'")
            self.shape_transform = glm.scale((0,0,0))         
            self.line_transform =  glm.scale((0,0,0))
            
        
    def calculate_joint_transform(self):
        if self.channels == None:
            self.joint_transform = glm.mat4()
        else:
            self.joint_transform = glm.mat4()
            for channel in self.channels:
                if channel.type == 'ROTATION':
                    if channel.axis == 'X':
                        J = glm.rotate(glm.radians(channel.value),glm.vec3(1,0,0))
                    elif channel.axis == 'Y':
                        J = glm.rotate(glm.radians(channel.value),glm.vec3(0,1,0))
                    elif channel.axis == 'Z':
                        J = glm.rotate(glm.radians(channel.value),glm.vec3(0,0,1))
                if channel.type == 'POSITION':
                    if channel.axis == 'X':
                        J = glm.translate(glm.vec3(channel.value,0,0))
                    elif channel.axis == 'Y':
                        J = glm.translate(glm.vec3(0,channel.value,0))
                    elif channel.axis == 'Z':
                        J = glm.translate(glm.vec3(0,0,channel.value))
                self.joint_transform = self.joint_transform * J

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def update_channel_list_value(channel_list, motion_data, line_num):
    for i in range(0,len(channel_list)):
        channel_list[i].set_value(motion_data[line_num][i])
    
def update_node_list_joint_transform(node_list):
    for node in node_list:
        node.calculate_joint_transform()
    
def calculate_angle_between_vectors(vector1, vector2):
    normalized_vector1 = glm.normalize(vector1)
    normalized_vector2 = glm.normalize(vector2)

    dot_product = glm.dot(normalized_vector1, normalized_vector2)
    angle = glm.acos(dot_product)

    return angle


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
        global g_P_toggle, g_rendering_mode_toggle, g_animate_toggle, g_aux_toggle
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                g_P_toggle = not g_P_toggle
                update_projection_matrix()
            elif key==GLFW_KEY_1:
                g_rendering_mode_toggle = 0
            elif key==GLFW_KEY_2:
                g_rendering_mode_toggle = 1
            elif key==GLFW_KEY_3:
                g_aux_toggle = not g_aux_toggle
            elif key==GLFW_KEY_SPACE:
                g_animate_toggle = not g_animate_toggle
            


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

def parse_bvh_file(path):
    node_list = [None]
    channel_list = []
    max_offset_length = 1
    min_offset_length = None

    with open(path, 'r') as file:
        lines = file.readlines()

        # variable declaration
        joint_name_list = []
        joint_name = None
        offset = None
        channels = []

        motion_data = []

        hierarchy_section = False
        motion_section = False
        
        node_section = False
        end_site_section = False
        
        parent_stack = [None]
        

        for line in lines:
            # 토큰 분할
            tokens = line.upper().strip().split()

            if tokens[0] == 'HIERARCHY':
                hierarchy_section = True
                motion_section = False
            elif tokens[0] == 'MOTION':
                hierarchy_section = False
                motion_section = True
            elif hierarchy_section:
                ## ROOT or JOINT
                if tokens[0] == 'ROOT' or tokens[0] == 'JOINT':
                    joint_name = (line.strip().split())[1] # 모두 대문자로 변환시킨 것을 풀어주기 위해, 새로 할당.
                    joint_name_list.append(joint_name)
                    node_section = True
                ## End Site
                elif tokens[0] == 'END':
                    end_site_section = True
                ## OFFSET
                elif tokens[0] == 'OFFSET':
                    offset = glm.vec3(float(tokens[1]),float(tokens[2]),float(tokens[3]))
                    offset_length = glm.length(offset)
                    if max_offset_length < offset_length :
                        max_offset_length = offset_length
                    # 참고로 다음 노드에 나오는 offset을 전에 node의 길이로 설정하게 됨.
                ## CHANNEL
                elif tokens[0] == 'CHANNELS':
                    for channel_tokens in tokens[2:]:
                        channel = Channel(channel_tokens[0],channel_tokens[1:])
                        channel.set_value(0) # channel value 초기화. 후에 업데이트 될 예정.
                        channels.append(channel)
                        channel_list.append(channel)
                    ## 필요한 정보(JOINT, OFFSET, CHANNEL)가 모두 모였으면 NODE 생성
                    if node_section and (offset != None) and (len(channels) != 0):
                        # create a hierarchical model - Node(parent, link_transform_from_parent, color,joint_name)
                        node = Node(parent_stack[-1],offset, glm.vec3(1, 1, 0),joint_name)
                        node.set_channels(channels) # channel 설정
                        node_list.append(node)
                        parent_stack.append(node)
                        ## 사용한 정보 폐기
                        node_section = False
                        offset = None
                        channels = []
                elif tokens[0] == '}':
                    if end_site_section:
                        end_site_section = False
                        Node(parent_stack[-1],offset, glm.vec3(1,1,1), "END") # end site
                        # min_offset_length 를 구함. 노드를 그릴 cube의 두께를 위해서
                        if offset_length != 0:
                            if min_offset_length == None:
                                min_offset_length = offset_length
                            elif min_offset_length > offset_length:
                                min_offset_length = offset_length
                    else:
                        del parent_stack[-1]
            elif motion_section:
                if tokens[0] == 'FRAMES:':
                    # 프레임 수
                    frame_count = int(tokens[1])
                elif tokens[0] == 'FRAME' and tokens[1] == 'TIME:':
                    # 프레임 간 시간 간격 추출
                    frame_time = float(tokens[2])
                else:
                    # 모션 데이터 추출
                    motion_data.append([float(token) for token in tokens])

    if min_offset_length == None:
        min_offset_length = 0.1
        
    return node_list[1:], max_offset_length, min_offset_length, channel_list, joint_name_list, frame_count, frame_time, motion_data

# load bvh file
def drop_callback(window, paths):
    bvh_path=paths[0]
    global g_node_list,g_max_offset_length, g_min_offset_length, g_channel_list, g_motion_data, g_motion_data_line_num, g_motion_data_line_max, g_animate_toggle, g_frame_time
    g_node_list, g_max_offset_length, g_min_offset_length, g_channel_list, joint_name_list, frame_count, g_frame_time, g_motion_data = parse_bvh_file(bvh_path)
    for node in g_node_list:
        node.calculate_shape_transform()
    g_motion_data_line_num = 0
    g_motion_data_line_max = len(g_motion_data)
    g_animate_toggle = False
    
    #update_channel_list_value(g_channel_list, g_motion_data, 0)
    update_node_list_joint_transform(g_node_list)

    # 파싱 결과 출력
    print("File name:", os.path.basename(paths[0]))
    print("Number of frames:", frame_count)
    print("FPS:", 1/g_frame_time)
    print("Number of joints:", len(joint_name_list))
    print("List of all joint names:", joint_name_list)

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

def prepare_vao_cube():
    # prepare vertex data (in main memory)0
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position      color
        -0.5 ,  1.0 ,  0.5 ,   0, 0, 1, # v0
        0.5 ,  0.0 ,  0.5 ,   0, 0, 1, # v2
        0.5 ,  1.0 ,  0.5 ,   0, 0, 1, # v1

        -0.5 ,  1.0 ,  0.5 ,   0, 0, 1, # v0
        -0.5 ,  0.0 ,  0.5 ,   0, 0, 1, # v3
        0.5 ,  0.0 ,  0.5 ,   0, 0, 1, # v2

        -0.5 ,  1.0 , -0.5 ,   0, 0,-1, # v4
        0.5 ,  1.0 , -0.5 ,   0, 0,-1, # v5
        0.5 ,  0.0 , -0.5 ,   0, 0,-1, # v6

        -0.5 ,  1.0 , -0.5 ,   0, 0,-1, # v4
        0.5 ,  0.0 , -0.5 ,   0, 0,-1, # v6
        -0.5 ,  0.0 , -0.5 ,   0, 0,-1, # v7

        -0.5 ,  1.0 ,  0.5 ,   0, 1, 0, # v0
        0.5 ,  1.0 ,  0.5 ,   0, 1, 0, # v1
        0.5 ,  1.0 , -0.5 ,   0, 1, 0, # v5

        -0.5 ,  1.0 ,  0.5 ,   0, 1, 0, # v0
        0.5 ,  1.0 , -0.5 ,   0, 1, 0, # v5
        -0.5 ,  1.0 , -0.5 ,   0, 1, 0, # v4

        -0.5 ,  0.0 ,  0.5 ,   0,-1, 0, # v3
        0.5 ,  0.0 , -0.5 ,   0,-1, 0, # v6
        0.5 ,  0.0 ,  0.5 ,   0,-1, 0, # v2

        -0.5 ,  0.0 ,  0.5 ,   0,-1, 0, # v3
        -0.5 ,  0.0 , -0.5 ,   0,-1, 0, # v7
        0.5 ,  0.0 , -0.5 ,   0,-1, 0, # v6

        0.5 ,  1.0 ,  0.5 ,   1, 0, 0, # v1
        0.5 ,  0.0 ,  0.5 ,   1, 0, 0, # v2
        0.5 ,  0.0 , -0.5 ,   1, 0, 0, # v6

        0.5 ,  1.0 ,  0.5 ,   1, 0, 0, # v1
        0.5 ,  0.0 , -0.5 ,   1, 0, 0, # v6
        0.5 ,  1.0 , -0.5 ,   1, 0, 0, # v5

        -0.5 ,  1.0 ,  0.5 ,  -1, 0, 0, # v0
        -0.5 ,  0.0 , -0.5 ,  -1, 0, 0, # v7
        -0.5 ,  0.0 ,  0.5 ,  -1, 0, 0, # v3

        -0.5 ,  1.0 ,  0.5 ,  -1, 0, 0, # v0
        -0.5 ,  1.0 , -0.5 ,  -1, 0, 0, # v4
        -0.5 ,  0.0 , -0.5 ,   -1, 0, 0, # v7
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

def prepare_vao_line():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position      color
         #y
         0 ,  0 ,  0 ,  1, 1, 0,
         0 ,  1 ,  0 ,  1, 1, 0,
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

def prepare_vao_line_with_offsets(offset1,offset2):
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position      color
        offset1[0],offset1[1],offset1[2], 1, 1, 0,
        offset2[0],offset2[1],offset2[2], 1, 1, 0,
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

def draw_frame(vao, MVP, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, unif_locs):
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 84)
    
def draw_node_with_line(vao, node, VP, unif_locs):
    retouch_scale_for_huge_model = glm.scale((1/g_max_offset_length, 1/g_max_offset_length,1/g_max_offset_length))
    M = retouch_scale_for_huge_model * node.get_global_transform() * node.line_transform
    MVP = VP * M
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 2)

    # aux line
    if len(node.children) > 2 and g_aux_toggle:
        for child in node.children:
            if  not (child.offset in [x.offset for x in node.children if x.name != child.name]):
                vao_aux_line = prepare_vao_line_with_offsets(node.offset,child.offset)
                M = retouch_scale_for_huge_model * node.get_global_transform()
                MVP = VP * M
                glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
                glBindVertexArray(vao_aux_line)
                glDrawArrays(GL_LINES, 0, 2)

def draw_node_with_cube(vao, node, VP, unif_locs):
    retouch_scale_for_huge_model = glm.scale((1/g_max_offset_length, 1/g_max_offset_length,1/g_max_offset_length))
    M = retouch_scale_for_huge_model * node.get_global_transform() * node.get_shape_transform()
    MVP = VP * M
    color = node.get_color()
    glUniformMatrix4fv(unif_locs['MVP'], 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(unif_locs['M'], 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(unif_locs['material_color'], color.r, color.g, color.b)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 36)

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'project3 2019019043 박종윤', None, None)
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
    vao_cube = prepare_vao_cube()
    vao_line = prepare_vao_line()
  

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
        # set uniform "view_pos" for specular
        glUseProgram(shader_lighting)
        glUniform3f(unif_locs_lighting['view_pos'], cam_pos.x, cam_pos.y, cam_pos.z)
        
        ## drawing skeleton
        if g_node_list != None:
            # recursively update global transformations of all nodes
            g_node_list[0].update_tree_global_transform()

            # draw nodes
            for node in g_node_list:
                # drawing with line
                if g_rendering_mode_toggle == 0:
                    glUseProgram(shader_color)
                    draw_node_with_line(vao_line, node, P*V, unif_locs_color)
            
                # drawing with cube
                elif g_rendering_mode_toggle == 1:
                    glUseProgram(shader_lighting)
                    glUniform3f(unif_locs_lighting['view_pos'], cam_pos.x, cam_pos.y, cam_pos.z)
                    draw_node_with_cube(vao_cube, node, P*V, unif_locs_lighting)
            
            # animate if it needs    
            if g_animate_toggle:
                global g_motion_data_line_num
                g_motion_data_line_num += 1
                if (g_motion_data_line_num == g_motion_data_line_max):
                    g_motion_data_line_num = 0
                update_channel_list_value(g_channel_list, g_motion_data, g_motion_data_line_num)
                update_node_list_joint_transform(g_node_list)
                time.sleep(g_frame_time)
        
        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
