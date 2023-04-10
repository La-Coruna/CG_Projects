from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

#for debug
debug_var=45
da=1.5
db=.1
dc=3.5
printer = None

# for mouse control
g_mouse_button_left_toggle = False
g_mouse_button_right_toggle = False
g_cursor_last_xpos = 0
g_cursor_last_ypos = 0

# for orbit
g_cam_azimuth = 0. # z축으로 부터의 각임.
g_cam_elevation = .1
g_cam_orbit_direction_toggle = False # blender의 viewr와 같이, 처음 클릭했을 때의 elevation을 기준으로 azimuth의 회전 방향을 결정.

# for pan
g_cam_pan = glm.vec3(.0,.0,.0)          # pan move ( cam pos = orbit + pan )

# camera's vector
g_cam_direction = glm.vec3(.0,.0,.0)    # w vector of camera
g_cam_right = glm.vec3(.0,.0,.0)        # u vector of camera
g_cam_up = glm.vec3(.0,.0,.0)           # v vector of camera

# for zoom
g_cam_distance = 3.6 # the distance between camera and target ## d가 0.2보다 작아지면 깨짐. (d=3.6 , fovy = 45)일 때 ortho 와 배율이 비슷. 작을 수록 확대 됨.
g_ortho_mag = .15 # the magnification of lens in ortho ## 작을 수록 확대 됨.

# for projection
g_view_height = 10.
g_view_width = g_view_height * 800/800

g_P = glm.perspective(np.deg2rad(45), g_view_width/g_view_height , 0.1, 100) ## fovy should be in ( 0 ~ pi )
g_P_toggle = True

g_vertex_shader_src = '''
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

g_fragment_shader_src = '''
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

    #g_view_height = 10. # g_view_height은 항상 고정.
    g_view_width = g_view_height * width/height
    update_projection_matrix()

def key_callback(window, key, scancode, action, mods):
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    #for debug
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            global debug_var, printer, da, db, dc, g_cam_pan
            global  g_cam_azimuth, g_cam_elevation # for orbit
            if key==GLFW_KEY_V: #! not for debuging
                global g_P_toggle
                g_P_toggle = not g_P_toggle
                update_projection_matrix()
            elif key==GLFW_KEY_1:
                da -= .1
                print(da)
                update_projection_matrix()
            elif key==GLFW_KEY_2:
                da += .1
                print(da)
                update_projection_matrix()
            elif key==GLFW_KEY_3:
                debug_var -= 1
                print(debug_var)
                update_projection_matrix()
            elif key==GLFW_KEY_4:
                debug_var += 1
                print(debug_var)
                update_projection_matrix()
            elif key==GLFW_KEY_5:
                debug_var = 5
            elif key==GLFW_KEY_J:
                db += .1
                print(db)
                update_projection_matrix()
            elif key==GLFW_KEY_H:
                db += -.1
                print(db)
                update_projection_matrix()
            elif key==GLFW_KEY_L:
                dc += .1
                print(dc)
                update_projection_matrix()
            elif key==GLFW_KEY_K:
                dc += -.1
                print(dc)
                update_projection_matrix()
            elif key==GLFW_KEY_SPACE:
                print(printer)
            elif key==GLFW_KEY_UP:
                g_cam_pan += -g_cam_direction
            elif key==GLFW_KEY_DOWN:
                g_cam_pan += g_cam_direction
                
            elif key==GLFW_KEY_C:
                print(g_cursor_last_xpos,g_cursor_last_ypos)


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
        g_ortho_mag = g_cam_distance * 0.0417
        if not g_P_toggle:
            update_projection_matrix()
    

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

def prepare_vao_triangle_blue():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 1.0, 1.0, # v0
         0.5, 0.0, 0.0,  1.0, 1.0, 1.0, # v1
         0.0, 0.5, 0.0,  0.0, 0.0, 1.0, # v2
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
         -1.0, 0.0, 0.0,  1.0, 1.0, 1.0, # x-axis start
         1.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, -1.0, 0.0,  1.0, 1.0, 1.0, # y-axis start
         0.0, 1.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -1.0,  1.0, 1.0, 1.0, # z-axis start
         0.0, 0.0, 1.0,  0.0, 0.0, 1.0, # z-axis end 
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

def prepare_vao_minibox():
    # prepare vertex data (in main memory)
    vs=[[], # for index to start from 1
        [0.01, 0, 0.01         ,1,0,1],
        [-0.01, 0, 0.01        ,1,0,1],
        [-0.01, 0, -0.01       ,1,0,1],
        [0.01, 0, -0.01        ,1,0,1],
        [0.01, 0.02, 0.01       ,1,0,1],
        [-0.01, 0.02, 0.01      ,1,0,1],
        [-0.01, 0.02, -0.01     ,1,0,1],
        [0.01, 0.02, -0.01      ,1,0,1],
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


def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'project1', None, None)
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

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_triangle = prepare_vao_triangle()
    vao_triangle_blue = prepare_vao_triangle_blue()
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()
    vao_box = prepare_vao_box()
    vao_minibox = prepare_vao_minibox()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        # use orthogonal projection (we'll see details later)
        P = g_P

        #TODO  Zoom 기능 구현
        # view matrix
        
        cam_target = glm.vec3(0,0,0) + g_cam_pan
        cam_orbit = g_cam_distance * glm.vec3(
            np.cos(g_cam_elevation) * np.sin(g_cam_azimuth),
            np.sin(g_cam_elevation),
            np.cos(g_cam_elevation) * np.cos(g_cam_azimuth)
            ) # cam_orbit
        cam_pos = cam_orbit + g_cam_pan
        up = glm.vec3(.0, .1, .0) if np.rad2deg(g_cam_elevation) < 90 or np.rad2deg(g_cam_elevation) > 270 else glm.vec3(.0, -1, .0)
        
        global g_cam_direction,g_cam_right,g_cam_up
        g_cam_direction = glm.normalize(cam_pos-cam_target) # actually opposite direction
        g_cam_right = glm.normalize(glm.cross(up,g_cam_direction))
        g_cam_up = glm.normalize(glm.cross(g_cam_direction,g_cam_right))
        
        #cam_zoom = -g_cam_direction * g_cam_zoom
        
        #! for debug
        global printer
        printer=("cam_pos: ",cam_pos,"cam_target: ",cam_target)
        
        V = glm.lookAt(cam_pos, cam_target, up)
        
        
        ##
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)
        
        # draw current grid
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, 84)
        
        # draw current box
        glBindVertexArray(vao_box)
        #glDrawArrays(GL_TRIANGLE_STRIP, 0, 16)
        glDrawArrays(GL_LINE_STRIP, 0, 16)
        
        ###### mini box for focus
        M=glm.translate(g_cam_pan)
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        # draw current minibox
        glBindVertexArray(vao_minibox)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 16)

        ##
        # animating
        t = glfwGetTime()

        # rotation
        th = np.radians(t*90)
        R = glm.rotate(th, glm.vec3(0,0,1))

        # tranlation
        T = glm.translate(glm.vec3(np.sin(t), .2, 0.2))

        # scaling
        S = glm.scale(glm.vec3(np.sin(t), np.sin(t), np.sin(t)))

        M = T
        # M = T
        # M = S
        # M = R @ T
        # M = T @ R

        # current frame: P*V*M
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw triangle w.r.t. the current frame
        glBindVertexArray(vao_triangle)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        ## draw triangle_blue
        M = glm.translate(glm.vec3(np.sin(t), .2, -0.2))
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glBindVertexArray(vao_triangle_blue)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
