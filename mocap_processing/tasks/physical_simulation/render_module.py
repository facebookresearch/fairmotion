import numpy as np

viewer = None
camera = None
gl_render = None
bullet_render = None
gl = None
glu = None
glut = None

flag = {}
flag["follow_cam"] = True
flag["ground"] = True
flag["origin"] = False
flag["shadow"] = True
flag["sim_model"] = True
flag["kin_model"] = True
flag["joint"] = False
flag["com_vel"] = False
flag["collision"] = True
flag["overlay"] = False
flag["target_pose"] = False
flag["auto_play"] = False
flag["fog"] = False
flag["facing_frame"] = False
flag["interaction"] = True

toggle = {}
toggle[b"0"] = "follow_cam"
toggle[b"1"] = "ground"
toggle[b"2"] = "origin"
toggle[b"3"] = "shadow"
toggle[b"4"] = "sim_model"
toggle[b"5"] = "kin_model"
toggle[b"6"] = "joint"
toggle[b"7"] = "com_vel"
toggle[b"8"] = "collision"
toggle[b"9"] = "overlay"
toggle[b"t"] = "target_pose"
toggle[b"a"] = "auto_play"
toggle[b"f"] = "fog"
toggle[b"F"] = "facing_frame"
toggle[b"i"] = "interaction"

tex_id_ground = None
file_tex_ground = "data/image/grid2.png"

COLORS_FOR_AGENTS = [
    np.array([30, 120, 180, 255]) / 255,
    np.array([215, 40, 40, 255]) / 255,
    np.array([225, 120, 190, 255]) / 255,
    np.array([150, 100, 190, 255]) / 255,
    np.array([140, 90, 80, 255]) / 255,
    np.array([50, 160, 50, 255]) / 255,
    np.array([255, 125, 15, 255]) / 255,
    np.array([125, 125, 125, 255]) / 255,
    np.array([255, 0, 255, 255]) / 255,
    np.array([0, 255, 125, 255]) / 255,
]

COLOR_AGENT = np.array([85, 160, 173, 255]) / 255.0

COLORS_FOR_EXPERTS = [
    np.array([30, 120, 180, 255]) / 255,
    np.array([215, 40, 40, 255]) / 255,
    np.array([225, 120, 190, 255]) / 255,
    np.array([150, 100, 190, 255]) / 255,
    np.array([140, 90, 80, 255]) / 255,
    np.array([50, 160, 50, 255]) / 255,
    np.array([255, 125, 15, 255]) / 255,
    np.array([125, 125, 125, 255]) / 255,
    np.array([255, 0, 255, 255]) / 255,
    np.array([0, 255, 125, 255]) / 255,
]

initialized = False


def initialize():
    global initialized
    if initialized:
        return
    global viewer, gl_render, camera, bullet_render, gl, glu, glut
    from basecode.render import glut_viewer as viewer
    from basecode.render import camera
    from basecode.render import gl_render
    from basecode.bullet import bullet_render
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    import OpenGL.GLUT as glut

    initialized = False
