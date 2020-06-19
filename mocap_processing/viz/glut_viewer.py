from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import sys
import numpy as np
import time
from collections import deque
from mocap_processing.viz import camera, utils

from PIL import Image

import pickle
import gzip

global window
ESCAPE = "\033"
window = 0
window_size = None
mouse_last_pos = None
pressed_button = None
state = {}
cam_cur = None
step_callback_func = None
keyboard_callback_func = None
render_callback_func = None
overlay_callback_func = None
idle_callback_func = None

time_checker_fps = utils.TimeChecker()
times_per_frame = deque(maxlen=10)
avg_fps = 0.0


def initGL(w, h):
    glDisable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_MULTISAMPLE)

    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

    glEnable(GL_DITHER)
    glShadeModel(GL_SMOOTH)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    ambient = [0.2, 0.2, 0.2, 1.0]
    diffuse = [0.6, 0.6, 0.6, 1.0]
    front_mat_shininess = [60.0]
    front_mat_specular = [0.2, 0.2, 0.2, 1.0]
    front_mat_diffuse = [0.5, 0.28, 0.38, 1.0]
    lmodel_ambient = [0.2, 0.2, 0.2, 1.0]
    lmodel_twoside = [GL_FALSE]

    position = [1.0, 0.0, 0.0, 0.0]
    position1 = [-1.0, 1.0, 1.0, 0.0]

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
    glLightfv(GL_LIGHT0, GL_POSITION, position)

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient)
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside)

    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
    glLightfv(GL_LIGHT1, GL_POSITION, position1)
    glDisable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDisable(GL_CULL_FACE)
    glEnable(GL_NORMALIZE)

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_COLOR_MATERIAL)


def resizeGL(w, h):
    global cam_cur, window_size
    window_size = (w, h)
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(cam_cur.fov, float(w) / float(h), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def idleFunc():
    if idle_callback_func is not None:
        idle_callback_func()


def drawGL():
    global times_per_frame, avg_fps
    time_per_frame = time_checker_fps.get_time(restart=True)
    times_per_frame.append(time_per_frame)
    avg_fps = 1.0 / np.mean(times_per_frame)

    # Clear The Screen And The Depth Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    global cam_cur, window_size

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        cam_cur.pos[0],
        cam_cur.pos[1],
        cam_cur.pos[2],
        cam_cur.origin[0],
        cam_cur.origin[1],
        cam_cur.origin[2],
        cam_cur.vup[0],
        cam_cur.vup[1],
        cam_cur.vup[2],
    )

    if render_callback_func is not None:
        render_callback_func()
    else:
        glutSolidSphere(0.3, 20, 20)

    if overlay_callback_func is not None:
        glClear(GL_DEPTH_BUFFER_BIT)
        glPushAttrib(GL_DEPTH_TEST)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, window_size[0], window_size[1], 0.0, 0.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        overlay_callback_func()
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glPopAttrib()

    glutSwapBuffers()


# The function called whenever a key is pressed.
# Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    global keyboard_callback_func
    if keyboard_callback_func is not None:
        handled = keyboard_callback_func(args[0])
        if handled:
            return
    if args[0] == b"\x1b":
        print("Hit ESC key to quit.")
        glutDestroyWindow(window)
        sys.exit()


def mouseFunc(button, state, x, y):
    global mouse_last_pos, pressed_button

    if state == 0:
        pressed_button = button
    else:
        pressed_button = None

    if state == 0:  # Mouse pressed
        mouse_last_pos = np.array([x, y])
    elif state == 1:
        mouse_last_pos = None

    if button == 3:
        cam_cur.zoom(0.95)
    elif button == 4:
        cam_cur.zoom(1.05)


def motionFunc(x, y):

    global mouse_last_pos, cam_cur
    scale = 0.01
    newPos = np.array([x, y])
    d = scale * (newPos - mouse_last_pos)
    if pressed_button == 0:
        cam_cur.rotate(d[1], -d[0], 0)
    elif pressed_button == 2:
        cam_cur.translate(np.array([d[0], d[1], 0]), frame_local=True)
    mouse_last_pos = newPos


def renderTimer(timer):
    glutPostRedisplay()
    glutTimerFunc(10, renderTimer, 1)


def run(
    title="glutgui_base",
    cam=None,
    size=(800, 600),
    keyboard_callback=None,
    render_callback=None,
    overlay_callback=None,
    idle_callback=None,
):

    # Init glut
    global window
    glutInit(())
    glutInitDisplayMode(
        GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_MULTISAMPLE
    )
    glutInitWindowSize(size[0], size[1])
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow(title)

    global window_size
    window_size = size

    global cam_cur
    if cam is None:
        cam_cur = camera.Camera(
            pos=np.array([0.0, 2.0, 4.0]),
            origin=np.array([0.0, 0.0, 0.0]),
            vup=np.array([0.0, 1.0, 0.0]),
            fov=45.0,
        )
    else:
        cam_cur = cam

    # Init callback
    global keyboard_callback_func
    global render_callback_func
    global overlay_callback_func
    global idle_callback_func
    keyboard_callback_func = keyboard_callback
    render_callback_func = render_callback
    overlay_callback_func = overlay_callback
    idle_callback_func = idle_callback

    # Init functions
    # glutFullScreen()
    glutDisplayFunc(drawGL)
    glutIdleFunc(idleFunc)
    glutReshapeFunc(resizeGL)
    glutKeyboardFunc(keyPressed)
    glutMouseFunc(mouseFunc)
    glutMotionFunc(motionFunc)
    glutTimerFunc(10, renderTimer, 1)
    initGL(size[0], size[1])
    timer_start = time.time()
    time_checker_fps.begin()

    # Run
    glutMainLoop()


def save_cam(filename):
    global cam_cur
    with gzip.open(filename, "wb") as f:
        pickle.dump(cam_cur, f)


def load_cam(filename):
    global cam_cur
    with gzip.open(filename, "rb") as f:
        cam_cur = pickle.load(f)


def save_screen(dir, name, format="png", render=False):
    if render:
        drawGL()
    x, y, width, height = glGetIntegerv(GL_VIEWPORT)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(os.path.join(dir, "%s.%s" % (name, format)), format=format)


if __name__ == "__main__":
    run()


def update_target_pos(pos, ignore_x=False, ignore_y=False, ignore_z=False):
    global cam_cur
    if np.array_equal(pos, cam_cur.origin):
        return
    d = pos - cam_cur.origin
    if ignore_x:
        d[0] = 0.0
    if ignore_y:
        d[1] = 0.0
    if ignore_z:
        d[2] = 0.0
    cam_cur.translate(d)
