# Copyright (c) Facebook, Inc. and its affiliates.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import sys
import numpy as np
from fairmotion.viz import camera, utils

from PIL import Image


class Viewer:
    """Viewer class builds general infrastructure to implement visualizer
    class for motion sequences.

    Attributes:
        title: Title displayed on visualizer window
        cam: Camera object for the scene
        size: Tuple; Visualizer window dimensions
        mouse_last_pos: Tuple; Stores last recorded position of mouse on the
            screen
        pressed_button: str; Stores last pressed keyboard character
        time_checker: Object of utils.TimeChecker class to keep track of UNIX
            time and playback time

    To create a custom visualizer, extend this class and implement the
    following methods:
        - render_callback
        - idle_callback
        - keyboard_callback (optional)
        - overlay_callback (optional)

    Once the viewer is initialized, call `run` method to display visualization
    """

    def __init__(
        self, title="glutgui_base", cam=None, size=(800, 600),
    ):
        self.title = title
        self.window = None
        self.window_size = size
        self.mouse_last_pos = None
        self.pressed_button = None

        self.time_checker = utils.TimeChecker()
        if cam is None:
            self.cam_cur = camera.Camera(
                pos=np.array([0.0, 2.0, 4.0]),
                origin=np.array([0.0, 0.0, 0.0]),
                vup=np.array([0.0, 1.0, 0.0]),
                fov=45.0,
            )
        else:
            self.cam_cur = cam

    def idle_callback(self):
        pass

    def overlay_callback(self):
        pass

    def keyboard_callback(self, key):
        return True

    def render_callback(self):
        glutSolidSphere(0.3, 20, 20)
        gl_render.render_ground(
            size=[100, 100], color=[0.8, 0.8, 0.8],
        )

    def _init_GL(self, w, h):
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

    def resize_GL(self, w, h):
        self.window_size = (w, h)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.cam_cur.fov, float(w) / float(h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def draw_GL(self):
        # Clear The Screen And The Depth Buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            *self.cam_cur.pos, *self.cam_cur.origin, *self.cam_cur.vup,
        )

        self.render_callback()

        if self.overlay_callback is not None:
            glClear(GL_DEPTH_BUFFER_BIT)
            glPushAttrib(GL_DEPTH_TEST)
            glDisable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(
                0.0, self.window_size[0], self.window_size[1], 0.0, 0.0, 1.0
            )

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            self.overlay_callback()
            glPopMatrix()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glPopAttrib()

        glutSwapBuffers()

    # The function called whenever a key is pressed.
    # Note the use of Python tuples to pass in: (key, x, y)
    def key_pressed(self, *args):
        handled = self.keyboard_callback(args[0])
        if handled:
            return
        if args[0] == b"\x1b":
            print("Hit ESC key to quit.")
            glutDestroyWindow(self.window)
            sys.exit()

    def mouse_func(self, button, state, x, y):
        if state == 0:
            self.pressed_button = button
        else:
            self.pressed_button = None

        if state == 0:  # Mouse pressed
            self.mouse_last_pos = np.array([x, y])
        elif state == 1:
            self.mouse_last_pos = None

        if button == 3:
            self.cam_cur.zoom(0.95)
        elif button == 4:
            self.cam_cur.zoom(1.05)

    def motion_func(self, x, y):
        scale = 0.01
        newPos = np.array([x, y])
        d = scale * (newPos - self.mouse_last_pos)
        if self.pressed_button == 0:
            self.cam_cur.rotate(d[1], -d[0], 0)
        elif self.pressed_button == 2:
            self.cam_cur.translate(np.array([d[0], d[1], 0]), frame_local=True)
        self.mouse_last_pos = newPos

    def render_timer(self, timer):
        glutPostRedisplay()
        glutTimerFunc(10, self.render_timer, 1)

    def run(self):
        # Init glut
        glutInit(())
        glutInitDisplayMode(
            GLUT_RGBA
            | GLUT_DOUBLE
            | GLUT_ALPHA
            | GLUT_DEPTH
            | GLUT_MULTISAMPLE
        )
        glutInitWindowSize(*self.window_size)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(self.title)

        # Init functions
        # glutFullScreen()
        glutDisplayFunc(self.draw_GL)
        glutIdleFunc(self.idle_callback)
        glutReshapeFunc(self.resize_GL)
        glutKeyboardFunc(self.key_pressed)
        glutMouseFunc(self.mouse_func)
        glutMotionFunc(self.motion_func)
        glutTimerFunc(10, self.render_timer, 1)
        self._init_GL(*self.window_size)
        self.time_checker.begin()

        # Run
        glutMainLoop()

    def save_screen(self, dir, name, format="png", render=False):
        image = self.get_screen(render)
        image.save(os.path.join(dir, "%s.%s" % (name, format)), format=format)

    def get_screen(self, render=False):
        if render:
            self.draw_GL()
        x, y, width, height = glGetIntegerv(GL_VIEWPORT)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
