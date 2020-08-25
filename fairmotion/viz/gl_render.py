# Copyright (c) Facebook, Inc. and its affiliates.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import numpy as np
from PIL import Image

from fairmotion.utils import constants
from fairmotion.ops import conversions, math as math_ops


def load_texture(file):
    im = Image.open(file)
    ix, iy, im_data = im.size[0], im.size[1], im.tobytes("raw", "RGBA", 0, -1)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, im_data
    )
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    return tex_id


COLOR_SEQUENCE = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 0.5],
    [0, 1, 0, 0.5],
    [0, 0, 1, 0.5],
    [1, 1, 0, 0.5],
    [1, 0, 1, 0.5],
    [0, 1, 1, 0.5],
]


def glTransform(T):
    glMultMatrixd(T.transpose().ravel())


def glColor(color):
    num_val = len(color)
    if num_val == 3:
        glColor3d(color[0], color[1], color[2])
    elif num_val > 3:
        glColor4d(color[0], color[1], color[2], color[3])
    else:
        raise NotImplemented


def render_cylinder_info(
    T, length, radius, scale=1.0, line_width=2.0, color=[0, 0, 0, 1], slice=10
):
    glDisable(GL_LIGHTING)

    glPushMatrix()
    glTransform(T)
    glScalef(scale, scale, scale)

    glColor(color)
    glPushMatrix()
    glTranslated(0.0, 0.0, -0.5 * length)

    render_circle(
        constants.eye_T(),
        r=radius,
        slice=slice,
        scale=1.0,
        line_width=line_width,
        color=color,
        draw_plane="xy",
    )

    glTranslated(0.0, 0.0, length)

    render_circle(
        constants.eye_T(),
        r=radius,
        slice=slice,
        scale=1.0,
        line_width=line_width,
        color=color,
        draw_plane="xy",
    )
    glPopMatrix()

    render_line(
        p1=[radius, 0.0, -0.5 * length],
        p2=[radius, 0.0, 0.5 * length],
        color=color,
        line_width=line_width,
    )

    glPopMatrix()

    glEnable(GL_LIGHTING)


def render_capsule_info(
    T, length, radius, scale=1.0, line_width=2.0, color=[0, 0, 0, 1], slice=10
):
    render_cylinder_info(T, length, radius, scale, line_width, color, slice)


def render_sphere_info(
    T, r=1.0, slice=10, scale=1.0, line_width=2.0, color=[0, 0, 0, 1]
):

    render_circle(
        T=T,
        r=r,
        slice=slice,
        scale=scale,
        line_width=line_width,
        color=color,
        draw_plane="xy",
    )
    render_circle(
        T=T,
        r=r,
        slice=slice,
        scale=scale,
        line_width=line_width,
        color=color,
        draw_plane="yz",
    )
    render_circle(
        T=T,
        r=r,
        slice=slice,
        scale=scale,
        line_width=line_width,
        color=color,
        draw_plane="zx",
    )


def render_cylinder(T, length, radius, scale=1.0, color=[0, 0, 0, 1], slice=16):
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluQuadricNormals(quadric, GLU_SMOOTH)

    glEnable(GL_DEPTH_TEST)
    glPushMatrix()
    glTransform(T)
    glScalef(scale, scale, scale)

    glColor(color)

    glTranslated(0.0, 0.0, -0.5 * length)
    gluCylinder(quadric, radius, radius, length, slice, 1)
    gluDisk(quadric, 0.0, radius, slice, 1)
    glTranslated(0.0, 0.0, length)
    gluDisk(quadric, 0.0, radius, slice, 1)

    glPopMatrix()


def render_capsule(T, length, radius, scale=1.0, color=[0, 0, 0, 1], slice=16):
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluQuadricNormals(quadric, GLU_SMOOTH)


    glEnable(GL_DEPTH_TEST)
    glPushMatrix()
    glTransform(T)
    glScalef(scale, scale, scale)

    glColor(color)

    glTranslated(0.0, 0.0, -0.5 * length)
    gluSphere(quadric, radius, slice, slice)
    gluCylinder(quadric, radius, radius, length, slice, 1)
    glTranslated(0.0, 0.0, length)
    gluSphere(quadric, radius, slice, slice)

    glPopMatrix()


def render_cube(
    T, size=[1.0, 1.0, 1.0], color=[0, 0, 0, 1], solid=True, line_width=1.0
):
    glPushMatrix()
    glTransform(T)
    glScalef(size[0], size[1], size[2])

    glColor(color)

    if solid:
        glutSolidCube(1.0)
    else:
        glLineWidth(line_width)
        glutWireCube(1.0)

    glPopMatrix()


def render_sphere(T, r=1.0, slice1=10, slice2=10, scale=1.0, color=[0, 0, 0]):
    glPushMatrix()
    glTransform(T)
    glScalef(scale, scale, scale)

    glColor(color)

    glutSolidSphere(r, slice1, slice2)

    glPopMatrix()


def render_disk(
    T,
    r_inner=0.0,
    r_outer=1.0,
    slice1=32,
    slice2=1,
    scale=1.0,
    color=[0.8, 0.8, 0.8, 1.0],
):
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluQuadricNormals(quadric, GLU_SMOOTH)

    glPushMatrix()
    glTransform(T)
    glScalef(scale, scale, scale)

    glColor(color)

    gluDisk(quadric, r_inner, r_outer, slice1, slice2)

    glPopMatrix()


def render_circle(
    T, r=1.0, slice=128, scale=1.0, line_width=1.0, color=[0, 0, 0], draw_plane="xy"
):
    glPushMatrix()
    glTransform(T)
    glScalef(scale, scale, scale)

    glColor(color)
    glLineWidth(line_width)

    glBegin(GL_LINE_LOOP)
    for i in range(slice):
        theta = 2.0 * i * math.pi / slice
        if draw_plane == "xy":
            glVertex3f(math.cos(theta) * r, math.sin(theta) * r, 0.0)
        elif draw_plane == "yz":
            glVertex3f(0.0, math.cos(theta) * r, math.sin(theta) * r)
        else:
            glVertex3f(math.cos(theta) * r, 0.0, math.sin(theta) * r)
    glEnd()

    glPopMatrix()


def render_point(p, scale=1.0, radius=1.0, color=[1.0, 1.0, 1.0, 1.0]):
    glPushMatrix()
    glTranslated(p[0], p[1], p[2])
    glScalef(scale, scale, scale)

    glColor(color)
    glutSolidSphere(radius, 10, 10)

    glPopMatrix()


def render_line(p1, p2, color=[0.0, 0.0, 0.0, 1.0], line_width=1.0):
    glLineWidth(line_width)

    glBegin(GL_LINES)
    glColor(color)
    glVertex3d(p1[0], p1[1], p1[2])
    glVertex3d(p2[0], p2[1], p2[2])
    glEnd()


def render_quad(
    p1,
    p2,
    p3,
    p4,
    n=None,
    color=[1.0, 1.0, 1.0, 1.0],
    tex_id=None,
    tex_param1=[0, 0],
    tex_param2=[1, 0],
    tex_param3=[1, 1],
    tex_param4=[0, 1],
):

    draw_tex = tex_id is not None
    if draw_tex:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
    else:
        glColor(color)

    if n is None:
        n = math_ops.normalize(np.cross(p3 - p2, p2 - p1))

    glBegin(GL_QUADS)

    if draw_tex:
        glTexCoord2f(tex_param1[0], tex_param1[1])
    glNormal3d(n[0], n[1], n[2])
    glVertex3d(p1[0], p1[1], p1[2])

    if draw_tex:
        glTexCoord2f(tex_param2[0], tex_param2[1])
    glNormal3d(n[0], n[1], n[2])
    glVertex3d(p2[0], p2[1], p2[2])

    if draw_tex:
        glTexCoord2f(tex_param3[0], tex_param3[1])
    glNormal3d(n[0], n[1], n[2])
    glVertex3d(p3[0], p3[1], p3[2])

    if draw_tex:
        glTexCoord2f(tex_param4[0], tex_param4[1])
    glNormal3d(n[0], n[1], n[2])
    glVertex3d(p4[0], p4[1], p4[2])

    glEnd()

    if draw_tex:
        glDisable(GL_TEXTURE_2D)


def render_tri(p1, p2, p3, color=[1.0, 1.0, 1.0, 1.0]):
    glColor(color)
    glBegin(GL_TRIANGLES)
    glVertex3d(p1[0], p1[1], p1[2])
    glVertex3d(p2[0], p2[1], p2[2])
    glVertex3d(p3[0], p3[1], p3[2])
    glEnd()


def render_tet(p1, p2, p3, p4, color=[1.0, 1.0, 1.0, 1.0]):
    render_tri(p1, p2, p3, color)
    render_tri(p1, p2, p4, color)
    render_tri(p2, p3, p4, color)
    render_tri(p3, p1, p4, color)


def render_tet_line(p1, p2, p3, p4, color=[0.0, 0.0, 0.0, 1.0], line_width=1.0):
    render_line(p1, p2, color, line_width)
    render_line(p2, p3, color, line_width)
    render_line(p3, p1, color, line_width)
    render_line(p1, p4, color, line_width)
    render_line(p2, p4, color, line_width)
    render_line(p3, p4, color, line_width)


def render_ground_texture(
    tex_id,
    size=[20.0, 20.0],
    dsize=[1.0, 1.0],
    axis="y",
    origin=True,
    use_arrow=True,
    circle_cut=False,
    circle_color=[1, 1, 1, 1],
    circle_offset=0.001,
):
    assert tex_id > 0

    lx = size[0]
    lz = size[1]
    dx = dsize[0]
    dz = dsize[1]
    nx = int(lx / dx) + 1
    nz = int(lz / dz) + 1

    if axis is "x":
        raise NotImplementedError
    elif axis is "y":
        up_vec = np.array([0.0, 1.0, 0.0])
        p1 = np.array([-0.5 * size[0], 0, -0.5 * size[0]])
        p2 = np.array([0.5 * size[0], 0, -0.5 * size[0]])
        p3 = np.array([0.5 * size[0], 0, 0.5 * size[0]])
        p4 = np.array([-0.5 * size[0], 0, 0.5 * size[0]])
    elif axis is "z":
        up_vec = np.array([0.0, 0.0, 1.0])
        p1 = np.array([-0.5 * size[0], -0.5 * size[0], 0])
        p2 = np.array([0.5 * size[0], -0.5 * size[0], 0])
        p3 = np.array([0.5 * size[0], 0.5 * size[0], 0])
        p4 = np.array([-0.5 * size[0], 0.5 * size[0], 0])

    render_quad(
        p1,
        p2,
        p3,
        p4,
        tex_id=tex_id,
        tex_param1=[0, 0],
        tex_param2=[size[0] / dsize[0], 0],
        tex_param3=[size[0] / dsize[0], size[1] / dsize[1]],
        tex_param4=[0, size[1] / dsize[0]],
    )

    if origin:
        render_transform(constants.eye_T(), use_arrow=use_arrow)

    if circle_cut:
        r_inner = min(0.5 * size[0], 0.5 * size[1])
        r_outer = 1.5 * max(0.5 * size[0], 0.5 * size[1])
        offset = circle_offset * up_vec
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(offset[0], offset[1], offset[2])
        if axis is "y":
            glRotated(-90.0, 1, 0, 0)
        render_disk(
            constants.eye_T(),
            r_inner=r_inner,
            r_outer=r_outer,
            slice1=64,
            slice2=32,
            scale=1.0,
            color=circle_color,
        )
        glPopMatrix()


def render_path(data, color=[0.0, 0.0, 0.0], scale=1.0, line_width=1.0, point_size=1.0):
    glColor(color)
    glLineWidth(line_width)
    glBegin(GL_LINE_STRIP)
    for d in data:
        R, p = conversions.T2Rp(d)
        glVertex3d(p[0], p[1], p[2])
    glEnd()

    for d in data:
        render_transform(d, scale, line_width, point_size)


def render_arrow(p1, p2, D=0.1, color=[1.0, 0.5, 0.0], closed=False):
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluQuadricNormals(quadric, GLU_SMOOTH)

    glColor(color)
    RADPERDEG = 0.0174533
    d = p2 - p1
    x = d[0]
    y = d[1]
    z = d[2]
    L = np.linalg.norm(d)

    glPushMatrix()

    glTranslated(p1[0], p1[1], p1[2])

    if x != 0.0 or y != 0.0:
        glRotated(math.atan2(y, x) / RADPERDEG, 0.0, 0.0, 1.0)
        glRotated(math.atan2(math.sqrt(x * x + y * y), z) / RADPERDEG, 0.0, 1.0, 0.0)
    elif z < 0:
        glRotated(180, 1.0, 0.0, 0.0)

    glTranslatef(0, 0, L - 4 * D)

    gluCylinder(quadric, 2 * D, 0.0, 4 * D, 32, 1)
    if closed:
        gluDisk(quadric, 0.0, 2 * D, 32, 1)

    glTranslatef(0, 0, -L + 4 * D)

    gluCylinder(quadric, D, D, L - 4 * D, 32, 1)
    if closed:
        gluDisk(quadric, 0.0, D, 32, 1)

    glPopMatrix()


def render_transform(
    T,
    scale=1.0,
    line_width=1.0,
    point_size=0.05,
    render_pos=True,
    render_ori=[True, True, True],
    color_pos=[0, 0, 0, 1],
    color_ori=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    use_arrow=False,
):
    glLineWidth(line_width)

    R, p = conversions.T2Rp(T)

    glPushMatrix()
    glTranslated(p[0], p[1], p[2])
    glScalef(scale, scale, scale)

    if render_pos:
        glColor(color_pos)
        glutSolidSphere(0.5 * point_size, 10, 10)

    if render_ori:
        o = np.zeros(3)
        if use_arrow:
            if render_ori[0]:
                render_arrow(o, o + R[:, 0], D=line_width * 0.02, color=color_ori[0])
            if render_ori[1]:
                render_arrow(o, o + R[:, 1], D=line_width * 0.02, color=color_ori[1])
            if render_ori[2]:
                render_arrow(o, o + R[:, 2], D=line_width * 0.02, color=color_ori[2])
        else:
            if render_ori[0]:
                render_line(o, o + R[:, 0], color=color_ori[0])
            if render_ori[1]:
                render_line(o, o + R[:, 1], color=color_ori[1])
            if render_ori[2]:
                render_line(o, o + R[:, 2], color=color_ori[2])

    glPopMatrix()


def render_ground(
    size=[20.0, 20.0],
    dsize=[1.0, 1.0],
    color=[0.0, 0.0, 0.0, 1.0],
    line_width=1.0,
    axis="y",
    origin=True,
    use_arrow=False,
    lighting=False,
):
    lx = size[0]
    lz = size[1]
    dx = dsize[0]
    dz = dsize[1]
    nx = int(lx / dx) + 1
    nz = int(lz / dz) + 1

    glColor(color)
    glLineWidth(line_width)
    if lighting:
        glEnable(GL_LIGHTING)

    if axis is "x":
        for i in np.linspace(-0.5 * lx, 0.5 * lx, nx):
            glBegin(GL_LINES)
            glVertex3d(0, i, -0.5 * lz)
            glVertex3d(0, i, 0.5 * lz)
            glEnd()
        for i in np.linspace(-0.5 * lz, 0.5 * lz, nz):
            glBegin(GL_LINES)
            glVertex3d(0, -0.5 * lx, i)
            glVertex3d(0, 0.5 * lx, i)
            glEnd()
    elif axis is "y":
        for i in np.linspace(-0.5 * lx, 0.5 * lx, nx):
            glBegin(GL_LINES)
            glVertex3d(i, 0, -0.5 * lz)
            glVertex3d(i, 0, 0.5 * lz)
            glEnd()
        for i in np.linspace(-0.5 * lz, 0.5 * lz, nz):
            glBegin(GL_LINES)
            glVertex3d(-0.5 * lx, 0, i)
            glVertex3d(0.5 * lx, 0, i)
            glEnd()
    elif axis is "z":
        for i in np.linspace(-0.5 * lx, 0.5 * lx, nx):
            glBegin(GL_LINES)
            glVertex3d(i, -0.5 * lz, 0)
            glVertex3d(i, 0.5 * lz, 0)
            glEnd()
        for i in np.linspace(-0.5 * lz, 0.5 * lz, nz):
            glBegin(GL_LINES)
            glVertex3d(-0.5 * lx, i, 0)
            glVertex3d(0.5 * lx, i, 0)
            glEnd()

    if origin:
        render_transform(constants.eye_T(), use_arrow=use_arrow)


def render_line_2D(p1, p2, line_width=1.0, color=[0, 0, 0, 1]):
    glColor(color)
    glLineWidth(line_width)
    glBegin(GL_LINES)
    glVertex2f(p1[0], p1[1])
    glVertex2f(p2[0], p2[1])
    glEnd()


def render_point_2D(p, size=5, color=[0, 0, 0, 1]):
    glColor(color)
    glPointSize(size)
    glBegin(GL_POINTS)
    glVertex2f(p[0], p[1])
    glEnd()


def render_quad_2D(p1, p2, p3, p4, color=[0, 0, 0, 1]):
    glColor(color)
    glBegin(GL_QUADS)
    glVertex2d(p1[0], p1[1])
    glVertex2d(p2[0], p2[1])
    glVertex2d(p3[0], p3[1])
    glVertex2d(p4[0], p4[1])
    glEnd()


def render_text(text, pos, font=GLUT_BITMAP_TIMES_ROMAN_10, color=[0, 0, 0, 1]):
    glPushAttrib(GL_DEPTH_TEST | GL_LIGHTING)

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glColor(color)

    glRasterPos2f(pos[0], pos[1])
    for ch in text:
        glutBitmapCharacter(font, ctypes.c_int(ord(ch)))

    glPopAttrib()


def render_pyramid(
    T=None, scale=1.0, base_x=1.0, base_z=1.0, height=1.0, color=[0, 0, 0, 1]
):
    glPushMatrix()

    if T is not None:
        glTransform(T)

    glScalef(scale, scale, scale)
    glColor(color)

    p1 = np.array([0.5 * base_x, 0, 0.5 * base_z])
    p2 = np.array([-0.5 * base_x, 0, 0.5 * base_z])
    p3 = np.array([-0.5 * base_x, 0, -0.5 * base_z])
    p4 = np.array([0.5 * base_x, 0, -0.5 * base_z])
    p5 = np.array([0, height, 0])

    render_quad(p1, p2, p3, p4, color=color)
    render_tri(p1, p2, p5, color=color)
    render_tri(p2, p3, p5, color=color)
    render_tri(p3, p4, p5, color=color)
    render_tri(p4, p1, p5, color=color)

    glPopMatrix()


def render_graph_base_2D(origin=(0, 0), axis_len=150, pad_len=30):
    p1 = (origin[0] - pad_len, origin[1] + pad_len)
    p2 = (origin[0] + pad_len + axis_len, origin[1] + pad_len)
    p3 = (origin[0] + pad_len + axis_len, origin[1] - pad_len - axis_len)
    p4 = (origin[0] - pad_len, origin[1] - pad_len - axis_len)
    render_quad_2D(p1, p2, p3, p4, color=[0.9, 0.9, 0.9, 0.8])

    # X-axis
    p1 = origin
    p2 = (origin[0] + axis_len, origin[1])
    render_line_2D(p1, p2, line_width=3.0)

    # Y-axis
    p1 = origin
    p2 = (origin[0], origin[1] - axis_len)
    render_line_2D(p1, p2, line_width=3.0)


def render_graph_data_point_2D(
    x_data,
    y_data,
    x_range=(0, 1),
    y_range=(0, 1),
    color=[0, 0, 0, 1],
    point_size=1.0,
    origin=(0, 0),
    axis_len=150,
    pad_len=30,
):
    assert len(x_data) == len(y_data)
    num_data = len(x_data)
    if num_data == 0:
        return
    x_range_len = x_range[1] - x_range[0]
    y_range_len = y_range[1] - y_range[0]
    for i in range(num_data):
        x_cur, y_cur = x_data[i], y_data[i]
        x = origin[0] + axis_len * (x_cur - x_range[0]) / x_range_len
        y = origin[1] - axis_len * (y_cur - y_range[0]) / y_range_len
        render_point_2D(p=(x, y), size=point_size, color=color)


def render_graph_data_line_2D(
    x_data,
    y_data,
    x_range=(0, 1),
    y_range=(0, 1),
    color=[0, 0, 0, 1],
    line_width=1.0,
    origin=(0, 0),
    axis_len=150,
    pad_len=30,
    multiple_data=False,
):
    if multiple_data:
        x = x_data
        y = y_data
        x_r = x_range
        y_r = y_range
        c = color
        l_w = line_width
    else:
        x = [x_data]
        y = [y_data]
        x_r = [x_range]
        y_r = [y_range]
        c = [color]
        l_w = [line_width]

    assert len(x) == len(y)
    for i in range(len(x)):
        num_data = len(x[i])
        if num_data <= 1:
            return
        x_prev, y_prev = x[i][0], y[i][0]
        x_range_len = x_r[i][1] - x_r[i][0]
        y_range_len = y_r[i][1] - y_r[i][0]
        for j in range(1, num_data):
            x_cur, y_cur = x[i][j], y[i][j]
            x0 = origin[0] + axis_len * (x_prev - x_r[i][0]) / x_range_len
            y0 = origin[1] - axis_len * (y_prev - y_r[i][0]) / y_range_len
            x1 = origin[0] + axis_len * (x_cur - x_r[i][0]) / x_range_len
            y1 = origin[1] - axis_len * (y_cur - y_r[i][0]) / y_range_len
            render_line_2D(p1=(x0, y0), p2=(x1, y1), line_width=l_w[i], color=c[i])
            x_prev, y_prev = x_cur, y_cur


def render_progress_bar_2D_horizontal(
    progress,
    origin=(0, 0),
    width=100,
    height=10,
    line_width=2.0,
    color_base=[0.5, 0.5, 0.5, 1],
    color_input=[0, 0, 0, 1],
):
    progress = np.clip(progress, 0.0, 1.0)
    p1 = (origin[0], origin[1])
    p2 = (origin[0] + progress * width, origin[1])
    p3 = (origin[0] + progress * width, origin[1] + height)
    p4 = (origin[0], origin[1] + height)
    p22 = (origin[0] + width, origin[1])
    p33 = (origin[0] + width, origin[1] + height)
    render_quad_2D(p1, p2, p3, p4, color=color_input)
    render_line_2D(p1=p1, p2=p22, line_width=line_width, color=color_base)
    render_line_2D(p1=p22, p2=p33, line_width=line_width, color=color_base)
    render_line_2D(p1=p33, p2=p4, line_width=line_width, color=color_base)
    render_line_2D(p1=p4, p2=p1, line_width=line_width, color=color_base)


def render_progress_bar_2D_vertical(
    progress,
    origin=(0, 0),
    width=10,
    height=100,
    line_width=2.0,
    color_base=[0.5, 0.5, 0.5, 1],
    color_input=[0, 0, 0, 1],
):
    progress = np.clip(progress, 0.0, 1.0)
    p1 = (origin[0], origin[1])
    p2 = (origin[0] + width, origin[1])
    p3 = (origin[0] + width, origin[1] + progress * height)
    p4 = (origin[0], origin[1] + progress * height)
    p33 = (origin[0] + width, origin[1] + height)
    p44 = (origin[0], origin[1] + height)
    render_quad_2D(p1, p2, p3, p4, color=color_input)
    render_line_2D(p1=p1, p2=p2, line_width=line_width, color=color_base)
    render_line_2D(p1=p2, p2=p33, line_width=line_width, color=color_base)
    render_line_2D(p1=p33, p2=p44, line_width=line_width, color=color_base)
    render_line_2D(p1=p44, p2=p1, line_width=line_width, color=color_base)


def render_progress_circle_2D(
    progress,
    origin=(0, 0),
    radius=100,
    line_width=2.0,
    color_base=[0.5, 0.5, 0.5, 1],
    color_input=[1, 0, 0, 1],
):
    p = np.array([origin[0], origin[1], 0])
    T = conversions.p2T(p)
    render_circle(T=T, r=radius, line_width=line_width, color=color_base)
    theta = 2 * math.pi * progress
    p += radius * np.array([math.cos(theta), math.sin(theta), 0])
    render_point_2D((p[0], p[1]), size=0.1 * radius, color=color_input)


def render_direction_input_2D(
    val,
    val_max,
    origin=(0, 0),
    radius=100,
    line_width=2.0,
    color_base=[0.5, 0.5, 0.5, 1],
    color_input=[1, 0, 0, 1],
):
    v = np.array([val[0] / val_max[0], val[1] / val_max[1]])
    v *= radius
    p = np.array([origin[0], origin[1], 0])
    T = conversions.p2T(p)
    render_circle(T=T, r=radius, line_width=line_width, color=color_base)
    render_line_2D(p1=origin, p2=origin + v, line_width=line_width, color=color_input)
    render_point_2D(origin, size=0.1 * radius, color=[0.5, 0.5, 0.5, 1])


def render_matrix(
    m,
    min_val=0.0,
    max_val=1.0,
    origin=(0, 0),
    width=100,
    height=100,
    min_color=[1, 1, 1],
    max_color=[1, 0, 0],
    line_width=1.0,
):
    assert min_val != max_val
    glPushMatrix()
    glTranslatef(origin[0], origin[1], 0)

    dim_x, dim_y = m.shape
    dx = width / dim_x
    dy = height / dim_y
    _min_color = np.array(min_color)
    _max_color = np.array(max_color)
    d_val = max_val - min_val
    d_color = _max_color - _min_color
    for i in range(dim_x):
        for j in range(dim_y):
            p1 = (dx * i, dy * j)
            p2 = (dx * i + dx, dy * j)
            p3 = (dx * i + dx, dy * j + dy)
            p4 = (dx * i, dy * j + dy)
            val = (m[i][j] - min_val) / d_val
            color = val * d_color + _min_color
            render_quad_2D(p1, p2, p3, p4, color=color)
    for i in range(dim_x):
        p1 = (dx * i, 0)
        p2 = (dx * i, height)
        render_line_2D(p1, p2, line_width=line_width)
    render_line_2D((width, 0), (width, height), line_width=line_width)
    for j in range(dim_y):
        p1 = (0, dy * j)
        p2 = (width, dy * j)
        render_line_2D(p1, p2, line_width=line_width)
    render_line_2D((0, height), (width, height), line_width=line_width)

    glPopMatrix()
