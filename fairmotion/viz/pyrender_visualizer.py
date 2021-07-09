import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import trimesh

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer
import tqdm

from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.utils import utils

import pyrender
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fairmotion.ops.conversions import E2R, R2Q

def _get_cam_rotation(p_cam, p_obj, vup):
    z = p_cam - p_obj
    z /= np.linalg.norm(z)
    x = np.cross(vup, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z]).transpose()


class MocapViewerOffline(animation.FuncAnimation):
    """
    MocapViewerOffline is an extension of the glut_viewer.Viewer class that implements
    requisite callback functions -- render_callback, keyboard_callback,
    idle_callback and overlay_callback.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1
    ```

    To visualize more than 1 motion sequence side by side, append more files
    to the `--bvh-files` argument. Set `--x-offset` to an appropriate float
    value to add space separation between characters in the row.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1 $BVH_FILE2 $BVH_FILE3 \
        --x-offset 2
    ```

    To visualize asfamc motion sequence:

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --asf-files tests/data/11.asf \
        --amc-files tests/data/11_01.amc
    ```

    """

    def __init__(
            self,
            motion,
            cam_pos,
            v_up_str,
            play_speed=1.0,
            scale=1.0,
            thickness=1.0,
            hide_origin=False
    ):
        animation.FuncAnimation.__init__(self, fig=plt.figure(figsize=(5, 5)), func=self.animate,
                                         frames=len(motion.poses), interval=50, blit=False)
        self.motion = motion
        self.play_speed = play_speed
        self.hide_origin = hide_origin
        self.file_idx = 0
        self.cur_time = 0.0
        self.scale = scale
        self.thickness = thickness
        self.cam_p = np.array(cam_pos)
        self.up_axis = utils.str_to_axis(v_up_str)
        self.ground_node = None
        self.init_pyrender()
        self.pt_pool = []
        self.cap_pool = []
        plt.axis('off')
        self.ims = None
        self.progress = tqdm.tqdm(total=len(motion.poses))  # Initialise

    def render_point(self, at_index, p, scale=1.0, radius=1.0, color=[1.0, 0.0, 0.0]):
        if at_index >= len(self.pt_pool):
            # create enough to allow at_index to work
            for i in range(len(self.pt_pool), at_index + 1):  # must include at_index too
                # get primitive first
                sphere_trimesh = trimesh.creation.icosphere(radius=radius, subdivisions=1)
                sphere_face_colors = np.zeros(sphere_trimesh.faces.shape)
                sphere_face_colors[:] = np.array(color)
                sphere_trimesh.visual.face_colors = sphere_face_colors
                sphere_mesh = Mesh.from_trimesh(sphere_trimesh, smooth=False)

                sphere_node = Node(mesh=sphere_mesh,
                                   name="sphere_" + str(i))  # , translation=np.array([-0.1, -0.10, 0.05]))
                self.scene.add_node(sphere_node)
                self.pt_pool.append(sphere_node)
        # okay now we know the node exists, just change it's position
        self.pt_pool[at_index].scale = [scale] * 3
        self.pt_pool[at_index].translation = p

    def render_capsule(self, at_index, p, Q, length, scale=1.0, color=[1.0, 0.0, 0.0]):
        if at_index >= len(self.cap_pool):
            # create enough to allow at_index to work
            for i in range(len(self.cap_pool), at_index + 1):  # must include at_index too
                # get primitive first
                sphere_trimesh = trimesh.creation.capsule(height=1.0, radius=1.0, count=[8, 8])
                sphere_face_colors = np.zeros(sphere_trimesh.faces.shape)
                sphere_face_colors[:] = np.array(color)
                sphere_trimesh.visual.face_colors = sphere_face_colors
                sphere_mesh = Mesh.from_trimesh(sphere_trimesh, smooth=False)

                sphere_node = Node(mesh=sphere_mesh,
                                   name="capsule_" + str(i))  # , translation=np.array([-0.1, -0.10, 0.05]))
                self.scene.add_node(sphere_node)
                self.cap_pool.append(sphere_node)
        # okay now we know the node exists, just change it's position
        self.cap_pool[at_index].scale = [0.1, 0.1, length]
        self.cap_pool[at_index].translation = p
        self.cap_pool[at_index].rotation = Q

    #         print("Q: " + str(Q))

    def _render_pose(self, pose, color):
        skel = pose.skel
        capnum = 0
        for ipt, j in enumerate(skel.joints):
            T = pose.get_transform(j, local=False)
            pos = 0.4 * conversions.T2p(T)

            self.render_point(ipt, pos, radius=0.03 * self.scale, color=color)

            if j.parent_joint is not None:
                # returns X that X dot vec1 = vec2
                pos_parent = 0.5 * conversions.T2p(
                    pose.get_transform(j.parent_joint, local=False)
                )
                p = 0.4 * (pos_parent + pos)
                l = np.linalg.norm(pos_parent - pos)
                #                 l=2.0
                r = 0.1 * self.thickness
                R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
                self.render_capsule(capnum,
                                    p,
                                    #                     conversions.p2T(p),
                                    R2Q(R),
                                    l / 2.0,
                                    #                     r * self.scale,
                                    0.1,
                                    color=color
                                    )
                capnum += 1

    def _render_characters(self, colors, frame):

        #         t = self.cur_time % motion.length()
        skel = self.motion.skel
        #         pose = motion.get_pose_by_frame(motion.time_to_frame(t))
        pose = self.motion.get_pose_by_frame(frame)
        #             pose = motion.get_pose_by_frame(0)
        color = colors[0 % len(colors)]

        self._render_pose(pose, color)

    def render_ground(self, size=[20.0, 20.0],
                      dsize=[1.0, 1.0],
                      color=[0.0, 0.0, 0.0, 1.0],
                      line_width=1.0,
                      axis="y",
                      origin=True,
                      use_arrow=False,
                      lighting=False):
        if self.ground_node is None:
            lx = size[0]
            lz = size[1]
            dx = dsize[0]
            dz = dsize[1]
            nx = int(lx / dx) + 1
            nz = int(lz / dz) + 1

            grid_pts = np.zeros((2 * nx + 2 * nz, 3))
            colors = np.zeros((2 * nx + 2 * nz, 4))
            colors[:] = np.array(color)

            if axis is "x":
                linei = 0
                for i in np.linspace(-0.5 * lx, 0.5 * lx, nx):
                    grid_pts[2 * linei] = [0, i, -0.5 * lz]
                    grid_pts[2 * linei + 1] = [0, i, 0.5 * lz]
                    linei += 1
                for i in np.linspace(-0.5 * lz, 0.5 * lz, nz):
                    grid_pts[2 * linei] = [0, -0.5 * lx, i]
                    grid_pts[2 * linei + 1] = [0, 0.5 * lx, i]
                    linei += 1
            elif axis is "y":
                linei = 0
                for i in np.linspace(-0.5 * lx, 0.5 * lx, nx):
                    grid_pts[2 * linei] = [i, 0, -0.5 * lz]
                    grid_pts[2 * linei + 1] = [i, 0, 0.5 * lz]
                    linei += 1
                for i in np.linspace(-0.5 * lz, 0.5 * lz, nz):
                    grid_pts[2 * linei] = [-0.5 * lx, 0, i]
                    grid_pts[2 * linei + 1] = [0.5 * lx, 0, i]
                    linei += 1
            elif axis is "z":
                linei = 0
                for i in np.linspace(-0.5 * lx, 0.5 * lx, nx):
                    grid_pts[2 * linei] = [i, -0.5 * lz, 0.]
                    grid_pts[2 * linei + 1] = [i, 0.5 * lz, 0.]
                    linei += 1
                for j, i in enumerate(np.linspace(-0.5 * lz, 0.5 * lz, nz)):
                    grid_pts[2 * linei] = [-0.5 * lx, i, 0.]
                    grid_pts[2 * linei + 1] = [0.5 * lx, i, 0.]
                    linei += 1
            grid = pyrender.Primitive(grid_pts, color_0=colors, mode=1)  # 1->LINES
            grid = pyrender.Mesh([grid])
            self.ground_node = Node(mesh=grid, name="ground_plane")
            self.scene.add_node(self.ground_node)

    def render_callback(self, frame_num):
        self.render_ground(
            size=[100, 100],
            color=[0.8, 0.8, 0.8, 1.0],
            axis="y",  # utils.axis_to_str(self.motions[0].skel.v_up_env),
            origin=not self.hide_origin,
            use_arrow=True,
        )

        colors = [
            np.array([123, 174, 85]) / 255.0,  # green
            np.array([255, 255, 0]) / 255.0,  # yellow
            np.array([85, 160, 173]) / 255.0,  # blue
        ]
        self._render_characters(colors, frame_num)
        color, depth = self.r.render(self.scene)
        return color

    def animate(self, frame_num):
        # print("Rendering frame %d..."%(frame_num))
        self.progress.update(frame_num)
        color = self.render_callback(frame_num)
        if self.ims is None:
            self.ims = plt.imshow(color, animated=True)
        else:
            self.ims.set_array(color)

        return self.ims,

    def idle_callback(self):
        time_elapsed = self.time_checker.get_time(restart=False)
        self.cur_time += self.play_speed * time_elapsed
        self.time_checker.begin()

    def init_pyrender(self):
        # ==============================================================================
        # Light creation
        # ==============================================================================
        self.spot_l = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                         innerConeAngle=np.pi / 16.0,
                                         outerConeAngle=np.pi / 6.0)

        # ==============================================================================
        # Camera creation
        # ==============================================================================

        cam = PerspectiveCamera(yfov=np.pi / 2.0)
        R = _get_cam_rotation(self.cam_p,
                              np.zeros((3)),
                              self.up_axis)

        self.cam_pose = conversions.Rp2T(R, self.cam_p)

        # ==============================================================================
        # Scene creation
        # ==============================================================================

        self.scene = Scene(ambient_light=np.array([0.1, 0.1, 0.1, 1.0]))
        self.spot_l_node = self.scene.add(self.spot_l, pose=self.cam_pose, name="spot_light")

        self.cam_node = self.scene.add(cam, pose=self.cam_pose, name="camera")
        self.r = OffscreenRenderer(viewport_width=320, viewport_height=240)


if __name__ == "__main__":
    from fairmotion.data import bvh
    filename = sys.argv[1]

    v_up_env = utils.str_to_axis('y')

    motion = bvh.load(
        file=filename,
        v_up_skel=v_up_env,
        v_face_skel=utils.str_to_axis('z'),
        v_up_env=v_up_env,
        scale=1.0)

    R = E2R([-np.pi / 2.0, 0.0, 0.0])
    motion = motion_ops.rotate(motion, R)
    motion = motion_ops.translate(motion, [0, -20.0, 0])

    viewer = MocapViewerOffline(
        motion=motion,
        cam_pos=[0.0, 1.1, -2.3],
        v_up_str='y',
        scale=20.33
    )

    viewer.to_html5_video()