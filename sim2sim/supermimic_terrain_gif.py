"""Render a measured S1 → S0 terrain run, including foot contacts and both world trajectories.

Run from the repository root with
    python sim2sim/supermimic_sim2sim.py --terrain-demo climb --gif climb.gif
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scan80000_policy import quat_to_mat
from sim2sim.scan80000_terrain import TerrainContacts, peak_box_height
from sim2sim.scan80000_view import ReferenceGhost, append_scan_markers, draw_planner_goal


@dataclass
class Frame:
    time: float
    qpos: np.ndarray
    reference_pos: np.ndarray
    reference_quat: np.ndarray
    reference_joints: np.ndarray


class TerrainReplayGIF:
    frame_stride = 4  # Every fourth 50 Hz S0 frame.

    def __init__(self, rt, planner, seed, show_heightmap, contacts=None):
        self.rt = rt
        self.planner = planner
        self.seed = seed
        self.show_heightmap = show_heightmap
        self.frames = []
        self.contacts = contacts if contacts is not None else TerrainContacts(
            rt, planner.target_xy, goal_height=planner.target_ground[2], goal_error=planner.goal_error)
        self.tops, self.corners = self.contacts.tops, self.contacts.corners
        self.height = peak_box_height(rt.motion.boxes)

    def measure_contacts(self, model, data):
        self.contacts.measure_contacts(model, data)

    @property
    def events(self):
        if self.contacts.goal_on_top:
            return [*self.contacts.top_times, self.contacts.first_hold_s]
        return self.contacts.events

    def capture(self):
        rt = self.rt
        if self.frames and self.frames[-1].time == rt.data.time:
            return
        i = rt.reference_frame
        self.frames.append(Frame(rt.data.time, rt.data.qpos.copy(), rt.ref.pos[i].copy(),
                                 rt.ref.quat[i].copy(), rt.ref.joints[i].copy()))

    def save(self, path):
        model = self.rt.model
        w, h = 558, 308
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)
        model.vis.headlight.ambient[:] = .48
        data = mujoco.MjData(model)
        ghost = ReferenceGhost(model, self.rt.q)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation = 108, -18
        camera.distance = 4.0 if self.height > .5 else 3.5
        overlay = ReplayOverlay(self)
        frames = []
        with mujoco.Renderer(model, height=h, width=w) as renderer:
            for i, frame in enumerate(self.frames):
                data.qpos[:] = frame.qpos
                mujoco.mj_forward(model, data)
                x, y, z = frame.qpos[:3]
                camera.lookat[:] = (x + .30, y, max(.55, z - .23))
                renderer.update_scene(data, camera=camera)
                scene = renderer.scene
                if self.show_heightmap:
                    pose = frame.qpos[:7]
                    values = self.rt.scanner.scan(pose[:3], quat_to_mat(pose[3:7]))
                    append_scan_markers(scene, self.rt.scanner, pose, values, radius=.012)
                ghost.append_pose(scene, frame.reference_pos, frame.reference_quat, frame.reference_joints)
                draw_planner_goal(scene, self.planner.target_ground, self.planner.target_yaw)
                frames.append(overlay.draw(renderer.render(), i))
        durations = [max(10, round(1000 * (b.time - a.time))) for a, b in zip(self.frames, self.frames[1:])]
        durations.append(round(1000 * self.frame_stride * self.rt.dt))
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=durations)


class ReplayOverlay:
    def __init__(self, replay):
        self.replay = replay
        # Prefer Chinese labels when a CJK font is installed. English also works with Pillow's bundled font.
        fonts = [Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
                 Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("C:/Windows/Fonts/msyh.ttc")]
        face = next((font for font in fonts if font.is_file()), None)
        self.chinese = face is not None
        if not face:
            try:
                face = ImageFont.truetype("DejaVuSans.ttf", 14).path
            except OSError:
                pass
        self.title, self.body, self.small = [ImageFont.truetype(str(face), size) if face else ImageFont.load_default()
                                            for size in (22, 14, 12)]
        actual = np.stack([f.qpos[:2] for f in replay.frames])
        reference = np.stack([f.reference_pos[:2] for f in replay.frames])
        target = replay.planner.target_xy
        points = np.vstack([actual, reference, target, *replay.corners])
        lo, hi = points.min(0), points.max(0)
        center = (lo + hi) / 2
        scale = min(166 / max(hi[0] - lo[0], 1), 113 / max(hi[1] - lo[1], 1))

        def project(xy):
            return tuple(np.rint([655, 155] + (xy - center) * [scale, -scale]).astype(int))

        self.trails = [([project(xy) for xy in reference], "#f998c6"), ([project(xy) for xy in actual], "#86c5ff")]
        self.polygons = [[project(xy) for xy in corners] for corners in replay.corners]
        self.goal = project(target)

    def label(self, chinese, english):
        return chinese if self.chinese else english

    def draw(self, rgb, i):
        r = self.replay
        f = r.frames[i]
        screen = Image.new("RGB", (756, 396), "#192530")
        screen.paste(Image.fromarray(rgb), (0, 51))
        draw = ImageDraw.Draw(screen)
        white, muted, green = "#f1f4f7", "#bac4d2", "#70e797"
        height = 100 * r.height
        draw.text((12, 8), self.label(f"{height:.1f} 厘米地形", f"{height:.1f} cm terrain"), font=self.title, fill=white)
        draw.text((230, 15), self.label("S1 在线生成  →  S0 物理跟踪", "S1 plan  /  S0 physics"), font=self.body, fill=muted)
        draw.text((600, 15), self.label("种子", "Seed") + f" {r.seed}   ·   {r.frames[-1].time:g} s", font=self.body, fill=muted)
        draw.text((570, 61), self.label("俯视轨迹", "World XY"), font=self.body, fill=white)
        draw.rounded_rectangle((567, 87, 744, 221), 5, fill="#283746")
        for polygon in self.polygons:
            draw.polygon(polygon, fill="#bb8d58")
        for path, color in self.trails:
            if i:
                draw.line(path[:i + 1], fill=color, width=2)
            x, y = path[i]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        x, y = self.goal
        draw.line((x - 4, y, x + 4, y), fill=green, width=2)
        draw.line((x, y - 4, x, y + 4), fill=green, width=2)
        draw.text((570, 232), self.label("实际足底接触", "Foot contacts"), font=self.body, fill=white)
        labels = [self.label("左脚踩上台面", "Left on top"), self.label("右脚踩上台面", "Right on top"),
                  self.label("两脚均落到另一侧", "Both landed beyond")]
        if r.contacts.goal_on_top:
            labels[2] = self.label("目标处双脚站稳 1 秒", "Held goal for 1 s")
        for j, (label, when) in enumerate(zip(labels, r.events)):
            hit = when is not None and f.time + 1e-8 >= when
            y = 261 + 25 * j
            draw.text((570, y), label, font=self.small, fill=green if hit else muted)
            draw.text((693, y), f"{when:.2f} s" if hit else "-", font=self.small, fill=white if hit else muted)
        distance = np.linalg.norm(f.qpos[:2] - r.planner.target_xy)
        draw.text((12, 367), f"{f.time:05.2f} s     " + self.label("距目标", "Goal") + f" {distance:.2f} m",
                  font=self.body, fill=white)
        legends = [(287, self.label("实际机器人", "Robot"), "#86c5ff"),
                   (400, self.label("S1 参考", "S1 reference"), "#f998c6"), (490, self.label("目标", "Goal"), green)]
        if r.show_heightmap:
            legends.append((565, self.label("地面彩点为 S0 高程采样", "Dots: S0 terrain samples"), muted))
        for x, label, color in legends:
            draw.text((x, 368), label, font=self.small, fill=color)
        return screen.convert("P", palette=Image.ADAPTIVE, colors=192)
