#!/usr/bin/env python3
"""Generate bc_track.world — an oval road with lane markings, plus the car.

The world is GENERATED rather than hand-written because the track geometry has
to be known in two other places: the expert driver follows this centre line,
and the evaluator measures how far the car strays from it. Writing the geometry
once, in Python, and emitting both the SDF and a centre-line CSV keeps the
simulation and the controller from disagreeing about where the road is.

    python -m behavioral_cloning.make_track   # writes behavioral_cloning/track.world
"""
import argparse
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def centreline(n=240, a=9.0, b=6.0, shape="eight"):
    """The track's centre line, as n points.

    "eight" is the default and it is not decoration. On an oval every corner
    turns the same way, so a network that ignores the camera and emits one
    constant steering angle drives it perfectly -- the demo would "work" while
    proving nothing. A figure-eight (lemniscate of Gerono) forces both
    directions, so the only way to stay on it is to actually look at the road.
    Measured on 6 recorded laps: the oval gives steering mean -0.094, std
    0.047 -- one constant would fit it. The eight gives mean +0.004, std
    0.161, with both signs well represented.
    """
    if shape == "ellipse":
        return [(a * math.cos(2 * math.pi * i / n),
                 b * math.sin(2 * math.pi * i / n)) for i in range(n)]
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        pts.append((a * math.cos(t), b * math.sin(t) * math.cos(t)))
    return pts


def _box(name, x, y, yaw, sx, sy, sz, rgba, z=None):
    z = sz / 2 if z is None else z
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>
      <link name="l">
        <visual name="v">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>"""


def road(n=240, a=9.0, b=6.0, width=3.0, shape="eight"):
    """Tarmac laid as n short slabs around the ellipse, plus dashed lane lines.

    Slabs rather than a mesh: no external assets, and the camera sees exactly
    the high-contrast road/grass boundary the network needs to learn from.
    """
    pts = centreline(n, a, b, shape)
    out = []
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[(i + 1) % n]
        yaw = math.atan2(ny - y, nx - x)
        seg = math.hypot(nx - x, ny - y) * 1.6      # overlap so there are no gaps
        out.append(_box(f"road_{i}", (x + nx) / 2, (y + ny) / 2, yaw,
                        seg, width, 0.02, "0.18 0.18 0.20 1", z=0.01))
        if i % 8 < 4:                                # dashed centre line
            out.append(_box(f"dash_{i}", (x + nx) / 2, (y + ny) / 2, yaw,
                            seg, 0.12, 0.02, "0.9 0.9 0.9 1", z=0.021))
        # solid edge lines: the strongest visual cue for where the lane is
        for side, tag in ((+1, "outer"), (-1, "inner")):
            ux, uy = -(ny - y), (nx - x)
            nrm = math.hypot(ux, uy) + 1e-9
            ox, oy = ux / nrm * side * width / 2, uy / nrm * side * width / 2
            out.append(_box(f"edge_{tag}_{i}", (x + nx) / 2 + ox,
                            (y + ny) / 2 + oy, yaw, seg, 0.14, 0.02,
                            "0.95 0.85 0.15 1", z=0.022))
    return "".join(out)


CAR = """
    <model name="bc_car">
      <pose>{x:.4f} {y:.4f} 0.12 0 0 {yaw:.4f}</pose>
      <link name="chassis">
        <inertial>
          <mass>2.0</mass>
          <inertia><ixx>0.02</ixx><iyy>0.04</iyy><izz>0.05</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="c">
          <geometry><box><size>0.40 0.22 0.10</size></box></geometry>
        </collision>
        <visual name="v">
          <geometry><box><size>0.40 0.22 0.10</size></box></geometry>
          <material><ambient>0.1 0.3 0.8 1</ambient><diffuse>0.1 0.3 0.8 1</diffuse></material>
        </visual>
        <!-- The camera looks slightly down, like a dashcam, so the road fills
             the lower half of the frame the way it does in the training data. -->
        <sensor name="cam" type="camera">
          <pose>0.20 0 0.22 0 0.16 0</pose>
          <always_on>1</always_on>
          <update_rate>20</update_rate>
          <camera>
            <horizontal_fov>1.20</horizontal_fov>
            <image><width>320</width><height>160</height><format>R8G8B8</format></image>
            <clip><near>0.02</near><far>60</far></clip>
          </camera>
          <plugin name="cam_plugin" filename="libgazebo_ros_camera.so">
            <ros><namespace>/</namespace>
              <remapping>~/image_raw:=camera/image_raw</remapping>
              <remapping>~/camera_info:=camera/camera_info</remapping>
            </ros>
            <camera_name>camera</camera_name>
            <frame_name>chassis</frame_name>
          </plugin>
        </sensor>
      </link>

      <link name="left_wheel">
        <pose>-0.10 0.135 -0.02 -1.5707 0 0</pose>
        <inertial><mass>0.2</mass>
          <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.05 0.05 0.05 1</diffuse></material></visual>
      </link>
      <link name="right_wheel">
        <pose>-0.10 -0.135 -0.02 -1.5707 0 0</pose>
        <inertial><mass>0.2</mass>
          <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.05 0.05 0.05 1</diffuse></material></visual>
      </link>
      <link name="caster">
        <pose>0.15 0 -0.07 0 0 0</pose>
        <inertial><mass>0.1</mass>
          <inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><sphere><radius>0.05</radius></sphere></geometry></visual>
      </link>

      <!-- The axis is expressed in the CHILD link frame. Each wheel link is
           rotated -90 deg about X so its cylinder lies across the car, which
           maps the link's local +Z onto the model's +Y. So the rolling axis is
           "0 0 1" HERE, not "0 1 0": writing the model-frame answer gives the
           wheels a vertical axis, and the car turns on the spot at 0.08 m/s
           while the odometry insists it is doing what it was told. -->
      <joint name="lw" type="revolute"><parent>chassis</parent><child>left_wheel</child>
        <axis><xyz>0 0 1</xyz>
          <limit><lower>-1e16</lower><upper>1e16</upper></limit>
          <dynamics><damping>0.005</damping><friction>0.0</friction></dynamics>
        </axis></joint>
      <joint name="rw" type="revolute"><parent>chassis</parent><child>right_wheel</child>
        <axis><xyz>0 0 1</xyz>
          <limit><lower>-1e16</lower><upper>1e16</upper></limit>
          <dynamics><damping>0.005</damping><friction>0.0</friction></dynamics>
        </axis></joint>
      <joint name="cw" type="fixed"><parent>chassis</parent><child>caster</child></joint>

      <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
        <ros><namespace>/</namespace></ros>
        <left_joint>lw</left_joint>
        <right_joint>rw</right_joint>
        <wheel_separation>0.27</wheel_separation>
        <wheel_diameter>0.20</wheel_diameter>
        <max_wheel_torque>10</max_wheel_torque>
        <max_wheel_acceleration>6.0</max_wheel_acceleration>
        <publish_odom>true</publish_odom>
        <publish_odom_tf>true</publish_odom_tf>
        <publish_wheel_tf>false</publish_wheel_tf>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>chassis</robot_base_frame>
      </plugin>
    </model>"""


def build(a=9.0, b=6.0, width=3.0, n=240, shape="eight"):
    pts = centreline(n, a, b, shape)
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    yaw = math.atan2(y1 - y0, x1 - x0)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="bc_track">
    <include><uri>model://sun</uri></include>
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.6 0.75 0.95 1</background>
           <shadows>false</shadows></scene>
    <!-- Grass-green ground so the tarmac stands out; the network has nothing
         but colour and edges to go on. -->
    <model name="ground">
      <static>true</static>
      <link name="l">
        <collision name="c"><geometry><plane><normal>0 0 1</normal>
          <size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal>
          <size>200 200</size></plane></geometry>
          <material><ambient>0.25 0.45 0.20 1</ambient><diffuse>0.25 0.45 0.20 1</diffuse></material></visual>
      </link>
    </model>
    {road(n, a, b, width, shape)}
    {CAR.format(x=x0, y=y0, yaw=yaw)}
    <physics type="ode"><max_step_size>0.004</max_step_size>
      <real_time_update_rate>250</real_time_update_rate></physics>
  </world>
</sdf>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=9.0)
    ap.add_argument("--b", type=float, default=9.0)
    ap.add_argument("--width", type=float, default=3.0)
    ap.add_argument("--segments", type=int, default=300)
    ap.add_argument("--shape", choices=("eight", "ellipse"), default="eight")
    ap.add_argument("--out", default=str(HERE / "track.world"))
    g = ap.parse_args()
    Path(g.out).write_text(build(g.a, g.b, g.width, g.segments, g.shape))
    csv = Path(g.out).with_name("centreline.csv")
    csv.write_text("\n".join(f"{x:.5f},{y:.5f}"
                             for x, y in centreline(g.segments, g.a, g.b,
                                                    g.shape)) + "\n")
    print(f"wrote {g.out}\nwrote {csv}")
