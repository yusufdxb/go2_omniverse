# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-2-Clause

import struct
import time

import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header, Float32MultiArray

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Imu

from isaaclab.sensors import CameraCfg, Camera
import omni.replicator.core as rep
from pxr import Gf
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils


def _to_numpy(arr):
    """warp.array / torch.Tensor / numpy / list -> numpy.ndarray.

    IsaacLab 4.5 / Isaac Sim 6.0 expose articulation buffers as warp arrays,
    which do not support Python item indexing or iteration; numpy does.
    """
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            return arr.detach().cpu().numpy()
    return np.asarray(arr)


_LIDAR_PUBLISH_FAILED = False
_LAST_LIDAR_PUB = 0.0


def update_meshes_for_cloud2(position_array, origin, rot):
    # pub_robo_data_ros2 already converts the articulation buffers to numpy via
    # _to_numpy, so origin/rot arrive as ndarrays with no .cpu(). Go through
    # _to_numpy so this works whether the caller hands over torch, warp, or numpy.
    q = _to_numpy(rot)
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    # The sensor sits at (0, 0, 0.4) in the BASE frame (see add_rtx_lidar's
    # translation), so the mount offset has to be rotated with the body before
    # the world translation is added. Adding it after rotation, as this did,
    # leaves the cloud tilted off the sensor whenever the robot pitches or rolls.
    rotated_vectors = rotation.apply(np.asarray(position_array) + [0.0, 0.0, 0.4])
    rotated_vectors += _to_numpy(origin)
    return rotated_vectors


def _create_point_cloud2(header, points):
    """Build a sensor_msgs/PointCloud2 from an (N,3) float32 array without sensor_msgs_py."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3).astype(np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def add_rtx_lidar(num_envs, robot_type, debug=False, config_file_name="Unitree_L1", variant=None):
    """Attach an RTX LiDAR per env and return one scan-buffer annotator each.

    Deliberately built on the low-level `IsaacSensorCreateRtxLidar` command plus
    `rep.create.render_product` rather than the `isaacsim.sensors.rtx.LidarRtx`
    convenience class. LidarRtx derives from isaacsim.core.prims, whose
    prim.py calls `SimulationManager._get_backend_utils()`. IsaacLab 4.5.22
    replaces SimulationManager with its backend-specific PhysxManager, which
    does not implement that method, so constructing LidarRtx raises
    `AttributeError: type object 'PhysxManager' has no attribute
    '_get_backend_utils'`. The command path touches none of that.

    Ouster configs are variants on a shared USD in Isaac 5.0+, so a bare name
    like "OS1_REV6_32ch10hz512res" must be split into config="OS1" plus that
    variant. Passing the full name still works via a deprecation shim, but the
    split is what the runtime actually wants.
    """
    import omni.kit.commands

    annotator_lst = []
    for i in range(num_envs):
        if robot_type == "g1":
            path = f'/World/envs/env_{i}/Robot/head_link/lidar_sensor'
            translation = Gf.Vec3d(0.0, 0.0, 0.0)
        else:
            path = f'/World/envs/env_{i}/Robot/base/lidar_sensor'
            translation = Gf.Vec3d(0.0, 0.0, 0.4)

        config = config_file_name
        sensor_variant = variant
        if sensor_variant is None and config_file_name.startswith("OS") and len(config_file_name) > 3:
            config, sensor_variant = config_file_name[:3], config_file_name

        _, prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=path,
            parent=None,
            config=config,
            variant=sensor_variant,
            translation=translation,
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        )
        if prim is None:
            # commands.py logs "Config 'X' not found" and returns None, after
            # which the command silently falls back to replicator's default
            # Example_Rotary profile. Refuse that: a cloud from a sensor nobody
            # asked for is worse than no cloud, because it looks healthy.
            raise RuntimeError(
                f"RTX LiDAR config {config!r} (variant={sensor_variant!r}) did not resolve. "
                "Config names must match a SUPPORTED_LIDAR_CONFIGS basename exactly, "
                "case included: 'HESAI_XT32_SD10', not 'Hesai_XT32_SD10'."
            )

        # 128x128 matches what isaacsim.sensors.rtx.LidarRtx creates for its own
        # render product. The earlier (1, 1) also produces points, but there is
        # no reason to diverge from the vendor's own path.
        render_product_path = rep.create.render_product(
            prim.GetPath().pathString, resolution=(128, 128)
        ).path

        if debug:
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloudBuffer")
            writer.attach([render_product_path])

        # Isaac Sim 6.0 dropped the "RtxSensorCpu" prefix from this annotator's
        # registered name; the pre-6.0 name is no longer in the registry.
        #
        # Use the NoAccumulator registration, not "IsaacCreateRTXLidarScanBuffer".
        # Both wrap the same OGN node type
        # (isaacsim.sensors.rtx.IsaacCreateRTXLidarScanBuffer), but Isaac 6.0
        # registers them differently in
        # exts/isaacsim.sensors.rtx/.../impl/extension.py::_register_nodes:
        # "IsaacExtractRTXSensorPointCloudNoAccumulator" is registered with
        # init_params={"enablePerFrameOutput": True}, while the plain
        # "IsaacCreateRTXLidarScanBuffer" is registered with no init_params at
        # all. Calling annotator.initialize(enablePerFrameOutput=True) on the
        # plain one is silently ignored: measured 195k to 204k points per
        # get_data() either way, which is full-revolution accumulation.
        #
        # That matters beyond bandwidth. A ~204k-point cloud is ~2.4 MB per
        # PointCloud2; publishing it at 20 Hz drove /robot0/odom down from
        # 108-133 Hz to 1.8 Hz and /robot0/point_cloud2 to zero messages
        # received over DDS. Accumulation mode does not just waste bytes, it
        # stalls the sim's main loop and the cloud never arrives at all.
        annotator = rep.AnnotatorRegistry.get_annotator(
            "IsaacExtractRTXSensorPointCloudNoAccumulator"
        )
        annotator.attach(render_product_path)
        annotator_lst.append(annotator)
    return annotator_lst


def add_camera(num_envs, robot_type):
    for i in range(num_envs):
        cameraCfg = CameraCfg(
            prim_path=f"/World/envs/env_{i}/Robot/base/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.32487, -0.00095, 0.05362), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

        if robot_type == "g1":
            cameraCfg.prim_path = f"/World/envs/env_{i}/Robot/head_link/front_cam"
            cameraCfg.offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")

        Camera(cameraCfg)


# Upper bound on points per published cloud. 8192 points is ~98 KB of
# PointCloud2 payload, which CycloneDDS delivers without tuning.
#
# Why a cap is needed at all: even with the NoAccumulator (per-frame) annotator,
# one get_data() here returns 195k to 204k points, because a rendered frame
# spans ~0.3 s of wall clock on this machine and the RTX sensor traces
# continuously across it. That is ~2.4 MB per PointCloud2, and at that size the
# messages simply never arrive: `ros2 topic hz /robot0/point_cloud2` received
# zero messages in 20 s while the publisher was running.
#
# This decimates real returns, it does not synthesize them. Every published
# point is a ray the RTX sensor actually traced this frame. If the sensor stops
# producing, there is nothing to stride over and nothing is published, so
# HELIX's staleness gate still sees a genuinely dead topic. The gate is
# rate-based, so density does not affect its verdict.
_MAX_CLOUD_POINTS = 8192


def _scan_points(annotator_lst, j):
    """Return this frame's (N,3) sensor-frame LiDAR points for robot j, or None.

    None means the annotator produced nothing on this frame, and the caller
    must publish nothing. There is deliberately no substitute source: filling
    /utlidar/cloud from the locomotion height scanner (or from anything else)
    would keep HELIX's staleness gate green while the LiDAR was dead, which is
    the exact failure the gate exists to catch.
    """
    if not annotator_lst:
        return None
    points = annotator_lst[j].get_data()['data']
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        return None
    if points.shape[0] > _MAX_CLOUD_POINTS:
        # Uniform stride, not a head slice: the returns arrive ordered by
        # azimuth, so taking the first N would publish a narrow wedge of the
        # scan instead of the whole field of view.
        stride = int(np.ceil(points.shape[0] / _MAX_CLOUD_POINTS))
        points = points[::stride]
    return points


def pub_robo_data_ros2(robot_type, num_envs, base_node, env, annotator_lst, start_time):
    # IsaacLab 4.5 / Isaac Sim 6.0 expose articulation buffers as warp arrays,
    # which do not support Python item indexing. Convert each buffer to numpy
    # once at the source so the publish helpers can index and iterate it.
    robot_data = env.unwrapped.scene["robot"].data
    joint_pos = _to_numpy(robot_data.joint_pos)
    root_state = _to_numpy(robot_data.root_state_w)
    lin_vel_b = _to_numpy(robot_data.root_lin_vel_b)
    ang_vel_b = _to_numpy(robot_data.root_ang_vel_b)
    for i in range(num_envs):
        base_node.publish_joints(robot_data.joint_names, joint_pos[i], i)
        base_node.publish_odom(root_state[i, :3], root_state[i, 3:7], i)
        base_node.publish_imu(root_state[i, 3:7], lin_vel_b[i, :], ang_vel_b[i, :], i)

        if robot_type == "go2":
            net_forces = _to_numpy(env.unwrapped.scene["contact_forces"].data.net_forces_w)
            base_node.publish_robot_state([
                net_forces[i][4][2],
                net_forces[i][8][2],
                net_forces[i][14][2],
                net_forces[i][18][2],
            ], i)

        try:
            # `start_time` is passed by value from the caller's loop and the
            # original code's `start_time = time.time()` below only rebound the
            # local name, so the caller kept handing back its original value and
            # this branch was taken on every single iteration. With a 200k-point
            # cloud that meant a full scipy rotation plus a 2.4 MB message build
            # per physics step. Keep the cadence in module state so 20 Hz means
            # 20 Hz.
            global _LAST_LIDAR_PUB
            if (time.time() - _LAST_LIDAR_PUB) > 1 / 20:
                for j in range(num_envs):
                    points = _scan_points(annotator_lst, j)
                    if points is None:
                        continue
                    point_cloud = update_meshes_for_cloud2(
                        points, root_state[j, :3], root_state[j, 3:7]
                    )
                    base_node.publish_lidar(point_cloud, j)
                _LAST_LIDAR_PUB = time.time()
        except Exception as e:
            # Report once. Swallowing this silently is how a permanently dead
            # point cloud stays invisible until something downstream complains.
            global _LIDAR_PUBLISH_FAILED
            if not _LIDAR_PUBLISH_FAILED:
                _LIDAR_PUBLISH_FAILED = True
                print(f"[go2_omniverse] lidar publish FAILED ({type(e).__name__}: {e})", flush=True)


class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__('go2_driver_node')
        qos_profile = QoSProfile(depth=10)

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_pub = []
        self.odom_pub = []
        self.imu_pub = []

        for i in range(num_envs):
            self.joint_pub.append(self.create_publisher(JointState, f'robot{i}/joint_states', qos_profile))
            # foot_force published as Float32MultiArray to avoid go2_interfaces dependency
            self.go2_state_pub.append(self.create_publisher(Float32MultiArray, f'robot{i}/foot_force', qos_profile))
            self.go2_lidar_pub.append(self.create_publisher(PointCloud2, f'robot{i}/point_cloud2', qos_profile))
            self.odom_pub.append(self.create_publisher(Odometry, f'robot{i}/odom', qos_profile))
            self.imu_pub.append(self.create_publisher(Imu, f'robot{i}/imu', qos_profile))
        # Publish TF as tf2_msgs/TFMessage on /tf — avoids tf2_ros dependency
        self.tf_pub = self.create_publisher(TFMessage, '/tf', qos_profile)

    def publish_joints(self, joint_names_lst, joint_state_lst, robot_num):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [f"robot{robot_num}/{n}" for n in joint_names_lst]
        joint_state.position = [float(v.item()) for v in joint_state_lst]
        self.joint_pub[robot_num].publish(joint_state)

    def publish_odom(self, base_pos, base_rot, robot_num):
        odom_trans = TransformStamped()
        odom_trans.header.stamp = self.get_clock().now().to_msg()
        odom_trans.header.frame_id = "odom"
        odom_trans.child_frame_id = f"robot{robot_num}/base_link"
        odom_trans.transform.translation.x = base_pos[0].item()
        odom_trans.transform.translation.y = base_pos[1].item()
        odom_trans.transform.translation.z = base_pos[2].item()
        odom_trans.transform.rotation.x = base_rot[1].item()
        odom_trans.transform.rotation.y = base_rot[2].item()
        odom_trans.transform.rotation.z = base_rot[3].item()
        odom_trans.transform.rotation.w = base_rot[0].item()
        self.tf_pub.publish(TFMessage(transforms=[odom_trans]))

        odom_topic = Odometry()
        odom_topic.header.stamp = self.get_clock().now().to_msg()
        odom_topic.header.frame_id = "odom"
        odom_topic.child_frame_id = f"robot{robot_num}/base_link"
        odom_topic.pose.pose.position.x = base_pos[0].item()
        odom_topic.pose.pose.position.y = base_pos[1].item()
        odom_topic.pose.pose.position.z = base_pos[2].item()
        odom_topic.pose.pose.orientation.x = base_rot[1].item()
        odom_topic.pose.pose.orientation.y = base_rot[2].item()
        odom_topic.pose.pose.orientation.z = base_rot[3].item()
        odom_topic.pose.pose.orientation.w = base_rot[0].item()
        self.odom_pub[robot_num].publish(odom_topic)

    def publish_imu(self, base_rot, base_lin_vel, base_ang_vel, robot_num):
        imu_trans = Imu()
        imu_trans.header.stamp = self.get_clock().now().to_msg()
        imu_trans.header.frame_id = f"robot{robot_num}/base_link"
        imu_trans.linear_acceleration.x = base_lin_vel[0].item()
        imu_trans.linear_acceleration.y = base_lin_vel[1].item()
        imu_trans.linear_acceleration.z = base_lin_vel[2].item()
        imu_trans.angular_velocity.x = base_ang_vel[0].item()
        imu_trans.angular_velocity.y = base_ang_vel[1].item()
        imu_trans.angular_velocity.z = base_ang_vel[2].item()
        imu_trans.orientation.x = base_rot[1].item()
        imu_trans.orientation.y = base_rot[2].item()
        imu_trans.orientation.z = base_rot[3].item()
        imu_trans.orientation.w = base_rot[0].item()
        self.imu_pub[robot_num].publish(imu_trans)

    def publish_robot_state(self, foot_force_lst, robot_num):
        msg = Float32MultiArray()
        msg.data = [float(v.item()) for v in foot_force_lst]
        self.go2_state_pub[robot_num].publish(msg)

    def publish_lidar(self, points, robot_num):
        header = Header(frame_id="odom")
        header.stamp = self.get_clock().now().to_msg()
        self.go2_lidar_pub[robot_num].publish(_create_point_cloud2(header, points))
