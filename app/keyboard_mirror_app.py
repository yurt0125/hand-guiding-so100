#!/usr/bin/env python3
"""SO100 keyboard teleoperation with MuJoCo mirror and hardware sync."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Set

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

from common.src import pinocchio_kinematic
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML_PATH = PROJECT_ROOT / "common" / "model" / "trs_so_arm100" / "scene.xml"
ARM_XML_PATH = PROJECT_ROOT / "common" / "model" / "trs_so_arm100" / "so_arm100.xml"


def _normalize_axis(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return a / norm


class KeyboardController:
    """Keyboard controller adapted for 5-DOF arm with tool-axis direction."""

    def __init__(self) -> None:
        self.home_x = 0.0
        self.home_y = -0.32
        self.home_z = 0.25
        self.home_tool_axis = np.array([0.0, 0.0, -1.0])

        self.x = self.home_x
        self.y = self.home_y
        self.z = self.home_z
        self.tool_axis = self.home_tool_axis.copy()

        # Unified runtime semantic:
        # internal gripper in [0, 1], then convert to LeRobot gripper.pos [0, 100] on send.
        self.gripper_position = 0.0
        self.gripper_min = 0.0
        self.gripper_max = 1.0
        self.gripper_sensitivity = 0.02

        self.x_min, self.x_max = -0.4, 0.4
        self.y_min, self.y_max = -0.4, 0.0
        self.z_min, self.z_max = 0.01, 0.4

        self.position_speed_m_s = 0.20
        self.orientation_speed_rad_s = 2.0
        self.gripper_speed_s = 2.0

        self.pressed_keys: Set[str] = set()
        self.listener: keyboard.Listener | None = None
        self.space_pressed_last_frame = False
        self.r_pressed_last_frame = False
        self.reset_requested = False
        self.is_ready = self.initialize_controller()

    def on_press(self, key: keyboard.KeyCode | keyboard.Key) -> None:
        if key == keyboard.Key.space:
            self.pressed_keys.add("space")
            return
        try:
            self.pressed_keys.add(key.char.lower())
        except AttributeError:
            return

    def on_release(self, key: keyboard.KeyCode | keyboard.Key) -> None:
        if key == keyboard.Key.space:
            self.pressed_keys.discard("space")
            return
        try:
            self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            return

    def initialize_controller(self) -> bool:
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        print("✅ 5-DOF 键盘遥控已启动（局部坐标系控制模式）。")
        print("[末端平移] W/S: 前后 | A/D: 左右 | Q/E: 上下")
        print("[末端旋转] I/K: 绕世界X轴旋转工具轴 | J/L: 绕世界Y轴旋转工具轴")
        print("[夹爪控制] C: 闭合 | V: 张开")
        print("[安全控制] R: 复位到安全位")
        print("🚀 按下 [空格键(Space)] 打印当前关节 qpos。")
        return True

    def handle_input(self, arm, current_qpos: np.ndarray, dt_s: float) -> None:
        if not self.is_ready:
            return

        space_pressed_now = "space" in self.pressed_keys
        if space_pressed_now and not self.space_pressed_last_frame:
            print("\n\n==========================================")
            print(f"✅ 记录成功！当前坐标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}")
            print("👉 请复制这行数组覆盖代码中的 guess_q:")
            print(f"guess_q = np.array({np.round(current_qpos, 4).tolist()})")
            print("==========================================\n")
        self.space_pressed_last_frame = space_pressed_now

        r_pressed_now = "r" in self.pressed_keys
        if r_pressed_now and not self.r_pressed_last_frame:
            self.reset_requested = True
        self.r_pressed_last_frame = r_pressed_now

        x_axis = 0.0
        y_axis = 0.0
        z_axis = 0.0
        if "w" in self.pressed_keys:
            y_axis = -1.0
        if "s" in self.pressed_keys:
            y_axis = 1.0
        if "a" in self.pressed_keys:
            x_axis = -1.0
        if "d" in self.pressed_keys:
            x_axis = 1.0
        if "q" in self.pressed_keys:
            z_axis = -1.0
        if "e" in self.pressed_keys:
            z_axis = 1.0

        dt_s = float(np.clip(dt_s, 1e-3, 0.1))
        delta_local = np.array([x_axis, y_axis, z_axis], dtype=float) * self.position_speed_m_s * dt_s
        rotation_end_effector = arm.fk(current_qpos)[:3, :3]
        delta_world = rotation_end_effector @ delta_local

        self.x = float(np.clip(self.x + delta_world[0], self.x_min, self.x_max))
        self.y = float(np.clip(self.y + delta_world[1], self.y_min, self.y_max))
        self.z = float(np.clip(self.z + delta_world[2], self.z_min, self.z_max))

        # Orientation: rotate tool axis around world axes.
        # Small-angle Rodrigues: v_new ≈ v + δ * (u × v)
        delta_axis = np.zeros(3)
        if "i" in self.pressed_keys:  # rotate around world X (-)
            delta_axis -= np.cross([1, 0, 0], self.tool_axis) * self.orientation_speed_rad_s * dt_s
        if "k" in self.pressed_keys:  # rotate around world X (+)
            delta_axis += np.cross([1, 0, 0], self.tool_axis) * self.orientation_speed_rad_s * dt_s
        if "j" in self.pressed_keys:  # rotate around world Y (+)
            delta_axis += np.cross([0, 1, 0], self.tool_axis) * self.orientation_speed_rad_s * dt_s
        if "l" in self.pressed_keys:  # rotate around world Y (-)
            delta_axis -= np.cross([0, 1, 0], self.tool_axis) * self.orientation_speed_rad_s * dt_s
        if np.linalg.norm(delta_axis) > 1e-12:
            self.tool_axis = _normalize_axis(self.tool_axis + delta_axis)

        if "c" in self.pressed_keys:
            self.gripper_position -= self.gripper_speed_s * dt_s
        if "v" in self.pressed_keys:
            self.gripper_position += self.gripper_speed_s * dt_s
        self.gripper_position = float(
            np.clip(self.gripper_position, self.gripper_min, self.gripper_max)
        )

    def cleanup(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class SO100TeleopViewer:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: KeyboardController,
        real_robot: SO100Follower | None = None,
        enable_viewer: bool = True,
        enable_status_output: bool = True,
        status_use_carriage_return: bool = True,
        control_hz: float = 100.0,
        profile_control: bool = False,
    ) -> None:
        self.model = model
        self.data = data
        self.controller = controller
        self.real_robot = real_robot

        self.arm = pinocchio_kinematic.Kinematics("JawOffset")
        self.arm.buildFromMJCF(str(ARM_XML_PATH))

        self.guess_q = np.array([0.0474, -3.0230, 2.8877, 0.7562, -1.5186, -0.0], dtype=float)
        self.last_dof = np.zeros(self.arm.model.nq, dtype=float)
        copy_length = min(len(self.guess_q), self.arm.model.nq)
        self.last_dof[:copy_length] = self.guess_q[:copy_length]
        self.data.qpos[:copy_length] = self.guess_q[:copy_length]
        mujoco.mj_forward(self.model, self.data)

        self.enable_viewer = enable_viewer
        self.enable_status_output = enable_status_output
        self.status_use_carriage_return = status_use_carriage_return
        self.control_period_s = 1.0 / max(float(control_hz), 1.0)
        self.profile_control = profile_control
        self.handle = mujoco.viewer.launch_passive(self.model, self.data) if self.enable_viewer else None
        self.frame_count = 0
        self.reset_hold_until = 0.0
        self.post_reset_soft_start_steps = 0
        self.stop_requested = False

        # Optional debug marker: visualize JawOffset site in MuJoCo viewer.
        self.jaw_offset_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "JawOffset"
        )
        if self.jaw_offset_site_id < 0:
            print("[WARN] site 'JawOffset' not found; EE marker disabled.")

    def _update_jawoffset_marker(self) -> None:
        if self.handle is None:
            return
        scene = self.handle.user_scn
        scene.ngeom = 0
        if self.jaw_offset_site_id < 0:
            return

        pos = self.data.site_xpos[self.jaw_offset_site_id].copy()

        # Marker 1: red sphere at JawOffset site.
        mat_identity = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            scene.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.006, 0.0, 0.0], dtype=np.float64),
            pos=pos,
            mat=mat_identity,
            rgba=np.array([1.0, 0.15, 0.15, 0.95], dtype=np.float32),
        )

        # Marker 2: cyan arrow showing current tool-axis direction.
        axis = self.controller.tool_axis.astype(np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-9:
            axis = axis / axis_norm
        else:
            axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Build orientation matrix whose x-axis aligns with arrow direction.
        x_axis = axis
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(x_axis, up))) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_axis = np.cross(up, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
        mat_arrow = np.column_stack([x_axis, y_axis, z_axis]).reshape(-1)

        mujoco.mjv_initGeom(
            scene.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.005, 0.010, 0.16], dtype=np.float64),
            pos=pos,
            mat=mat_arrow,
            rgba=np.array([0.1, 0.9, 1.0, 0.95], dtype=np.float32),
        )

        scene.ngeom = 2

    def run_loop(self) -> None:
        if self.handle is not None:
            self.handle.cam.distance = 2.0
            self.handle.cam.azimuth = 90
            self.handle.cam.elevation = -20

        try:
            last_status_print_time = 0.0
            last_loop_time = time.perf_counter()
            profile_last_print_time = last_loop_time
            profile_count = 0
            profile_loop_sum = 0.0
            profile_ik_sum = 0.0
            profile_send_sum = 0.0
            profile_loop_max = 0.0
            profile_ik_max = 0.0
            profile_send_max = 0.0
            status_interval_s = 0.2  # 5 Hz
            while True:
                loop_start_time = time.perf_counter()
                dt_s = loop_start_time - last_loop_time
                last_loop_time = loop_start_time
                if self.stop_requested:
                    break
                if self.handle is not None and not self.handle.is_running():
                    break
                self.frame_count += 1
                mujoco.mj_forward(self.model, self.data)

                current_qpos = self.data.qpos[: self.arm.model.nq]
                transform_actual = self.arm.fk(current_qpos)
                position_actual = transform_actual[:3, 3]

                self.controller.handle_input(self.arm, current_qpos, dt_s)
                if self.controller.reset_requested:
                    self.perform_safe_reset(
                        reason="按键复位",
                        settle_seconds=1.0,
                    )
                    self.controller.reset_requested = False
                    self.controller.pressed_keys.clear()
                    self.controller.space_pressed_last_frame = False
                    self.controller.r_pressed_last_frame = False
                    # 复位后直接进入下一控制周期，避免本帧使用旧位姿回写目标造成“弹回”
                    time.sleep(self.control_period_s)
                    continue

                if time.time() < self.reset_hold_until:
                    if self.handle is not None:
                        self._update_jawoffset_marker()
                        self.handle.sync()
                    time.sleep(self.control_period_s)
                    continue

                # 5DOF IK: position + tool axis direction
                p_target = np.array([self.controller.x, self.controller.y, self.controller.z])
                axis_target = self.controller.tool_axis

                ik_start_time = time.perf_counter()
                dof, _info = self.arm.ik(p_target, axis_target, current_arm_motor_q=self.last_dof)
                ik_dt = time.perf_counter() - ik_start_time
                if self.post_reset_soft_start_steps > 0:
                    max_joint_step = 0.04
                    dof = self.last_dof + np.clip(dof - self.last_dof, -max_joint_step, max_joint_step)
                    self.post_reset_soft_start_steps -= 1
                self.last_dof = dof

                position_error_norm = np.linalg.norm(p_target - position_actual)
                now = time.time()
                if self.enable_status_output and (now - last_status_print_time >= status_interval_s):
                    ax = self.controller.tool_axis
                    status_text = (
                        f"XYZ[{self.controller.x:.3f},{self.controller.y:.3f},{self.controller.z:.3f}] | "
                        f"AXIS[{ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f}] | "
                        f"GRIP[{self.controller.gripper_position:.2f}]"
                    )
                    if self.status_use_carriage_return:
                        print(f"\r{status_text:<96}", end="", flush=True)
                    else:
                        print(status_text, flush=True)
                    last_status_print_time = now

                if self.frame_count > 60 and position_error_norm > 0.05:
                    self.controller.x, self.controller.y, self.controller.z = (
                        float(position_actual[0]),
                        float(position_actual[1]),
                        float(position_actual[2]),
                    )
                    dof = current_qpos

                if len(dof) >= 6:
                    dof[5] = self.controller.gripper_position

                self.data.qpos[:6] = dof[:6]
                mujoco.mj_step(self.model, self.data)

                if self.real_robot is not None:
                    simulation_degrees = [math.degrees(value) for value in dof[:5]]
                    real_pan = simulation_degrees[0] * -1.0 + 3.0
                    real_lift = simulation_degrees[1] * 1.0 + 90.0
                    real_elbow = simulation_degrees[2] * 1.0 - 90.0
                    real_wrist = simulation_degrees[3] * 1.0 + 0.0
                    real_roll = simulation_degrees[4] * 1.0 + 90.0
                    action_dict = {
                        "shoulder_pan.pos": real_pan,
                        "shoulder_lift.pos": real_lift,
                        "elbow_flex.pos": real_elbow,
                        "wrist_flex.pos": real_wrist,
                        "wrist_roll.pos": real_roll,
                        "gripper.pos": float(np.clip(self.controller.gripper_position * 100.0, 0.0, 100.0)),
                    }
                    send_start_time = time.perf_counter()
                    self.real_robot.send_action(action_dict)
                    send_dt = time.perf_counter() - send_start_time
                else:
                    send_dt = 0.0

                if self.handle is not None:
                    self._update_jawoffset_marker()
                    self.handle.sync()
                elapsed_s = time.perf_counter() - loop_start_time
                time.sleep(max(0.0, self.control_period_s - elapsed_s))
                if self.profile_control:
                    loop_dt = time.perf_counter() - loop_start_time
                    profile_count += 1
                    profile_loop_sum += loop_dt
                    profile_ik_sum += ik_dt
                    profile_send_sum += send_dt
                    profile_loop_max = max(profile_loop_max, loop_dt)
                    profile_ik_max = max(profile_ik_max, ik_dt)
                    profile_send_max = max(profile_send_max, send_dt)
                    now_perf = time.perf_counter()
                    if now_perf - profile_last_print_time >= 1.0:
                        print(
                            "\n[PROFILE] "
                            f"loop avg/max={profile_loop_sum/profile_count*1000:.1f}/{profile_loop_max*1000:.1f}ms | "
                            f"ik avg/max={profile_ik_sum/profile_count*1000:.1f}/{profile_ik_max*1000:.1f}ms | "
                            f"send avg/max={profile_send_sum/profile_count*1000:.1f}/{profile_send_max*1000:.1f}ms",
                            flush=True,
                        )
                        profile_last_print_time = now_perf
                        profile_count = 0
                        profile_loop_sum = 0.0
                        profile_ik_sum = 0.0
                        profile_send_sum = 0.0
                        profile_loop_max = 0.0
                        profile_ik_max = 0.0
                        profile_send_max = 0.0
        except KeyboardInterrupt:
            print("\n\n🛑 收到 Ctrl+C，执行安全复位后退出...")
            self.perform_safe_reset(reason="Ctrl+C 退出保护", settle_seconds=1.0)
        finally:
            if self.handle is not None and self.handle.is_running():
                self.handle.close()

    def build_real_action_from_dof(self, dof: np.ndarray) -> dict[str, float]:
        simulation_degrees = [math.degrees(value) for value in dof[:5]]
        real_pan = simulation_degrees[0] * -1.0 + 3.0
        real_lift = simulation_degrees[1] * 1.0 + 90.0
        real_elbow = simulation_degrees[2] * 1.0 - 90.0
        real_wrist = simulation_degrees[3] * 1.0 + 0.0
        real_roll = simulation_degrees[4] * 1.0 + 90.0
        return {
            "shoulder_pan.pos": real_pan,
            "shoulder_lift.pos": real_lift,
            "elbow_flex.pos": real_elbow,
            "wrist_flex.pos": real_wrist,
            "wrist_roll.pos": real_roll,
            "gripper.pos": float(np.clip(self.controller.gripper_position * 100.0, 0.0, 100.0)),
        }

    def perform_safe_reset(self, reason: str, settle_seconds: float) -> None:
        print(f"\n安全复位：{reason}")
        reset_transform = self.arm.fk(self.guess_q[: self.arm.model.nq])
        reset_position = reset_transform[:3, 3]
        reset_tool_axis = _normalize_axis(reset_transform[:3, 2])

        self.controller.x = float(reset_position[0])
        self.controller.y = float(reset_position[1])
        self.controller.z = float(reset_position[2])
        self.controller.tool_axis = reset_tool_axis.copy()
        self.controller.gripper_position = 0.0

        self.last_dof = self.guess_q.copy()
        self.data.qpos[:6] = self.guess_q[:6]
        mujoco.mj_forward(self.model, self.data)

        if self.real_robot is not None:
            safe_action = self.build_real_action_from_dof(self.guess_q)
            repeat_count = max(1, int(settle_seconds / 0.02))
            for _ in range(repeat_count):
                self.real_robot.send_action(safe_action)
                time.sleep(0.02)

        self.reset_hold_until = time.time() + 0.4
        self.post_reset_soft_start_steps = 25
        print("✅ 已复位到安全姿态。")


def run_session(
    use_real_robot: bool,
    enable_viewer: bool,
    enable_status_output: bool,
    status_use_carriage_return: bool,
    port: str | None = None,
    control_hz: float = 100.0,
    profile_control: bool = False,
) -> None:
    print("=" * 50)
    if use_real_robot and enable_viewer:
        print("🤖 SO-100 仿真与实机镜像同步版")
    elif use_real_robot and not enable_viewer:
        print("🤖 SO-100 真机键盘控制版（无仿真窗口）")
    else:
        print("🧪 SO-100 仿真键盘控制版（不连接真机）")
    print("=" * 50)

    real_robot: SO100Follower | None = None
    if use_real_robot:
        resolved_port = (port or "").strip() or "/dev/ttyACM0"
        robot_config = SO100FollowerConfig(port=resolved_port, id="single_arm", use_degrees=True)
        real_robot = SO100Follower(robot_config)
        real_robot.connect()
        print(f"✅ 真实机械臂连接成功！port={resolved_port}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    data = mujoco.MjData(model)
    controller = KeyboardController()

    viewer: SO100TeleopViewer | None = None
    try:
        viewer = SO100TeleopViewer(
            model,
            data,
            controller,
            real_robot=real_robot,
            enable_viewer=enable_viewer,
            enable_status_output=enable_status_output,
            status_use_carriage_return=status_use_carriage_return,
            control_hz=control_hz,
            profile_control=profile_control,
        )
        viewer.run_loop()
    finally:
        if viewer is not None:
            viewer.perform_safe_reset(reason="程序退出前保护", settle_seconds=0.8)
        controller.cleanup()
        if real_robot is not None:
            real_robot.disconnect()
            print("\n🛑 机械臂已安全断开，程序完全退出。")
        else:
            print("\n🛑 仿真程序已退出。")
        sys.exit(0)


def main(port: str | None = None) -> None:
    run_session(
        use_real_robot=True,
        enable_viewer=True,
        enable_status_output=True,
        status_use_carriage_return=True,
        port=port,
    )


def main_sim_only() -> None:
    run_session(
        use_real_robot=False,
        enable_viewer=True,
        enable_status_output=True,
        status_use_carriage_return=True,
    )


def main_real_only(port: str | None = None) -> None:
    run_session(
        use_real_robot=True,
        enable_viewer=False,
        enable_status_output=True,
        status_use_carriage_return=False,
        port=port,
    )


if __name__ == "__main__":
    main()
