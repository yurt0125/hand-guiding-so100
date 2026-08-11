#!/usr/bin/env python3
import sys
import time
from typing import Set

import numpy as np
from pynput import keyboard

from common.common_control import CommonRunner
from common.config import DEFAULT_CONTROL_GAIN, DEFAULT_FILTER_BACKEND, DEFAULT_USB_PORT, KEYBOARD_WORKSPACE
from common.common_robot import EndEffectorTarget, SO100Hardware, SO100Kinematics


def _normalize_axis(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return a / norm


class KeyboardSource:
    def __init__(self):
        self.pressed_keys: Set[str] = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.pos_sensitivity = 0.002
        self.ori_sensitivity = 0.02
        self.gripper_sensitivity = 5.0
        self.target = None

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char.lower())
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass

    def seed_from_target(self, target: EndEffectorTarget):
        self.target = EndEffectorTarget(**target.__dict__)

    def update(self, kin, current_qpos_rad: np.ndarray) -> EndEffectorTarget:
        if self.target is None:
            raise RuntimeError('KeyboardSource not seeded')

        # Position: local-frame increments
        x_axis = y_axis = z_axis = 0.0
        if 'w' in self.pressed_keys:
            y_axis = -1.0
        if 's' in self.pressed_keys:
            y_axis = 1.0
        if 'a' in self.pressed_keys:
            x_axis = -1.0
        if 'd' in self.pressed_keys:
            x_axis = 1.0
        if 'q' in self.pressed_keys:
            z_axis = -1.0
        if 'e' in self.pressed_keys:
            z_axis = 1.0

        delta_local = np.array([x_axis, y_axis, z_axis], dtype=float) * self.pos_sensitivity
        R = kin.fk(current_qpos_rad)[:3, :3]
        delta_world = R @ delta_local
        self.target.x += float(delta_world[0])
        self.target.y += float(delta_world[1])
        self.target.z += float(delta_world[2])

        # Orientation: rotate tool axis around world axes
        # Small-angle Rodrigues: v_new = v + delta * (u x v)
        axis = self.target.tool_axis()
        delta_axis = np.zeros(3)
        if 'i' in self.pressed_keys:  # pitch up: rotate around world Y
            delta_axis += np.cross([0, 1, 0], axis) * self.ori_sensitivity
        if 'k' in self.pressed_keys:  # pitch down
            delta_axis -= np.cross([0, 1, 0], axis) * self.ori_sensitivity
        if 'j' in self.pressed_keys:  # roll left: rotate around world X
            delta_axis -= np.cross([1, 0, 0], axis) * self.ori_sensitivity
        if 'l' in self.pressed_keys:  # roll right
            delta_axis += np.cross([1, 0, 0], axis) * self.ori_sensitivity

        if np.linalg.norm(delta_axis) > 1e-12:
            axis = _normalize_axis(axis + delta_axis)
            self.target.tool_axis_x = float(axis[0])
            self.target.tool_axis_y = float(axis[1])
            self.target.tool_axis_z = float(axis[2])

        # Gripper
        if 'c' in self.pressed_keys:
            self.target.gripper += self.gripper_sensitivity
        if 'v' in self.pressed_keys:
            self.target.gripper -= self.gripper_sensitivity
        self.target.gripper = float(np.clip(self.target.gripper, 0.0, 100.0))
        self.target.source = 'keyboard'
        self.target.timestamp = time.time()
        return EndEffectorTarget(**self.target.__dict__)

    def cleanup(self):
        self.listener.stop()


def main():
    port = input(f'请输入 USB 端口 (回车默认 {DEFAULT_USB_PORT}): ').strip() or DEFAULT_USB_PORT
    kin = SO100Kinematics()
    hardware = SO100Hardware(port=port, use_degrees=False)
    hardware.connect()
    runner = CommonRunner(
        kin=kin,
        hardware=hardware,
        workspace=KEYBOARD_WORKSPACE,
        kp=DEFAULT_CONTROL_GAIN,
        filter_backend=DEFAULT_FILTER_BACKEND,
    )
    init_target = runner.initialize_from_robot()
    source = KeyboardSource()
    source.seed_from_target(init_target)

    print('\n[键盘操作说明 - 局部坐标系]')
    print('W/S: 前后 | A/D: 左右 | Q/E: 上下')
    print('I/K: pitch (俯仰) | J/L: roll (侧倾)')
    print('C/V: gripper')

    try:
        while True:
            current_q = hardware.get_qpos_rad(kin.arm.model.nq)
            raw_target = source.update(kin, current_q)
            result = runner.step(raw_target)
            axis = result.safe_target.tool_axis()
            print(
                f"\rRAW[{result.raw_target.x:.3f},{result.raw_target.y:.3f},{result.raw_target.z:.3f}] "
                f"SAFE[{result.safe_target.x:.3f},{result.safe_target.y:.3f},{result.safe_target.z:.3f}] "
                f"AXIS[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}] "
                f"GRIP[{result.safe_target.gripper:.1f}] IK[{result.ik_ok}]",
                end='', flush=True
            )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print('\n退出')
    finally:
        source.cleanup()
        hardware.disconnect()
        sys.exit(0)


if __name__ == '__main__':
    main()
