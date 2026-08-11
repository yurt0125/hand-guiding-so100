import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from pynput import keyboard
import src.pinocchio_kinematic as pinocchio_kinematic

# 配置你的 SO-100 机械臂路径
SCENE_XML_PATH = 'model/trs_so_arm100/scene.xml'
ARM_XML_PATH = 'model/trs_so_arm100/so_arm100.xml'

class KeyboardController:
    """键盘控制器类（适配 5-DOF 机械臂与独立夹爪控制）"""
    
    def __init__(self):
        # 初始平移位置 (修正后的正确坐标)
        self.x = 0.0
        self.y = -0.32
        self.z = 0.25
        
        # 初始姿态
        self.roll = np.pi
        self.pitch = 0.0
        
        # --- 夹爪控制变量 ---
        self.gripper_pos = 0.0
        self.gripper_min = -1.0 
        self.gripper_max = 1.0  
        self.gripper_sensitivity = 0.05
        
        # 限位保护
        self.x_min, self.x_max = -0.4, 0.4
        self.y_min, self.y_max = -0.4, 0
        self.z_min, self.z_max = 0.01, 0.4
        
        self.pos_sensitivity = 0.002
        self.ori_sensitivity = 0.02
        
        # 键盘监听
        self.pressed_keys = set()
        self.listener = None
        self.space_pressed_last_frame = False  # 防空格连按标志位
        self.is_ready = self.init_controller()

    def _matrix_to_rpy(self, R):
        pitch = -np.arcsin(R[2, 0])
        if abs(R[2, 0]) < 0.999999:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        return roll, pitch, yaw

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.pressed_keys.add('space')
        else:
            try: self.pressed_keys.add(key.char.lower())
            except AttributeError: pass

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.pressed_keys.discard('space')
        else:
            try: self.pressed_keys.discard(key.char.lower())
            except AttributeError: pass

    def init_controller(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        print("✅ 5-DOF 键盘遥控已启动（局部坐标系控制模式）。")
        print("[末端平移] W/S: 前后 | A/D: 左右 | Q/E: 上下")
        print("[末端旋转] I/K: Pitch (俯仰) | U/O: Roll (翻滚)")
        print("[夹爪控制] C: 闭合 | V: 张开")
        print("🚀 【新增功能】：按下 [空格键(Space)] 打印当前关节 qpos！")
        return True
        
    def is_connected(self):
        return self.is_ready
        
    def handle_input(self, arm, current_qpos):
        if not self.is_connected(): return

        # --- 空格键打印 qpos 逻辑 (加了边缘检测，按一次只打印一次) ---
        space_pressed_now = 'space' in self.pressed_keys
        if space_pressed_now and not self.space_pressed_last_frame:
            print(f"\n\n==========================================")
            print(f"✅ 记录成功！当前坐标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}")
            print(f"👉 请复制这行数组覆盖代码中的 guess_q:")
            print(f"guess_q = np.array({np.round(current_qpos, 4).tolist()})")
            print(f"==========================================\n")
        self.space_pressed_last_frame = space_pressed_now
        # -------------------------------------------------------------
                    
        # 1. 平移控制 (W/S前后, A/D左右, Q/E上下)
        x_axis, y_axis, z_axis = 0.0, 0.0, 0.0
        # W/S 控制前后 -> 对应局部的 y_axis
        if 'w' in self.pressed_keys: y_axis = -1.0
        if 's' in self.pressed_keys: y_axis = 1.0
        
        # A/D 控制左右 -> 对应局部的 x_axis
        if 'a' in self.pressed_keys: x_axis = -1.0
        if 'd' in self.pressed_keys: x_axis = 1.0
        
        # Q/E 控制上下 -> 对应局部的 z_axis
        if 'q' in self.pressed_keys: z_axis = -1.0
        if 'e' in self.pressed_keys: z_axis = 1.0

        # 【局部 -> 世界】坐标转换
        delta_local = np.array([x_axis, y_axis, z_axis]) * self.pos_sensitivity
        tf_current = arm.fk(current_qpos)
        R_ee = tf_current[:3, :3]
        delta_world = R_ee @ delta_local  # 转换为世界坐标系的增量
        
        self.x = np.clip(self.x + delta_world[0], self.x_min, self.x_max)
        self.y = np.clip(self.y + delta_world[1], self.y_min, self.y_max)
        self.z = np.clip(self.z + delta_world[2], self.z_min, self.z_max)

        # 2. 旋转控制 (I/K Pitch, U/O Roll)
        
        # 数学上的 Roll (绕局部 X 轴) 对应视觉上的 Pitch (上下点头)
        if 'i' in self.pressed_keys: self.roll += self.ori_sensitivity
        if 'k' in self.pressed_keys: self.roll -= self.ori_sensitivity
        
        # 数学上的 Pitch (绕局部 Y 轴) 对应视觉上的 Roll (绕手臂轴线扭转)
        if 'j' in self.pressed_keys: self.pitch -= self.ori_sensitivity
        if 'l' in self.pressed_keys: self.pitch += self.ori_sensitivity
        
        # 3. 夹爪控制
        if 'c' in self.pressed_keys: self.gripper_pos -= self.gripper_sensitivity
        if 'v' in self.pressed_keys: self.gripper_pos += self.gripper_sensitivity
        self.gripper_pos = np.clip(self.gripper_pos, self.gripper_min, self.gripper_max)

    def cleanup(self):
        if self.listener is not None: self.listener.stop()


class SO100TeleopViewer:
    def __init__(self, model, data, controller):
        self.model = model
        self.data = data
        self.controller = controller
        
        self.arm = pinocchio_kinematic.Kinematics("Jaw")
        self.arm.buildFromMJCF(ARM_XML_PATH)
        
        # ========================================================
        # 【步骤3】在这里填入你按空格测出来的 guess_q 数组
        # 如果是全0，机器臂开机就是“朝天指”。
        # ========================================================
        self.guess_q = np.array([0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0])# <--- 替换这里！
        # 将猜想值注入作为求解器的上一帧状态 (Warm Start)
        self.last_dof = np.zeros(self.arm.model.nq)
        copy_len = min(len(self.guess_q), self.arm.model.nq)
        self.last_dof[:copy_len] = self.guess_q[:copy_len]
        
        # 同时将仿真器的物理状态也强制设定过去，防止开机第一帧瞬移拉扯
        self.data.qpos[:copy_len] = self.guess_q[:copy_len]
        mujoco.mj_forward(self.model, self.data)
        # ========================================================

        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.frame_count = 0

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    def build_transform(self, x, y, z, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R = Rz @ Ry @ Rx
        
        tf = np.eye(4)
        tf[:3, :3] = R
        tf[:3, 3] = [x, y, z]
        return tf

    def run_loop(self):
        self.handle.cam.distance = 2.0
        self.handle.cam.azimuth = 90
        self.handle.cam.elevation = -20

        while self.is_running():
            self.frame_count += 1
            mujoco.mj_forward(self.model, self.data)
            
            # 获取当前实际位姿
            current_qpos = self.data.qpos[:self.arm.model.nq]
            tf_actual = self.arm.fk(current_qpos)
            R_actual = tf_actual[:3, :3]
            pos_actual = tf_actual[:3, 3]
            
            # 【核心优势】：提取实际的 Yaw 角
            _, _, actual_yaw = self.controller._matrix_to_rpy(R_actual)

            # 更新控制器输入
            self.controller.handle_input(self.arm, current_qpos)
            x, y, z = self.controller.x, self.controller.y, self.controller.z
            roll, pitch = self.controller.roll, self.controller.pitch

            # 欺骗 IK 求解器：用期望的 XYZ+Roll+Pitch 结合实际的 Yaw
            tf_target = self.build_transform(x, y, z, roll, pitch, actual_yaw)
            
            # 求解 IK
            dof, info = self.arm.ik(tf_target, current_arm_motor_q=self.last_dof)
            self.last_dof = dof

            # 误差监控
            pos_target = tf_target[:3, 3]
            delta_pos = pos_target - pos_actual
            pos_err_norm = np.linalg.norm(delta_pos)

            print(f"\r🎯 坐标: x={x:.3f}, y={y:.3f}, z={z:.3f} | 爪: {self.controller.gripper_pos:.3f}", end="")

            # 【防飞车保护】前60帧让它平滑过渡，之后如果偏离过大就冻结
            if self.frame_count > 60:
                POS_ERR_LIMIT = 0.05
                if pos_err_norm > POS_ERR_LIMIT:
                    self.controller.x, self.controller.y, self.controller.z = pos_actual
                    dof = current_qpos

            # 夹爪独立控制（覆盖第6个轴）
            if len(dof) >= 6:
                dof[5] = self.controller.gripper_pos
                
            self.data.qpos[:6] = dof[:6]
            mujoco.mj_step(self.model, self.data)
            
            self.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    data = mujoco.MjData(model)
    
    controller = KeyboardController()
    
    try:
        viewer = SO100TeleopViewer(model, data, controller)
        viewer.run_loop()
    finally:
        controller.cleanup()
