import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import numpy as np

import draccus
from spatialmath import SE3

# 显式导入所有相关的机械臂配置
from lerobot.robots import (
    RobotConfig, 
    make_robot_from_config,
    so101_follower,  
    so100_follower,
    koch_follower
)
from lerobot.utils.utils import init_logging
from lerobot.utils.import_utils import register_third_party_plugins

# 导入 lerobot-kinematics 的模型和解算器
try:
    from lerobot_kinematics.lerobot.lerobot_Kinematics import get_robot
    from lerobot_kinematics.IK import IK_LM
except ImportError as e:
    logging.error(f"导入失败: {e}")
    exit(1)

@dataclass
class TargetPoseConfig:
    x: float = 0.18     
    y: float = 0.0      
    z: float = 0.15     
    roll: float = 0.0   
    pitch: float = 1.57 
    yaw: float = 0.0    
    gripper_open: bool = False

@dataclass
class IKControlConfig:
    robot: RobotConfig | None = None
    target: TargetPoseConfig = TargetPoseConfig()

@draccus.wrap()
def move_to_target(cfg: IKControlConfig):
    init_logging()
    logging.info("=========== 接收到的配置参数 ===========")
    logging.info(pformat(asdict(cfg)))

    if not cfg.robot:
        raise ValueError("必须提供 robot 配置")

    device = make_robot_from_config(cfg.robot)
    device.connect(calibrate=False)
    
    try:
        # ==========================================
        # 步骤 2：读取当前真实关节角度
        # ==========================================
        obs_dict = device.get_observation() 
        
        # 仅提取前 4 个空间运动关节
        current_qpos_deg = np.array([
            obs_dict["shoulder_pan.pos"],
            obs_dict["shoulder_lift.pos"],
            obs_dict["elbow_flex.pos"],
            obs_dict["wrist_flex.pos"],
        ])
        
        # 提取第 5 个关节（保持不变）
        current_wrist_roll_deg = obs_dict["wrist_roll.pos"]
        logging.info(f"读取到的机械臂当前角度: {current_qpos_deg}")
        
        # 转换为弧度送入 IK 解算器
        current_qpos_rad = np.deg2rad(current_qpos_deg)

        # ==========================================
        # 步骤 3：获取机械臂模型并运行 IK 解算器
        # ==========================================
        robot_ets = get_robot("so100") 
        Tep = SE3(cfg.target.x, cfg.target.y, cfg.target.z) * SE3.RPY([cfg.target.roll, cfg.target.pitch, cfg.target.yaw], order='xyz')
        
        # mask=[1,1,1,0,0,0] 忽略姿态，只求到达该 xyz 坐标，防止 violating joint limits
        solver = IK_LM(mask=np.array([1, 1, 1, 0, 0, 0]))
        logging.info("正在执行逆运动学求解...")
        
        solution = solver.solve(robot_ets, Tep, q0=current_qpos_rad)
        
        if not solution.success:
            logging.error(f"IK 解算失败，原因: {solution.reason}")
            return
            
        target_qpos_rad = solution.q
        target_qpos_deg = np.rad2deg(target_qpos_rad)
        logging.info(f"IK 解算成功！目标关节角度: {target_qpos_deg}")

        # ==========================================
        # 步骤 4：将计算结果下发给机械臂执行
        # ==========================================
        action = {
            "shoulder_pan.pos": float(target_qpos_deg[0]),
            "shoulder_lift.pos": float(target_qpos_deg[1]),
            "elbow_flex.pos": float(target_qpos_deg[2]),
            "wrist_flex.pos": float(target_qpos_deg[3]),
            "wrist_roll.pos": float(current_wrist_roll_deg), # 保持当前手腕旋转角度
            "gripper.pos": 100.0 if cfg.target.gripper_open else -100.0  # 根据你的标定调整
        }
        
        logging.info("发送指令给机械臂...")
        device.send_action(action)
        
        # 【关键修复】：加上延迟，等待机械臂运动完毕后再断开连接
        logging.info("等待机械臂移动到位 (3秒)...")
        time.sleep(3.0)
        logging.info("移动完成！")

    except Exception as e:
        logging.error(f"发生异常: {e}")
    finally:
        logging.info("断开机械臂连接，释放电机扭力...")
        device.disconnect()

def main():
    register_third_party_plugins()
    move_to_target()

if __name__ == "__main__":
    main()
