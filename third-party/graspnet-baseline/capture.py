import pyrealsense2 as rs
import numpy as np
import cv2
import os
import scipy.io as scio
from PIL import Image

# 创建保存数据的文件夹
SAVE_DIR = "./doc/my_grasp_data"
os.makedirs(SAVE_DIR, exist_ok=True)

print("正在初始化 RealSense D435 相机...")

# 1. 配置数据流 (强制 1280x720，帧率 6fps 以匹配硬件极限)
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)

# 启动相机
profile = pipeline.start(config)

# 2. 获取并保存相机内参
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# 构造成 3x3 的内参矩阵
intrinsics_matrix = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])

# 写入 meta.mat (模拟官方 GraspNet 格式)
meta_data = {
    'intrinsic_matrix': intrinsics_matrix,
    'factor_depth': np.array([[1000.0]]) # D435 深度比例通常为 1000
}
scio.savemat(os.path.join(SAVE_DIR, 'meta.mat'), meta_data)
print(f"相机内参已保存至: {SAVE_DIR}/meta.mat")

# 自动生成 1280x720 的全白 workspace_mask.png
mask = np.ones((720, 1280), dtype=np.uint8) * 255
cv2.imwrite(os.path.join(SAVE_DIR, 'workspace_mask.png'), mask)
print(f"掩码图已生成至: {SAVE_DIR}/workspace_mask.png")

# 3. 创建对齐对象 (将深度图严格对齐到彩色图的视角)
align_to = rs.stream.color
align = rs.align(align_to)

print("\n[操作说明]")
print(" - 按下 [空格键] 保存当前画面的 彩色图 和 深度图")
print(" - 每次按下空格，都会覆盖当前的 color.png 和 depth.png")
print(" - 按下 [Q] 键 或 [ESC] 键退出程序\n")

try:
    while True:
        # 等待一对连贯的帧
        frames = pipeline.wait_for_frames()
        
        # 将深度帧对齐到彩色帧
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 转换为 numpy 数组 (此时是原生的 RGB 格式)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image_rgb = np.asanyarray(color_frame.get_data())

        # 【修复颜色】转成 BGR 格式专门给 OpenCV 显示用，防止变成“阿凡达”
        color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

        # 将深度图转为伪彩色图以便人类观看
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 左右拼接彩色图和伪彩色深度图显示
        images_to_show = np.hstack((color_image_bgr, depth_colormap))
        
        # 为了方便预览，将显示窗口缩小一半 (实际保存的照片依然是 1280x720)
        images_to_show = cv2.resize(images_to_show, (1280, 360))
            
        cv2.imshow('RealSense D435 (Space: Save | Q: Quit)', images_to_show)

        # 监听键盘按键
        key = cv2.waitKey(1) & 0xFF

        # 按下空格键 (ASCII码 32)
        if key == 32:
            color_path = os.path.join(SAVE_DIR, 'color.png')
            depth_path = os.path.join(SAVE_DIR, 'depth.png')
            
            # 【保存彩色图】直接保存原生的 RGB 数据，保证交给 demo.py 的颜色是正确的
            Image.fromarray(color_image_rgb).save(color_path)
            
            # 【保存深度图】必须保存为 16位无损 PNG，否则会丢失真实的距离信息
            cv2.imwrite(depth_path, depth_image.astype(np.uint16))
            
            print(f"✅ 成功截取并覆盖 1280x720 高清图片 -> {SAVE_DIR} 目录")
            print("你现在可以去运行: python demo.py --checkpoint_path logs/log_rs/checkpoint-rs.tar")
            
        # 按下 'q' 或 ESC 键退出
        elif key == ord('q') or key == 27:
            print("退出采集程序...")
            break

finally:
    # 停止数据流并关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()
