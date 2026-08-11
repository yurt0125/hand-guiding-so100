import pyrealsense2 as rs

ctx = rs.context()
for i in range(len(ctx.devices)):
    dev = ctx.devices[i]
    print(f"找到设备: {dev.get_info(rs.camera_info.name)}")
    
    # 获取所有的传感器 (彩色和深度)
    sensors = dev.query_sensors()
    for sensor in sensors:
        print(f"\n--- 传感器: {sensor.get_info(rs.camera_info.name)} ---")
        # 列出该传感器所有支持的流格式
        for profile in sensor.get_stream_profiles():
            # 我们只关心视频流
            if profile.is_video_stream_profile():
                v_profile = profile.as_video_stream_profile()
                # 过滤出 1280x720 或者 1280x800 的大分辨率
                if v_profile.width() >= 1280:
                    print(f"流类型: {v_profile.stream_type()}, 格式: {v_profile.format()}, 分辨率: {v_profile.width()}x{v_profile.height()}, 帧率: {v_profile.fps()}fps")
