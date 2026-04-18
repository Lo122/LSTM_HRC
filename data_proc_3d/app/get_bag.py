import pyrealsense2 as rs
import numpy as np
import cv2

# ==== 输入 ====
bag_file = r"C:\Users\loy49\Documents\realsense\20260415_101942.bag"   # 你的文件路径
pixel_x = 320           # 想查询的像素点 u
pixel_y = 240           # 想查询的像素点 v

# ==== 创建 pipeline ====
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)

# 开启深度流
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

profile = pipeline.start(config)

device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)
playback.set_repeat(True) 

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # ✅ 安全判断
        if not depth_frame:
            print("No depth frame")
            continue

        # if not color_frame:
        #     print("No color frame")
        #     continue
        depth_value = depth_frame.get_distance(pixel_x, pixel_y)
        # ✅ 只有存在才用
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        print(f"Depth value at pixel ({pixel_x}, {pixel_y}): {depth_value}")

        # cv2.imshow("depth", depth_image)
        cv2.imshow("color", color_image)
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()