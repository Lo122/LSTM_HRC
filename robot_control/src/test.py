import time


from rtde_control import RTDEControlInterface as RTDEControl


# Connect to the robot
# Replace '169.254.130.206' with your robot's actual IP address
robot_ip = "169.254.130.206"
rtde_c = RTDEControl(robot_ip)

print(rtde_c.getRobotStatus)

# Example: Move the robot to a specific joint configuration
# Joint values are in radians [q1, q2, q3, q4, q5, q6]


# Disconnect when finished


# Enable manual movement
rtde_c.teachMode()


print("Robot is now in teach mode. Move it manually.")
time.sleep(10)

init_time = time.time()
print("ending teach mode")
rtde_c.endTeachMode()
end_time = time.time()-init_time

print(end_time)
print("Teach mode ended.")


rtde_c.disconnect()
