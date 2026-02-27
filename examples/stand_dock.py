import os
import h5py, time
import numpy as np
from spot_teleop.spot_controller import SpotRobotController
from bosdyn.client.math_helpers  import SE3Pose, Quat

robot_ip = os.environ.get("SPOT_ROBOT_IP", "192.168.1.138")
user     = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")

print(f"Connecting to Spot at {robot_ip} ...")
print(f"user: {user}, password: {password}")

spot = SpotRobotController(robot_ip, user, password)
spot.undock()

time.sleep(3.0)

# spot.reset_pose()  # x,y,z, qx,qy,qz,qw

spot.stow_arm()

time.sleep(3.0)
spot.dock()