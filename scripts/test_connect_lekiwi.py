import time

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import busy_wait

FPS = 30

robot_config = LeKiwiClientConfig(remote_ip="127.0.0.1", id="my_lekiwi", has_arm=False)
robot = LeKiwiClient(robot_config)

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")
else:
    print("Robot is connected!")


stop_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

try:
    while True:
        t0 = time.perf_counter()

        # observation = robot.get_observation()

        # action = policy(observation)
        # base_action = robot._from_bi_wheel_to_base_action(action)

        # robot.send_action(base_action)

        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
finally:
    if robot.is_connected:
        robot.send_action(stop_action)
        robot.disconnect()

# robot.disconnect()