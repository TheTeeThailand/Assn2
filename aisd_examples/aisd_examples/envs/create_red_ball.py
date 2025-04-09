import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import math

class RedBall(Node):
    def __init__(self):
        super().__init__('redball')
        self.br = CvBridge()
        self.redball_position = 320
        self.create3_is_stopped = True

        self.create_subscription(Image, '/target_redball', self.image_callback, 10)
        self.create_subscription(Bool, '/stop_status', self.stop_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg)
        self.redball_position = frame.shape[1] // 2

    def stop_callback(self, msg):
        self.create3_is_stopped = msg.data

    def step(self, action):
        twist = Twist()
        twist.angular.z = (action - 320) / 320 * math.pi/2
        self.cmd_pub.publish(twist)
        self.create3_is_stopped = False

class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        rclpy.init()
        self.redball = RedBall()
        self.observation_space = spaces.Discrete(640)
        self.action_space = spaces.Discrete(640)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self.redball.redball_position, {}

    def step(self, action):
        self.redball.step(action)
        rclpy.spin_once(self.redball)

        while not self.redball.create3_is_stopped:
            rclpy.spin_once(self.redball)

        obs = self.redball.redball_position
        reward = -abs(obs - 320)
        self.step_count += 1
        done = self.step_count >= 100
        return obs, reward, done, False, {}

    def render(self):
        pass

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()

