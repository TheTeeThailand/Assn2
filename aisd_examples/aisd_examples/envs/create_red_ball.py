import gymnasium as gym
from gymnasium import spaces
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import math


class RedBall(Node):
    """
    A Node to analyse red balls in images and publish the results
    """
    def __init__(self):
        super().__init__('redball')

        self.subscription = self.create_subscription(
            Image,
            '/custom_ns/camera1/image_raw',
            self.listener_callback,
            10
        )

        self.br = CvBridge()
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.last_x = None

    def listener_callback(self, msg):
        print("📸 Received image")

        frame = self.br.imgmsg_to_cv2(msg)
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask for red (two ranges)
        lower_red1 = (0, 100, 100)
        upper_red1 = (10, 255, 255)
        mask1 = cv2.inRange(hsv_conv_img, lower_red1, upper_red1)

        lower_red2 = (160, 100, 100)
        upper_red2 = (180, 255, 255)
        mask2 = cv2.inRange(hsv_conv_img, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Blur and morph
        blurred = cv2.GaussianBlur(red_mask, (9, 9), 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            ((x, y), radius) = cv2.minEnclosingCircle(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-5) if perimeter > 0 else 0

            print(f"🔍 radius={radius:.1f}, area={area:.1f}, circularity={circularity:.2f}, x={int(x)}, y={int(y)}")

            if radius > 5 and area > 150 and circularity > 0.7 and 20 < x < 620 and 20 < y < 460:
                center = (int(x), int(y))
                circled = cv2.circle(frame, center, int(radius), (0, 255, 0), 3)

                self.last_x = int(x)
                self.target_publisher.publish(self.br.cv2_to_imgmsg(circled))
                print(f"🎯 Ball detected at: x={int(x)}, y={int(y)}, radius={int(radius)}")
                return

        self.last_x = None
        print("❌ No red ball in this frame")
        self.get_logger().info('no ball detected')


class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Discrete(640)
        self.action_space = spaces.Discrete(3)  # 0 = left, 1 = right, 2 = no move
        self.step_count = 0

        rclpy.init()
        self.redball = RedBall()
        self.cmd_pub = self.redball.create_publisher(Twist, '/cmd_vel', 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.redball.last_x = None
        rclpy.spin_once(self.redball, timeout_sec=0.5)
        return 0, {}  # Default state if no ball

    def step(self, action):
        self.step_count += 1

        twist = Twist()
        if action == 0:
            twist.angular.z = 0.5   # rotate left
        elif action == 1:
            twist.angular.z = -0.5  # rotate right
        else:
            twist.angular.z = 0.0   # stay

        self.cmd_pub.publish(twist)

        rclpy.spin_once(self.redball, timeout_sec=0.5)
        print(f"[STEP {self.step_count}]")

        if self.redball.last_x is not None:
            obs = int(self.redball.last_x)
            reward = 1.0 - abs(obs - 320) / 320.0
            print(f"🎯 RL obs={obs}, reward={reward:.2f}")
        else:
            obs = 0
            reward = -1.0
            print("❌ No red ball detected for RL")

        terminated = self.step_count >= 100
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()

