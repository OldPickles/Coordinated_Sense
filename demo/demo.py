import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

import argparse
import time

class CustomEnv(gym.Env):
    """Custom Environment that can run in both command-line and graphical modes"""
    metadata = {'render.modes': ['human', 'None']}

    def __init__(self, render_mode="None", width=800, height=600):
        """
        Initialize the environment
        :param render_mode: "human" for graphical mode, "None" for headless mode
        :param width: width of the graphical window
        :param height: height of the graphical window
        """
        super(CustomEnv, self).__init__()
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(3)  # 0: 减速, 1: 保持, 2: 加速
        self.observation_space = spaces.Box(low=np.array([0, -1], dtype=np.float64),
                                            high=np.array([100, 1], dtype=np.float64)
                                            , dtype=np.float64)

        # 初始化状态
        self.position = 0
        self.velocity = 0
        self.goal_position = 100

        # 初始化图形界面（如果需要）
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Custom Environment")
            self.clock = pygame.time.Clock()

    def reset(self):
        """Reset the environment to its initial state"""
        self.position = 0
        self.velocity = 0
        return np.array([self.position, self.velocity])

    def step(self, action):
        """Take an action and update the environment"""
        if action == 0:  # 减速
            self.velocity -= 0.5
        elif action == 1:  # 保持
            self.velocity = self.velocity
        elif action == 2:  # 加速
            self.velocity += 0.5

        self.position += self.velocity
        self.velocity = np.clip(self.velocity, -1, 1)
        self.position = np.clip(self.position, 0, 100)

        done = bool(self.position >= self.goal_position)
        reward = -1.0
        if done:
            reward = 100.0

        return np.array([self.position, self.velocity]), reward, done, {}

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Custom Environment")
                self.clock = pygame.time.Clock()

            self.screen.fill((255, 255, 255))
            # 绘制轨道
            pygame.draw.line(self.screen, (0, 0, 0), (100, 500), (700, 500), 5)
            # 绘制目标点
            pygame.draw.circle(self.screen, (0, 255, 0), (int(100 + (self.goal_position / 100) * 600), 500), 10)
            # 绘制小车
            car_x = 100 + (self.position / 100) * 600
            pygame.draw.rect(self.screen, (255, 0, 0), (car_x - 10, 490, 20, 20))
            pygame.display.flip()
            self.clock.tick(30)
        elif mode == 'None':
            pass

    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.quit()

# 测试环境
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--render_mode", type=str, default="None", help="渲染模式")

    # 在本地PC上运行（图形化模式）
    env = CustomEnv(render_mode=parse.parse_args().render_mode)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break
    env.close()

    # 在服务器上运行（无头模式）
    env = CustomEnv(render_mode="None")
    env.reset()
    for _ in range(1000):
        print(f"position: {env.position}, velocity: {env.velocity}")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break
    env.close()