import collections
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

class DQNSeaquest(gym.Wrapper):
    def __init__(self, render_mode="rgb_array", repeat=4, device = 'cpu'):
        env = gym.make('ALE/Seaquest-v5', render_mode=render_mode)
        super(DQNSeaquest, self).__init__(env)
        self.repeat = repeat
        self.device = device
        self.lives = env.ale.lives()
        self.frame_buffer = []
        self.image_shape = (84, 84)

    def reset(self):
        self.frame_buffer = [] 
        
        observation, _ = self.env.reset()
        self.lives = self.env.ale.lives()
        observation = self.preprocess(observation)
        return observation

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):    
            observation, reward, done,truncated, info = self.env.step(action)
            total_reward += reward

            current_lives = info['lives']
            # print(info, current_lives)

            if current_lives < self.lives:
                total_reward = total_reward - 1
                self.lives = current_lives

            # print(f"total_reward:{total_reward}, Lives {self.lives}, reward:{reward}")


            self.frame_buffer.append(observation)

            if done:
                break
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.preprocess(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return max_frame, total_reward, done, info
    
    def preprocess(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert('L')
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0
        img = img.to(self.device)
        return img
