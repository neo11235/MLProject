import os
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

from breakout import *
from model import *
from agent import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

environment = DQNBreakout(device=device) #, render_mode="human")

action_space = environment.action_space.n
model = Atarinet(nb_actions=action_space)
model.to(device)
model.load()

agent = Agent(model, 
            device=device,
            epsilon=0.10, 
            min_epsilon=0.10, 
            nb_warmup=1000, 
            nb_action=action_space, 
            memomy_capacity=4000, 
            batch_size=64, 
            learning_rate=0.00001)

agent.train(env = environment, epochs=100000)


# state = environment.reset()

# print(model.forward(state))

# for _ in range(100):
#     action = environment.action_space.sample()
#     next_state, reward, done, info = environment.step(action)
#     print(next_state.shape)


