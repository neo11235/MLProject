import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make('ALE/Seaquest-v5')

observation, info = env.reset(seed=42)

print(type(observation))
print(observation.shape)
print(info)
plt.ion()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    plt.imshow(observation)
    plt.show()    
    plt.pause(.2)
    if terminated or truncated:
        observation, info = env.reset()
plt.ioff()
env.close()
