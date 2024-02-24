import gymnasium as gym
import matplotlib.pyplot as plt
from pynput import keyboard
# from pynput.keyboard import Key
import threading

current_action = 0
action_lock = threading.Lock()
def on_press(key):
    global current_action
    with action_lock:
        try:
            if key.char == 'a':
                print('space')
        except AttributeError:
            if key == keyboard.Key.up:
                current_action = 2
            elif key == keyboard.Key.down:
                current_action = 5
            elif key == keyboard.Key.left:
                current_action = 4
            elif key == keyboard.Key.right:
                current_action = 3
            elif key == keyboard.Key.space:
                current_action = 13


def listen_for_keys():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    env = gym.make('ALE/Seaquest-v5', render_mode="human")
    # env = gym.make('ALE/Breakout-v5')
    # keyboard.hook(on_arrow_key)
    # with keyboard.Listener(on_press=on_arrow_key) as listener:
    #     listener.join()
    listener_thread = threading.Thread(target=listen_for_keys)
    listener_thread.start()


    observation, info = env.reset(seed=42)

    # print(type(observation))
    # print(observation.shape)
    # print(info)
    global current_action
    # plt.ion()
    for _ in range(1000):
        # action = env.action_space.sample()
        with action_lock:
            action = current_action
            current_action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        if abs(reward) > 1e-9:
            print(reward)
            print(info)
            print(observation.shape)
            print(reward)
            print(type(reward))
        # exit(0)
        
        # plt.imshow(observation)
        # plt.show()    
        # plt.pause(.05)
        if terminated or truncated:
            observation, info = env.reset()
    # plt.ioff()
    env.close()

if __name__ == '__main__':
    main()