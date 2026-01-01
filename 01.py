#!/usr/bin/env python
# coding: utf-8

# ## Setup the env from scratch

# In[1]:


# gymnasium_chrome_dino/__init__.py
import gymnasium as gym

gym.register(
    id='ChromeDino-v0',
    entry_point='gymnasium_chrome_dino.envs:ChromeDinoEnv',
    kwargs={'render': True, 'accelerate': False, 'autoscale': False}
)

gym.register(
    id='ChromeDinoNoBrowser-v0',
    entry_point='gymnasium_chrome_dino.envs:ChromeDinoEnv',
    kwargs={'render': False, 'accelerate': False, 'autoscale': False}
)


# In[2]:


# gymnasium_chrome_dino/envs.py
import base64
import io
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium import spaces

class ChromeDinoEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, render, accelerate, autoscale):
        self.game = DinoGame(render, accelerate, autoscale)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(150, 600, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)  # NOOP, UP, DOWN, SPACE
        self.gametime_reward = 0.1
        self.gameover_penalty = -1
        self.current_frame = np.zeros((150, 600, 3), dtype=np.uint8)
        self._action_set = [0, 1, 2, 3]

    def _observe(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)
        i = rgba2rgb(i)
        a = np.array(i)
        self.current_frame = a
        return self.current_frame

    def step(self, action):
        if action == 1:  # UP
            self.game.press_up()
        elif action == 2:  # DOWN
            self.game.press_down()
        elif action == 3:  # SPACE
            self.game.press_space()
        # action == 0 is NOOP

        observation = self._observe()
        reward = self.gametime_reward
        terminated = False
        truncated = False
        info = {}

        if self.game.is_crashed():
            reward = self.gameover_penalty
            terminated = True

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restart()
        return self._observe(), {}

    def render(self):
        return self.current_frame

    def close(self):
        self.game.close()

    def get_score(self):
        return self.game.get_score()

    def set_acceleration(self, enable):
        if enable:
            self.game.restore_parameter('config.ACCELERATION')
        else:
            self.game.set_parameter('config.ACCELERATION', 0)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

ACTION_MEANING = {
    0 : "NOOP",
    1 : "UP",
    2 : "DOWN",
    3 : "SPACE",
}


# In[3]:


# gymnasium_chrome_dino/game.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class DinoGame():
    def __init__(self, render=False, accelerate=False, autoscale=False):
        options = Options()
        options.add_argument('--disable-infobars')
        options.add_argument('--mute-audio')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=800,600')
        if not render:
            options.add_argument('--headless=new')

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

        self.driver.get('https://elvisyjlin.github.io/t-rex-runner/')
        self.defaults = self.get_parameters()  # default parameters
        if not accelerate:
            self.set_parameter('config.ACCELERATION', 0)
        if not autoscale:
            self.driver.execute_script('Runner.instance_.setArcadeModeContainerScale = function(){};')
        self.press_space()

    def get_parameters(self):
        params = {}
        params['config.ACCELERATION'] = self.driver.execute_script('return Runner.config.ACCELERATION;')
        return params

    def is_crashed(self):
        return self.driver.execute_script('return Runner.instance_.crashed;')

    def is_inverted(self):
        return self.driver.execute_script('return Runner.instance_.inverted;')

    def is_paused(self):
        return self.driver.execute_script('return Runner.instance_.paused;')

    def is_playing(self):
        return self.driver.execute_script('return Runner.instance_.playing;')

    def press_space(self):
        return self.driver.find_element('tag name', 'body').send_keys(Keys.SPACE)

    def press_up(self):
        return self.driver.find_element('tag name', 'body').send_keys(Keys.UP)

    def press_down(self):
        return self.driver.find_element('tag name', 'body').send_keys(Keys.DOWN)

    def pause(self):
        return self.driver.execute_script('Runner.instance_.stop();')

    def resume(self):
        return self.driver.execute_script('Runner.instance_.play();')

    def restart(self):
        return self.driver.execute_script('Runner.instance_.restart();')

    def close(self):
        self.driver.quit()

    def get_score(self):
        digits = self.driver.execute_script('return Runner.instance_.distanceMeter.digits;');
        return int(''.join(digits))

    def get_canvas(self):
        return self.driver.execute_script('return document.getElementsByClassName("runner-canvas")[0].toDataURL().substring(22);')

    def set_parameter(self, key, value):
        self.driver.execute_script('Runner.{} = {};'.format(key, value))

    def restore_parameter(self, key):
        self.set_parameter(key, self.defaults[key])


# In[4]:


# gymnasium_chrome_dino/utils/helpers.py
def rgba2rgb(im):
    bg = Image.new("RGB", im.size, (255, 255, 255))  # fill background as white color
    bg.paste(im, mask=im.split()[3])  # 3 is the alpha channel
    return bg

import time
class Timer():
    def __init__(self):
        self.t0 = time.time()
    def tick(self):
        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1
        return dt


# In[5]:


# gymnasium_chrome_dino/utils/wrappers.py
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

cv2.ocl.setUseOpenCL(False)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class TimerEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.timer = Timer()

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.timer.tick()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['timedelta'] = self.timer.tick()
        return obs, reward, terminated, truncated, info

def make_dino(env, timer=True, frame_stack=True):
    env = WarpFrame(env, 160, 80)
    if timer:
        env = TimerEnv(env)
    if frame_stack:
        env = gym.wrappers.FrameStack(env, 4)
    return env


# In[6]:


env = gym.make('ChromeDino-v0', render=True, accelerate=False, autoscale=False)
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
env.close()


# In[6]:





# In[ ]:




