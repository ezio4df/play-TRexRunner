import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
from pygame.locals import *

# Import game classes from the original code
# You may need to adjust based on how the original repo structures classes
from dino import Dino
from obstacle_manager import ObstacleManager
from background import Background
from score import Score

# Initialize pygame (but not display if headless)
pygame.init()
pygame.font.init()

class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.window_width = 800
        self.window_height = 400

        # Action space: 0 = do nothing, 1 = jump, 2 = duck
        self.action_space = spaces.Discrete(3)

        # Observation: [dino_y, dino_vel_y, nearest_obstacle_dist, obstacle_height, game_speed, is_ducking]
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0, 5, 0], dtype=np.float32),
            high=np.array([self.window_height, 20, self.window_width, 200, 30, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Game objects
        self.dino = None
        self.obstacles = None
        self.background = None
        self.score = None
        self.game_speed = 0
        self.done = False
        self.clock = pygame.time.Clock()

        # For rendering
        self.window = None
        self.clock = pygame.time.Clock()

    def _get_obs(self):
        nearest = self.obstacles.get_nearest_obstacle()
        return np.array([
            self.dino.rect.y,
            self.dino.vel_y,
            nearest.x - self.dino.rect.x if nearest else self.window_width,
            nearest.height if nearest else 0,
            self.game_speed,
            float(self.dino.is_ducking)
        ], dtype=np.float32)

    def _get_info(self):
        return {"score": self.score.get_score()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_speed = 8
        self.done = False

        self.dino = Dino()
        self.obstacles = ObstacleManager()
        self.background = Background()
        self.score = Score()

        # Optional: advance a few frames to avoid immediate collision
        for _ in range(10):
            self.obstacles.update(self.game_speed)
            self.background.update(self.game_speed)

        if self.render_mode == "human" and self.window is None:
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Dino RL")

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Handle action
        if action == 1:  # jump
            self.dino.jump()
        elif action == 2:  # duck
            self.dino.duck(True)
        else:
            self.dino.duck(False)

        # Update game state
        self.dino.update()
        self.obstacles.update(self.game_speed)
        self.background.update(self.game_speed)
        self.score.update()

        # Collision check
        collision = self.obstacles.check_collision(self.dino.rect)
        self.done = collision

        reward = 1.0  # +1 per frame survived
        if self.done:
            reward = -100

        # Increase difficulty
        if self.score.get_score() % 100 == 0 and self.score.get_score() > 0:
            self.game_speed += 0.5

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, self.done, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # 'human' mode is handled in step()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            return

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))  # white background

        self.background.draw(canvas)
        self.dino.draw(canvas)
        self.obstacles.draw(canvas)
        self.score.draw(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()