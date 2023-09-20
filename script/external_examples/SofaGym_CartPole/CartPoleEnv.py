"""
The first step is to create a custom gym environment for the SOFA scene we want to simulate and train. 
For in-depth documentation on this, you can check the gymnasium tutorial for creating custom environments.
"""

import os
import math
import numpy as np

from gym import spaces

from sofagym import AbstractEnv 
from sofagym.rpc_server import start_scene

class CartPoleEnv(AbstractEnv):
    """
    All gym environments are defined as classes that must inherit from the base gym.Env class and override the necessary methods. 
    In SofaGym, our AbstractEnv class inherits from gym.Env to act as the base class for creating the SOFA gym environments. 
    To create a new environment from a SOFA scene, you need to create a class for your environment that inherits from the AbstractEnv class. 
    For this, we will create a new file CartPoleEnv.py.
    
    Sub-class of AbstractEnv, dedicated to the cartpole scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    """
    After creating the new environment class, the default configuration for the SOFA scene should be defined as a dictionary. 
    The minimum required data that must be defined can be found in AbstractEnv documentation. 
    Other config data can be added if needed, according to the scene.
    """
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "CartPole",
                      "deterministic": True,
                      "source": [0, 0, 160],
                      "target": [0, 0, 0],
                      "goalList": [[0]],
                      "start_node": None,
                      "scale_factor": 10,
                      "dt": 0.001,
                      "timer_limit": 80,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/CartPole",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "init_x": 0,
                      "max_move": 24,
                      }
    
    def __init__(self, config = None):
        """
        Here we initialize any necessary variables and assign their values based on the previously defined config. 
        For the CartPole env, we define the maximum value cart of the cart's motion on the x-axis as x_threshold. 
        We also define the maximum allowed angle for the pole to tilt before it's considered to be falling.
        We must also define the type of actions and observations the gym environment will use by defining the action_space and observation_space. In the CartPole case, we can only control the cart to move either left or right, so we have only two actions that we define as Discrete(2). The observation space is defined as Box() which consists of 4 continuous values representing the cart x position, the cart velocity, the pole angle, and the pole angular velocity. The minimum and maximum values for the observations are also defined according to the allowed thresholds for the positions and velocities of the cart and pole.
        """
        super().__init__(config)

        self.x_threshold = 100
        self.theta_threshold_radians = self.config["max_move"] * math.pi / 180
        self.config.update({'max_angle': self.theta_threshold_radians})
        
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        nb_actions = 2
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        self.observation_space = spaces.Box(high, -high, dtype=np.float32)

    """
    The second part is to override the step and reset methods from the AbstractEnv class. For the step method, no additions are required. 
    For the reset method, we need to restart the scene using start_scene and return the first observation. 
    It might also be needed to update some parameters or configs like the randomized goal position in the case of some environments.
    """
    def step(self, action):
        return super().step(action)

    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()

        self.config.update({'goalPos': self.goal})

        obs = start_scene(self.config, self.nb_actions)
        
        return np.array(obs['observation'])